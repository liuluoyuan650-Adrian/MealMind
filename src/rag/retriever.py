from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.rag.config import RAGSettings
from src.rag.embedding import SentenceTransformerEmbedder


NEGATIVE_ALIASES = {
    "辣": ("辣", "辣椒", "麻辣", "香辣"),
    "香菜": ("香菜",),
    "海鲜": ("海鲜", "虾", "鱼", "蟹", "贝类"),
    "牛肉": ("牛肉",),
    "乳制品": ("乳制品", "牛奶", "奶油", "芝士", "奶酪"),
    "面条": ("面条", "面食"),
    "米饭": ("米饭", "饭类"),
    "沙拉": ("沙拉",),
    "油炸": ("油炸", "炸物"),
}
NEGATIVE_PREFIX = r"(?:不想吃|不想要|不爱吃|不喜欢|不能吃|不要|不吃|别吃|避免|忌口|拒绝)"


def split_tags(value: str) -> set[str]:
    """将 CSV 中以空格、逗号或斜杠分隔的标签转换为集合。"""

    if not value:
        return set()
    normalized = str(value).replace("/", " ").replace(",", " ").replace("，", " ")
    return {item.strip() for item in normalized.split() if item.strip()}


@dataclass(frozen=True)
class RetrievedDish:
    document_id: str
    score: float
    text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "score": round(self.score, 4),
            "text": self.text,
            "metadata": self.metadata,
        }


def extract_negative_preferences(query: str) -> set[str]:
    """提取显式否定条件；这些条件不能只依赖向量相似度判断。"""

    exclusions: set[str] = set()
    if re.search(r"(?:不辣|免辣|零辣)", query):
        exclusions.add("辣")
    for label, aliases in NEGATIVE_ALIASES.items():
        for alias in aliases:
            pattern = rf"{NEGATIVE_PREFIX}\s*(?:太|很|特别)?\s*{re.escape(alias)}"
            if re.search(pattern, query):
                exclusions.add(label)
                break
    return exclusions


def violates_exclusions(metadata: dict[str, Any], exclusions: set[str]) -> bool:
    """在向量召回后执行硬过滤，确保否定偏好不会被相似词反向命中。"""

    if not exclusions:
        return False
    fields = (
        "name",
        "category",
        "taste",
        "ingredients",
        "avoid_tags",
        "tags",
    )
    tokens: set[str] = set()
    for field in fields:
        tokens |= split_tags(str(metadata.get(field, "")))

    if "辣" in exclusions:
        if int(metadata.get("spicy", 0)) > 0:
            return True
        if any("辣" in token and token != "不辣" for token in tokens):
            return True

    for label in exclusions - {"辣"}:
        aliases = NEGATIVE_ALIASES[label]
        if any(alias in token or token in alias for alias in aliases for token in tokens):
            return True
    return False


class FaissDishRetriever:
    """加载本地 FAISS 索引，完成查询向量化、召回和否定条件过滤。"""

    def __init__(
        self,
        index_dir: Path,
        embedding_model: str | None = None,
        embedder: SentenceTransformerEmbedder | None = None,
    ) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("缺少 faiss-cpu，请先执行 pip install -r requirements.txt") from exc

        manifest_path = index_dir / "manifest.json"
        documents_path = index_dir / "documents.json"
        index_path = index_dir / "dishes.faiss"
        for path in (manifest_path, documents_path, index_path):
            if not path.exists():
                raise FileNotFoundError(f"RAG 索引文件不存在：{path}，请先运行建库脚本")

        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.documents = json.loads(documents_path.read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(index_path))
        model_name = embedding_model or str(self.manifest["embedding_model"])
        self.embedder = embedder or SentenceTransformerEmbedder(model_name)
        if self.index.ntotal != len(self.documents):
            raise RuntimeError("FAISS 索引条数与 metadata 条数不一致，请重新建库")

    @classmethod
    def from_settings(cls, settings: RAGSettings | None = None):
        settings = settings or RAGSettings.load()
        if os.getenv("MEALMIND_FORCE_KEYWORD_RETRIEVER", "").strip() == "1":
            return KeywordDishRetriever.from_settings(settings, reason="forced by environment")
        # 已建好的索引必须使用 manifest 中记录的模型进行查询。
        # MEALMIND_EMBEDDING_MODEL 只在重新建库时生效，避免启动 API 时环境变量
        # 与旧索引配置发生漂移。
        try:
            return cls(settings.index_dir)
        except RuntimeError as exc:
            if "Embedding" not in str(exc) and "sentence-transformers" not in str(exc):
                raise
            return KeywordDishRetriever.from_settings(settings, reason=str(exc))

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDish]:
        if not query.strip():
            raise ValueError("query 不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        query_vector = self.embedder.encode_query(query.strip())
        if query_vector.shape[1] != self.index.d:
            raise RuntimeError(
                f"查询向量维度 {query_vector.shape[1]} 与索引维度 {self.index.d} 不一致，"
                "请使用同一 Embedding 模型重新构建索引"
            )
        exclusions = extract_negative_preferences(query)
        # 有硬过滤时扩大召回池，避免前几个结果全部被过滤后不足 Top-K。
        candidate_k = min(self.index.ntotal, max(top_k * 5, top_k))
        scores, indices = self.index.search(query_vector, candidate_k)

        results: list[RetrievedDish] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            document = self.documents[int(index)]
            metadata = document["metadata"]
            if violates_exclusions(metadata, exclusions):
                continue
            results.append(
                RetrievedDish(
                    document_id=document["document_id"],
                    score=float(score),
                    text=document["text"],
                    metadata=metadata,
                )
            )
            if len(results) >= top_k:
                break
        return results


class KeywordDishRetriever:
    """Fallback retriever used when the embedding model cannot be loaded locally."""

    FIELD_WEIGHTS = {
        "name": 5.0,
        "category": 3.0,
        "taste": 2.5,
        "ingredients": 2.0,
        "tags": 2.0,
        "scene": 1.5,
        "temperature": 1.5,
        "calorie_level": 1.0,
    }

    PREFERENCE_HINTS = {
        "热": ("热", "热乎", "带汤", "汤", "下雨", "暖"),
        "汤": ("汤", "带汤", "热乎", "下雨"),
        "不辣": ("不辣", "清淡", "不要辣", "免辣"),
        "辣": ("辣", "麻辣", "香辣"),
        "低卡": ("低卡", "减脂", "健康", "少油"),
        "饱腹": ("饱腹", "管饱", "主食"),
        "便宜": ("便宜", "实惠", "预算"),
    }

    def __init__(self, index_dir: Path, reason: str = "") -> None:
        documents_path = index_dir / "documents.json"
        if not documents_path.exists():
            raise FileNotFoundError(f"RAG metadata file not found: {documents_path}")
        self.documents = json.loads(documents_path.read_text(encoding="utf-8"))
        self.reason = reason

    @classmethod
    def from_settings(
        cls, settings: RAGSettings | None = None, reason: str = ""
    ) -> "KeywordDishRetriever":
        settings = settings or RAGSettings.load()
        return cls(settings.index_dir, reason=reason)

    @staticmethod
    def _contains_any(text: str, aliases: tuple[str, ...]) -> bool:
        return any(alias and alias in text for alias in aliases)

    def _score_document(self, query: str, document: dict[str, Any]) -> float:
        metadata = document["metadata"]
        score = 0.0
        compact_query = query.replace(" ", "")

        for field, weight in self.FIELD_WEIGHTS.items():
            value = str(metadata.get(field, ""))
            if not value:
                continue
            field_tokens = split_tags(value) | {value}
            for token in field_tokens:
                if token and token in compact_query:
                    score += weight
                elif token and len(token) >= 2 and token in document["text"]:
                    common_chars = set(token) & set(compact_query)
                    if len(common_chars) >= min(2, len(set(token))):
                        score += weight * 0.25

        for label, aliases in self.PREFERENCE_HINTS.items():
            if self._contains_any(compact_query, aliases):
                searchable = " ".join(
                    str(metadata.get(field, ""))
                    for field in ("name", "taste", "ingredients", "tags", "scene", "temperature", "calorie_level")
                )
                if label in searchable or any(alias in searchable for alias in aliases):
                    score += 2.0

        rating = float(metadata.get("rating", 0) or 0)
        score += rating * 0.05
        return score

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDish]:
        if not query.strip():
            raise ValueError("query 不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        exclusions = extract_negative_preferences(query)
        ranked: list[tuple[float, dict[str, Any]]] = []
        for document in self.documents:
            metadata = document["metadata"]
            if violates_exclusions(metadata, exclusions):
                continue
            ranked.append((self._score_document(query, document), document))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedDish(
                document_id=document["document_id"],
                score=float(score),
                text=document["text"],
                metadata=document["metadata"],
            )
            for score, document in ranked[:top_k]
        ]
