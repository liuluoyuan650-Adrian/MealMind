from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.rag.config import RAGSettings
from src.rag.embedding import SentenceTransformerEmbedder


REQUIRED_COLUMNS = {
    "name",
    "category",
    "price",
    "taste",
    "spicy",
    "calorie_kcal",
    "calorie_level",
    "satiety",
    "scene",
    "temperature",
    "ingredients",
    "avoid_tags",
    "rating",
    "tags",
}


@dataclass(frozen=True)
class DishDocument:
    """一条可向量化的菜品文本，以及不参与向量计算的原始元数据。"""

    document_id: str
    text: str
    metadata: dict[str, Any]


def _clean_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_dish_text(metadata: dict[str, Any]) -> str:
    """将表格行转换成字段明确的自然语言，降低短标签之间的语义歧义。"""

    spicy_text = "不辣" if int(metadata["spicy"]) == 0 else f"辣度 {metadata['spicy']}"
    description = str(metadata.get("description", "")).strip()
    parts = [
        f"菜名：{metadata['name']}。",
        f"类别：{metadata['category']}。",
        f"口味：{metadata['taste']}，{spicy_text}。",
        f"温度：{metadata['temperature']}。",
        f"主要食材：{metadata['ingredients']}。",
        f"适用场景：{metadata['scene']}。",
        f"特点标签：{metadata['tags']}。",
        f"价格：{metadata['price']} 元。",
        f"热量：{metadata['calorie_kcal']} 千卡，热量等级{metadata['calorie_level']}。",
        f"饱腹感：{metadata['satiety']}/5。",
    ]
    if description:
        parts.insert(2, f"描述：{description}。")
    return "".join(parts)


def load_dish_documents(csv_path: Path) -> list[DishDocument]:
    """读取并校验菜品表，生成稳定 ID、检索文本和结构化 metadata。"""

    data = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"菜品数据缺少必要字段：{', '.join(sorted(missing))}")
    if data["name"].duplicated().any():
        duplicated = data.loc[data["name"].duplicated(), "name"].tolist()
        raise ValueError(f"菜名必须唯一，发现重复项：{duplicated}")

    documents: list[DishDocument] = []
    for index, row in data.iterrows():
        metadata = {column: _clean_value(row[column]) for column in data.columns}
        metadata["price"] = int(float(row["price"]))
        metadata["spicy"] = int(float(row["spicy"]))
        metadata["calorie_kcal"] = int(float(row["calorie_kcal"]))
        metadata["satiety"] = int(float(row["satiety"]))
        metadata["rating"] = float(row["rating"])
        documents.append(
            DishDocument(
                document_id=f"dish-{index + 1:04d}",
                text=build_dish_text(metadata),
                metadata=metadata,
            )
        )
    return documents


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_faiss_index(settings: RAGSettings | None = None) -> dict[str, Any]:
    """离线向量化全部菜品，并把 FAISS 索引和元数据持久化到本地。"""

    settings = settings or RAGSettings.load()
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("缺少 faiss-cpu，请先执行 pip install -r requirements.txt") from exc

    documents = load_dish_documents(settings.dish_data_path)
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    vectors = embedder.encode_documents(
        [document.text for document in documents], batch_size=settings.batch_size
    )
    if len(vectors) != len(documents):
        raise RuntimeError("Embedding 数量与菜品数量不一致，已终止建库")

    # 向量已归一化，IndexFlatIP 的内积等价于余弦相似度。
    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(vectors)

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(settings.index_dir / "dishes.faiss"))
    (settings.index_dir / "documents.json").write_text(
        json.dumps([asdict(item) for item in documents], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "format_version": 1,
        "embedding_model": settings.embedding_model,
        "vector_dimension": int(vectors.shape[1]),
        "document_count": len(documents),
        "source_file": str(settings.dish_data_path),
        "source_sha256": _sha256(settings.dish_data_path),
    }
    (settings.index_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
