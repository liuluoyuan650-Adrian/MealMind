from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


class SentenceTransformerEmbedder:
    """对 sentence-transformers 做一层薄封装，便于测试和替换本地模型。"""

    def __init__(self, model_name_or_path: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "缺少 sentence-transformers，请先执行 pip install -r requirements.txt"
            ) from exc

        self.model_name_or_path = model_name_or_path
        model_path = Path(model_name_or_path).expanduser()
        if model_path.exists():
            # 显式本地目录不应触发任何 Hugging Face 网络请求。
            self.model = SentenceTransformer(str(model_path), local_files_only=True)
            return

        try:
            # 已下载过的仓库模型优先从缓存加载。新版 Transformers 即使权重已缓存，
            # 默认仍可能联网检查 adapter_config.json，离线运行时会因此失败。
            self.model = SentenceTransformer(model_name_or_path, local_files_only=True)
        except (OSError, ValueError):
            try:
                # 本地没有完整缓存时，首次加载才允许 sentence-transformers 下载模型。
                self.model = SentenceTransformer(model_name_or_path)
            except Exception as exc:
                raise RuntimeError(
                    f"无法加载 Embedding 模型 {model_name_or_path}。"
                    "首次运行需要连接 Hugging Face；也可以下载模型后，将 "
                    "MEALMIND_EMBEDDING_MODEL 设置为本地模型目录。"
                ) from exc

    def encode_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype="float32")

    def encode_query(self, query: str) -> np.ndarray:
        # BGE 中文模型建议为检索查询增加 instruction；其他模型保持原始查询。
        if "bge" in self.model_name_or_path.lower():
            query = f"为这个句子生成表示以用于检索相关文章：{query}"
        vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vector, dtype="float32")
