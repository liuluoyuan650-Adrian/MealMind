from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

# DeepSeek 配置：不要把真实 API Key 写进代码；请通过环境变量 DEEPSEEK_API_KEY 设置。
DEEPSEEK_API_KEY = "sk-dd48fb56abf24e9a988128721b8920df"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-pro"
DEEPSEEK_TIMEOUT_SECONDS = 20


@dataclass(frozen=True)
class RAGSettings:
    """RAG 相关路径和模型配置，均可通过环境变量覆盖。"""

    embedding_model: str
    dish_data_path: Path
    index_dir: Path
    batch_size: int = 32

    @classmethod
    def load(cls) -> "RAGSettings":
        return cls(
            embedding_model=os.getenv(
                "MEALMIND_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
            ).strip(),
            dish_data_path=Path(
                os.getenv("MEALMIND_DISH_DATA", str(ROOT / "data" / "dish_dataset.csv"))
            ),
            index_dir=Path(
                os.getenv("MEALMIND_RAG_INDEX_DIR", str(ROOT / "data" / "faiss_index"))
            ),
            batch_size=int(os.getenv("MEALMIND_EMBEDDING_BATCH_SIZE", "32")),
        )
