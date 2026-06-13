from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

# DeepSeek 配置：可以直接把新生成的 API Key 填在这里。
# 注意：不要把含真实密钥的代码上传到 GitHub 或发给其他人。
DEEPSEEK_API_KEY = "sk-b25ba03174bb42f19ecbd23d9ef9614c"
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
