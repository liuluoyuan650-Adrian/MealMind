from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.data_ingestion import build_faiss_index


def main() -> None:
    manifest = build_faiss_index()
    print("RAG 索引构建完成：")
    for key, value in manifest.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
