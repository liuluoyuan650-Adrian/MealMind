from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.rag.service import RAGRecommendationService


app = FastAPI(title="MealMind RAG API", version="1.0.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class RecommendationRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500, description="用户自然语言饮食需求")
    top_k: int = Field(default=5, ge=1, le=20)


class RecommendationResponse(BaseModel):
    query: str
    answer: str
    retrieved_dishes: list[dict]


@lru_cache(maxsize=1)
def get_service() -> RAGRecommendationService:
    # 首次请求时才加载 FAISS 和 Embedding 模型，避免 API 进程导入阶段阻塞。
    return RAGRecommendationService.from_defaults()


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest) -> dict:
    try:
        return get_service().recommend(request.query, request.top_k).to_dict()
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
