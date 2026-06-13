from __future__ import annotations

from dataclasses import dataclass

from src.rag.llm import LLMClient, get_default_llm_client
from src.rag.prompts import SYSTEM_PROMPT, build_user_prompt
from src.rag.retriever import FaissDishRetriever, RetrievedDish


@dataclass(frozen=True)
class RAGRecommendationResult:
    query: str
    answer: str
    retrieved_dishes: list[RetrievedDish]

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "retrieved_dishes": [dish.to_dict() for dish in self.retrieved_dishes],
        }


class RAGRecommendationService:
    """串联 Retriever、Prompt 和 DeepSeek，提供单一的业务调用入口。"""

    def __init__(self, retriever: FaissDishRetriever, llm_client: LLMClient) -> None:
        self.retriever = retriever
        self.llm_client = llm_client

    @classmethod
    def from_defaults(cls) -> "RAGRecommendationService":
        return cls(FaissDishRetriever.from_settings(), get_default_llm_client())

    def recommend(self, query: str, top_k: int = 5) -> RAGRecommendationResult:
        query = query.strip()
        if not query:
            raise ValueError("query 不能为空")
        if not self.llm_client.is_configured:
            raise RuntimeError(
                "DeepSeek API 未配置，请设置 DEEPSEEK_API_KEY 或 MEALMIND_LLM_API_KEY"
            )

        dishes = self.retriever.retrieve(query, top_k=top_k)
        if not dishes:
            return RAGRecommendationResult(
                query=query,
                answer="没有检索到满足当前忌口和偏好条件的本地菜品，请适当放宽条件。",
                retrieved_dishes=[],
            )

        answer = self.llm_client.chat(
            SYSTEM_PROMPT,
            build_user_prompt(query, dishes),
            max_tokens=900,
            temperature=0.3,
        )
        if not answer:
            detail = self.llm_client.last_error or "未知错误"
            raise RuntimeError(f"DeepSeek 生成推荐失败：{detail}")
        return RAGRecommendationResult(query=query, answer=answer, retrieved_dishes=dishes)


def recommend_from_query(query: str, top_k: int = 5) -> dict:
    """最简函数接口：接收自然语言 query，返回推荐话术及检索证据。"""

    return RAGRecommendationService.from_defaults().recommend(query, top_k).to_dict()
