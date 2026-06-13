from __future__ import annotations

import os
from dataclasses import dataclass

from src.rag.llm import LLMClient, get_default_llm_client
from src.rag.prompts import SYSTEM_PROMPT, build_user_prompt
from src.rag.retriever import FaissDishRetriever, RetrievedDish


def _clean_answer_style(answer: str) -> str:
    replacements = {
        "“": "",
        "”": "",
        "‘": "",
        "’": "",
        "\"": "",
        "'": "",
        "**": "",
        "__": "",
    }
    for old, new in replacements.items():
        answer = answer.replace(old, new)
    return answer.strip()


def _spicy_text(metadata: dict) -> str:
    spicy = int(metadata.get("spicy", 0) or 0)
    return "不辣" if spicy == 0 else f"辣度 {spicy}/5"


def _fallback_answer(dishes: list[RetrievedDish]) -> str:
    lines = ["根据本地菜单，优先推荐这几道："]
    for index, dish in enumerate(dishes[:3], start=1):
        metadata = dish.metadata
        name = metadata.get("name", "未命名菜品")
        taste = metadata.get("taste", "口味信息未知")
        price = metadata.get("price", "价格未知")
        calorie = metadata.get("calorie_kcal", "热量未知")
        satiety = metadata.get("satiety", "饱腹感未知")
        temperature = metadata.get("temperature", "")
        ingredients = metadata.get("ingredients", "")
        tags = str(metadata.get("tags", ""))
        scene = str(metadata.get("scene", ""))

        opening = f"{temperature}食" if temperature else "这道菜"
        if "汤" in tags or "汤" in str(metadata.get("category", "")) or "汤" in name:
            opening = "热汤" if temperature == "热" else "汤品"

        reasons = [opening, _spicy_text(metadata), f"{taste}口味"]
        if ingredients:
            reasons.append(f"主要食材是 {ingredients}")
        if scene:
            reasons.append(f"适合{scene.replace(' ', '、')}")
        reasons.append(f"热量 {calorie} kcal")
        reasons.append(f"饱腹感 {satiety}/5")
        reason = "，".join(reasons)
        lines.append(f"{index}. {name}：{reason}，价格 {price} 元。")
    return "\n".join(lines)


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

        dishes = self.retriever.retrieve(query, top_k=top_k)
        if not dishes:
            return RAGRecommendationResult(
                query=query,
                answer="没有检索到满足当前忌口和偏好条件的本地菜品，请适当放宽条件。",
                retrieved_dishes=[],
            )

        if os.getenv("MEALMIND_TEMPLATE_ANSWER", "").strip() == "1":
            return RAGRecommendationResult(
                query=query,
                answer=_clean_answer_style(_fallback_answer(dishes)),
                retrieved_dishes=dishes,
            )

        if not self.llm_client.is_configured:
            return RAGRecommendationResult(
                query=query,
                answer=_clean_answer_style(_fallback_answer(dishes)),
                retrieved_dishes=dishes,
            )

        answer = self.llm_client.chat(
            SYSTEM_PROMPT,
            build_user_prompt(query, dishes),
            max_tokens=420,
            temperature=0.3,
        )
        if not answer:
            answer = _fallback_answer(dishes)
        return RAGRecommendationResult(query=query, answer=_clean_answer_style(answer), retrieved_dishes=dishes)


def recommend_from_query(query: str, top_k: int = 5) -> dict:
    """最简函数接口：接收自然语言 query，返回推荐话术及检索证据。"""

    return RAGRecommendationService.from_defaults().recommend(query, top_k).to_dict()
