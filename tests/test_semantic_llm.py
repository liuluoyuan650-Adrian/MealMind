from __future__ import annotations

import unittest

from src.nlp import IntentClassifier, ParsedNeed, parse_need
from src.recommender import recommend


class FakeLLMClient:
    is_configured = True

    def interpret_need(self, text: str) -> dict:
        return {
            "intent": "健康餐",
            "taste_tags": ["清淡"],
            "health_tags": ["低热量", "低脂"],
            "scene_tags": ["减脂"],
            "temperature": None,
            "avoid_tags": ["油炸"],
            "staple_preferences": [],
            "ingredient_preferences": [],
            "budget": None,
            "people_count": None,
            "confidence": 0.92,
            "needs_clarification": False,
            "clarification_question": "",
            "clarification_options": [],
        }

    def generate_recommendation_reasons(self, need: dict, dishes: list[dict]) -> dict[str, str]:
        return {
            dish["name"]: f"{dish['name']}与当前口味和健康需求匹配，分量信息也便于比较。"
            for dish in dishes
        }


class SemanticAndLLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.classifier = IntentClassifier()
        cls.classifier.fit()

    def test_unknown_phrase_is_mapped_by_semantic_layer(self) -> None:
        parsed = parse_need(
            "今天想吃点不罪恶的，但又要能吃饱",
            self.classifier,
            record_unknown=False,
        )
        self.assertIn("不罪恶", parsed.unknown_expressions)
        self.assertTrue({"低热量", "低脂"} & set(parsed.health))
        self.assertIn("饱腹", parsed.scenes)
        self.assertFalse(parsed.needs_clarification)
        self.assertEqual(parsed.understanding_source, "规则+语义向量")

    def test_llm_fallback_structures_low_confidence_phrase(self) -> None:
        parsed = parse_need(
            "想吃点赛博养生的",
            self.classifier,
            llm_client=FakeLLMClient(),
            record_unknown=False,
        )
        self.assertTrue(parsed.llm_fallback_used)
        self.assertIn("低热量", parsed.health)
        self.assertIn("油炸", parsed.avoid)
        self.assertFalse(parsed.needs_clarification)

    def test_unresolved_phrase_requests_clarification_without_api(self) -> None:
        parsed = parse_need(
            "来点量子口味的",
            self.classifier,
            record_unknown=False,
        )
        self.assertTrue(parsed.needs_clarification)
        self.assertTrue(parsed.clarification_options)

    def test_known_budget_and_scene_do_not_trigger_clarification(self) -> None:
        cases = (
            "我今天想吃清淡一点的，预算 30 以内，不要辣。",
            "我们三个人吃饭，一个人不吃辣，人均 50 以内。",
            "晚上有点饿，想吃夜宵，不要太贵，最好热乎一点。",
        )
        for text in cases:
            with self.subTest(text=text):
                parsed = parse_need(text, self.classifier, record_unknown=False)
                self.assertFalse(parsed.needs_clarification)
                self.assertFalse(parsed.unknown_expressions)

    def test_llm_generates_reason_for_each_selected_dish(self) -> None:
        need = ParsedNeed(raw_text="清淡一点", tastes=["清淡"])
        results = recommend(need, top_k=3, llm_client=FakeLLMClient())
        self.assertEqual(len(results), 3)
        self.assertTrue((results["recommend_reason_source"] == "大模型生成").all())
        self.assertTrue(results["recommend_reason"].str.contains("健康需求匹配").all())


if __name__ == "__main__":
    unittest.main()
