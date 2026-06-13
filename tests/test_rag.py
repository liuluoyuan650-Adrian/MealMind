from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from src.rag.api import app
from src.rag.data_ingestion import build_dish_text
from src.rag.prompts import SYSTEM_PROMPT, build_user_prompt
from src.rag.retriever import RetrievedDish, extract_negative_preferences, violates_exclusions


class RAGTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = {
            "name": "麻辣豆腐",
            "category": "热菜",
            "price": 20,
            "taste": "麻辣 咸香",
            "spicy": 2,
            "calorie_kcal": 380,
            "calorie_level": "中",
            "satiety": 3,
            "scene": "晚餐",
            "temperature": "热",
            "ingredients": "豆腐 辣椒",
            "avoid_tags": "辣椒",
            "rating": 4.5,
            "tags": "热食 麻辣",
        }

    def test_build_dish_text_contains_searchable_fields(self) -> None:
        text = build_dish_text(self.metadata)
        self.assertIn("菜名：麻辣豆腐", text)
        self.assertIn("主要食材：豆腐 辣椒", text)
        self.assertIn("价格：20 元", text)

    def test_negative_spicy_preference_is_hard_filtered(self) -> None:
        exclusions = extract_negative_preferences("今天不想吃辣，来点热乎的")
        self.assertEqual(exclusions, {"辣"})
        self.assertTrue(violates_exclusions(self.metadata, exclusions))

    def test_non_spicy_tag_does_not_count_as_spicy(self) -> None:
        metadata = dict(
            self.metadata,
            name="清炖豆腐",
            spicy=0,
            taste="清淡",
            ingredients="豆腐",
            avoid_tags="无",
            tags="热食 不辣",
        )
        self.assertFalse(violates_exclusions(metadata, {"辣"}))

    def test_prompt_limits_generation_to_retrieved_context(self) -> None:
        dish = RetrievedDish("dish-1", 0.9, build_dish_text(self.metadata), self.metadata)
        prompt = build_user_prompt("不要辣", [dish])
        self.assertIn("程序识别出的必须排除项", prompt)
        self.assertIn('"辣"', prompt)
        self.assertIn("不得推荐上下文之外的菜名", SYSTEM_PROMPT)

    def test_home_page_is_user_facing_recommendation_ui(self) -> None:
        response = TestClient(app).get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("今天想吃什么", response.text)
        self.assertIn("帮我推荐", response.text)

    def test_frontend_assets_are_available(self) -> None:
        client = TestClient(app)
        self.assertEqual(client.get("/static/styles.css").status_code, 200)
        self.assertEqual(client.get("/static/app.js").status_code, 200)


if __name__ == "__main__":
    unittest.main()
