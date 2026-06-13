from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def get_button(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


class StreamlitAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["MEALMIND_UNKNOWN_LOG_PATH"] = str(
            Path(self.temp_dir.name) / "unknown_phrases.jsonl"
        )

    def tearDown(self) -> None:
        os.environ.pop("MEALMIND_UNKNOWN_LOG_PATH", None)
        self.temp_dir.cleanup()

    def make_app(self) -> AppTest:
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertFalse(app.exception)
        return app

    def test_normal_recommendation_flow(self) -> None:
        app = self.make_app()
        app.text_area[0].set_value("今天想吃点不罪恶的，但又要能吃饱")
        get_button(app, "开始推荐").click()
        app.run(timeout=30)

        self.assertFalse(app.exception)
        self.assertIn("2. 推荐菜品", [header.value for header in app.header])
        self.assertTrue(any("鸡胸肉蔬菜饭" in item.value for item in app.subheader))

    def test_clarification_flow(self) -> None:
        app = self.make_app()
        app.text_area[0].set_value("来点量子口味的")
        get_button(app, "开始推荐").click()
        app.run(timeout=30)

        self.assertFalse(app.exception)
        self.assertIn("2. 需求澄清", [header.value for header in app.header])
        self.assertTrue(app.warning)
        app.selectbox[0].select(app.selectbox[0].options[0])
        get_button(app, "确认需求并继续推荐").click()
        app.run(timeout=30)

        self.assertFalse(app.exception)
        self.assertIn("2. 推荐菜品", [header.value for header in app.header])


if __name__ == "__main__":
    unittest.main()
