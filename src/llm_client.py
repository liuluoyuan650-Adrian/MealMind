from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src import llm_config


@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: int = 20

    @classmethod
    def load(cls) -> "LLMSettings":
        return cls(
            api_key=os.getenv("MEALMIND_LLM_API_KEY", llm_config.LLM_API_KEY).strip(),
            base_url=os.getenv("MEALMIND_LLM_BASE_URL", llm_config.LLM_BASE_URL).strip(),
            model=os.getenv("MEALMIND_LLM_MODEL", llm_config.LLM_MODEL).strip(),
            timeout_seconds=int(
                os.getenv("MEALMIND_LLM_TIMEOUT", str(llm_config.LLM_TIMEOUT_SECONDS))
            ),
        )


def _extract_json(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


class LLMClient:
    """Minimal OpenAI-compatible Chat Completions client.

    Keeping the HTTP layer small avoids forcing the application to install a
    provider-specific SDK. The base URL and model can be replaced for another
    compatible provider.
    """

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or LLMSettings.load()
        self.last_error: str | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.api_key and self.settings.base_url and self.settings.model)

    def _chat_json(self, system_prompt: str, user_prompt: str, max_tokens: int) -> dict[str, Any] | None:
        if not self.is_configured:
            return None

        endpoint = f"{self.settings.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        request = Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            self.last_error = None
            return _extract_json(str(content))
        except (HTTPError, URLError, TimeoutError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def interpret_need(self, text: str) -> dict[str, Any] | None:
        system_prompt = (
            "你是餐饮需求结构化助手。只理解用户需求，不直接推荐菜品。"
            "必须只输出一个 JSON 对象，不要 Markdown，不要补充解释。"
        )
        user_prompt = f"""
请把下面的自然语言需求转换为结构化标签。只能从给定标签中选择；不确定时保持空数组。

允许值：
- taste_tags: 清淡、辣、甜、酸甜、咸香、鲜香
- health_tags: 低热量、低脂、高蛋白
- scene_tags: 减脂、夜宵、治愈、饱腹、多人、快速
- temperature: 热、冷或 null
- avoid_tags: 辣椒、香菜、海鲜、牛肉、乳制品、面、面条、米饭、沙拉、油炸、高脂肪
- staple_preferences: 米饭、饭、面、面条、粉、粥、汤、甜品、饮品
- ingredient_preferences: 鸡肉、牛肉、猪肉、鱼、虾、蔬菜、鸡蛋、豆腐、甜、肉、冰品

还需返回 intent、budget、people_count、confidence（0 到 1）、needs_clarification、
clarification_question 和 clarification_options。clarification_options 只能使用上述标签。

JSON 字段必须完整：
{{"intent":"","taste_tags":[],"health_tags":[],"scene_tags":[],"temperature":null,
"avoid_tags":[],"staple_preferences":[],"ingredient_preferences":[],"budget":null,
"people_count":null,"confidence":0.0,"needs_clarification":false,
"clarification_question":"","clarification_options":[]}}

用户需求：{text}
""".strip()
        return self._chat_json(system_prompt, user_prompt, max_tokens=500)

    def generate_recommendation_reasons(
        self, need: dict[str, Any], dishes: list[dict[str, Any]]
    ) -> dict[str, str]:
        if not dishes:
            return {}
        system_prompt = (
            "你是菜品推荐解释助手。根据已经完成的规则打分结果，为每个菜品写一句中文推荐理由。"
            "不得编造输入中没有的营养数据，不得改变菜品，不得声称治疗疾病。"
            "每条理由 25 到 70 个汉字，只输出 JSON。"
        )
        user_prompt = (
            "用户需求：\n"
            + json.dumps(need, ensure_ascii=False)
            + "\n候选菜品：\n"
            + json.dumps(dishes, ensure_ascii=False)
            + '\n返回格式：{"reasons":[{"name":"菜品名","reason":"推荐理由"}]}'
        )
        result = self._chat_json(system_prompt, user_prompt, max_tokens=max(500, len(dishes) * 140))
        if not result or not isinstance(result.get("reasons"), list):
            return {}

        valid_names = {str(item["name"]) for item in dishes}
        reasons: dict[str, str] = {}
        for item in result["reasons"]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            reason = str(item.get("reason", "")).strip()
            if name in valid_names and 8 <= len(reason) <= 160:
                reasons[name] = reason
        return reasons


def get_default_llm_client() -> LLMClient:
    return LLMClient()

