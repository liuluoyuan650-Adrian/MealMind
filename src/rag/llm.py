from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.rag import config


@dataclass(frozen=True)
class LLMSettings:
    """DeepSeek 配置；环境变量优先于代码配置。"""

    api_key: str
    base_url: str
    model: str
    timeout_seconds: int = 20

    @classmethod
    def load(cls) -> "LLMSettings":
        return cls(
            api_key=os.getenv(
                "MEALMIND_LLM_API_KEY",
                os.getenv("DEEPSEEK_API_KEY", config.DEEPSEEK_API_KEY),
            ).strip(),
            base_url=os.getenv(
                "MEALMIND_LLM_BASE_URL", config.DEEPSEEK_BASE_URL
            ).strip(),
            model=os.getenv("MEALMIND_LLM_MODEL", config.DEEPSEEK_MODEL).strip(),
            timeout_seconds=int(
                os.getenv(
                    "MEALMIND_LLM_TIMEOUT", str(config.DEEPSEEK_TIMEOUT_SECONDS)
                )
            ),
        )


class LLMClient:
    """用于 RAG 生成阶段的 OpenAI 兼容 Chat Completions 客户端。"""

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or LLMSettings.load()
        self.last_error: str | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.api_key and self.settings.base_url and self.settings.model)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str | None:
        if not self.is_configured:
            return None

        endpoint = f"{self.settings.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
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
            return str(content).strip()
        except (
            HTTPError,
            URLError,
            TimeoutError,
            KeyError,
            IndexError,
            TypeError,
            json.JSONDecodeError,
        ) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None


def get_default_llm_client() -> LLMClient:
    return LLMClient()
