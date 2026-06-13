from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
UNKNOWN_PHRASE_DATA = ROOT / "data" / "unknown_phrases.jsonl"


def _append_record(record: dict[str, Any], path: Path | None = None) -> None:
    path = path or Path(os.getenv("MEALMIND_UNKNOWN_LOG_PATH", str(UNKNOWN_PHRASE_DATA)))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def record_unknown_expression(
    raw_text: str,
    expressions: list[str],
    semantic_matches: list[dict[str, Any]],
    source: str,
    needs_clarification: bool,
) -> None:
    _append_record(
        {
            "event": "unknown_expression",
            "raw_text": raw_text,
            "expressions": expressions,
            "semantic_matches": semantic_matches,
            "source": source,
            "needs_clarification": needs_clarification,
        }
    )


def record_user_feedback(raw_text: str, expression: str, selected_label: str) -> None:
    _append_record(
        {
            "event": "user_feedback",
            "raw_text": raw_text,
            "expression": expression,
            "selected_label": selected_label,
        }
    )
