from __future__ import annotations


CHINESE_NUMBERS = {
    "一": 1,
    "两": 2,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def split_tags(value: str) -> set[str]:
    if not value:
        return set()
    normalized = str(value).replace("/", " ").replace(",", " ").replace("，", " ")
    return {item.strip() for item in normalized.split() if item.strip()}


def contains_any(text: str, words: list[str]) -> bool:
    return any(word in text for word in words)
