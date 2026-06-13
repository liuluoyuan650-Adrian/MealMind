from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.utils import CHINESE_NUMBERS, contains_any
from src.active_learning import record_unknown_expression
from src.semantic_understanding import SemanticMatch, SemanticTagMatcher, extract_unknown_expressions

if TYPE_CHECKING:
    from src.llm_client import LLMClient


ROOT = Path(__file__).resolve().parents[1]
INTENT_DATA = ROOT / "data" / "intent_train.csv"


TASTE_DICT = {
    "清淡": ["清淡", "淡一点", "不油", "少油", "低盐", "不要太重口", "热乎清淡"],
    "辣": ["辣", "麻辣", "香辣", "重口"],
    "甜": ["甜", "甜品", "蛋糕", "奶茶", "热可可", "双皮奶"],
    "酸甜": ["酸甜", "番茄", "糖醋"],
    "咸香": ["咸香", "香一点"],
    "鲜香": ["鲜", "鲜香", "汤"],
}

SCENE_DICT = {
    "减脂": ["减脂", "低卡", "低热量", "瘦身", "控制体重", "健康"],
    "夜宵": ["夜宵", "晚上饿", "半夜", "睡前"],
    "治愈": ["心情不好", "治愈", "压力大", "累", "疲惫", "舒服"],
    "饱腹": ["吃饱", "顶饿", "饱腹", "管饱", "很饿"],
    "多人": ["我们", "朋友", "聚餐", "三个人", "两个人", "多人"],
    "快速": ["快", "赶时间", "马上", "快速"],
}

HEALTH_DICT = {
    "低热量": ["低热量", "低卡", "热量不要太高", "别太油", "少油"],
    "高蛋白": ["高蛋白", "蛋白质", "健身", "鸡胸", "牛肉"],
    "低脂": ["低脂", "减脂", "不油腻", "别太油"],
}

TEMPERATURE_DICT = {
    "热": ["热乎", "热的", "热食", "热饮", "汤", "暖"],
    "冷": ["凉的", "冷的", "沙拉", "冰", "冷饮"],
}

AVOID_TRIGGERS = ["不吃", "不要", "不能吃", "别放", "过敏", "排除"]
AVOID_ITEMS = ["辣", "辣椒", "香菜", "海鲜", "牛肉", "乳制品", "面", "面条", "米饭", "沙拉"]
STAPLE_ITEMS = ["米饭", "饭", "面", "面条", "粉", "粥", "汤", "甜品", "饮品"]
PREFERENCE_ITEMS = ["鸡肉", "牛肉", "猪肉", "鱼", "虾", "蔬菜", "鸡蛋", "豆腐", "甜", "肉", "冰品"]
ICE_PREFERENCE_WORDS = ["冰的", "冰点", "冰品", "冰淇淋", "雪糕", "冰沙"]
DRINK_PREFERENCE_WORDS = ["喝", "饮品", "饮料", "奶茶", "冷饮", "热饮", "喝东西"]
SPICY_AVOID_PATTERNS = [
    "不辣",
    "免辣",
    "微辣都不要",
    "不要辣",
    "别辣",
    "不能吃辣",
    "不吃辣",
    "别给我推荐麻辣",
]
SPICY_PREFERENCE_PATTERNS = ["想吃辣", "吃辣", "要辣", "麻辣", "香辣", "重口"]
BUDGET_EXPRESSIONS = ["不要太贵", "不太贵", "便宜", "便宜一点", "预算餐", "没多少钱"]

SEMANTIC_AUTO_THRESHOLD = 0.22
SEMANTIC_SECONDARY_THRESHOLD = 0.17


def _known_expressions() -> set[str]:
    words: set[str] = set()
    for mapping in (TASTE_DICT, SCENE_DICT, HEALTH_DICT, TEMPERATURE_DICT):
        for label, expressions in mapping.items():
            words.add(label)
            words.update(expressions)
    words.update(AVOID_ITEMS)
    words.update(STAPLE_ITEMS)
    words.update(PREFERENCE_ITEMS)
    words.update(ICE_PREFERENCE_WORDS)
    words.update(DRINK_PREFERENCE_WORDS)
    words.update(SPICY_AVOID_PATTERNS)
    words.update(SPICY_PREFERENCE_PATTERNS)
    words.update(BUDGET_EXPRESSIONS)
    return words


KNOWN_EXPRESSIONS = _known_expressions()


@dataclass
class ParsedNeed:
    raw_text: str
    intent: str = "未知"
    intent_confidence: float = 0.0
    budget: int | None = None
    people_count: int | None = None
    tastes: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    health: list[str] = field(default_factory=list)
    scenes: list[str] = field(default_factory=list)
    temperature: str | None = None
    staple_preference: list[str] = field(default_factory=list)
    ingredient_preference: list[str] = field(default_factory=list)
    understanding_source: str = "规则"
    semantic_confidence: float = 0.0
    semantic_matches: list[dict[str, Any]] = field(default_factory=list)
    unknown_expressions: list[str] = field(default_factory=list)
    llm_fallback_used: bool = False
    needs_clarification: bool = False
    clarification_question: str = ""
    clarification_options: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "原始输入": self.raw_text,
            "意图": self.intent,
            "意图置信度": round(self.intent_confidence, 3),
            "预算": self.budget,
            "人数": self.people_count,
            "口味": self.tastes,
            "忌口/排除": self.avoid,
            "健康需求": self.health,
            "场景": self.scenes,
            "温度偏好": self.temperature,
            "主食偏好": self.staple_preference,
            "食材偏好": self.ingredient_preference,
            "理解来源": self.understanding_source,
            "语义匹配置信度": round(self.semantic_confidence, 3),
            "语义匹配": self.semantic_matches,
            "未知表达": self.unknown_expressions,
            "大模型兜底": self.llm_fallback_used,
            "需要澄清": self.needs_clarification,
            "澄清问题": self.clarification_question,
            "澄清选项": self.clarification_options,
        }


class IntentClassifier:
    def __init__(self) -> None:
        self.model: Pipeline | None = None

    def fit(self) -> None:
        data = pd.read_csv(INTENT_DATA)
        self.model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 3))),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
        self.model.fit(data["text"], data["intent"])

    def predict(self, text: str) -> tuple[str, float]:
        if self.model is None:
            self.fit()
        assert self.model is not None
        intent = self.model.predict([text])[0]
        if hasattr(self.model.named_steps["clf"], "predict_proba"):
            probs = self.model.predict_proba([text])[0]
            confidence = float(max(probs))
        else:
            confidence = 0.0
        return str(intent), confidence


def extract_budget(text: str) -> int | None:
    patterns = [
        r"(?:预算|人均)?\s*(\d{1,3})\s*(?:元|块|以内|以下|左右)",
        r"(?:不超过|低于|少于|小于)\s*(\d{1,3})",
        r"(\d{1,3})\s*(?:以内|以下)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    if contains_any(text, BUDGET_EXPRESSIONS):
        return 30
    return None


def extract_people_count(text: str) -> int | None:
    candidates: list[int] = []
    for match in re.finditer(r"(\d+)\s*个?人", text):
        candidates.append(int(match.group(1)))
    for word, number in CHINESE_NUMBERS.items():
        if f"{word}个人" in text or f"{word}人" in text:
            candidates.append(number)
    if "我和朋友" in text:
        candidates.append(2)
    return max(candidates) if candidates else None


def extract_by_dict(text: str, mapping: dict[str, list[str]]) -> list[str]:
    return [label for label, words in mapping.items() if label in text or contains_any(text, words)]


def wants_no_spicy(text: str) -> bool:
    return contains_any(text, SPICY_AVOID_PATTERNS)


def wants_spicy(text: str) -> bool:
    return contains_any(text, SPICY_PREFERENCE_PATTERNS) and not wants_no_spicy(text)


def extract_avoid(text: str) -> list[str]:
    avoid: set[str] = set()
    if wants_no_spicy(text):
        avoid.add("辣椒")
    for trigger in AVOID_TRIGGERS:
        if trigger not in text:
            continue
        for item in AVOID_ITEMS:
            if item in text:
                if item == "辣":
                    avoid.add("辣椒")
                else:
                    avoid.add(item)
    return sorted(avoid)


def extract_preferences(text: str, items: list[str]) -> list[str]:
    preferences = []
    for item in items:
        if item in text and item not in extract_avoid(text):
            preferences.append(item)
    if contains_any(text, ICE_PREFERENCE_WORDS) and "冰品" in items:
        preferences.append("冰品")
    if contains_any(text, DRINK_PREFERENCE_WORDS) and "饮品" in items:
        preferences.append("饮品")
    return preferences


def _append_unique(target: list[str], values: list[str]) -> None:
    for value in values:
        if value and value not in target:
            target.append(value)


def _apply_semantic_match(parsed: ParsedNeed, match: SemanticMatch) -> None:
    if match.slot == "taste":
        _append_unique(parsed.tastes, [match.label])
    elif match.slot == "health":
        _append_unique(parsed.health, [match.label])
        if match.label in {"低热量", "低脂"}:
            _append_unique(parsed.scenes, ["减脂"])
    elif match.slot == "scene":
        _append_unique(parsed.scenes, [match.label])
    elif match.slot == "temperature":
        parsed.temperature = match.label


def _safe_list(payload: dict[str, Any], key: str, allowed: set[str]) -> list[str]:
    value = payload.get(key, [])
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item) in allowed]


def _apply_llm_result(parsed: ParsedNeed, payload: dict[str, Any]) -> float:
    _append_unique(
        parsed.tastes,
        _safe_list(payload, "taste_tags", {"清淡", "辣", "甜", "酸甜", "咸香", "鲜香"}),
    )
    _append_unique(
        parsed.health,
        _safe_list(payload, "health_tags", {"低热量", "低脂", "高蛋白"}),
    )
    _append_unique(
        parsed.scenes,
        _safe_list(payload, "scene_tags", {"减脂", "夜宵", "治愈", "饱腹", "多人", "快速"}),
    )
    _append_unique(
        parsed.avoid,
        _safe_list(
            payload,
            "avoid_tags",
            {"辣椒", "香菜", "海鲜", "牛肉", "乳制品", "面", "面条", "米饭", "沙拉", "油炸", "高脂肪"},
        ),
    )
    _append_unique(
        parsed.staple_preference,
        _safe_list(payload, "staple_preferences", set(STAPLE_ITEMS)),
    )
    _append_unique(
        parsed.ingredient_preference,
        _safe_list(payload, "ingredient_preferences", set(PREFERENCE_ITEMS)),
    )

    temperature = payload.get("temperature")
    if temperature in {"热", "冷"}:
        parsed.temperature = str(temperature)
    intent = str(payload.get("intent", "")).strip()
    if intent:
        parsed.intent = intent
    for field_name in ("budget", "people_count"):
        value = payload.get(field_name)
        if isinstance(value, int) and value > 0 and getattr(parsed, field_name) is None:
            setattr(parsed, field_name, value)

    try:
        confidence = max(0.0, min(float(payload.get("confidence", 0.0)), 1.0))
    except (TypeError, ValueError):
        confidence = 0.0
    parsed.needs_clarification = bool(payload.get("needs_clarification", False)) or confidence < 0.55
    parsed.clarification_question = str(payload.get("clarification_question", "")).strip()
    parsed.clarification_options = _safe_list(
        payload,
        "clarification_options",
        {"清淡", "辣", "甜", "酸甜", "咸香", "鲜香", "低热量", "低脂", "高蛋白", "减脂", "夜宵", "治愈", "饱腹", "多人", "快速", "热", "冷"},
    )
    return confidence


def parse_need(
    text: str,
    classifier: IntentClassifier | None = None,
    llm_client: "LLMClient | None" = None,
    record_unknown: bool = True,
) -> ParsedNeed:
    classifier = classifier or IntentClassifier()
    intent, confidence = classifier.predict(text)
    parsed = ParsedNeed(
        raw_text=text,
        intent=intent,
        intent_confidence=confidence,
        budget=extract_budget(text),
        people_count=extract_people_count(text),
        tastes=extract_by_dict(text, TASTE_DICT),
        avoid=extract_avoid(text),
        health=extract_by_dict(text, HEALTH_DICT),
        scenes=extract_by_dict(text, SCENE_DICT),
        staple_preference=extract_preferences(text, STAPLE_ITEMS),
        ingredient_preference=extract_preferences(text, PREFERENCE_ITEMS),
    )

    temperature_hits = extract_by_dict(text, TEMPERATURE_DICT)
    if temperature_hits:
        parsed.temperature = temperature_hits[0]

    if confidence >= 0.35:
        if parsed.intent == "减脂餐" and "减脂" not in parsed.scenes:
            parsed.scenes.append("减脂")
        if parsed.intent == "夜宵" and "夜宵" not in parsed.scenes:
            parsed.scenes.append("夜宵")
        if parsed.intent == "治愈餐" and "治愈" not in parsed.scenes:
            parsed.scenes.append("治愈")
        if parsed.intent == "快速饱腹餐" and "饱腹" not in parsed.scenes:
            parsed.scenes.append("饱腹")
        if parsed.intent == "甜食饮品推荐" and "甜" not in parsed.tastes:
            parsed.tastes.append("甜")
    if "饮品" in parsed.staple_preference:
        parsed.intent = "饮品推荐"
    if "冰品" in parsed.ingredient_preference:
        parsed.intent = "冰品甜食推荐"
        if "甜" not in parsed.tastes:
            parsed.tastes.append("甜")
    if wants_spicy(text) and "辣" not in parsed.tastes:
        parsed.tastes.append("辣")
    if wants_spicy(text) and parsed.intent == "不辣餐":
        parsed.intent = "重口辣味餐"
    if parsed.intent == "不辣餐" and wants_no_spicy(text) and "辣椒" not in parsed.avoid:
        parsed.avoid.append("辣椒")
    if "辣椒" in parsed.avoid:
        parsed.tastes = [taste for taste in parsed.tastes if taste != "辣"]

    parsed.unknown_expressions = extract_unknown_expressions(text, KNOWN_EXPRESSIONS)
    matcher = SemanticTagMatcher()
    unresolved: list[str] = []
    semantic_applied = False
    all_matches: list[SemanticMatch] = []
    for expression in parsed.unknown_expressions:
        matches = matcher.match(expression, top_k=3)
        all_matches.extend(matches)
        if not matches or matches[0].score < SEMANTIC_AUTO_THRESHOLD:
            unresolved.append(expression)
            continue
        top_score = matches[0].score
        for match in matches:
            if match.score >= SEMANTIC_AUTO_THRESHOLD or (
                match.score >= SEMANTIC_SECONDARY_THRESHOLD and match.score >= top_score * 0.6
            ):
                _apply_semantic_match(parsed, match)
                semantic_applied = True

    parsed.semantic_matches = [match.to_dict() for match in all_matches]
    parsed.semantic_confidence = max((match.score for match in all_matches), default=0.0)
    if semantic_applied:
        parsed.understanding_source = "规则+语义向量"

    if unresolved and llm_client is not None and llm_client.is_configured:
        llm_result = llm_client.interpret_need(text)
        if llm_result:
            llm_confidence = _apply_llm_result(parsed, llm_result)
            parsed.semantic_confidence = max(parsed.semantic_confidence, llm_confidence)
            parsed.llm_fallback_used = True
            parsed.understanding_source = "规则+大模型兜底"
            unresolved = [] if not parsed.needs_clarification else unresolved

    if unresolved and not parsed.llm_fallback_used:
        option_scores: dict[str, float] = {}
        for match in all_matches:
            option_scores[match.label] = max(option_scores.get(match.label, 0.0), match.score)
        parsed.clarification_options = [
            label for label, _ in sorted(option_scores.items(), key=lambda item: item[1], reverse=True)[:4]
        ]
        if not parsed.clarification_options:
            parsed.clarification_options = ["低热量", "高蛋白", "清淡", "饱腹"]
        parsed.needs_clarification = True
        guessed = "、".join(parsed.clarification_options[:3])
        parsed.clarification_question = f"我还不完全理解“{unresolved[0]}”。你更偏向{guessed}中的哪一种？"

    if parsed.needs_clarification and not parsed.clarification_question:
        parsed.clarification_question = "这个表达有些模糊，请选择一个更接近的用餐需求。"

    if parsed.unknown_expressions and record_unknown:
        record_unknown_expression(
            raw_text=text,
            expressions=parsed.unknown_expressions,
            semantic_matches=parsed.semantic_matches,
            source=parsed.understanding_source,
            needs_clarification=parsed.needs_clarification,
        )

    return parsed


def merge_needs(
    old: ParsedNeed,
    new_text: str,
    classifier: IntentClassifier | None = None,
    llm_client: "LLMClient | None" = None,
) -> ParsedNeed:
    new = parse_need(new_text, classifier, llm_client=llm_client)
    merged = ParsedNeed(raw_text=f"{old.raw_text}；{new_text}")
    merged.intent = new.intent if new.intent != "未知" else old.intent
    merged.intent_confidence = max(old.intent_confidence, new.intent_confidence)
    merged.budget = new.budget if new.budget is not None else old.budget
    merged.people_count = new.people_count if new.people_count is not None else old.people_count
    merged.temperature = new.temperature if new.temperature is not None else old.temperature
    merged.tastes = sorted(set(old.tastes + new.tastes))
    merged.avoid = sorted(set(old.avoid + new.avoid))
    merged.health = sorted(set(old.health + new.health))
    merged.scenes = sorted(set(old.scenes + new.scenes))
    merged.staple_preference = sorted(set(old.staple_preference + new.staple_preference))
    merged.ingredient_preference = sorted(set(old.ingredient_preference + new.ingredient_preference))
    merged.understanding_source = (
        "规则+用户确认" if old.needs_clarification and not new.needs_clarification else new.understanding_source
    )
    merged.semantic_confidence = max(old.semantic_confidence, new.semantic_confidence)
    merged.semantic_matches = old.semantic_matches + new.semantic_matches
    merged.unknown_expressions = sorted(set(old.unknown_expressions + new.unknown_expressions))
    merged.llm_fallback_used = old.llm_fallback_used or new.llm_fallback_used
    merged.needs_clarification = new.needs_clarification
    merged.clarification_question = new.clarification_question
    merged.clarification_options = new.clarification_options
    return merged
