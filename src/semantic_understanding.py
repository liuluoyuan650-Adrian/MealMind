from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class TagDefinition:
    label: str
    slot: str
    description: str
    examples: tuple[str, ...]


@dataclass(frozen=True)
class SemanticMatch:
    expression: str
    label: str
    slot: str
    score: float

    def to_dict(self) -> dict:
        return {
            "expression": self.expression,
            "label": self.label,
            "slot": self.slot,
            "score": round(self.score, 3),
        }


TAG_DEFINITIONS = (
    TagDefinition("低热量", "health", "热量较低，轻负担，适合控制体重和不想吃得有罪恶感", ("不罪恶", "没负担", "轻一点", "低卡", "少长胖")),
    TagDefinition("低脂", "health", "少油少脂，不油腻，吃起来干净清爽，没有饮食罪恶感", ("不罪恶", "不负担", "干净一点", "别太油", "科技与狠活少一点", "健康点")),
    TagDefinition("高蛋白", "health", "鸡肉牛肉鸡蛋虾仁等蛋白质较高，适合健身", ("补充蛋白质", "健身餐", "多点肉", "蛋白质高")),
    TagDefinition("清淡", "taste", "少油少盐，味道不重，口感清爽", ("清爽点", "干净点", "淡口", "舒服一点", "别太重")),
    TagDefinition("咸香", "taste", "现炒热菜，有香气和锅气，味道较足", ("有锅气", "现炒的", "香一点", "家常炒菜")),
    TagDefinition("辣", "taste", "麻辣香辣或重口味，辣度明显", ("刺激一点", "带劲", "重口", "麻辣")),
    TagDefinition("甜", "taste", "甜味、甜品或能带来满足感的食物", ("来点甜的", "甜口", "糖分慰藉")),
    TagDefinition("治愈", "scene", "温暖舒适，让心情变好并带来满足感", ("续命", "安慰一下", "舒服的", "治愈系", "心累")),
    TagDefinition("饱腹", "scene", "能吃饱、顶饿，适合作为正餐", ("扛饿", "管饱", "顶一阵", "能吃饱")),
    TagDefinition("热", "temperature", "热乎的饭菜、汤、面、粥、热饮或刚出锅的现炒菜", ("暖胃", "热腾腾", "暖一点", "现做的", "有锅气", "现炒的")),
    TagDefinition("快速", "scene", "适合赶时间、工作忙或需要快速解决一餐", ("打工人", "赶时间", "快速解决", "马上能吃")),
    TagDefinition("夜宵", "scene", "适合晚上、半夜或睡前少量进食", ("深夜吃点", "熬夜", "半夜垫垫")),
)


class SemanticTagMatcher:
    def __init__(self, definitions: tuple[TagDefinition, ...] = TAG_DEFINITIONS) -> None:
        self.definitions = definitions
        documents = [
            " ".join((item.label, item.description, *item.examples)) for item in definitions
        ]
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), sublinear_tf=True)
        self.tag_vectors = self.vectorizer.fit_transform(documents)

    def match(self, expression: str, top_k: int = 3) -> list[SemanticMatch]:
        vector = self.vectorizer.transform([expression])
        scores = cosine_similarity(vector, self.tag_vectors)[0]
        ranked = scores.argsort()[::-1][:top_k]
        return [
            SemanticMatch(
                expression=expression,
                label=self.definitions[index].label,
                slot=self.definitions[index].slot,
                score=float(scores[index]),
            )
            for index in ranked
            if scores[index] > 0
        ]


GENERIC_WORDS = (
    "今天", "现在", "有点", "我想", "想吃点", "想吃", "想要", "来点", "给我", "推荐", "吃点", "一点",
    "一些", "东西", "食物", "饭菜", "最好", "比较", "可以", "有没有", "不想", "不要", "太", "的",
)


def extract_unknown_expressions(text: str, known_words: set[str]) -> list[str]:
    clauses = re.split(r"[，。！？；,;\n]|但是|而且|另外|同时|还要|不过|但又|最好", text)
    unknown: list[str] = []
    for clause in clauses:
        cleaned = clause.strip()
        if not cleaned:
            continue
        if re.search(r"\d+\s*(?:元|块|人|以内|以下|左右)", cleaned) or (
            any(marker in cleaned for marker in ("预算", "人均", "不超过", "低于", "少于"))
            and re.search(r"\d", cleaned)
        ):
            continue
        if any(word and word in cleaned for word in known_words):
            continue
        compact = cleaned
        for word in GENERIC_WORDS:
            compact = compact.replace(word, "")
        compact = re.sub(r"[\s，。！？；,.!?;]", "", compact)
        if any(word and word in compact for word in known_words):
            continue
        if len(compact) >= 2 and not compact.isdigit():
            unknown.append(compact)
    return list(dict.fromkeys(unknown))
