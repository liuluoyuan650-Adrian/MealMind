from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.nlp import ParsedNeed
from src.utils import split_tags

if TYPE_CHECKING:
    from src.llm_client import LLMClient


ROOT = Path(__file__).resolve().parents[1]
DISH_DATA = ROOT / "data" / "dish_dataset.csv"


def load_dishes() -> pd.DataFrame:
    data = pd.read_csv(DISH_DATA)
    data["spicy"] = data["spicy"].astype(int)
    data["price"] = data["price"].astype(int)
    data["calorie_kcal"] = data["calorie_kcal"].astype(int)
    data["satiety"] = data["satiety"].astype(int)
    data["rating"] = data["rating"].astype(float)
    return data


def is_mixed_drink_and_food_need(need: ParsedNeed) -> bool:
    return "饮品" in need.staple_preference and "辣" in need.tastes


def is_spicy_food(row: pd.Series) -> bool:
    taste_tags = split_tags(row.get("taste", "")) | split_tags(row.get("tags", ""))
    return int(row["spicy"]) > 0 or "辣" in taste_tags or any("辣" in item for item in taste_tags)


def is_drink_or_ice_item(row: pd.Series) -> bool:
    tags = split_tags(row.get("tags", ""))
    category = str(row.get("category", ""))
    return category in {"饮品", "甜品"} or bool(tags & {"饮品", "冰品", "冰淇淋", "冰沙", "冷饮"})


def violates_avoid(row: pd.Series, need: ParsedNeed) -> bool:
    avoid_tags = split_tags(row.get("avoid_tags", ""))
    ingredients = split_tags(row.get("ingredients", ""))
    tags = split_tags(row.get("tags", ""))
    category = str(row.get("category", ""))
    name = str(row.get("name", ""))
    mixed_drink_food = is_mixed_drink_and_food_need(need)
    spicy_food = is_spicy_food(row)

    if need.temperature and row["temperature"] != need.temperature and not (mixed_drink_food and spicy_food):
        return True
    if "冰品" in need.ingredient_preference:
        if not is_drink_or_ice_item(row) and not (mixed_drink_food and spicy_food):
            return True
    if "饮品" in need.staple_preference and category != "饮品" and "饮品" not in tags:
        if not (mixed_drink_food and spicy_food):
            return True
    if mixed_drink_food and not (is_drink_or_ice_item(row) or spicy_food):
        return True
    if "辣椒" in need.avoid and int(row["spicy"]) > 0:
        return True
    if "面" in need.avoid or "面条" in need.avoid:
        if "面" in category or "面" in name or "面条" in ingredients:
            return True
    if "米饭" in need.avoid:
        if "饭" in category or "饭" in name or "米饭" in ingredients:
            return True
    if "沙拉" in need.avoid and ("沙拉" in category or "沙拉" in name or "沙拉" in tags):
        return True

    for item in need.avoid:
        if item in avoid_tags or item in ingredients or item in tags or item in name:
            return True
    return False


def score_budget(price: int, budget: int | None) -> tuple[int, str | None]:
    if budget is None:
        return 12, None
    if price <= budget:
        return 20, f"价格 {price} 元，在预算 {budget} 元以内"
    if price <= budget * 1.2:
        return 10, f"价格 {price} 元，略高于预算但仍接近"
    if price <= budget * 1.5:
        return 5, f"价格 {price} 元，明显高于预算"
    return -20, f"价格 {price} 元，超过预算较多"


def score_row(row: pd.Series, need: ParsedNeed) -> dict:
    reasons: list[str] = []
    score = 0

    if violates_avoid(row, need):
        return {"score": -999, "reasons": ["不满足忌口或排除条件"], "filtered": True}

    avoid_score = 20
    score += avoid_score
    if need.avoid:
        reasons.append("不包含已识别的忌口或排除项")

    budget_score, budget_reason = score_budget(int(row["price"]), need.budget)
    score += budget_score
    if budget_reason:
        reasons.append(budget_reason)

    row_taste = split_tags(row["taste"])
    row_tags = split_tags(row["tags"])
    taste_pool = row_taste | row_tags
    taste_match = {
        taste
        for taste in need.tastes
        if taste in taste_pool or any(taste in item or item in taste for item in taste_pool)
    }
    if need.tastes:
        if "辣" in need.tastes and int(row["spicy"]) > 0:
            taste_match.add("辣")
        taste_score = 15 if taste_match else 4
        if "辣" in need.tastes and int(row["spicy"]) == 0:
            taste_score = -8
        if taste_match:
            reasons.append(f"口味匹配：{', '.join(sorted(taste_match))}")
    else:
        taste_score = 8
    score += taste_score

    row_scene = split_tags(row["scene"]) | row_tags
    scene_match = set(need.scenes) & row_scene
    if need.scenes:
        scene_score = 15 if scene_match else 5
        if scene_match:
            reasons.append(f"适合场景：{', '.join(sorted(scene_match))}")
    else:
        scene_score = 7
    score += scene_score

    row_health = row_tags | {str(row["calorie_level"])}
    health_match = set(need.health) & row_health
    if need.health:
        if health_match:
            health_score = 10
        elif ("低热量" in need.health or "低脂" in need.health) and row["calorie_level"] == "高":
            health_score = -12
            reasons.append("热量偏高，不完全符合低热量需求")
        elif ("低热量" in need.health or "低脂" in need.health) and row["calorie_level"] == "中":
            health_score = 3
        else:
            health_score = 3
        if health_match:
            reasons.append(f"健康需求匹配：{', '.join(sorted(health_match))}")
    elif "减脂" in need.scenes and (row["calorie_level"] == "低" or "低卡" in row_tags or "低脂" in row_tags):
        health_score = 10
        reasons.append("热量或脂肪负担较低，适合减脂需求")
    elif "减脂" in need.scenes and row["calorie_level"] == "高":
        health_score = -8
    else:
        health_score = 5
    score += health_score

    if "饱腹" in need.scenes:
        satiety_score = min(int(row["satiety"]) * 2, 10)
        reasons.append(f"饱腹感 {int(row['satiety'])}/5")
    else:
        satiety_score = min(int(row["satiety"]) + 2, 8)
    score += satiety_score

    if need.temperature:
        temp_score = 8 if row["temperature"] == need.temperature else 2
        if row["temperature"] == need.temperature:
            reasons.append(f"符合{need.temperature}食偏好")
    else:
        temp_score = 5
    score += temp_score

    pref_pool = split_tags(row["ingredients"]) | row_tags | split_tags(row["category"])
    preferences = need.ingredient_preference + need.staple_preference
    pref_match = {
        pref
        for pref in preferences
        if pref in pref_pool or any(pref in item or item in pref for item in pref_pool)
    }
    if pref_match:
        score += 7 + max(len(pref_match) - 1, 0) * 3
        if "冰品" in pref_match:
            score += 8
        reasons.append(f"包含偏好：{', '.join(sorted(pref_match))}")

    rating_score = min(max((float(row["rating"]) - 4.0) * 8, 0), 5)
    score += rating_score

    return {
        "score": max(0, min(round(score, 1), 100)),
        "reasons": reasons[:5],
        "filtered": False,
    }


def generate_reason(row: pd.Series, reasons: list[str]) -> str:
    if not reasons:
        return f"推荐{row['name']}，它整体评分较高，价格适中，适合作为候选菜品。"
    reason_text = "；".join(reasons)
    return f"推荐{row['name']}，因为{reason_text}。"


def _enrich_reasons(
    selected: pd.DataFrame,
    need: ParsedNeed,
    llm_client: "LLMClient | None",
) -> pd.DataFrame:
    selected = selected.copy().reset_index(drop=True)
    selected["recommend_reason_source"] = "规则生成"
    if llm_client is None or not llm_client.is_configured or selected.empty:
        return selected

    dishes = []
    for _, row in selected.iterrows():
        dishes.append(
            {
                "name": str(row["name"]),
                "category": str(row["category"]),
                "price": int(row["price"]),
                "calorie_kcal": int(row["calorie_kcal"]),
                "satiety": int(row["satiety"]),
                "taste": str(row["taste"]),
                "tags": str(row["tags"]),
                "ingredients": str(row["ingredients"]),
                "recommend_score": float(row["recommend_score"]),
                "match_reasons": list(row["match_reasons"]),
            }
        )
    generated = llm_client.generate_recommendation_reasons(need.to_dict(), dishes)
    for index, row in selected.iterrows():
        reason = generated.get(str(row["name"]))
        if reason:
            selected.at[index, "recommend_reason"] = reason
            selected.at[index, "recommend_reason_source"] = "大模型生成"
    return selected


def recommend(
    need: ParsedNeed,
    top_k: int = 5,
    llm_client: "LLMClient | None" = None,
) -> pd.DataFrame:
    dishes = load_dishes()
    results = []
    for _, row in dishes.iterrows():
        scored = score_row(row, need)
        if scored["filtered"]:
            continue
        item = row.to_dict()
        item["recommend_score"] = scored["score"]
        item["match_reasons"] = scored["reasons"]
        item["recommend_reason"] = generate_reason(row, scored["reasons"])
        results.append(item)

    ranked = pd.DataFrame(results)
    if ranked.empty:
        return ranked
    ranked = ranked.sort_values(["recommend_score", "rating"], ascending=[False, False])
    if is_mixed_drink_and_food_need(need):
        drink_mask = ranked.apply(is_drink_or_ice_item, axis=1)
        spicy_mask = ranked.apply(is_spicy_food, axis=1)
        drink_items = ranked[drink_mask].head(max(1, top_k // 2))
        spicy_items = ranked[spicy_mask & ~drink_mask].head(top_k - len(drink_items))
        combined = pd.concat([drink_items, spicy_items], ignore_index=True)
        if len(combined) < top_k:
            combined_names = set(combined["name"])
            fallback = ranked[~ranked["name"].isin(combined_names)].head(top_k - len(combined))
            combined = pd.concat([combined, fallback], ignore_index=True)
        selected = combined.head(top_k).reset_index(drop=True)
    else:
        selected = ranked.head(top_k).reset_index(drop=True)
    return _enrich_reasons(selected, need, llm_client)
