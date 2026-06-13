from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nlp import IntentClassifier, parse_need
from src.recommender import recommend


TEST_CASES = [
    "我今天想吃清淡一点的，预算 30 以内，不要辣。",
    "我在减脂，想吃饱一点，但热量不要太高。",
    "我们三个人吃饭，一个人不吃辣，一个人想吃肉，人均 50 以内。",
    "晚上有点饿，想吃夜宵，不要太贵，最好热乎一点。",
    "今天心情不好，想吃点甜的或者治愈一点的。",
]


def main() -> None:
    classifier = IntentClassifier()
    classifier.fit()
    for idx, text in enumerate(TEST_CASES, start=1):
        print("=" * 80)
        print(f"测试用例 {idx}: {text}")
        parsed = parse_need(text, classifier)
        print("解析结果:")
        for key, value in parsed.to_dict().items():
            print(f"  {key}: {value}")
        results = recommend(parsed, top_k=5)
        print("推荐结果:")
        for rank, row in results.iterrows():
            print(
                f"  {rank + 1}. {row['name']} | 分数 {row['recommend_score']} | "
                f"价格 {row['price']} 元 | 热量 {row['calorie_kcal']} kcal"
            )
            print(f"     {row['recommend_reason']}")


if __name__ == "__main__":
    main()
