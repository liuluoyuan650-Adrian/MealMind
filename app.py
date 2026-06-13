from __future__ import annotations

import pandas as pd
import streamlit as st

from src.active_learning import record_user_feedback
from src.llm_client import get_default_llm_client
from src.nlp import INTENT_DATA, IntentClassifier, merge_needs, parse_need
from src.recommender import recommend


st.set_page_config(page_title="MealMind 菜品推荐系统", page_icon="🍽", layout="wide")


@st.cache_resource
def get_classifier(_train_data_mtime: int) -> IntentClassifier:
    classifier = IntentClassifier()
    classifier.fit()
    return classifier


def render_need(parsed) -> None:
    data = parsed.to_dict()
    view = {
        "意图": data["意图"],
        "意图置信度": data["意图置信度"],
        "预算": "未指定" if data["预算"] is None else f"{data['预算']} 元以内",
        "人数": "未指定" if data["人数"] is None else f"{data['人数']} 人",
        "口味": "、".join(data["口味"]) or "未指定",
        "忌口/排除": "、".join(data["忌口/排除"]) or "无",
        "健康需求": "、".join(data["健康需求"]) or "未指定",
        "场景": "、".join(data["场景"]) or "未指定",
        "温度偏好": data["温度偏好"] or "未指定",
        "主食偏好": "、".join(data["主食偏好"]) or "未指定",
        "食材偏好": "、".join(data["食材偏好"]) or "未指定",
        "理解来源": data["理解来源"],
        "语义置信度": data["语义匹配置信度"],
        "未知表达": "、".join(data["未知表达"]) or "无",
    }
    rows = [(key, str(value)) for key, value in view.items()]
    st.dataframe(
        pd.DataFrame(rows, columns=["解析项", "结果"]),
        width="stretch",
        hide_index=True,
    )


def render_cards(results: pd.DataFrame) -> None:
    if results.empty:
        st.warning("没有找到完全满足条件的菜品，可以放宽预算或减少忌口条件。")
        return

    for i, row in results.iterrows():
        with st.container(border=True):
            left, right = st.columns([2, 1])
            with left:
                st.subheader(f"{i + 1}. {row['name']}")
                st.write(row["recommend_reason"])
                st.caption(f"标签：{row['tags']}｜理由来源：{row['recommend_reason_source']}")
            with right:
                st.metric("推荐分", f"{row['recommend_score']}")
                st.write(f"价格：{row['price']} 元")
                st.write(f"热量：{row['calorie_kcal']} kcal")
                st.write(f"辣度：{row['spicy']}")
                st.write(f"饱腹感：{row['satiety']}/5")


classifier = get_classifier(INTENT_DATA.stat().st_mtime_ns)
llm_client = get_default_llm_client()

if "parsed_need" not in st.session_state:
    st.session_state.parsed_need = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "recommendation_key" not in st.session_state:
    st.session_state.recommendation_key = None
if "recommendation_results" not in st.session_state:
    st.session_state.recommendation_results = None

st.title("MealMind：基于自然语言理解的个性化菜品推荐系统")
st.caption("输入一句吃饭需求，系统会解析意图、预算、口味、忌口和场景，并推荐具体菜品。")

examples = [
    "我今天有点累，想吃点热乎清淡的，预算 30 以内，不要辣。",
    "我在减脂，想吃饱一点，但热量不要太高。",
    "我们三个人吃饭，一个人不吃辣，一个人想吃肉，人均 50 以内。",
    "晚上有点饿，想吃夜宵，不要太贵，最好热乎一点。",
    "今天心情不好，想吃点甜的或者治愈一点的。",
]

with st.sidebar:
    st.header("演示案例")
    chosen = st.radio("选择一个测试输入", examples, index=0)
    if st.button("填入示例"):
        st.session_state.current_input = chosen
    st.divider()
    st.write("核心流程")
    st.write("规则词典 → 语义向量 → 大模型兜底/澄清 → 菜品过滤 → 规则打分 → 推荐理由")
    st.divider()
    st.write("大模型状态")
    if llm_client.is_configured:
        st.success(f"已启用：{llm_client.settings.model}")
    else:
        st.info("未配置 API，当前使用本地语义匹配与规则理由。")

query = st.text_area(
    "今天想吃什么？",
    value=st.session_state.get("current_input", examples[0]),
    height=100,
)

col_a, col_b, col_c = st.columns([1, 1, 4])
with col_a:
    start = st.button("开始推荐", type="primary")
with col_b:
    clear = st.button("清空条件")

if clear:
    st.session_state.parsed_need = None
    st.session_state.query_history = []
    st.session_state.recommendation_key = None
    st.session_state.recommendation_results = None
    st.rerun()

if start:
    parsed = parse_need(query, classifier, llm_client=llm_client)
    st.session_state.parsed_need = parsed
    st.session_state.query_history = [query]
    st.session_state.recommendation_key = None
    st.session_state.recommendation_results = None

if st.session_state.parsed_need is not None:
    parsed = st.session_state.parsed_need
    st.header("1. NLP 需求解析结果")
    render_need(parsed)

    if parsed.needs_clarification:
        st.header("2. 需求澄清")
        st.warning(parsed.clarification_question)
        selected_option = st.selectbox("请选择最接近的需求", parsed.clarification_options)
        if st.button("确认需求并继续推荐", type="primary"):
            expression = parsed.unknown_expressions[0] if parsed.unknown_expressions else parsed.raw_text
            record_user_feedback(parsed.raw_text, expression, selected_option)
            st.session_state.parsed_need = merge_needs(
                parsed,
                selected_option,
                classifier,
                llm_client=llm_client,
            )
            st.session_state.query_history.append(f"确认需求：{selected_option}")
            st.session_state.recommendation_key = None
            st.session_state.recommendation_results = None
            st.rerun()
    else:
        recommendation_key = repr(parsed.to_dict())
        if st.session_state.recommendation_key != recommendation_key:
            st.session_state.recommendation_results = recommend(
                parsed,
                top_k=5,
                llm_client=llm_client,
            )
            st.session_state.recommendation_key = recommendation_key
        results = st.session_state.recommendation_results

        st.header("2. 推荐菜品")
        render_cards(results)

        if not results.empty:
            st.header("3. 推荐分数可视化")
            chart_data = results[["name", "recommend_score"]].set_index("name")
            st.bar_chart(chart_data)

        st.header("4. 多轮补充条件")
        follow_up = st.text_input("继续补充需求，例如：不要面，想吃米饭")
        if st.button("更新推荐"):
            if follow_up.strip():
                st.session_state.parsed_need = merge_needs(
                    parsed,
                    follow_up.strip(),
                    classifier,
                    llm_client=llm_client,
                )
                st.session_state.query_history.append(follow_up.strip())
                st.session_state.recommendation_key = None
                st.session_state.recommendation_results = None
                st.rerun()

    with st.expander("对话历史"):
        for idx, item in enumerate(st.session_state.query_history, start=1):
            st.write(f"{idx}. {item}")
else:
    st.info("输入需求后点击“开始推荐”。")
