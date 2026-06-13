# MealMind：基于自然语言理解的个性化菜品推荐系统

MealMind 是一个面向日常吃饭场景的 NLP 应用系统。用户输入一句自然语言需求，例如“我今天有点累，想吃点热乎清淡的，预算 30 以内，不要辣”，系统会自动解析意图、预算、口味、忌口、健康需求和情绪场景，并从菜品数据集中推荐合适菜品。

## 核心功能

- 自然语言需求输入
- 用餐意图识别
- 槽位抽取：预算、人数、口味、忌口、主食偏好、健康需求、情绪场景
- 菜品数据库检索与硬性过滤
- 100 分制可解释推荐排序
- 推荐理由生成
- 多轮补充条件更新
- Streamlit 可视化演示界面

## 技术栈

- Python
- Streamlit
- pandas
- scikit-learn
- TF-IDF + Logistic Regression
- 正则表达式与关键词词典
- 规则打分推荐算法
- TF-IDF 字符向量语义匹配
- OpenAI 兼容的大模型 API（可选）

## 三层语义理解

系统按以下顺序理解用户需求：

1. 规则词典识别常见口味、预算、忌口和场景。
2. 对未命中的短语进行本地语义向量匹配，例如把“不罪恶”映射为“低热量/低脂”。
3. 语义置信度不足时调用大模型转换为结构化标签；未配置 API 或模型仍不确定时，界面会让用户选择澄清项。

未知表达和用户确认结果会追加到 `data/unknown_phrases.jsonl`，可用于后续扩充词典或训练集。菜品过滤和排序始终由本地推荐算法完成，大模型只负责低置信度理解和推荐理由生成。

## 配置大模型 API

直接在 `src/llm_config.py` 中填写 `LLM_API_KEY`。默认使用 OpenAI 兼容的 `/chat/completions` 接口，也可以修改 `LLM_BASE_URL` 和 `LLM_MODEL` 接入其他兼容服务。

更推荐通过环境变量配置：

```powershell
$env:MEALMIND_LLM_API_KEY="你的密钥"
$env:MEALMIND_LLM_BASE_URL="https://api.openai.com/v1"
$env:MEALMIND_LLM_MODEL="gpt-4.1-mini"
streamlit run app.py
```

API 未配置、超时或返回格式异常时，系统会自动回退到本地语义理解和规则推荐理由。

## 运行方式

```bash
cd C:\Users\Vean\Desktop\MealMind
streamlit run app.py
```

## 测试方式

```bash
cd C:\Users\Vean\Desktop\MealMind
python tests/run_test_cases.py
```

## 项目结构

```text
MealMind/
  app.py
  README.md
  requirements.txt
  data/
    dish_dataset.csv
    intent_train.csv
  src/
    active_learning.py
    llm_client.py
    llm_config.py
    nlp.py
    recommender.py
    semantic_understanding.py
    utils.py
  tests/
    run_test_cases.py
    test_semantic_llm.py
```
