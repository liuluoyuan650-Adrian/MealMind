# MealMind RAG 美食推荐系统

MealMind 现在只保留 RAG 推荐链路：使用中文 Embedding 模型将本地菜品向量化，
通过 FAISS 召回候选菜品，再由 DeepSeek 严格依据候选信息生成推荐话术。

## 核心流程

```text
用户自然语言需求
  -> BAAI/bge-small-zh-v1.5 查询向量
  -> FAISS Top-K 相似度检索
  -> 否定条件硬过滤（不辣、不要海鲜等）
  -> 受约束的 RAG Prompt
  -> DeepSeek 生成 2-3 道菜的推荐话术
```

## 项目结构

```text
MealMind/
  data/
    dish_dataset.csv          # 菜品事实源
    faiss_index/              # 本地向量索引和 metadata
  scripts/
    build_rag_index.py        # 离线建库脚本
  src/
    rag/
      api.py                  # FastAPI 接口
      config.py               # RAG 配置
      data_ingestion.py       # 数据加载和结构化文本生成
      embedding.py            # sentence-transformers 封装
      llm.py                  # DeepSeek API 客户端
      prompts.py              # 防幻觉 Prompt
      retriever.py            # FAISS 检索和否定条件过滤
      service.py              # 完整 RAG 推荐服务
  tests/
    test_rag.py
```

## 首次安装

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

当前仓库已经包含构建好的 FAISS 索引。只有修改 `data/dish_dataset.csv` 或更换
Embedding 模型后，才需要重新建库：

```powershell
.\.venv\Scripts\python.exe scripts\build_rag_index.py
```

默认 Embedding 模型为 `BAAI/bge-small-zh-v1.5`。首次建库需要下载模型，之后会
优先从本地缓存加载。也可以在建库前指定本地模型目录：

```powershell
$env:MEALMIND_EMBEDDING_MODEL="D:\models\bge-small-zh-v1.5"
.\.venv\Scripts\python.exe scripts\build_rag_index.py
```

## 配置 DeepSeek

打开 `src/rag/config.py`，填写：

```python
DEEPSEEK_API_KEY = "你的新 DeepSeek API Key"
```

保存后即可直接启动。不要把包含真实密钥的项目上传到公开仓库。

也可以使用环境变量临时覆盖代码配置：

```powershell
$env:DEEPSEEK_API_KEY="替换成你的真实密钥"
```

默认配置：

```text
Base URL: https://api.deepseek.com
Model: deepseek-v4-pro
```

可以通过 `MEALMIND_LLM_BASE_URL` 和 `MEALMIND_LLM_MODEL` 覆盖。

## 启动项目

项目现在只有一个入口：

```powershell
cd C:\PythonProject\MealMind
$env:DEEPSEEK_API_KEY="替换成你的真实密钥"
.\.venv\Scripts\python.exe -m uvicorn src.rag.api:app --host 127.0.0.1 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

根路径是面向普通用户的推荐页面，输入自然语言需求后点击“帮我推荐”即可。

开发者接口文档仍保留在：

```text
http://127.0.0.1:8000/docs
```

在 Swagger 的 `POST /recommend` 中可以输入：

```json
{
  "query": "今天下雨，想吃点热乎的带汤的，不要辣",
  "top_k": 5
}
```

## API

- `GET /`：美食推荐网页
- `GET /health`：服务健康检查
- `POST /recommend`：执行检索和 DeepSeek 推荐生成

Python 函数调用：

```python
from src.rag.service import recommend_from_query

result = recommend_from_query("想吃热乎的汤，不要辣")
print(result["answer"])
```

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
