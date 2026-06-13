# MealMind RAG 美食推荐系统

MealMind 使用本地菜品数据完成检索增强推荐。系统优先使用中文 Embedding 和
FAISS 检索候选菜品，再将候选信息交给 DeepSeek 生成自然语言推荐；当模型或网络
不可用时，会自动使用本地兜底逻辑，避免页面直接报错。

## 当前配置

- Embedding：`BAAI/bge-small-zh-v1.5`
- 向量数据库：FAISS 本地索引
- 生成模型：`deepseek-v4-pro`
- 深度思考：显式关闭，调用时发送 `thinking: {"type": "disabled"}`
- 网页检索数量：5 道候选菜
- DeepSeek 最终推荐：从 5 道候选中挑选 2-3 道
- 页面展示：显示 DeepSeek 推荐话术，并展示 5 道本地候选菜卡片
- DeepSeek 最大输出：`420` tokens

深度思考是否开启与推荐数量互不影响。候选数量由前端请求中的 `top_k: 5` 决定，
最终推荐数量由 `src/rag/prompts.py` 中的 2-3 道约束决定。

## 核心流程

```text
用户自然语言需求
  -> BGE 查询向量
  -> FAISS 相似度召回并执行忌口硬过滤
  -> 选出 5 道本地候选菜
  -> 将 5 道候选作为 RAG 上下文传给 DeepSeek
  -> DeepSeek 从中推荐 2-3 道并解释理由
  -> 页面同时展示 5 道相关候选菜卡片
```

## 自动兜底

系统保留两级本地兜底：

1. Embedding 模型加载失败时，自动改用本地关键词检索。
2. DeepSeek 未配置或调用失败时，自动根据本地菜品信息生成模板推荐。

也可以通过环境变量强制启用：

```powershell
$env:MEALMIND_FORCE_KEYWORD_RETRIEVER="1"
$env:MEALMIND_TEMPLATE_ANSWER="1"
```

这两个变量主要用于断网演示。完整 RAG 模式不要设置它们。

## 项目结构

```text
MealMind/
  data/
    dish_dataset.csv          # 菜品事实源
    faiss_index/              # FAISS 索引和菜品 metadata
  scripts/
    build_rag_index.py        # 离线建库脚本
  src/rag/
    api.py                    # FastAPI 接口和网页入口
    config.py                 # Embedding 与 DeepSeek 配置
    data_ingestion.py         # 数据读取和结构化文本生成
    embedding.py              # sentence-transformers 封装
    llm.py                    # DeepSeek API 客户端
    prompts.py                # RAG Prompt 与输出约束
    retriever.py              # FAISS 检索、关键词兜底和忌口过滤
    service.py                # 推荐主流程和模板兜底
    static/                   # 普通用户网页前端
  tests/
    test_rag.py
  run_demo.ps1                # 离线稳定演示脚本
```

## 安装与建库

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

仓库已包含 `data/faiss_index/`。只有修改菜品数据或更换 Embedding 模型时才需重建：

```powershell
.\.venv\Scripts\python.exe scripts\build_rag_index.py
```

首次下载 BGE 模型需要访问 Hugging Face。下载完成后会优先从本地缓存加载。

## 配置 DeepSeek

可在 `src/rag/config.py` 中填写 API Key，也可以使用环境变量。环境变量优先：

```powershell
$env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"
```

不要将包含真实密钥的代码提交到公开仓库。已经公开过的密钥应立即作废并重新生成。

## 启动完整 RAG

完整模式使用 FAISS + BGE 检索，并优先使用 DeepSeek 生成话术：

```powershell
cd C:\PythonProject\MealMind
$env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"
Remove-Item Env:MEALMIND_FORCE_KEYWORD_RETRIEVER -ErrorAction SilentlyContinue
Remove-Item Env:MEALMIND_TEMPLATE_ANSWER -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m uvicorn src.rag.api:app --host 127.0.0.1 --port 8000
```

浏览器访问：

```text
http://127.0.0.1:8000
```

## 接口

Swagger 文档：`http://127.0.0.1:8000/docs`

`POST /recommend` 示例：

```json
{
  "query": "今天下雨，想吃点热乎的带汤的，不要辣",
  "top_k": 5
}
```

`top_k` 表示传入生成阶段的本地候选数量，不表示 DeepSeek 必须最终推荐同样数量。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
