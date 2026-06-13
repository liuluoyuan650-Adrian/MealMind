# MealMind 运行说明

## 模式一：完整 RAG 模式

这是正常使用方式：

- BGE + FAISS 从本地菜单检索 5 道候选菜
- 显式关闭 DeepSeek 深度思考
- 将 5 道候选菜传给 DeepSeek
- DeepSeek 从中推荐 2-3 道
- 如果 DeepSeek 临时失败，自动改用本地模板生成话术

启动命令：

```powershell
cd C:\PythonProject\MealMind
$env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"
Remove-Item Env:MEALMIND_FORCE_KEYWORD_RETRIEVER -ErrorAction SilentlyContinue
Remove-Item Env:MEALMIND_TEMPLATE_ANSWER -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m uvicorn src.rag.api:app --host 127.0.0.1 --port 8000
```

## 模式二：离线稳定演示模式

没有 Hugging Face 网络或不希望消耗 DeepSeek API 时，可以运行：

```powershell
.\run_demo.ps1
```

该脚本会设置：

```powershell
$env:MEALMIND_FORCE_KEYWORD_RETRIEVER="1"
$env:MEALMIND_TEMPLATE_ANSWER="1"
```

因此演示模式使用本地关键词检索和本地模板话术，不会调用 BGE Embedding，也不会
调用 DeepSeek。它适合答辩或断网演示，但不等同于完整 RAG 运行效果。

## 打开页面

服务启动后访问：

```text
http://127.0.0.1:8000
```

页面会显示：

- DeepSeek 或模板生成的推荐话术
- 5 道本地候选菜卡片
- 每道菜的热量、辣度、饱腹感、评分和匹配信息
- 重新推荐和更换需求按钮

开发接口：

```text
http://127.0.0.1:8000/docs
```

## 停止服务

在运行 Uvicorn 的 PowerShell 窗口按：

```text
Ctrl+C
```

不建议在文档中记录固定进程号，因为每次启动产生的 PID 都不同。

## 常见问题

### 页面显示 5 道，推荐文案只有 2-3 道

这是当前设计。系统检索 5 道候选并展示为卡片，DeepSeek 根据 Prompt 从中挑选
2-3 道重点推荐。

### DeepSeek 是否开启深度思考

没有。当前请求显式发送：

```json
{"thinking": {"type": "disabled"}}
```

### 修改菜品数据后怎么办

重新构建索引：

```powershell
.\.venv\Scripts\python.exe scripts\build_rag_index.py
```

### 如何运行测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
