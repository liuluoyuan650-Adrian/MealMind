$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    py -3.10 -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120

$env:MEALMIND_FORCE_KEYWORD_RETRIEVER = "1"
$env:MEALMIND_TEMPLATE_ANSWER = "1"

Write-Host "MealMind is starting at http://127.0.0.1:8000"
.\.venv\Scripts\python.exe -m uvicorn src.rag.api:app --host 127.0.0.1 --port 8000
