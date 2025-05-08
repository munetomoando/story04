#!/bin/bash
set -e  # エラー発生時に即終了

# FastAPI をメインプロセスとして実行
exec uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT --workers 1