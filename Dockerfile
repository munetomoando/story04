# ---- base image ----
FROM python:3.11-slim

# OSパッケージ（必要に応じて追加）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl fonts-noto-cjk fontconfig && \
    rm -rf /var/lib/apt/lists/* && \
    fc-cache -fv

# 作業ディレクトリ
WORKDIR /app

# 依存を先にコピーしてキャッシュを効かせる
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# アプリ本体コピー
COPY . /app

COPY data_seed /app/data_seed

# デフォルトポート（Fly.io: 8080, Render: 10000 等 — PORT 環境変数で上書き可能）
ENV PORT=8080
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
