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

# ポートは fly の標準 8080 に合わせる
ENV PORT=8080
# Uvicorn を 0.0.0.0:8080 で起動
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
