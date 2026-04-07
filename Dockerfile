# ベースイメージ
FROM python:3.12-slim

# 作業ディレクトリ
WORKDIR /app

# 依存関係コピー
COPY requirements_docker.txt .

# インストール
RUN pip install --no-cache-dir -r requirements_docker.txt

# アプリコピー
COPY . .

# ポート開放
# FastAPIはデフォルトで8000番ポートを使用するため、ここで開放します。
EXPOSE 8000

# 起動コマンド
# FastAPIアプリを起動するためのコマンドです。uvicornを使用して、src.api.mainモジュールのappオブジェクトをホスト
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]