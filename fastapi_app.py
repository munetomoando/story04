from fastapi import FastAPI, Query, Form, Depends, Request
import pandas as pd
from recommendation import recommend_for_all_users, user_object_matrix, user_zscore_matrix, user_similarity, get_username
from utils import load_object_names
import os
import csv
import bcrypt
import logging
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Ellipse
from io import BytesIO
import base64
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles
from recommendation import get_recommendations_for_user
from dotenv import load_dotenv
import asyncio
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import APIRouter
import urllib.parse
from scipy.stats import chi2



# .envファイルの読み込み
load_dotenv()

app = FastAPI()

_cached_merged_df = None
_cached_object_id_to_name = None
_cached_user_dict = None
_cached_recommend_df = None

@app.api_route("/", methods=["GET", "HEAD"])
async def login_page(request: Request, error_message: str = ""):
    """ログインページを表示（GETおよびHEAD対応）"""
    user = request.session.get("user_id")
    if user:
        return RedirectResponse(url=f"/rating", status_code=303)
    return templates.TemplateResponse("index.html", {"request": request, "error_message": error_message})

# ✅ 環境変数からシークレットキーを取得（設定されていない場合はデフォルト値）
SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "default-secret-key")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# テンプレート設定
templates = Jinja2Templates(directory="templates")

# users.csv のパスを明示的に指定
USERS_FILE = "/opt/render/project/src/users.csv"
OBJECTS_FILE = "/opt/render/project/src/objects.csv"
RATINGS_FILE = "/opt/render/project/src/ratings.csv"


# ✅ `objects.csv` の読み込み（エラーハンドリング付き）
object_dict = {}
try:
    objects_df = pd.read_csv(OBJECTS_FILE, encoding="utf-8-sig", dtype={"object_id": str})
    object_dict = dict(zip(objects_df["object_id"].astype(str), objects_df["object_name"].fillna("Unknown")))
except FileNotFoundError:
    logging.warning(f"Warning: {OBJECTS_FILE} not found. Using empty object dictionary.")
    object_dict = {}

# `ratings.csv` の存在をチェックしてデータフレームを初期化
if os.path.exists(RATINGS_FILE):
    ratings_df = pd.read_csv(RATINGS_FILE, encoding="utf-8-sig", dtype={"user_id": str, "object_id": str})
    ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce")  # 数値変換（エラー時は NaN）

    # `object_id` を明示的に str に変換
    ratings_df["object_id"] = ratings_df["object_id"].astype(str)

    # `object_id` を `object_name` に変換（object_id が `object_dict` にある場合のみ）
    if not ratings_df.empty:
        ratings_df["object_name"] = ratings_df["object_id"].map(object_dict)
        print("🔍 `ratings_df` の `object_name` のサンプル:\n", ratings_df.head())  # デバッグ用出力
else:
    # `ratings.csv` が存在しない場合は作成
    with open(RATINGS_FILE, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["user_id", "object_id", "rating"])  # ヘッダーを追加
    ratings_df = pd.DataFrame(columns=["user_id", "object_id", "rating"])  # 空のDataFrameを作成

app.mount("/static", StaticFiles(directory="static", html=True), name="static")

def update_user_similarity():
    global user_similarity

    # `user_object_matrix` を取得し、NaN を 0 に置き換え
    user_object_matrix_filled = user_object_matrix.fillna(0)

    # コサイン類似度を計算
    new_similarity_matrix = pd.DataFrame(
        cosine_similarity(user_object_matrix_filled),
        index=user_object_matrix_filled.index,
        columns=user_object_matrix_filled.index
    )

    # `user_similarity` を更新
    user_similarity = new_similarity_matrix

@app.get("/routes")
async def get_routes():
    return [{"path": route.path, "name": route.name} for route in app.router.routes]

@app.get("/")
def show_login_page(request: Request, error_message: str = ""):
    """ログインページを表示"""
    user = request.session.get("user")
    return templates.TemplateResponse("index.html", {"request": request, "error_message": error_message, "messages": []})

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """ ユーザーログイン処理 """

    # `users.csv` が存在しない場合、エラーメッセージを表示
    if not os.path.exists(USERS_FILE):
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error_message": "ユーザーが登録されていません。"
        })

    # `users.csv` からデータを読み込む
    with open(USERS_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader, None)  # ヘッダーをスキップ

        for row in reader:
            stored_user_id, stored_username, stored_password_hash = row

            # 🔍 ユーザー名が一致するか確認
            if stored_username == username:
                # 🔍 パスワードが一致するか確認
                if bcrypt.checkpw(password.encode(), stored_password_hash.encode()):
                    request.session["username"] = username
                    request.session["user_id"] = stored_user_id

                    return RedirectResponse(url="/rating", status_code=303)

                return templates.TemplateResponse("index.html", {
                    "request": request,
                    "error_message": "パスワードが間違っています。"
                })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "error_message": "ユーザー名が存在しません。"
    })

router = APIRouter()

@router.post("/logout")
async def logout(request: Request):
    """ セッションをクリアし、クッキーを削除してログアウト """
    request.session.clear()  # ✅ セッションを削除
    response = RedirectResponse(url="/index", status_code=303)
    response.delete_cookie("session")  # ✅ クッキーも削除
    return response

@app.get("/index")
async def show_index_page(request: Request):
    """ 明示的に index.html を表示するルート """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/objects")
def get_objects():
    """レストラン一覧を取得"""
    return {"object_names": list(object_dict.values())}  # API ではリストとして返す

@app.get("/recommend/{user_id}")
def get_recommendations(request: Request, user_id: str):
    """特定のユーザーに対する推薦結果を取得"""
    user = request.session.get("user_id")
    if not user or user != user_id:
        return RedirectResponse(url="/", status_code=303)
    
    # 全ユーザーの推薦データを取得（データフレーム）
    recommendation_df, user_similarity_data = recommend_for_all_users()

    # 指定した `user_id` に対する推薦データをフィルタリング
    user_recommendations = recommendation_df[recommendation_df["user_id"] == user_id]

    # ユーザー名を取得
    username = get_username(user_id)

    # 推薦リストを辞書に変換
    recommendations = [
        {
            "object_id": str(row["object_id"]),
            "object_name": object_dict.get(str(row["object_id"]), "不明"),
            "recommendation_score": round(row["recommendation_score"], 2)  # 0.25 以上のものを考慮
        }
        for _, row in user_recommendations.iterrows()
        if pd.notna(row["recommendation_score"])
    ]

    return templates.TemplateResponse("recommendations.html", {
        "request": request,
        "username": username,
        "recommendations": recommendations,
        "has_similar_users": len(recommendations) > 0
    })

# ユーザー名の存在チェック関数
def is_username_taken(username: str) -> bool:
    """ ユーザー名が既に登録されているか確認 """
    if not os.path.exists(USERS_FILE):
        return False  # ファイルがない場合は存在しない
    
    with open(USERS_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[1] == username:
                return True  # 既に存在する
    return False  # 存在しない

# ユーザー名の存在チェックAPI（フロントエンド用）
@app.get("/check_username")
def check_username(username: str = Query(...)):
    """ ユーザー名の重複チェックAPI """
    return {"exists": is_username_taken(username)}

@app.get("/register")
async def show_register_page(request: Request):
    """新規登録ページを開く際にログアウトする"""
    request.session.clear()  # ✅ セッションを削除してログアウト
    response = templates.TemplateResponse("register.html", {"request": request})
    response.delete_cookie("session")  # ✅ クッキーも削除
    return response

# 新規ユーザー登録処理
@app.post("/register")
async def register_user(request: Request, username: str = Form(...), password: str = Form(...)):
    """ 新規ユーザー登録処理 """
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["user_id", "username", "password_hash"])  # ヘッダー追加

    # 既存のユーザーを取得
    existing_users = []
    with open(USERS_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader, None)  # ヘッダーをスキップ
        for row in reader:
            # 万が一行が不正な場合の対策として長さチェック
            if len(row) >= 2:
                existing_users.append(row[1])

    if username in existing_users:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error_message": "このユーザー名は既に登録されています。"
        })

    user_id = str(len(existing_users) + 1)
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    # 末尾に改行があるかをバイナリモードで確認
    newline_needed = False
    if os.path.getsize(USERS_FILE) > 0:
        with open(USERS_FILE, "rb") as file:
            file.seek(-1, os.SEEK_END)
            last_byte = file.read(1)
            if last_byte != b'\n':
                newline_needed = True

    with open(USERS_FILE, "a", newline="", encoding="utf-8") as file:
        if newline_needed:
            file.write("\n")
        writer = csv.writer(file)
        writer.writerow([user_id, username, hashed_password])
    
    _cached_merged_df, _cached_object_id_to_name, _cached_recommend_df, _cached_user_dict = create_merged_df()

    return RedirectResponse(url="/", status_code=303)

def is_object_exists(object_name: str) -> bool:
    """ 指定された評価対象が `objects.csv` に既に存在するかチェック """
    if not os.path.exists(OBJECTS_FILE):
        return False  # ファイルが存在しなければ、評価対象は存在しない
    
    with open(OBJECTS_FILE, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        existing_objects = {row[0] for row in reader if row}  # 既存のオブジェクト名を取得
    
    return object_name in existing_objects

@app.get("/add_objects", response_class=HTMLResponse)
async def show_add_objects_page(request: Request, success: bool = False, message: str = ""):
    messages = [message] if success and message else []
    return templates.TemplateResponse(
        "add_objects.html",
        {
            "request": request,
            "messages": messages  # ✅ messages が未定義の場合、空リストを渡す
        }
    )

@app.post("/add_objects")
async def add_objects(request: Request, object_names: str = Form(...)):
    """ 新しい評価対象（飲食店）を `objects.csv` に追加 """

    #  カンマ区切りで複数の飲食店名をリストに変換
    new_objects = [name.strip() for name in object_names.split(",") if name.strip()]

    if not new_objects:
        return templates.TemplateResponse("add_objects.html", {
            "request": request,
            "error_message": "評価対象を入力してください。"
        })

    #  ファイルがなければ作成
    if not os.path.exists(OBJECTS_FILE):
        with open(OBJECTS_FILE, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["object_id", "object_name"])  # ヘッダー追加

    #  既存のオブジェクトを取得（辞書形式: {object_name: object_id}）
    existing_objects = {}
    with open(OBJECTS_FILE, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        next(reader, None)  # ヘッダーをスキップ
        for row in reader:
            if len(row) == 2:
                existing_objects[row[1]] = str(row[0])  # {object_name: object_id}

    #  追加するオブジェクトのリストを作成
    added_objects = []
    next_id = str(len(existing_objects) + 1) 
    for obj in new_objects:
        if obj not in existing_objects:  # 重複しない場合のみ追加
            added_objects.append((next_id, obj))
            next_id = str(int(next_id) + 1)  # object_id を増やす

    if not added_objects:
        return templates.TemplateResponse("add_objects.html", {
            "request": request,
            "error_message": "すべての評価対象が既に登録されています。"
        })

    #  CSV に新しい評価対象を追加
    with open(OBJECTS_FILE, "a", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerows(added_objects)  # まとめて書き込む

    _cached_merged_df, _cached_object_id_to_name, _cached_recommend_df, _cached_user_dict = create_merged_df()

    # ✅ メッセージをURLクエリパラメータとして渡す（エンコード処理）
    message = urllib.parse.quote("評価対象が追加されました！")

    # ✅ 登録成功後に `messages` をクエリパラメータとして渡す
    return RedirectResponse(url=f"/add_objects?success=true&message={message}", status_code=303)

@app.get("/rating")
async def show_rating_page(request: Request):
    """評価ページを表示"""

    # ユーザーがログインしていない場合はリダイレクト
    username = request.session.get("username")
    user_id = request.session.get("user_id")
    if not username:
        return RedirectResponse(url="/", status_code=303)

    # `user_id` を文字列として統一（`ratings.csv` のデータ型不一致対策）
    user_id = str(user_id)

    # objects.csv から評価対象を取得（object_idの降順にソート）
    objects = {}
    if os.path.exists(OBJECTS_FILE):
        objects_df = pd.read_csv(OBJECTS_FILE, encoding="utf-8-sig")

        # object_id を整数型に変換して降順ソート
        objects_df["object_id"] = objects_df["object_id"].astype(str)
        objects_df["object_id"] = objects_df["object_id"].astype(int)

        objects_df = objects_df.sort_values("object_id", ascending=False)
        objects_df["object_id"] = objects_df["object_id"].astype(str)

        # object_id をキー、object_name を値とする辞書に変換
        objects = dict(zip(objects_df["object_id"], objects_df["object_name"]))

    # ratings.csv が存在しない場合は作成
    if not os.path.exists(RATINGS_FILE):
        with open(RATINGS_FILE, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["user_id", "object_id", "rating"])  # ヘッダーを追加

    # `ratings.csv` から最新の評価のみ取得
    past_ratings = {object_id: None for object_id in objects.keys()}  # 未評価のオブジェクトは None にする
    if os.path.exists(RATINGS_FILE) and os.stat(RATINGS_FILE).st_size > 0:
        df = pd.read_csv(RATINGS_FILE, encoding="utf-8-sig", dtype={"user_id": str, "object_id": str})

        # 最新の評価のみ取得（`user_id` が文字列になっていることを確認）
        latest_ratings = (df[df["user_id"] == user_id].sort_values(by=["user_id", "object_id", "rating"]).groupby("object_id")["rating"].last().to_dict())

        # `past_ratings` に最新の評価を反映
        for object_id in objects.keys():
            if object_id in latest_ratings:
                past_ratings[object_id] = latest_ratings[object_id]  # 評価済みのオブジェクトは最新の評価を設定

    # **評価値のリスト**
    rating_values = ["", "1", "2", "3", "4", "5"]
    rating_labels = ["?", "1", "2", "3", "4", "5"]

    # **Python 側で `zip()` を使ってリスト化**
    rating_pairs = list(zip(rating_values, rating_labels))
    
    return templates.TemplateResponse("rating.html", {
        "request": request,
        "username": username,
        "objects": objects,  # object_id → object_name に変換
        "past_ratings": past_ratings,  # 最新の評価を表示
        "rating_pairs": rating_pairs,  # 評価値と表示ラベルのペア
    })


@app.post("/submit_ratings")
async def submit_ratings(request: Request):
    """評価を ratings.csv に更新"""

    logging.info("submit_ratings() called")  # ✅ ログを追加
    form_data = await request.form()
    ratings = {key.replace("ratings[", "").replace("]", ""): value for key, value in form_data.items() if "ratings[" in key}
    delete_flags = {key.replace("delete_", ""): value for key, value in form_data.items() if key.startswith("delete_")}

    if not ratings and not delete_flags:
        logging.warning("No ratings provided")  # ✅ ログを追加
        return JSONResponse(content={"detail": "No ratings provided"}, status_code=400)

    user_id = request.session["user_id"]
    logging.info(f"Processing ratings for user_id: {user_id}")  # ✅ ログを追加

    # `ratings.csv` を DataFrame として読み込む
    if os.path.exists(RATINGS_FILE):
        df = pd.read_csv(RATINGS_FILE, encoding="utf-8-sig")
    else:
        df = pd.DataFrame(columns=["user_id", "object_id", "rating"])
    
    # 確実に user_id を int 型に変換
    if not df.empty:
        df["user_id"] = df["user_id"].astype(int)


    # ユーザーの過去の評価を削除（更新するため）
    df = df[df["user_id"] != int(user_id)]

    # 削除フラグが `true` のものは、新しい評価に追加しない
    new_ratings = pd.DataFrame(
        [[user_id, object_id, int(rating)] for object_id, rating in ratings.items() if rating and delete_flags.get(object_id) != "true"],
        columns=["user_id", "object_id", "rating"]
    )

    # 最新のデータで `ratings.csv` を上書き
    df = pd.concat([df, new_ratings], ignore_index=True).drop_duplicates() 

    try:
        df.to_csv(RATINGS_FILE, index=False, encoding="utf-8-sig")
        logging.info("ratings.csv successfully updated")  # ✅ ログを追加
    except Exception as e:
        logging.error(f"Error writing to ratings.csv: {e}")  # ✅ エラーログを追加
        return JSONResponse(content={"detail": "Failed to update ratings.csv"}, status_code=500)
    
    # 書き込み後に短い遅延を入れる
    await asyncio.sleep(0.1)
    # 更新を反映
    try:
        await update_ratings_no_reload()
        logging.info("update_ratings_no_reload() completed")  # ✅ ログを追加
    except Exception as e:
        logging.error(f"Error in update_ratings_no_reload(): {e}")  # ✅ エラーログを追加
        return JSONResponse(content={"detail": "Failed to update ratings"}, status_code=500)

    logging.info("Redirecting to recommendations")  # ✅ ログを追加
    return RedirectResponse(url="/recommendations", status_code=303)


async def update_ratings_no_reload():
    """`ratings.csv` を最新のデータに更新"""
    global _cached_merged_df, _cached_object_id_to_name, _cached_user_dict, _cached_recommend_df

    try:
        _cached_merged_df, _cached_object_id_to_name, _cached_recommend_df, _cached_user_dict = create_merged_df()
        logging.info("Cache successfully updated in update_ratings_no_reload()")
    except Exception as e:
        logging.error(f"Error in update_ratings_no_reload(): {e}")
        raise

@app.get("/recommendations")
async def recommendations_page(request: Request):
    """おすすめ結果を表示"""
    username = request.session.get("username")
    user_id = request.session.get("user_id")
    if not username:
        return RedirectResponse(url="/", status_code=303)
    
    # ✅ `copy()` を使ってスライスの影響を防ぐ
    recommendations_df, user_similarity_data = recommend_for_all_users()  # ✅ タプルを展開して代入
    user_recommendations = recommendations_df[recommendations_df["user_id"] == user_id].copy()

    # ✅ `object_id` から `object_name` を取得
    user_recommendations["object_name"] = user_recommendations["object_id"].map(object_dict)

    # ✅ `recommendation_score` を小数点以下2桁に丸める
    user_recommendations["recommendation_score"] = user_recommendations["recommendation_score"].round(2)

    return templates.TemplateResponse("recommendations.html", {
        "request": request,
        "username": username,
        "recommendations": user_recommendations.to_dict(orient="records"),
    })


def categorize_recommendation(score):
    try:
        score_val = float(score)
    except (ValueError, TypeError):
        return "-"
    if score_val < 0.25:
        return "-"
    elif score_val >= 0.75:
        return "★★★"
    elif score_val >= 0.5:
        return "★★"
    elif score_val >= 0.25:
        return "★"
    
def format_cell(row):
    # rating のフォーマット
    try:
        rating_val = float(row["rating"])
        rating = f"{rating_val:.1f}"
    except (ValueError, TypeError):
        rating = str(row["rating"])
        
    # z_score のフォーマット
    try:
        z_val = float(row["z_score"])
        z_score = f"{z_val:.2f}"
    except (ValueError, TypeError):
        z_score = str(row["z_score"])
        
    # recommendation_score のフォーマット
    try:
        rec_val = float(row["recommendation_score"])
        rec_score = f"{rec_val:.2f}"
    except (ValueError, TypeError):
        rec_score = str(row["recommendation_score"])

    stars = categorize_recommendation(row["recommendation_score"])
    return f"R: {rating}\nZ: {z_score}\nRS: {rec_score}\n{stars}"


def create_merged_df():
    global _cached_merged_df, _cached_object_id_to_name, _cached_user_dict, _cached_recommend_df, user_similarity

    # --- 推薦データとユーザー類似度の計算 ---
    recommend_df, computed_user_similarity = recommend_for_all_users()
    # ここでグローバルのuser_similarityも更新する
    user_similarity = computed_user_similarity
    # --- データの読み込み ---
    users = pd.read_csv(USERS_FILE, encoding="utf-8-sig") if os.path.exists(USERS_FILE) else pd.DataFrame(columns=["user_id", "username"])
    objects = pd.read_csv(OBJECTS_FILE, encoding="utf-8-sig") if os.path.exists(OBJECTS_FILE) else pd.DataFrame(columns=["object_id", "object_name"])
    ratings = pd.read_csv(RATINGS_FILE, encoding="utf-8-sig") if os.path.exists(RATINGS_FILE) else pd.DataFrame(columns=["user_id", "object_id", "rating"])

    # ✅ `user_id` をすべて `str` に統一
    users["user_id"] = users["user_id"].astype(str)
    ratings["user_id"] = ratings["user_id"].astype(str)
    ratings["object_id"] = ratings["object_id"].astype(str)
    objects["object_id"] = objects["object_id"].astype(str)

    # --- ユーザーID, オブジェクトID のマッピング ---
    _cached_user_dict = dict(zip(users["user_id"], users["username"]))
    object_dict_local = dict(zip(objects["object_id"], objects["object_name"]))
    ratings["username"] = ratings["user_id"].map(_cached_user_dict)
    ratings["object_name"] = ratings["object_id"].map(object_dict_local)

    recommend_df["user_id"] = recommend_df["user_id"].astype(str)
    recommend_df["object_id"] = recommend_df["object_id"].astype(str)
    recommend_df["recommendation_score"] = recommend_df["recommendation_score"].astype(float)

    # --- `ratings` が空なら適切なデフォルトデータを設定 ---
    if ratings.empty:
        ratings = pd.DataFrame(columns=["user_id", "object_id", "rating"])

    # --- ユーザー × オブジェクトのピボットテーブル ---
    user_object_matrix_df = ratings.pivot(index="user_id", columns="object_id", values="rating").reset_index()

    # --- user_zscore_matrix の計算 ---
    user_zscore_matrix_df = user_object_matrix_df.copy()
    numeric_cols = user_zscore_matrix_df.columns.drop("user_id", errors="ignore")
    user_zscore_matrix_df[numeric_cols] = user_zscore_matrix_df[numeric_cols].apply(
        lambda x: (x - x.mean()) / (x.std() if x.std() != 0 else 1), axis=1
    )

    # --- `user_ratings` を作成 ---
    user_ratings = user_object_matrix_df.melt(id_vars=["user_id"], var_name="object_id", value_name="rating")

    # --- `user_z_scores` を作成 ---
    user_z_scores = user_zscore_matrix_df.melt(id_vars=["user_id"], var_name="object_id", value_name="z_score")

    # --- `merged_df` を作成 ---
    merged_df = pd.merge(user_ratings, user_z_scores, on=["user_id", "object_id"], how="outer")

    # --- `recommend_df` との結合 ---
    merged_df = pd.merge(merged_df, recommend_df, on=["user_id", "object_id"], how="left")


    # --- `recommendation_score` の欠損値を埋める ---
    if "recommendation_score" in merged_df.columns:
        merged_df["recommendation_score"] = merged_df["recommendation_score"].fillna(0).astype(float)
    else:
        print("❌ `recommendation_score` カラムが `merged_df` にありません。マージの処理を確認してください。")

    # --- 欠損値の処理 ---
    merged_df["rating"] = merged_df["rating"].fillna("-")
    merged_df["z_score"] = merged_df["z_score"].fillna("-")

    merged_df["username"] = merged_df["user_id"].map(_cached_user_dict)
    merged_df["object_name"] = merged_df["object_id"].map(object_dict_local)

    merged_df["cell_info"] = merged_df.apply(format_cell, axis=1)

    # --- object_id → object_name の辞書作成 ---
    object_id_to_name = dict(zip(objects["object_id"], objects["object_name"].fillna("Unknown")))

    # merged_df の生成処理の最後で
    _cached_merged_df = merged_df
    _cached_object_id_to_name = object_id_to_name
    _cached_recommend_df = recommend_df
    
    print("🔍 `merged_df` の先頭:\n", merged_df.head())

    return merged_df, object_id_to_name, recommend_df, _cached_user_dict

@app.get("/admin_reviews", response_class=HTMLResponse)
async def admin_reviews(request: Request):
    global _cached_merged_df, _cached_object_id_to_name, _cached_recommend_df, _cached_user_dict, user_similarity

    _cached_merged_df, _cached_object_id_to_name, _cached_recommend_df, _cached_user_dict = create_merged_df()
    
    if user_similarity is None or user_similarity.empty:
        return HTMLResponse(content="ユーザー類似度データが利用できません", status_code=500)
    
    # グラフ生成やピボットテーブルの作成などをまとめて実行
    (
        heatmap_base64,
        similarity_hist_base64,
        cluster_df,
        cluster_plot_base64,
        recommendation_hist_base64,
        pivot_compact,
        object_name_cols
    ) = update_heatmap()

    # ✅ クラスタ表の2種類の並び替え
    cluster_sorted_by_username = cluster_df.sort_values(
        ["username"], key=lambda x: x.str.casefold()
    )
    cluster_sorted_by_cluster = cluster_df.sort_values(
        ["cluster", "username"], 
        key=lambda x: x.str.casefold() if x.name == "username" else x
    )

    # 最後にテンプレートを返す
    return templates.TemplateResponse(
        "admin_reviews.html",
        {
            "request": request,
            "pivot_table": pivot_compact.to_dict(orient="records"),
            "column_names": object_name_cols,
            "heatmap_base64": heatmap_base64,
            "similarity_hist_base64": similarity_hist_base64,
            "cluster_sorted_by_username": cluster_sorted_by_username.to_dict(orient="records"),
            "cluster_sorted_by_cluster": cluster_sorted_by_cluster.to_dict(orient="records"),
            "cluster_plot_base64": cluster_plot_base64,
            "recommendation_hist_base64": recommendation_hist_base64
        },
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )

# =============================
# グラフなどを生成する補助関数
# =============================
def update_heatmap():
    """
    update_heatmap() は、admin_reviews() 内で利用する可視化処理をまとめた関数。
    グラフやピボットテーブルなどを作り、Base64文字列やDataFrameを返す。
    """
    global user_similarity, _cached_merged_df, _cached_object_id_to_name, _cached_user_dict, _cached_recommend_df

    # ========== ピボットテーブルの作成 ==========
    # _cached_merged_df から pivot_compact を作る
    _cached_merged_df["user_id"] = _cached_merged_df["user_id"].astype(str)
    _cached_merged_df["object_id"] = _cached_merged_df["object_id"].astype(str)
    _cached_merged_df["cell_info"] = _cached_merged_df.apply(format_cell, axis=1)

    pivot_compact = _cached_merged_df.pivot_table(
        index=["user_id", "username"],
        columns="object_id",
        values="cell_info",
        aggfunc="first"
    ).reset_index()

    # object_id → object_name
    object_mapping = _cached_merged_df.set_index("object_id")["object_name"].to_dict()

    # カラムの再構築
    new_columns = ["user_id", "username"] + list(pivot_compact.columns[2:])
    pivot_compact.columns = new_columns

    # カラムのうち数字（object_id）だけ取り出し、それをソート
    valid_columns = [col for col in pivot_compact.columns[2:] if col.isdigit()]
    missing_keys = [obj for obj in valid_columns if obj not in object_mapping]
    if missing_keys:
        print(f"⚠️ object_mapping に存在しないキー: {missing_keys}")

    pivot_compact["user_id"] = pivot_compact["user_id"].astype(int)
    pivot_compact = pivot_compact.sort_values(by="user_id")
    # ソート後に再度文字列型に戻す（必要であれば）
    pivot_compact["user_id"] = pivot_compact["user_id"].astype(str)

    ordered_object_names = sorted(object_mapping.keys(), key=lambda x: int(x))
    existing_columns = [col for col in ordered_object_names if col in pivot_compact.columns]
    ordered_columns = ["user_id", "username"] + existing_columns
    pivot_compact = pivot_compact[ordered_columns] 

    pivot_compact = pivot_compact.rename(columns=object_mapping)

    # 余分にユーザー情報の列などを除いた実際の object_name 列だけを取得
    object_name_cols = pivot_compact.columns[2:].tolist()

    # ========== Heatmap (User Similarity) ==========
    # user_similarity のインデックスを数値順にソート
    sorted_index = sorted(user_similarity.index, key=lambda x: int(x))
    user_similarity = user_similarity.loc[sorted_index, sorted_index]

    user_labels = [
        f"{uid} ({_cached_user_dict.get(str(uid), 'Unknown')})" for uid in user_similarity.index
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        user_similarity[::-1],
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        ax=ax,
        xticklabels=user_labels,
        yticklabels=user_labels[::-1]
    )
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    img_buf = BytesIO()
    plt.savefig(img_buf, format="png")
    plt.close(fig)
    img_buf.seek(0)
    heatmap_base64 = base64.b64encode(img_buf.getvalue()).decode("utf-8")

    # ========== ユーザー類似度のヒストグラム ==========
    similarity_values = user_similarity.values.flatten()
    similarity_values = similarity_values[similarity_values < 1.0]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(similarity_values, bins=20, kde=True, ax=ax)
    ax.set_xlabel("User Similarity", fontsize=12)
    ax.set_ylabel("頻度", fontsize=12)
    ax.set_title("Histogram of User Similarity", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    img_buf = BytesIO()
    plt.savefig(img_buf, format="png", bbox_inches="tight")
    plt.close(fig)
    img_buf.seek(0)
    similarity_hist_base64 = base64.b64encode(img_buf.getvalue()).decode("utf-8")

    # まず、最新のユーザー情報を取得する関数を定義
    def update_user_info_from_csv():
        user_df = pd.read_csv(USERS_FILE, encoding="utf-8-sig", dtype={"user_id": str})
        return dict(zip(user_df["user_id"], user_df["username"]))

    # get_username() も最新の _cached_user_dict を使うように定義
    def get_username(user_id):
        # user_id を文字列にしてから辞書参照することで確実に取得
        return _cached_user_dict.get(str(user_id), "Unknown")

    # ========== ユーザークラスタリング ==========
    # クラスタリング直前に、最新のユーザー情報で _cached_user_dict を更新
    _cached_user_dict = update_user_info_from_csv()
    # PCA + KMeans
    num_users = user_similarity.shape[0]
    num_clusters = max(2, int(num_users / 4) + 1)  #クラスター数の決定ルール（平均4人になるように分割）

    if isinstance(user_similarity, np.ndarray):
        user_similarity_df = pd.DataFrame(
            user_similarity,
            index=_cached_user_dict.keys(),
            columns=_cached_user_dict.keys()
        )
    else:
        user_similarity_df = user_similarity

    pca = PCA(n_components=2)
    user_embedding = pca.fit_transform(user_similarity_df)
    user_ids = list(user_similarity_df.index)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(user_embedding)

    cluster_df = pd.DataFrame({
        "user_id": [str(uid) for uid in user_ids],
        "username": [get_username(uid) for uid in user_ids],
        "cluster": clusters + 1
    }).sort_values(["cluster"], ascending=True).sort_values(["username"], key=lambda x: x.str.casefold())

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["red", "blue", "green", "purple", "orange", "cyan", "magenta", "brown"]
    num_colors = min(num_clusters, len(colors))

    for cluster_id in range(kmeans.n_clusters):
        cluster_users = cluster_df[cluster_df["cluster"] == cluster_id + 1]
        indices = cluster_users.index.to_numpy()
        cluster_points = user_embedding[indices, :2]

        ax.scatter(
            cluster_points[:, 0], cluster_points[:, 1],
            label=f"Cluster {cluster_id + 1}",
            alpha=0.6,
            color=colors[cluster_id % len(colors)]
        )
        for (x, y), username in zip(cluster_points, cluster_users["username"]):
            ax.text(x, y, username, fontsize=10, ha='right', va='bottom', color="black")

        # クラスタ楕円
        if len(cluster_points) > 1:
            x_mean, y_mean = cluster_points[:, 0].mean(), cluster_points[:, 1].mean()
            cov = np.cov(cluster_points, rowvar=False)
            # 対称行列なので eigh を利用（固有値は昇順に返る）
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            order = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[order]
            eigenvecs = eigenvecs[:, order]
            # 95%信頼楕円のためのスケーリング因子
            confidence_level = 0.95
            chi2_val = np.sqrt(chi2.ppf(confidence_level, df=2))
            # 幅と高さ（楕円の長軸・短軸は2倍の標準偏差×スケーリング因子）
            width = 2 * chi2_val * np.sqrt(eigenvals[0])
            height = 2 * chi2_val * np.sqrt(eigenvals[1])
            # 回転角度は arctan2 を用いて計算
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))

            ell = Ellipse(
                xy=(x_mean, y_mean),
                width=width,
                height=height,
                angle=angle,
                edgecolor=colors[cluster_id % len(colors)],
                facecolor="none",
                linewidth=2
            )
            ax.add_patch(ell)
        elif len(cluster_points) == 1:
            x, y = cluster_points[0]
            ell = Ellipse(
                xy=(x, y),
                width=0.2,
                height=0.2,
                edgecolor=colors[cluster_id % len(colors)],
                facecolor="none",
                linewidth=2,
                linestyle="dashed"
            )
            ax.add_patch(ell)

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("User Clustering Visualization")
    ax.legend()
    img_buf = BytesIO()
    plt.savefig(img_buf, format="png")
    plt.close(fig)
    img_buf.seek(0)
    cluster_plot_base64 = base64.b64encode(img_buf.getvalue()).decode("utf-8")

    # ========== 推薦スコアのヒストグラム ==========
    recommendation_scores = _cached_recommend_df["recommendation_score"].dropna()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(recommendation_scores, bins=20, kde=True, ax=ax)
    ax.set_xlabel("Recommendation Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Recommendation Scores")
    img_buf = BytesIO()
    plt.savefig(img_buf, format="png")
    plt.close(fig)
    img_buf.seek(0)
    recommendation_hist_base64 = base64.b64encode(img_buf.getvalue()).decode("utf-8")

    # 必要な情報をまとめて返す
    return (
        heatmap_base64,
        similarity_hist_base64,
        cluster_df,
        cluster_plot_base64,
        recommendation_hist_base64,
        pivot_compact,
        object_name_cols
    )
