from fastapi import FastAPI, Query, Form, Depends, Request, HTTPException, UploadFile, File
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import pandas as pd
from recommendation import recommend_for_all_users, recommend_for_single_user, categorize_recommendation, explain_recommendations, user_object_matrix, user_zscore_matrix, get_username
import os
import sqlite3
import bcrypt
import logging
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語フォント設定（Noto Sans CJK が利用可能な場合）
for _font_name in ['Noto Sans CJK JP', 'Noto Sans CJK', 'IPAGothic', 'IPAPGothic']:
    if any(_font_name in f.name for f in fm.fontManager.ttflist):
        matplotlib.rcParams['font.family'] = _font_name
        break
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.patches import Ellipse
from io import BytesIO
import base64
import pathlib
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import asyncio
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import APIRouter
import urllib.parse
from scipy.stats import chi2
import datetime
from database import get_db, init_db, migrate_from_csv




# .envファイルの読み込み
load_dotenv()

app = FastAPI(
    docs_url="/admin/api-docs",
    redoc_url=None,
    openapi_url="/admin/openapi.json",
)

# レートリミット設定
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    from starlette.responses import HTMLResponse
    return HTMLResponse(
        content="<h2>リクエストが多すぎます。しばらく待ってから再度お試しください。</h2>",
        status_code=429
    )

import threading

# =============================
# アプリケーション設定定数
# =============================
BREAK_EVERY = 20                  # 評価の休憩間隔（件数）
MIN_RATINGS_FOR_POPULAR = 3       # 人気ランキングに載せる最低評価件数
FEW_RATINGS_THRESHOLD = 3         # ダッシュボード「評価少ない」アラートの閾値
INACTIVE_DAYS_THRESHOLD = 14      # ダッシュボード「離脱ユーザー」の日数閾値
STORE_VIEWS_RANKING_LIMIT = 20    # 店舗閲覧ランキングの表示件数
_ADMIN_PAGE_SIZE = 50             # 管理ページのページネーションサイズ

_cached_merged_df = None
_cached_object_id_to_name = None
_cached_user_dict = None
_cached_recommend_df = None
user_similarity = pd.DataFrame()  # app.py 独自のコピー（recommendation.py のものとは別）
_recommend_lock = threading.Lock()  # 推薦キャッシュの競合防止

# 管理画面チャート・ヒートマップのキャッシュ（TTL ベース）
_chart_cache: dict[str, tuple[float, object]] = {}  # key -> (expire_timestamp, data)
_CHART_CACHE_TTL = 300  # 5分

def _get_cached(key: str):
    """TTL ベースのキャッシュ取得。期限切れまたは未キャッシュなら None"""
    import time
    entry = _chart_cache.get(key)
    if entry and entry[0] > time.time():
        return entry[1]
    return None

def _set_cached(key: str, data):
    """TTL ベースのキャッシュ保存"""
    import time
    _chart_cache[key] = (time.time() + _CHART_CACHE_TTL, data)

@app.api_route("/", methods=["GET", "HEAD"])
async def login_page(request: Request, error_message: str = ""):
    """ログインページを表示（GETおよびHEAD対応）"""
    user = request.session.get("user_id")
    if user:
        return RedirectResponse(url=f"/rating", status_code=303)
    return templates.TemplateResponse("index.html", {"request": request, "error_message": error_message})

# 管理者認証情報（環境変数から取得）
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "")
INVITE_CODE = os.getenv("INVITE_CODE", "")

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse as StarletteRedirect
import secrets as _secrets

class AdminAuthMiddleware(BaseHTTPMiddleware):
    """全 /admin/* ルートに管理者セッションを要求するミドルウェア"""
    async def dispatch(self, request, call_next):
        if request.url.path.startswith("/admin") and request.url.path != "/admin/login":
            if not request.session.get("is_admin"):
                return StarletteRedirect(url="/admin/login", status_code=303)
        return await call_next(request)

CSRF_EXEMPT_PATHS: set[str] = set()  # 全POSTエンドポイントでCSRF検証を実施


class CSRFMiddleware(BaseHTTPMiddleware):
    """CSRFトークン生成・検証ミドルウェア（bodyキャッシュ方式）"""

    async def dispatch(self, request, call_next):
        if "csrf_token" not in request.session:
            request.session["csrf_token"] = _secrets.token_hex(16)

        if request.method == "POST" and request.url.path not in CSRF_EXEMPT_PATHS:
            # body を読み取ってキャッシュ
            body = await request.body()

            content_type = request.headers.get("content-type", "")
            if "multipart" in content_type:
                # multipart/form-data から csrf_token フィールドを抽出
                import re as _csrf_re
                token = ""
                match = _csrf_re.search(
                    rb'name="csrf_token"\r?\n\r?\n([^\r\n]+)',
                    body,
                )
                if match:
                    token = match.group(1).decode("utf-8", errors="ignore").strip()
                if token != request.session.get("csrf_token", ""):
                    from starlette.responses import Response
                    return Response("CSRF token mismatch", status_code=403)
            else:
                from urllib.parse import parse_qs
                params = parse_qs(body.decode("utf-8", errors="ignore"))
                token = params.get("csrf_token", [""])[0]
                if token != request.session.get("csrf_token", ""):
                    from starlette.responses import Response
                    return Response("CSRF token mismatch", status_code=403)

            # body を再読み取り可能にする
            async def receive():
                return {"type": "http.request", "body": body}
            request._receive = receive

        return await call_next(request)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """セキュリティヘッダーを全レスポンスに付与"""
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response

# ミドルウェアは後に追加したものが先に実行される
# 実行順: SessionMiddleware → CSRFMiddleware → AdminAuthMiddleware → SecurityHeaders
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(AdminAuthMiddleware)
app.add_middleware(CSRFMiddleware)

# セッション秘密鍵（未設定時はファイルに永続化して再起動後も維持）
SECRET_KEY = os.getenv("SESSION_SECRET_KEY")
if not SECRET_KEY:
    _key_file = pathlib.Path(os.getenv("DATA_DIR", "/data")) / ".session_secret_key"
    try:
        if _key_file.exists():
            SECRET_KEY = _key_file.read_text().strip()
        if not SECRET_KEY:
            SECRET_KEY = _secrets.token_hex(32)
            _key_file.parent.mkdir(parents=True, exist_ok=True)
            _key_file.write_text(SECRET_KEY)
            logging.info("Generated and persisted new session secret key")
    except Exception:
        SECRET_KEY = _secrets.token_hex(32)
        logging.warning("Could not persist session secret key to file, using ephemeral key")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)


# テンプレート設定
_jinja_templates = Jinja2Templates(directory="templates")


class _CSRFTemplates:
    """TemplateResponse に csrf_token を自動注入するラッパー"""
    def __getattr__(self, name):
        return getattr(_jinja_templates, name)

    def TemplateResponse(self, name, context, **kwargs):
        request = context.get("request")
        if request and "csrf_token" not in context:
            context["csrf_token"] = request.session.get("csrf_token", "")
        return _jinja_templates.TemplateResponse(name, context, **kwargs)

templates = _CSRFTemplates()

DATA_DIR = pathlib.Path(os.getenv("DATA_DIR", "/data"))
_docker_seed = pathlib.Path("/app/data_seed")
SEED_DIR = _docker_seed if _docker_seed.exists() else pathlib.Path(__file__).parent / "data_seed"
STORE_IMAGES_DIR = DATA_DIR / "store_images"
STORE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

@app.on_event("startup")
def startup():
    """DB 初期化・CSV→SQLite マイグレーション・キャッシュ復元"""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        init_db()
        migrate_from_csv(DATA_DIR, SEED_DIR)

        global object_dict, _cached_recommend_df
        # object_dict を SQLite からロード
        with get_db() as conn:
            rows = conn.execute("SELECT object_id, object_name FROM objects").fetchall()
            object_dict = {str(r["object_id"]): r["object_name"] for r in rows}

        # 推薦キャッシュを recommendations テーブルから復元
        try:
            with get_db() as conn:
                rows = conn.execute(
                    "SELECT CAST(user_id AS TEXT) as user_id, CAST(object_id AS TEXT) as object_id, recommendation_score FROM recommendations"
                ).fetchall()
                if rows:
                    _cached_recommend_df = pd.DataFrame(
                        [(r["user_id"], r["object_id"], r["recommendation_score"]) for r in rows],
                        columns=["user_id", "object_id", "recommendation_score"]
                    )
                    logging.info("Loaded recommendations from SQLite cache")
        except Exception as e:
            logging.warning(f"recommendations cache load failed: {e}")

        # 2年未ログインユーザーの論理削除 + 猶予30日超の物理削除
        # 2年未ログインユーザーの論理削除 + 猶予30日超の物理削除
        # 注意: login_logs にレコードがないユーザー（ログ導入前に登録）は対象外
        try:
            two_years_ago = (datetime.datetime.now() - datetime.timedelta(days=730)).isoformat()
            two_years_30d_ago = (datetime.datetime.now() - datetime.timedelta(days=760)).isoformat()
            with get_db() as conn:
                # 2年未ログイン → inactive にマーク
                # login_logs にレコードが1件以上あり、かつ最終ログインが2年以上前のユーザーのみ
                conn.execute(
                    "UPDATE users SET status = 'inactive' WHERE (status = 'active' OR status IS NULL) AND user_id IN ("
                    "  SELECT u.user_id FROM users u JOIN login_logs ll ON u.user_id = ll.user_id "
                    "  GROUP BY u.user_id HAVING MAX(ll.logged_in_at) < ?"
                    ")",
                    (two_years_ago,)
                )
                marked = conn.execute("SELECT changes()").fetchone()[0]
                if marked:
                    logging.info(f"Marked {marked} users as inactive (2+ years no login)")

                # inactive かつ猶予30日超 → 物理削除
                inactive_to_delete = conn.execute(
                    "SELECT u.user_id FROM users u WHERE u.status = 'inactive' AND u.user_id IN ("
                    "  SELECT u2.user_id FROM users u2 JOIN login_logs ll ON u2.user_id = ll.user_id "
                    "  GROUP BY u2.user_id HAVING MAX(ll.logged_in_at) < ?"
                    ")",
                    (two_years_30d_ago,)
                ).fetchall()

                for row in inactive_to_delete:
                    uid = row["user_id"]
                    conn.execute("DELETE FROM ratings WHERE user_id = ?", (uid,))
                    conn.execute("DELETE FROM recommendations WHERE user_id = ?", (uid,))
                    conn.execute("DELETE FROM reviews WHERE user_id = ?", (uid,))
                    conn.execute("DELETE FROM login_logs WHERE user_id = ?", (uid,))
                    conn.execute("DELETE FROM page_views WHERE user_id = ?", (uid,))
                    conn.execute("DELETE FROM review_reports WHERE user_id = ?", (uid,))
                    conn.execute("DELETE FROM users WHERE user_id = ?", (uid,))
                if inactive_to_delete:
                    logging.info(f"Permanently deleted {len(inactive_to_delete)} inactive users")
        except Exception as e:
            logging.warning(f"inactive user cleanup failed: {e}")

        # 登録後7日以内にログインがないユーザーを削除
        try:
            seven_days_ago = (datetime.datetime.now() - datetime.timedelta(days=7)).isoformat()
            with get_db() as conn:
                # created_at があり、7日以上前に登録、login_logs にレコードがないユーザー
                stale_users = conn.execute(
                    "SELECT u.user_id FROM users u "
                    "WHERE u.created_at IS NOT NULL AND u.created_at < ? "
                    "AND u.user_id NOT IN (SELECT DISTINCT user_id FROM login_logs)",
                    (seven_days_ago,)
                ).fetchall()

                for row in stale_users:
                    uid = row["user_id"]
                    conn.execute("DELETE FROM ratings WHERE user_id = ?", (uid,))
                    conn.execute("DELETE FROM recommendations WHERE user_id = ?", (uid,))
                    conn.execute("DELETE FROM reviews WHERE user_id = ?", (uid,))
                    conn.execute("DELETE FROM page_views WHERE user_id = ?", (uid,))
                    conn.execute("DELETE FROM review_reports WHERE user_id = ?", (uid,))
                    conn.execute("DELETE FROM users WHERE user_id = ?", (uid,))
                if stale_users:
                    logging.info(f"Deleted {len(stale_users)} users who never logged in within 7 days of registration")
        except Exception as e:
            logging.warning(f"stale user cleanup failed: {e}")

        # 類似度行列をプリロード（おすすめ表示の高速化）
        try:
            import recommendation as _rec_module
            _rec_module.update_user_similarity_from_db()
            logging.info("User similarity matrix preloaded")
        except Exception as e:
            logging.warning(f"similarity preload failed: {e}")
    except Exception as e:
        logging.error(f"startup error: {e}")

# object_dict は startup() で SQLite からロードされる
object_dict = {}

app.mount("/static/store_images", StaticFiles(directory=str(STORE_IMAGES_DIR)), name="store_images")
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

def get_user_group_code(user_id) -> str:
    """ユーザーのグループコードを取得"""
    if not user_id:
        return ""
    try:
        with get_db() as conn:
            row = conn.execute("SELECT group_code FROM users WHERE user_id = ?", (int(user_id),)).fetchone()
        return row["group_code"] if row and row["group_code"] else ""
    except Exception:
        return ""


def get_store_image_map() -> dict:
    """object_id -> 画像URL のマッピングを返す"""
    image_map = {}
    if STORE_IMAGES_DIR.exists():
        for f in STORE_IMAGES_DIR.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
                image_map[f.stem] = f"/static/store_images/{f.name}"
    return image_map


def get_popular_objects(top_n=5):
    """全ユーザーの平均評価が高い店舗上位 top_n を返す"""
    try:
        with get_db() as conn:
            rows = conn.execute("""
                SELECT CAST(r.object_id AS TEXT) as object_id,
                       o.object_name,
                       AVG(r.rating) as mean,
                       COUNT(*) as cnt
                FROM ratings r
                JOIN objects o ON r.object_id = o.object_id
                GROUP BY r.object_id
                HAVING COUNT(*) >= ?
                ORDER BY mean DESC
                LIMIT ?
            """, (MIN_RATINGS_FOR_POPULAR, top_n)).fetchall()
        return [
            {
                "object_id": r["object_id"],
                "object_name": r["object_name"],
                "mean": round(r["mean"], 1),
                "count": r["cnt"],
            }
            for r in rows
        ]
    except Exception as e:
        logging.warning(f"get_popular_objects error: {e}")
        return []


def save_recommendations_to_db(recommend_df):
    """推薦結果を recommendations テーブルに保存する"""
    if recommend_df is None or recommend_df.empty:
        return
    try:
        now = datetime.datetime.now().isoformat()
        with get_db() as conn:
            conn.execute("DELETE FROM recommendations")
            conn.executemany(
                "INSERT INTO recommendations (user_id, object_id, recommendation_score, updated_at) VALUES (?, ?, ?, ?)",
                [
                    (int(row["user_id"]), int(row["object_id"]), float(row["recommendation_score"]), now)
                    for _, row in recommend_df.iterrows()
                ]
            )
        logging.info("Saved recommendations to SQLite")
    except Exception as e:
        logging.warning(f"save_recommendations_to_db error: {e}")


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


@app.post("/login")
@limiter.limit("10/minute")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """ユーザーログイン処理"""
    with get_db() as conn:
        row = conn.execute(
            "SELECT user_id, username, password_hash, status FROM users WHERE username = ?",
            (username,)
        ).fetchone()

    if row is None:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error_message": "ユーザー名が存在しません。"
        })

    # 凍結ユーザーのログインを拒否
    if row["status"] == "banned":
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error_message": "このアカウントは利用できません。"
        })

    if not bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error_message": "パスワードが間違っています。"
        })

    request.session["username"] = row["username"]
    request.session["user_id"] = str(row["user_id"])

    # inactive ユーザーの復帰
    try:
        with get_db() as conn:
            conn.execute("UPDATE users SET status = 'active' WHERE user_id = ? AND status = 'inactive'", (row["user_id"],))
    except Exception:
        pass

    # ログイン履歴を記録
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO login_logs (user_id, logged_in_at) VALUES (?, ?)",
                (row["user_id"], datetime.datetime.now().isoformat())
            )
    except Exception:
        pass

    return RedirectResponse(url="/rating", status_code=303)

router = APIRouter()

@router.post("/logout")
async def logout(request: Request):
    """ セッションをクリアし、クッキーを削除してログアウト """
    request.session.clear()  # ✅ セッションを削除
    response = RedirectResponse(url="/index", status_code=303)
    response.delete_cookie("session")  # ✅ クッキーも削除
    return response

app.include_router(router)

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
        "current_group_code": get_user_group_code(request.session.get("user_id")),
        "recommendations": recommendations,
        "has_similar_users": len(recommendations) > 0
    })

@app.get("/check_username")
@limiter.limit("20/minute")
async def check_username(request: Request, username: str = Query(...)):
    """ユーザー名の重複チェックAPI"""
    with get_db() as conn:
        row = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
    return {"exists": row is not None}

@app.get("/register")
async def show_register_page(request: Request):
    """新規登録ページを開く際にログアウトする"""
    request.session.clear()
    response = templates.TemplateResponse("register.html", {
        "request": request,
        "invite_required": bool(INVITE_CODE),
    })
    response.delete_cookie("session")
    return response

@app.post("/register")
@limiter.limit("5/minute")
async def register_user(request: Request, username: str = Form(...), pw: str = Form(alias="password"), invite_code: str = Form("")):
    """新規ユーザー登録処理"""
    # 招待コードチェック
    if INVITE_CODE and invite_code.strip() != INVITE_CODE:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error_message": "招待コードが正しくありません。",
        })

    # パスワード強度チェック（サーバー側）
    import re as _re
    if len(pw) < 8 or not _re.search(r'[A-Za-z]', pw) or not _re.search(r'\d', pw) or not _re.search(r'[@$!%*#?&]', pw):
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error_message": "パスワードは8文字以上で、英字・数字・記号(@$!%*#?&)を各1つ以上含めてください。",
        })

    pw_hash = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, pw_hash, datetime.datetime.now().isoformat())
            )
        logging.info(f"New user registered: {username}")
    except sqlite3.IntegrityError:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error_message": "このユーザー名は既に登録されています。"
        })
    except Exception as e:
        logging.error(f"Error registering user: {e}")
        return JSONResponse(content={"detail": "Failed to register user"}, status_code=500)

    with _recommend_lock:
        _cached_merged_df, _cached_object_id_to_name, _cached_recommend_df, _cached_user_dict = create_merged_df()
    return RedirectResponse(url="/", status_code=303)

def is_object_exists(object_name: str) -> bool:
    """指定された評価対象が objects テーブルに既に存在するかチェック"""
    with get_db() as conn:
        row = conn.execute("SELECT 1 FROM objects WHERE object_name = ?", (object_name,)).fetchone()
    return row is not None

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
    """新しい評価対象（飲食店）を objects テーブルに追加"""
    new_objects = [name.strip() for name in object_names.split(",") if name.strip()]

    if not new_objects:
        return templates.TemplateResponse("add_objects.html", {
            "request": request,
            "error_message": "評価対象を入力してください。"
        })

    added_count = 0
    with get_db() as conn:
        for obj in new_objects:
            try:
                conn.execute("INSERT OR IGNORE INTO objects (object_name) VALUES (?)", (obj,))
                added_count += 1
            except Exception as e:
                logging.warning(f"add_objects insert error: {e}")

    if added_count == 0:
        return templates.TemplateResponse("add_objects.html", {
            "request": request,
            "error_message": "すべての評価対象が既に登録されています。"
        })

    # object_dict を再ロード
    with get_db() as conn:
        rows = conn.execute("SELECT object_id, object_name FROM objects").fetchall()
        object_dict.clear()
        object_dict.update({str(r["object_id"]): r["object_name"] for r in rows})
    logging.info(f"object_dict reloaded after add_objects ({added_count} added)")

    with _recommend_lock:
        _cached_merged_df, _cached_object_id_to_name, _cached_recommend_df, _cached_user_dict = create_merged_df()

    message = urllib.parse.quote("評価対象が追加されました！")
    return RedirectResponse(url=f"/add_objects?success=true&message={message}", status_code=303)




@app.get("/help", response_class=HTMLResponse)
async def help_page(request: Request):
    """ヘルプページ"""
    username = request.session.get("username")
    if not username:
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("help.html", {
        "request": request,
        "username": username,
        "current_group_code": get_user_group_code(request.session.get("user_id")),
    })


@app.get("/change_password", response_class=HTMLResponse)
async def show_change_password_page(request: Request):
    """パスワード変更ページを表示"""
    username = request.session.get("username")
    if not username:
        return RedirectResponse(url="/", status_code=303)
    error = request.query_params.get("error", "")
    success = request.query_params.get("success", "")
    return templates.TemplateResponse("change_password.html", {
        "request": request,
        "username": username,
        "current_group_code": get_user_group_code(request.session.get("user_id")),
        "error": error,
        "success": success,
    })


@app.post("/change_password")
async def change_password(request: Request, current_password: str = Form(...), new_password: str = Form(...)):
    """パスワード変更処理"""
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse(url="/", status_code=303)

    import re as _re

    # 新パスワードの強度チェック
    if len(new_password) < 8 or not _re.search(r'[A-Za-z]', new_password) or not _re.search(r'\d', new_password) or not _re.search(r'[@$!%*#?&]', new_password):
        return RedirectResponse(url="/change_password?error=weak", status_code=303)

    with get_db() as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE user_id = ?", (int(user_id),)).fetchone()
        if not row:
            return RedirectResponse(url="/", status_code=303)

        # 現在のパスワードを確認
        if not bcrypt.checkpw(current_password.encode(), row["password_hash"].encode()):
            return RedirectResponse(url="/change_password?error=wrong", status_code=303)

        # 新パスワードをハッシュ化して更新
        new_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
        conn.execute("UPDATE users SET password_hash = ? WHERE user_id = ?", (new_hash, int(user_id)))

    return RedirectResponse(url="/change_password?success=1", status_code=303)


@app.get("/export_my_data")
async def export_my_data(request: Request):
    """ログインユーザー自身のデータを JSON でダウンロード"""
    import json as _json

    user_id = request.session.get("user_id")
    username = request.session.get("username")
    if not user_id:
        return RedirectResponse(url="/", status_code=303)

    uid = int(user_id)
    with get_db() as conn:
        user_row = conn.execute(
            "SELECT user_id, username, created_at, status, group_code FROM users WHERE user_id = ?",
            (uid,)
        ).fetchone()

        ratings = conn.execute(
            "SELECT object_id, rating, rated_at FROM ratings WHERE user_id = ?", (uid,)
        ).fetchall()

        reviews = conn.execute(
            "SELECT review_id, object_id, comment, created_at, deleted_at FROM reviews WHERE user_id = ?", (uid,)
        ).fetchall()

        login_logs = conn.execute(
            "SELECT logged_in_at FROM login_logs WHERE user_id = ? ORDER BY logged_in_at DESC", (uid,)
        ).fetchall()

    data = {
        "exported_at": datetime.datetime.now().isoformat(),
        "user": {
            "user_id": user_row["user_id"],
            "username": user_row["username"],
            "created_at": user_row["created_at"],
            "status": user_row["status"],
            "group_code": user_row["group_code"],
        } if user_row else None,
        "ratings": [
            {"object_id": r["object_id"], "object_name": object_dict.get(str(r["object_id"]), ""),
             "rating": r["rating"], "rated_at": r["rated_at"]}
            for r in ratings
        ],
        "reviews": [
            {"review_id": r["review_id"], "object_id": r["object_id"],
             "object_name": object_dict.get(str(r["object_id"]), ""),
             "comment": r["comment"], "created_at": r["created_at"],
             "deleted": r["deleted_at"] is not None}
            for r in reviews
        ],
        "login_history": [r["logged_in_at"] for r in login_logs],
    }

    json_bytes = _json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    today = datetime.datetime.now().strftime("%Y%m%d")
    filename = f"mydata_{username}_{today}.json"

    from starlette.responses import Response
    return Response(
        content=json_bytes,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/map")
async def show_map_page(request: Request):
    """地図ページを表示"""
    username = request.session.get("username")
    if not username:
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("map.html", {
        "request": request,
        "username": username,
        "current_group_code": get_user_group_code(request.session.get("user_id")),
    })


@app.get("/api/objects/locations")
async def get_object_locations(request: Request):
    """位置情報付き店舗一覧を返すAPI"""
    user = request.session.get("user_id")
    if not user:
        return JSONResponse(content={"detail": "Unauthorized"}, status_code=401)

    with get_db() as conn:
        rows = conn.execute(
            "SELECT object_id, object_name, latitude, longitude, genre FROM objects "
            "WHERE latitude IS NOT NULL AND longitude IS NOT NULL"
        ).fetchall()

    image_map = get_store_image_map()
    return JSONResponse(content=[
        {
            "object_id": r["object_id"],
            "object_name": r["object_name"],
            "latitude": r["latitude"],
            "longitude": r["longitude"],
            "genre": r["genre"] or "",
            "image_url": image_map.get(str(r["object_id"])),
        }
        for r in rows
    ])


@app.get("/store/{object_id}", response_class=HTMLResponse)
async def store_detail_page(request: Request, object_id: str):
    """店舗詳細ページ（名前・写真・地図・評価分布・口コミ）"""
    username = request.session.get("username")
    user_id = request.session.get("user_id")
    if not username:
        return RedirectResponse(url="/", status_code=303)

    # ページビューを記録
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO page_views (user_id, object_id, viewed_at) VALUES (?, ?, ?)",
                (int(user_id), int(object_id), datetime.datetime.now().isoformat())
            )
    except Exception:
        pass

    with get_db() as conn:
        obj = conn.execute(
            "SELECT object_id, object_name, latitude, longitude FROM objects WHERE object_id = ?",
            (int(object_id),)
        ).fetchone()
        if obj is None:
            return RedirectResponse(url="/recommendations", status_code=303)

        # 評価の分布を取得
        rating_rows = conn.execute(
            "SELECT rating, COUNT(*) as cnt FROM ratings WHERE object_id = ? GROUP BY rating ORDER BY rating DESC",
            (int(object_id),)
        ).fetchall()

        total_ratings = sum(r["cnt"] for r in rating_rows)

        # 口コミを取得
        comments = conn.execute(
            "SELECT rv.review_id, rv.user_id, rv.comment, rv.created_at, u.username "
            "FROM reviews rv JOIN users u ON rv.user_id = u.user_id "
            "WHERE rv.object_id = ? AND rv.deleted_at IS NULL ORDER BY rv.created_at DESC",
            (int(object_id),)
        ).fetchall()

    image_map = get_store_image_map()
    store = {
        "object_id": str(obj["object_id"]),
        "object_name": obj["object_name"],
        "latitude": obj["latitude"],
        "longitude": obj["longitude"],
        "image_url": image_map.get(str(obj["object_id"])),
    }

    # 5〜1の分布を辞書で作成（0件の星も含む）
    dist = {r["rating"]: r["cnt"] for r in rating_rows}
    rating_distribution = [{"stars": s, "count": dist.get(s, 0)} for s in range(5, 0, -1)]

    current_user_id = str(request.session.get("user_id"))
    comment_list = [
        {
            "review_id": c["review_id"],
            "comment": c["comment"],
            "created_at": c["created_at"][:10],
            "is_mine": str(c["user_id"]) == current_user_id,
        }
        for c in comments
    ]

    return templates.TemplateResponse("store_detail.html", {
        "request": request,
        "username": username,
        "current_group_code": get_user_group_code(request.session.get("user_id")),
        "store": store,
        "rating_distribution": rating_distribution,
        "total_ratings": total_ratings,
        "comments": comment_list,
    })


@app.get("/post_review", response_class=HTMLResponse)
async def show_post_review_page(request: Request, object_id: str = ""):
    """口コミ投稿ページ（既存口コミがあれば編集モード）"""
    username = request.session.get("username")
    user_id = request.session.get("user_id")
    if not username:
        return RedirectResponse(url="/", status_code=303)

    with get_db() as conn:
        obj_rows = conn.execute(
            "SELECT object_id, object_name FROM objects ORDER BY object_id ASC"
        ).fetchall()

        # 選択された店舗に既存口コミがあるか確認
        existing_comment = ""
        if object_id:
            row = conn.execute(
                "SELECT comment FROM reviews WHERE user_id = ? AND object_id = ? AND deleted_at IS NULL",
                (int(user_id), int(object_id))
            ).fetchone()
            if row:
                existing_comment = row["comment"]

    objects_list = [{"object_id": str(r["object_id"]), "object_name": r["object_name"]} for r in obj_rows]

    return templates.TemplateResponse("post_review.html", {
        "request": request,
        "username": username,
        "current_group_code": get_user_group_code(request.session.get("user_id")),
        "objects": objects_list,
        "selected_object_id": object_id,
        "existing_comment": existing_comment,
    })


@app.post("/post_review")
@limiter.limit("10/minute")
async def submit_review(request: Request, object_id: str = Form(...), comment: str = Form(...)):
    """口コミを保存（既存があれば更新）"""
    username = request.session.get("username")
    user_id = request.session.get("user_id")
    if not username:
        return RedirectResponse(url="/", status_code=303)

    comment = comment.strip()[:500]  # 最大500文字
    if not comment:
        return RedirectResponse(url=f"/post_review?object_id={object_id}", status_code=303)

    try:
        now = datetime.datetime.now().isoformat()
        with get_db() as conn:
            existing = conn.execute(
                "SELECT review_id FROM reviews WHERE user_id = ? AND object_id = ? AND deleted_at IS NULL",
                (int(user_id), int(object_id))
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE reviews SET comment = ?, created_at = ? WHERE review_id = ?",
                    (comment, now, existing["review_id"])
                )
            else:
                conn.execute(
                    "INSERT INTO reviews (user_id, object_id, comment, created_at) VALUES (?, ?, ?, ?)",
                    (int(user_id), int(object_id), comment, now)
                )
    except Exception as e:
        logging.error(f"Failed to save review: {e}")
        return RedirectResponse(url=f"/post_review?object_id={object_id}", status_code=303)

    return RedirectResponse(url=f"/store/{object_id}", status_code=303)


@app.get("/edit_review/{review_id}", response_class=HTMLResponse)
async def show_edit_review_page(request: Request, review_id: int):
    """口コミ編集ページ"""
    username = request.session.get("username")
    user_id = request.session.get("user_id")
    if not username:
        return RedirectResponse(url="/", status_code=303)

    with get_db() as conn:
        review = conn.execute(
            "SELECT rv.review_id, rv.user_id, rv.object_id, rv.comment, o.object_name "
            "FROM reviews rv JOIN objects o ON rv.object_id = o.object_id "
            "WHERE rv.review_id = ? AND rv.deleted_at IS NULL",
            (review_id,)
        ).fetchone()

    if review is None or str(review["user_id"]) != str(user_id):
        return RedirectResponse(url="/recommendations", status_code=303)

    return templates.TemplateResponse("edit_review.html", {
        "request": request,
        "username": username,
        "current_group_code": get_user_group_code(request.session.get("user_id")),
        "review": {
            "review_id": review["review_id"],
            "object_id": review["object_id"],
            "object_name": review["object_name"],
            "comment": review["comment"],
        },
    })


@app.post("/edit_review/{review_id}")
async def update_review(request: Request, review_id: int, comment: str = Form(...)):
    """口コミを更新"""
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse(url="/", status_code=303)

    comment = comment.strip()[:500]  # 最大500文字
    with get_db() as conn:
        review = conn.execute(
            "SELECT user_id, object_id FROM reviews WHERE review_id = ?", (review_id,)
        ).fetchone()
        if review is None or str(review["user_id"]) != str(user_id):
            return RedirectResponse(url="/recommendations", status_code=303)

        if comment:
            conn.execute(
                "UPDATE reviews SET comment = ? WHERE review_id = ?",
                (comment, review_id)
            )
        object_id = review["object_id"]

    return RedirectResponse(url=f"/store/{object_id}", status_code=303)


@app.post("/delete_review")
async def delete_review(request: Request, review_id: int = Form(...)):
    """口コミを削除"""
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse(url="/", status_code=303)

    with get_db() as conn:
        review = conn.execute(
            "SELECT user_id, object_id FROM reviews WHERE review_id = ?", (review_id,)
        ).fetchone()
        if review is None or str(review["user_id"]) != str(user_id):
            return RedirectResponse(url="/recommendations", status_code=303)

        conn.execute(
            "UPDATE reviews SET deleted_at = ?, deleted_by = 'user' WHERE review_id = ?",
            (datetime.datetime.now().isoformat(), review_id)
        )
        object_id = review["object_id"]

    return RedirectResponse(url=f"/store/{object_id}", status_code=303)


@app.get("/rating")
async def show_rating_page(request: Request):
    """評価ページを表示"""
    username = request.session.get("username")
    user_id = request.session.get("user_id")
    if not username:
        return RedirectResponse(url="/", status_code=303)

    user_id = str(user_id)

    with get_db() as conn:
        # セッションのユーザーが実在し、利用可能か確認
        user_row = conn.execute("SELECT status FROM users WHERE user_id = ?", (int(user_id),)).fetchone()
        if not user_row:
            logging.warning(f"Session user_id={user_id} not found in DB, forcing logout")
            request.session.clear()
            return RedirectResponse(url="/", status_code=303)
        if user_row["status"] == "banned":
            request.session.clear()
            return RedirectResponse(url="/", status_code=303)

        # 評価対象を評価件数が多い順に取得（activeユーザーの評価のみカウント）
        obj_rows = conn.execute(
            "SELECT o.object_id, o.object_name, "
            "COUNT(r.user_id) as rating_count "
            "FROM objects o "
            "LEFT JOIN (SELECT rat.object_id, rat.user_id FROM ratings rat "
            "  JOIN users u ON rat.user_id = u.user_id "
            "  WHERE u.status IS NULL OR u.status IN ('active','warned')) r "
            "ON o.object_id = r.object_id "
            "GROUP BY o.object_id "
            "ORDER BY rating_count DESC, o.object_id ASC"
        ).fetchall()

        # ユーザーの評価を取得
        rating_rows = conn.execute(
            "SELECT object_id, rating FROM ratings WHERE user_id = ?",
            (int(user_id),)
        ).fetchall()

    past_ratings = {str(r["object_id"]): r["rating"] for r in rating_rows}
    image_map = get_store_image_map()

    unrated_objects = [
        {"id": str(r["object_id"]), "name": r["object_name"], "image_url": image_map.get(str(r["object_id"]))}
        for r in obj_rows if str(r["object_id"]) not in past_ratings
    ]
    rated_objects = [
        {"id": str(r["object_id"]), "name": r["object_name"],
         "rating": past_ratings[str(r["object_id"])],
         "image_url": image_map.get(str(r["object_id"]))}
        for r in obj_rows if str(r["object_id"]) in past_ratings
    ]

    return templates.TemplateResponse("rating.html", {
        "request": request,
        "username": username,
        "current_group_code": get_user_group_code(request.session.get("user_id")),
        "unrated_objects": unrated_objects,
        "rated_objects": rated_objects,
        "past_ratings": past_ratings,
        "break_every": BREAK_EVERY,
    })


@app.post("/submit_ratings")
async def submit_ratings(request: Request):
    """評価を ratings テーブルに更新（DELETE + INSERT）"""
    logging.info("submit_ratings() called")
    form_data = await request.form()
    ratings = {
        key.replace("ratings[", "").replace("]", ""): value
        for key, value in form_data.items() if "ratings[" in key
    }

    if not ratings:
        logging.info("No ratings in submission, redirecting to recommendations")
        return RedirectResponse(url="/recommendations", status_code=303)

    uid = int(request.session["user_id"])
    logging.info(f"Processing ratings for user_id: {uid}")

    try:
        with get_db() as conn:
            conn.execute("DELETE FROM ratings WHERE user_id = ?", (uid,))
            conn.executemany(
                "INSERT INTO ratings (user_id, object_id, rating, rated_at) VALUES (?, ?, ?, ?)",
                [(uid, int(oid), int(rating), datetime.datetime.now().isoformat()) for oid, rating in ratings.items() if rating]
            )
        logging.info("ratings updated in SQLite")
    except Exception as e:
        logging.error(f"Error writing ratings: {e}")
        return JSONResponse(content={"detail": "Failed to update ratings"}, status_code=500)

    # 即時: 類似度行列を更新し、送信ユーザーのみ推薦を再計算
    try:
        import recommendation as _rec_module
        _rec_module.update_user_similarity_from_db()
        user_rec = recommend_for_single_user(uid)
        # ロックでキャッシュの競合を防止
        with _recommend_lock:
            global _cached_recommend_df
            if _cached_recommend_df is not None and not _cached_recommend_df.empty:
                _cached_recommend_df = _cached_recommend_df[_cached_recommend_df["user_id"] != str(uid)]
                _cached_recommend_df = pd.concat([_cached_recommend_df, user_rec], ignore_index=True)
            else:
                _cached_recommend_df = user_rec
        logging.info(f"Instant recommendation updated for user_id={uid}")
    except Exception as e:
        logging.error(f"Error in instant recommendation: {e}")

    # バックグラウンド: 全ユーザー再計算（他ユーザーへの波及を反映）
    asyncio.ensure_future(_background_full_recalc())

    return RedirectResponse(url="/recommendations", status_code=303)


async def _background_full_recalc():
    """バックグラウンドで全ユーザーの推薦を再計算"""
    global _cached_merged_df, _cached_object_id_to_name, _cached_user_dict, _cached_recommend_df
    await asyncio.sleep(0.5)  # メインレスポンスを優先
    try:
        merged_df, obj_to_name, recommend_df, user_dict = create_merged_df()
        with _recommend_lock:
            _cached_merged_df = merged_df
            _cached_object_id_to_name = obj_to_name
            _cached_recommend_df = recommend_df
            _cached_user_dict = user_dict
        _chart_cache.clear()  # 推薦データ更新時にチャートキャッシュをクリア
        logging.info("Background full recalculation completed")
    except Exception as e:
        logging.error(f"Background recalculation error: {e}")




@app.get("/recommendations")
async def recommendations_page(request: Request):
    """おすすめ結果を表示（キャッシュ利用）"""
    username = request.session.get("username")
    user_id = request.session.get("user_id")
    if not username:
        return RedirectResponse(url="/", status_code=303)

    user_id = str(user_id)
    with get_db() as conn:
        if not conn.execute("SELECT 1 FROM users WHERE user_id = ?", (int(user_id),)).fetchone():
            request.session.clear()
            return RedirectResponse(url="/", status_code=303)

    global _cached_recommend_df
    with _recommend_lock:
        if _cached_recommend_df is None or _cached_recommend_df.empty:
            logging.info("Recommendation cache is empty, computing now")
            recommendations_df, _ = recommend_for_all_users()
            _cached_recommend_df = recommendations_df
            save_recommendations_to_db(_cached_recommend_df)

        user_recommendations = _cached_recommend_df[_cached_recommend_df["user_id"] == user_id].copy()
    user_recommendations["object_name"] = user_recommendations["object_id"].map(object_dict)
    user_recommendations["recommendation_score"] = user_recommendations["recommendation_score"].round(2)
    user_recommendations = user_recommendations[user_recommendations["recommendation_score"] >= 0.25]
    user_recommendations = user_recommendations.sort_values("recommendation_score", ascending=False)

    # 推薦理由を生成
    rec_object_ids = user_recommendations["object_id"].tolist()
    explanations = {}
    if rec_object_ids:
        try:
            explanations = explain_recommendations(user_id, rec_object_ids)
        except Exception as e:
            logging.warning(f"Failed to generate recommendation explanations: {e}")

    rec_list = user_recommendations.to_dict(orient="records")
    for rec in rec_list:
        exp = explanations.get(str(rec["object_id"]), {})
        rec["reason"] = exp.get("reason", "")
        rec["sub_reason"] = exp.get("sub_reason", "")

    popular_objects = get_popular_objects()

    return templates.TemplateResponse("recommendations.html", {
        "request": request,
        "username": username,
        "current_group_code": get_user_group_code(request.session.get("user_id")),
        "recommendations": rec_list,
        "popular_objects": popular_objects,
    })


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

    stars = categorize_recommendation(row["recommendation_score"], default="-")
    return f"R: {rating}\nZ: {z_score}\nRS: {rec_score}\n{stars}"


def create_merged_df(recommend_df=None):
    global _cached_merged_df, _cached_object_id_to_name, _cached_user_dict, _cached_recommend_df, user_similarity

    # --- 推薦データとユーザー類似度の計算 ---
    if recommend_df is None:
        recommend_df, computed_user_similarity = recommend_for_all_users()
        user_similarity = computed_user_similarity
    else:
        # キャッシュ利用時はユーザー類似度のみ更新（推薦再計算をスキップ）
        import recommendation as _rec_module
        _rec_module.update_user_similarity_from_db()
        user_similarity = _rec_module.user_similarity.copy()

    # --- SQLite からデータを読み込む ---
    with get_db() as conn:
        users = pd.read_sql_query(
            "SELECT CAST(user_id AS TEXT) as user_id, username FROM users", conn
        )
        objects = pd.read_sql_query(
            "SELECT CAST(object_id AS TEXT) as object_id, object_name FROM objects", conn
        )
        ratings = pd.read_sql_query(
            "SELECT CAST(user_id AS TEXT) as user_id, CAST(object_id AS TEXT) as object_id, rating FROM ratings",
            conn
        )

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
        logging.warning("recommendation_score column missing from merged_df")

    # --- 欠損値の処理 ---
    merged_df["rating"] = merged_df["rating"].fillna("-")
    merged_df["z_score"] = merged_df["z_score"].fillna("-")

    merged_df["username"] = merged_df["user_id"].map(_cached_user_dict)
    merged_df["object_name"] = merged_df["object_id"].map(object_dict_local)

    merged_df["cell_info"] = merged_df.apply(format_cell, axis=1)

    # --- object_id → object_name の辞書作成 ---
    object_id_to_name = dict(zip(objects["object_id"], objects["object_name"].fillna("Unknown")))

    # 推薦結果を SQLite にキャッシュ保存
    save_recommendations_to_db(recommend_df)

    return merged_df, object_id_to_name, recommend_df, _cached_user_dict

@app.get("/admin/reviews", response_class=HTMLResponse)
async def admin_reviews(request: Request, page: int = 1):
    global _cached_merged_df, _cached_object_id_to_name, _cached_recommend_df, _cached_user_dict, user_similarity

    with _recommend_lock:
        if _cached_recommend_df is not None and not _cached_recommend_df.empty:
            _cached_merged_df, _cached_object_id_to_name, _, _cached_user_dict = create_merged_df(recommend_df=_cached_recommend_df)
        else:
            _cached_merged_df, _cached_object_id_to_name, _cached_recommend_df, _cached_user_dict = create_merged_df()

    if user_similarity is None or user_similarity.empty:
        return HTMLResponse(content="ユーザー類似度データが利用できません", status_code=500)

    # ピボットテーブルと推薦スコアヒストグラムを生成（キャッシュ利用）
    heatmap_result = _get_cached("admin_reviews_heatmap")
    if heatmap_result is None:
        heatmap_result = update_heatmap()
        if heatmap_result is not None and heatmap_result[0] is not None:
            _set_cached("admin_reviews_heatmap", heatmap_result)
    if heatmap_result is None or heatmap_result[0] is None:
        recommendation_hist_base64 = None
        pivot_compact = pd.DataFrame()
        object_name_cols = []
    else:
        recommendation_hist_base64, pivot_compact, object_name_cols = heatmap_result

    # ページネーション（ユーザー行単位）
    page = max(1, page)
    total_rows = len(pivot_compact)
    total_pages = max(1, (total_rows + _ADMIN_PAGE_SIZE - 1) // _ADMIN_PAGE_SIZE)
    offset = (page - 1) * _ADMIN_PAGE_SIZE
    pivot_page = pivot_compact.iloc[offset:offset + _ADMIN_PAGE_SIZE] if not pivot_compact.empty else pivot_compact

    # pending リクエスト一覧を取得
    requests_df = _load_requests()
    pending_requests = requests_df[requests_df["status"] == "pending"].to_dict(orient="records") if not requests_df.empty else []

    # 最後にテンプレートを返す
    return templates.TemplateResponse(
        "admin_reviews.html",
        {
            "request": request,
            "pending_requests": pending_requests,
            "pivot_table": pivot_page.to_dict(orient="records"),
            "column_names": object_name_cols,
            "recommendation_hist_base64": recommendation_hist_base64,
            "current_page": page,
            "total_pages": total_pages,
            "total_rows": total_rows,
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

    if _cached_merged_df is None or (hasattr(_cached_merged_df, 'empty') and _cached_merged_df.empty):
        return None, None, []

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

    # ========== 推薦スコアのヒストグラム ==========
    if _cached_recommend_df is None or _cached_recommend_df.empty:
        return None, None, []
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
        recommendation_hist_base64,
        pivot_compact,
        object_name_cols
    )


def build_heatmap_page_data(group_code=None):
    """
    /admin/heatmap ページ用データを生成して返す。
    group_code 指定時はそのグループのユーザーのみで分析。
    """
    import recommendation as _rec_module
    _rec_module.update_user_similarity_from_db()
    sim = _rec_module.user_similarity.copy()
    if sim.empty:
        return None

    with get_db() as conn:
        if group_code:
            rows = conn.execute(
                "SELECT CAST(user_id AS TEXT) as user_id, username FROM users WHERE group_code = ?",
                (group_code,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT CAST(user_id AS TEXT) as user_id, username FROM users"
            ).fetchall()
    user_dict = {r["user_id"]: r["username"] for r in rows}

    # グループフィルター時は類似度行列を絞り込む
    if group_code:
        filtered_ids = [uid for uid in sim.index if str(uid) in user_dict]
        if len(filtered_ids) < 1:
            return None
        sim = sim.loc[filtered_ids, filtered_ids]

    sorted_index = sorted(sim.index, key=lambda x: int(x))
    sim = sim.loc[sorted_index, sorted_index]
    user_labels = [f"{uid} ({user_dict.get(str(uid), 'Unknown')})" for uid in sim.index]

    # ========== ヒートマップ ==========
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        sim[::-1],
        cmap="coolwarm",
        annot=False,
        ax=ax,
        xticklabels=user_labels,
        yticklabels=user_labels[::-1]
    )
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    img_buf = BytesIO()
    plt.savefig(img_buf, format="png")
    plt.close(fig)
    img_buf.seek(0)
    heatmap_base64 = base64.b64encode(img_buf.getvalue()).decode("utf-8")

    # ========== 類似度ヒストグラム ==========
    similarity_values = sim.values.flatten()
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

    # ========== クラスタリング ==========
    num_users = sim.shape[0]
    if num_users < 2:
        # ユーザーが1人以下ではクラスタリング不可
        return (
            heatmap_base64,
            similarity_hist_base64,
            pd.DataFrame(columns=["user_id", "username", "cluster"]),
            pd.DataFrame(columns=["user_id", "username", "cluster"]),
            None,
        )
    num_clusters = max(2, int(num_users / 4) + 1)
    num_clusters = min(num_clusters, num_users)  # クラスタ数がユーザー数を超えないように
    pca = PCA(n_components=min(2, num_users))
    user_embedding = pca.fit_transform(sim)
    user_ids = list(sim.index)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(user_embedding)

    cluster_df = pd.DataFrame({
        "user_id": [str(uid) for uid in user_ids],
        "username": [user_dict.get(str(uid), "Unknown") for uid in user_ids],
        "cluster": clusters + 1
    })
    cluster_sorted_by_username = cluster_df.sort_values(
        ["username"], key=lambda x: x.str.casefold()
    )
    cluster_sorted_by_cluster = cluster_df.sort_values(
        ["cluster", "username"],
        key=lambda x: x.str.casefold() if x.name == "username" else x
    )

    # ========== クラスタ散布図 ==========
    colors = ["red", "blue", "green", "purple", "orange", "cyan", "magenta", "brown"]
    fig, ax = plt.subplots(figsize=(8, 6))
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
        if len(cluster_points) > 1:
            x_mean, y_mean = cluster_points[:, 0].mean(), cluster_points[:, 1].mean()
            cov = np.cov(cluster_points, rowvar=False)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            order = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[order]
            eigenvecs = eigenvecs[:, order]
            chi2_val = np.sqrt(chi2.ppf(0.95, df=2))
            width = 2 * chi2_val * np.sqrt(eigenvals[0])
            height = 2 * chi2_val * np.sqrt(eigenvals[1])
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            ell = Ellipse(
                xy=(x_mean, y_mean), width=width, height=height, angle=angle,
                edgecolor=colors[cluster_id % len(colors)], facecolor="none", linewidth=2
            )
            ax.add_patch(ell)
        elif len(cluster_points) == 1:
            x, y = cluster_points[0]
            ell = Ellipse(
                xy=(x, y), width=0.2, height=0.2,
                edgecolor=colors[cluster_id % len(colors)], facecolor="none",
                linewidth=2, linestyle="dashed"
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

    return (
        heatmap_base64,
        similarity_hist_base64,
        cluster_sorted_by_username,
        cluster_sorted_by_cluster,
        cluster_plot_base64,
    )


@app.get("/admin/heatmap", response_class=HTMLResponse)
async def admin_heatmap(request: Request, group_code: str = ""):
    """ユーザー類似度ヒートマップ・分析ページ"""
    with get_db() as conn:
        group_codes_list = conn.execute("SELECT code, label FROM group_codes ORDER BY label").fetchall()

    heatmap_cache_key = f"heatmap_{group_code or 'all'}"
    result = _get_cached(heatmap_cache_key)
    if result is None:
        result = build_heatmap_page_data(group_code=group_code or None)
        if result is not None:
            _set_cached(heatmap_cache_key, result)
    if result is None:
        return templates.TemplateResponse("admin_heatmap.html", {
            "request": request,
            "group_codes": [{"code": g["code"], "label": g["label"]} for g in group_codes_list],
            "selected_group_code": group_code,
            "heatmap_base64": None,
            "similarity_hist_base64": None,
            "cluster_sorted_by_username": [],
            "cluster_sorted_by_cluster": [],
            "cluster_plot_base64": None,
            "no_data": True,
        })

    heatmap_base64, similarity_hist_base64, cluster_sorted_by_username, cluster_sorted_by_cluster, cluster_plot_base64 = result
    return templates.TemplateResponse("admin_heatmap.html", {
        "request": request,
        "group_codes": [{"code": g["code"], "label": g["label"]} for g in group_codes_list],
        "selected_group_code": group_code,
        "heatmap_base64": heatmap_base64,
        "similarity_hist_base64": similarity_hist_base64,
        "cluster_sorted_by_username": cluster_sorted_by_username.to_dict(orient="records") if hasattr(cluster_sorted_by_username, 'to_dict') else cluster_sorted_by_username,
        "cluster_sorted_by_cluster": cluster_sorted_by_cluster.to_dict(orient="records") if hasattr(cluster_sorted_by_cluster, 'to_dict') else cluster_sorted_by_cluster,
        "cluster_plot_base64": cluster_plot_base64,
        "no_data": False,
    })


# =============================
# 店舗追加リクエスト（一般ユーザー）
# =============================

def _load_requests() -> pd.DataFrame:
    try:
        with get_db() as conn:
            return pd.read_sql_query(
                "SELECT CAST(request_id AS TEXT) as request_id, CAST(user_id AS TEXT) as user_id, "
                "object_name, status, created_at FROM object_requests",
                conn
            )
    except Exception as e:
        logging.error(f"Failed to load object_requests: {e}")
        return pd.DataFrame(columns=["request_id", "user_id", "object_name", "status", "created_at"])


@app.get("/request_object", response_class=HTMLResponse)
async def show_request_object_page(request: Request):
    """店舗追加リクエストページ"""
    username = request.session.get("username")
    user_id = request.session.get("user_id")
    if not username:
        return RedirectResponse(url="/", status_code=303)

    requests_df = _load_requests()
    user_requests = requests_df[requests_df["user_id"] == str(user_id)].to_dict(orient="records") if not requests_df.empty else []

    return templates.TemplateResponse("request_object.html", {
        "request": request,
        "username": username,
        "current_group_code": get_user_group_code(request.session.get("user_id")),
        "user_requests": user_requests,
    })


@app.post("/request_object")
async def submit_request_object(request: Request, object_name: str = Form(...)):
    """店舗追加リクエストを送信"""
    username = request.session.get("username")
    user_id = request.session.get("user_id")
    if not username:
        return RedirectResponse(url="/", status_code=303)

    object_name = object_name.strip()
    if not object_name:
        return templates.TemplateResponse("request_object.html", {
            "request": request,
            "username": username,
        "current_group_code": get_user_group_code(request.session.get("user_id")),
            "user_requests": [],
            "error_message": "店舗名を入力してください。",
        })

    # 重複チェック（既存店舗）
    if object_name in object_dict.values():
        requests_df = _load_requests()
        return templates.TemplateResponse("request_object.html", {
            "request": request,
            "username": username,
        "current_group_code": get_user_group_code(request.session.get("user_id")),
            "user_requests": requests_df[requests_df["user_id"] == str(user_id)].to_dict(orient="records"),
            "error_message": f"「{object_name}」は既に登録済みです。",
        })

    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO object_requests (user_id, object_name, status, created_at) VALUES (?, ?, 'pending', ?)",
                (int(user_id), object_name, datetime.datetime.now().isoformat())
            )
    except Exception as e:
        logging.error(f"Failed to save request: {e}")
        raise HTTPException(status_code=500, detail="リクエストの保存に失敗しました。")

    return RedirectResponse(url="/request_object?success=1", status_code=303)


# =============================
# =============================
# 店舗管理画面（管理者）
# =============================

@app.get("/admin/objects", response_class=HTMLResponse)
async def admin_objects_page(request: Request):
    with get_db() as conn:
        obj_rows = conn.execute(
            "SELECT object_id, object_name, latitude, longitude, genre FROM objects ORDER BY object_id ASC"
        ).fetchall()
    image_map = get_store_image_map()
    objects_list = [
        {
            "object_id": str(r["object_id"]),
            "object_name": r["object_name"],
            "image_url": image_map.get(str(r["object_id"])),
            "latitude": r["latitude"],
            "longitude": r["longitude"],
            "genre": r["genre"] or "",
        }
        for r in obj_rows
    ]
    pending_requests = [
        r for r in _load_requests().to_dict(orient="records")
        if r.get("status") == "pending"
    ]
    with get_db() as conn:
        genre_rows = conn.execute("SELECT name FROM genres ORDER BY sort_order, name").fetchall()
    genre_choices = [r["name"] for r in genre_rows] if genre_rows else ["その他"]

    return templates.TemplateResponse("admin_objects.html", {
        "request": request,
        "objects": objects_list,
        "pending_requests": pending_requests,
        "genres": genre_choices,
    })


@app.post("/admin/objects/set_location")
async def set_object_location(
    object_id: str = Form(...),
    latitude: str = Form(""),
    longitude: str = Form(""),
):
    """店舗の緯度経度を設定"""
    lat = float(latitude) if latitude.strip() else None
    lng = float(longitude) if longitude.strip() else None
    with get_db() as conn:
        conn.execute(
            "UPDATE objects SET latitude = ?, longitude = ? WHERE object_id = ?",
            (lat, lng, int(object_id))
        )
    return RedirectResponse(url="/admin/objects", status_code=303)


@app.post("/admin/objects/set_genre")
async def set_object_genre(object_id: str = Form(...), genre: str = Form("")):
    """店舗のジャンルを設定"""
    with get_db() as conn:
        conn.execute(
            "UPDATE objects SET genre = ? WHERE object_id = ?",
            (genre.strip() or None, int(object_id))
        )
    return RedirectResponse(url="/admin/objects", status_code=303)


@app.post("/admin/objects/upload_image")
async def upload_store_image(object_id: str = Form(...), image: UploadFile = File(...)):
    """店舗画像をアップロード"""
    with get_db() as conn:
        if not conn.execute("SELECT 1 FROM objects WHERE object_id = ?", (int(object_id),)).fetchone():
            return RedirectResponse(url="/admin/objects", status_code=303)
    STORE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ext = image.filename.rsplit(".", 1)[-1].lower() if "." in image.filename else "jpg"
    if ext not in {"jpg", "jpeg", "png", "gif", "webp"}:
        ext = "jpg"
    # 同じ object_id の既存画像を削除
    for old in STORE_IMAGES_DIR.glob(f"{object_id}.*"):
        old.unlink()
    content = await image.read()
    MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
    if len(content) > MAX_IMAGE_SIZE:
        return RedirectResponse(url="/admin/objects", status_code=303)
    with open(STORE_IMAGES_DIR / f"{object_id}.{ext}", "wb") as f:
        f.write(content)
    return RedirectResponse(url="/admin/objects", status_code=303)


@app.post("/admin/objects/delete_image")
async def delete_store_image(object_id: str = Form(...)):
    """店舗画像を削除"""
    for f in STORE_IMAGES_DIR.glob(f"{object_id}.*"):
        f.unlink()
    return RedirectResponse(url="/admin/objects", status_code=303)


@app.post("/admin/objects/delete")
async def delete_object(object_id: str = Form(...)):
    """店舗を削除（関連する評価・推薦スコアも削除）"""
    oid = int(object_id)
    with get_db() as conn:
        conn.execute("DELETE FROM ratings WHERE object_id = ?", (oid,))
        conn.execute("DELETE FROM recommendations WHERE object_id = ?", (oid,))
        conn.execute("DELETE FROM objects WHERE object_id = ?", (oid,))
    object_dict.pop(object_id, None)
    for f in STORE_IMAGES_DIR.glob(f"{object_id}.*"):
        f.unlink()
    logging.info(f"Deleted object_id={object_id} and related data")
    return RedirectResponse(url="/admin/objects", status_code=303)


# 店舗追加リクエスト承認/却下（管理者）
# =============================

@app.post("/admin/approve_request")
async def approve_request(request: Request, request_id: str = Form(...)):
    """店舗追加リクエストを承認し objects テーブルに追加"""
    with get_db() as conn:
        row = conn.execute(
            "SELECT object_name FROM object_requests WHERE request_id = ?",
            (int(request_id),)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="リクエストが見つかりません。")

        object_name = row["object_name"]
        if object_name not in object_dict.values():
            try:
                conn.execute("INSERT OR IGNORE INTO objects (object_name) VALUES (?)", (object_name,))
                new_row = conn.execute(
                    "SELECT object_id FROM objects WHERE object_name = ?", (object_name,)
                ).fetchone()
                if new_row:
                    object_dict[str(new_row["object_id"])] = object_name
                    logging.info(f"Approved: added '{object_name}' (id={new_row['object_id']})")
            except Exception as e:
                logging.error(f"Failed to add object: {e}")
                raise HTTPException(status_code=500, detail="店舗の追加に失敗しました。")

        conn.execute(
            "UPDATE object_requests SET status = 'approved' WHERE request_id = ?",
            (int(request_id),)
        )

    return RedirectResponse(url="/admin/objects", status_code=303)


@app.post("/admin/reject_request")
async def reject_request(request: Request, request_id: str = Form(...)):
    """店舗追加リクエストを却下"""
    with get_db() as conn:
        affected = conn.execute(
            "UPDATE object_requests SET status = 'rejected' WHERE request_id = ?",
            (int(request_id),)
        ).rowcount
    if affected == 0:
        raise HTTPException(status_code=404, detail="リクエストが見つかりません。")
    return RedirectResponse(url="/admin/objects", status_code=303)


@app.post("/set_group_code")
async def set_group_code(request: Request, group_code: str = Form("")):
    """ユーザーのグループコードを設定"""
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse(url="/", status_code=303)

    code = group_code.strip()
    if code:
        with get_db() as conn:
            valid = conn.execute("SELECT 1 FROM group_codes WHERE code = ?", (code,)).fetchone()
        if not valid:
            referer = request.headers.get("referer", "/rating")
            return RedirectResponse(url=referer + ("&" if "?" in referer else "?") + "group_error=1", status_code=303)

    with get_db() as conn:
        conn.execute("UPDATE users SET group_code = ? WHERE user_id = ?", (code or None, int(user_id)))

    return RedirectResponse(url=request.headers.get("referer", "/rating"), status_code=303)


@app.get("/admin/groups", response_class=HTMLResponse)
async def admin_groups_page(request: Request):
    """グループ管理ページ"""
    with get_db() as conn:
        groups = conn.execute(
            "SELECT gc.code, gc.label, gc.created_at, "
            "(SELECT COUNT(*) FROM users u WHERE u.group_code = gc.code) as user_count "
            "FROM group_codes gc ORDER BY gc.created_at DESC"
        ).fetchall()

    return templates.TemplateResponse("admin_groups.html", {
        "request": request,
        "groups": [{"code": g["code"], "label": g["label"],
                    "created_at": g["created_at"][:10] if g["created_at"] else "",
                    "user_count": g["user_count"]} for g in groups],
    })


@app.post("/admin/groups/create")
async def admin_create_group(code: str = Form(...), label: str = Form(...)):
    """グループコードを作成"""
    with get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO group_codes (code, label, created_at) VALUES (?, ?, ?)",
            (code.strip(), label.strip(), datetime.datetime.now().isoformat())
        )
    return RedirectResponse(url="/admin/groups", status_code=303)


@app.post("/admin/groups/delete")
async def admin_delete_group(code: str = Form(...)):
    """グループコードを削除（所属ユーザーのコードもクリア）"""
    with get_db() as conn:
        conn.execute("UPDATE users SET group_code = NULL WHERE group_code = ?", (code,))
        conn.execute("DELETE FROM group_codes WHERE code = ?", (code,))
    return RedirectResponse(url="/admin/groups", status_code=303)


# =============================
# ジャンル管理（管理者）
# =============================

@app.get("/admin/genres", response_class=HTMLResponse)
async def admin_genres_page(request: Request):
    """ジャンル管理ページ"""
    with get_db() as conn:
        genres = conn.execute(
            "SELECT genre_id, name, sort_order, "
            "(SELECT COUNT(*) FROM objects o WHERE o.genre = g.name) as store_count "
            "FROM genres g ORDER BY sort_order, name"
        ).fetchall()

    return templates.TemplateResponse("admin_genres.html", {
        "request": request,
        "genres": [{"genre_id": g["genre_id"], "name": g["name"],
                    "sort_order": g["sort_order"], "store_count": g["store_count"]}
                   for g in genres],
    })


@app.post("/admin/genres/create")
async def admin_create_genre(name: str = Form(...)):
    """ジャンルを追加"""
    name = name.strip()
    if name:
        with get_db() as conn:
            max_order = conn.execute("SELECT MAX(sort_order) as m FROM genres").fetchone()["m"] or 0
            conn.execute(
                "INSERT OR IGNORE INTO genres (name, sort_order) VALUES (?, ?)",
                (name, max_order + 1)
            )
    return RedirectResponse(url="/admin/genres", status_code=303)


@app.post("/admin/genres/delete")
async def admin_delete_genre(genre_id: int = Form(...)):
    """ジャンルを削除（使用中の店舗のジャンルは NULL にリセット）"""
    with get_db() as conn:
        row = conn.execute("SELECT name FROM genres WHERE genre_id = ?", (genre_id,)).fetchone()
        if row:
            conn.execute("UPDATE objects SET genre = NULL WHERE genre = ?", (row["name"],))
            conn.execute("DELETE FROM genres WHERE genre_id = ?", (genre_id,))
    return RedirectResponse(url="/admin/genres", status_code=303)


@app.post("/admin/genres/reorder")
async def admin_reorder_genre(genre_id: int = Form(...), direction: str = Form(...)):
    """ジャンルの並び順を上下に移動"""
    with get_db() as conn:
        current = conn.execute("SELECT genre_id, sort_order FROM genres WHERE genre_id = ?", (genre_id,)).fetchone()
        if not current:
            return RedirectResponse(url="/admin/genres", status_code=303)

        if direction == "up":
            neighbor = conn.execute(
                "SELECT genre_id, sort_order FROM genres WHERE sort_order < ? ORDER BY sort_order DESC LIMIT 1",
                (current["sort_order"],)
            ).fetchone()
        else:
            neighbor = conn.execute(
                "SELECT genre_id, sort_order FROM genres WHERE sort_order > ? ORDER BY sort_order ASC LIMIT 1",
                (current["sort_order"],)
            ).fetchone()

        if neighbor:
            conn.execute("UPDATE genres SET sort_order = ? WHERE genre_id = ?", (neighbor["sort_order"], current["genre_id"]))
            conn.execute("UPDATE genres SET sort_order = ? WHERE genre_id = ?", (current["sort_order"], neighbor["genre_id"]))

    return RedirectResponse(url="/admin/genres", status_code=303)


@app.get("/admin/users", response_class=HTMLResponse)
async def admin_users_page(request: Request, sort: str = "id_asc", page: int = 1):
    """ユーザー管理ページ（ページネーション付き）"""
    order_map = {
        "id_asc": "u.user_id ASC",
        "id_desc": "u.user_id DESC",
        "name_asc": "u.username ASC",
        "name_desc": "u.username DESC",
        "created_asc": "u.created_at ASC",
        "created_desc": "u.created_at DESC",
        "last_login_asc": "last_login ASC",
        "last_login_desc": "last_login DESC",
        "login_count_asc": "login_count ASC",
        "login_count_desc": "login_count DESC",
        "rating_count_asc": "rating_count ASC",
        "rating_count_desc": "rating_count DESC",
        "review_count_asc": "review_count ASC",
        "review_count_desc": "review_count DESC",
    }
    order_clause = order_map.get(sort, "u.user_id ASC")
    page = max(1, page)
    offset = (page - 1) * _ADMIN_PAGE_SIZE

    with get_db() as conn:
        total_count = conn.execute("SELECT COUNT(*) as cnt FROM users").fetchone()["cnt"]
        rows = conn.execute(
            f"SELECT u.user_id, u.username, u.created_at, u.status, "
            f"MAX(ll.logged_in_at) as last_login, "
            f"COUNT(DISTINCT ll.log_id) as login_count, "
            f"(SELECT COUNT(*) FROM ratings r WHERE r.user_id = u.user_id) as rating_count, "
            f"(SELECT COUNT(*) FROM reviews rv WHERE rv.user_id = u.user_id AND rv.deleted_at IS NULL) as review_count "
            f"FROM users u LEFT JOIN login_logs ll ON u.user_id = ll.user_id "
            f"GROUP BY u.user_id "
            f"ORDER BY {order_clause} "
            f"LIMIT {_ADMIN_PAGE_SIZE} OFFSET {offset}"
        ).fetchall()

    total_pages = max(1, (total_count + _ADMIN_PAGE_SIZE - 1) // _ADMIN_PAGE_SIZE)

    users = [
        {
            "user_id": r["user_id"],
            "username": r["username"],
            "created_at": r["created_at"][:10] if r["created_at"] else "-",
            "status": r["status"] or "active",
            "last_login": r["last_login"][:10] if r["last_login"] else "-",
            "login_count": r["login_count"],
            "rating_count": r["rating_count"],
            "review_count": r["review_count"],
        }
        for r in rows
    ]

    # 仮パスワード表示（セッションから読み取り後クリア）
    reset_pw = request.session.pop("_reset_pw", "")
    reset_uid = request.session.pop("_reset_uid", "")

    return templates.TemplateResponse("admin_users.html", {
        "request": request,
        "users": users,
        "current_sort": sort,
        "current_page": page,
        "total_pages": total_pages,
        "total_count": total_count,
        "reset_pw": reset_pw,
        "reset_uid": reset_uid,
    })


@app.post("/admin/users/warn")
async def admin_warn_user(user_id: int = Form(...)):
    """ユーザーに警告"""
    with get_db() as conn:
        conn.execute("UPDATE users SET status = 'warned' WHERE user_id = ?", (user_id,))
    return RedirectResponse(url="/admin/users", status_code=303)


@app.post("/admin/users/ban")
async def admin_ban_user(user_id: int = Form(...)):
    """ユーザーを凍結（warned ユーザーのみ）"""
    with get_db() as conn:
        status = conn.execute("SELECT status FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if status and status["status"] in ("warned", "banned"):
            conn.execute("UPDATE users SET status = 'banned' WHERE user_id = ?", (user_id,))
    return RedirectResponse(url="/admin/users", status_code=303)


@app.post("/admin/users/delete")
async def admin_delete_user(user_id: int = Form(...)):
    """ユーザーを削除（banned ユーザーのみ）"""
    with get_db() as conn:
        status = conn.execute("SELECT status FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if status and status["status"] == "banned":
            for table in ["ratings", "recommendations", "reviews", "login_logs", "page_views", "review_reports"]:
                conn.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
    return RedirectResponse(url="/admin/users", status_code=303)


@app.post("/admin/users/restore")
async def admin_restore_user(user_id: int = Form(...)):
    """ユーザーのステータスを active に戻す"""
    with get_db() as conn:
        conn.execute("UPDATE users SET status = 'active' WHERE user_id = ?", (user_id,))
    return RedirectResponse(url="/admin/users", status_code=303)


@app.post("/admin/users/reset_password")
async def admin_reset_password(request: Request, user_id: int = Form(...)):
    """管理者がユーザーのパスワードをリセット（仮パスワードを設定）"""
    import string
    import random
    # 8文字の仮パスワードを生成（英字+数字+記号）
    chars = string.ascii_letters + string.digits
    temp_pw = ''.join(random.choices(chars, k=6)) + random.choice("@$!%*#?&") + random.choice(string.digits)
    pw_hash = bcrypt.hashpw(temp_pw.encode(), bcrypt.gensalt()).decode()
    with get_db() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE user_id = ?", (pw_hash, user_id))
    logging.info(f"Password reset for user_id={user_id}")
    # セッションに一時保存（URL に露出させない）
    request.session["_reset_pw"] = temp_pw
    request.session["_reset_uid"] = str(user_id)
    return RedirectResponse(url="/admin/users", status_code=303)


@app.get("/admin/comments", response_class=HTMLResponse)
async def admin_comments_page(request: Request, sort: str = "date_desc", page: int = 1, tab: str = "active"):
    """口コミ管理ページ（ソート・ページネーション・タブ対応）"""
    order_map = {
        "date_desc": "rv.created_at DESC",
        "date_asc": "rv.created_at ASC",
        "user_asc": "u.username ASC, rv.created_at DESC",
        "user_desc": "u.username DESC, rv.created_at DESC",
        "store_asc": "o.object_name ASC, rv.created_at DESC",
        "store_desc": "o.object_name DESC, rv.created_at DESC",
    }
    order_clause = order_map.get(sort, "rv.created_at DESC")
    page = max(1, page)
    offset = (page - 1) * _ADMIN_PAGE_SIZE
    show_deleted = tab == "deleted"

    with get_db() as conn:
        # 両タブの件数を取得（タブに件数バッジを表示するため）
        active_count = conn.execute("SELECT COUNT(*) as cnt FROM reviews WHERE deleted_at IS NULL").fetchone()["cnt"]
        deleted_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM reviews WHERE deleted_at IS NOT NULL"
        ).fetchone()["cnt"]
        total_count = deleted_count if show_deleted else active_count

        if show_deleted:
            rows = conn.execute(
                f"SELECT rv.review_id, u.username, o.object_name, rv.comment, rv.created_at, "
                f"rv.deleted_at, rv.deleted_by "
                f"FROM reviews rv "
                f"JOIN users u ON rv.user_id = u.user_id "
                f"JOIN objects o ON rv.object_id = o.object_id "
                f"WHERE rv.deleted_at IS NOT NULL "
                f"ORDER BY rv.deleted_at DESC "
                f"LIMIT {_ADMIN_PAGE_SIZE} OFFSET {offset}"
            ).fetchall()
            report_rows = []
        else:
            rows = conn.execute(
                f"SELECT rv.review_id, u.username, o.object_name, rv.comment, rv.created_at "
                f"FROM reviews rv "
                f"JOIN users u ON rv.user_id = u.user_id "
                f"JOIN objects o ON rv.object_id = o.object_id "
                f"WHERE rv.deleted_at IS NULL "
                f"ORDER BY {order_clause} "
                f"LIMIT {_ADMIN_PAGE_SIZE} OFFSET {offset}"
            ).fetchall()

            # 通報データを取得（現在ページの review_id に限定）
            review_ids = [r["review_id"] for r in rows]
            if review_ids:
                placeholders = ",".join("?" * len(review_ids))
                report_rows = conn.execute(
                    f"SELECT rr.report_id, rr.review_id, rr.reason, rr.created_at, u.username as reporter "
                    f"FROM review_reports rr JOIN users u ON rr.user_id = u.user_id "
                    f"WHERE rr.review_id IN ({placeholders}) "
                    f"ORDER BY rr.created_at DESC",
                    review_ids
                ).fetchall()
            else:
                report_rows = []

    total_pages = max(1, (total_count + _ADMIN_PAGE_SIZE - 1) // _ADMIN_PAGE_SIZE)

    # review_id ごとに通報をグループ化
    reports_by_review = {}
    for rr in report_rows:
        rid = rr["review_id"]
        if rid not in reports_by_review:
            reports_by_review[rid] = []
        reports_by_review[rid].append({
            "report_id": rr["report_id"],
            "reporter": rr["reporter"],
            "reason": rr["reason"],
            "created_at": rr["created_at"][:10] if rr["created_at"] else "",
        })

    comments = [
        {
            "review_id": r["review_id"],
            "username": r["username"],
            "object_name": r["object_name"],
            "comment": r["comment"],
            "created_at": r["created_at"][:10] if r["created_at"] else "",
            "reports": reports_by_review.get(r["review_id"], []),
            "deleted_at": r["deleted_at"][:10] if show_deleted and r["deleted_at"] else "",
            "deleted_by": {"admin": "管理者", "user": "本人"}.get(r["deleted_by"], "") if show_deleted else "",
        }
        for r in rows
    ]

    return templates.TemplateResponse("admin_comments.html", {
        "request": request,
        "comments": comments,
        "current_sort": sort,
        "current_page": page,
        "total_pages": total_pages,
        "total_count": total_count,
        "current_tab": tab,
        "active_count": active_count,
        "deleted_count": deleted_count,
    })


@app.post("/admin/restore_review")
async def admin_restore_review(review_id: int = Form(...)):
    """削除済み口コミを復元"""
    with get_db() as conn:
        conn.execute(
            "UPDATE reviews SET deleted_at = NULL, deleted_by = NULL WHERE review_id = ?",
            (review_id,)
        )
    return RedirectResponse(url="/admin/comments?tab=deleted", status_code=303)


@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request, error: str = ""):
    """管理者ログインページ"""
    if request.session.get("is_admin"):
        return RedirectResponse(url="/admin/dashboard", status_code=303)
    return templates.TemplateResponse("admin_login.html", {"request": request, "error": error})


@app.post("/admin/login")
async def admin_login(request: Request, username: str = Form(...), password: str = Form(...)):
    """管理者ログイン処理"""
    if not ADMIN_PASSWORD_HASH:
        return templates.TemplateResponse("admin_login.html", {
            "request": request, "error": "管理者アカウントが設定されていません。"
        })
    if username != ADMIN_USERNAME:
        return templates.TemplateResponse("admin_login.html", {
            "request": request, "error": "ユーザー名またはパスワードが間違っています。"
        })
    if not bcrypt.checkpw(password.encode(), ADMIN_PASSWORD_HASH.encode()):
        return templates.TemplateResponse("admin_login.html", {
            "request": request, "error": "ユーザー名またはパスワードが間違っています。"
        })
    request.session["is_admin"] = True
    return RedirectResponse(url="/admin/dashboard", status_code=303)


@app.post("/admin/logout")
async def admin_logout(request: Request):
    """管理者ログアウト"""
    request.session.pop("is_admin", None)
    return RedirectResponse(url="/admin/login", status_code=303)


def _make_ts_chart(datasets, title, ylabel):
    """時系列折れ線グラフを Base64 PNG として生成"""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for label, rows, color in datasets:
        if rows:
            days_list = [r["day"] for r in rows]
            counts = [r["cnt"] for r in rows]
            ax.plot(days_list, counts, marker='o', markersize=4, label=label, color=color, linewidth=2)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if ax.get_xlim()[1] - ax.get_xlim()[0] > 10:
        ax.tick_params(axis='x', rotation=45, labelsize=9)
    else:
        ax.tick_params(axis='x', rotation=45, labelsize=10)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _make_bar_chart(rows, title):
    """横棒グラフを Base64 PNG として生成"""
    if not rows:
        return None
    names = [r["object_name"][:15] for r in rows][::-1]
    counts = [r["cnt"] for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(10, max(3, len(names) * 0.35)))
    bars = ax.barh(names, counts, color='#2e7d32', height=0.6)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Views', fontsize=11)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, str(cnt),
                va='center', fontsize=10)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _dashboard_query_data(since_date, inactive_threshold_date):
    """ダッシュボード用の全データを DB から取得して辞書で返す"""
    with get_db() as conn:
        new_users = conn.execute(
            "SELECT DATE(created_at) as day, COUNT(*) as cnt FROM users "
            "WHERE created_at IS NOT NULL AND DATE(created_at) >= ? GROUP BY day ORDER BY day",
            (since_date,)
        ).fetchall()

        active_users = conn.execute(
            "SELECT DATE(logged_in_at) as day, COUNT(DISTINCT user_id) as cnt FROM login_logs "
            "WHERE DATE(logged_in_at) >= ? GROUP BY day ORDER BY day",
            (since_date,)
        ).fetchall()

        daily_ratings = conn.execute(
            "SELECT DATE(rated_at) as day, COUNT(*) as cnt FROM ratings "
            "WHERE rated_at IS NOT NULL AND DATE(rated_at) >= ? GROUP BY day ORDER BY day",
            (since_date,)
        ).fetchall()

        daily_reviews = conn.execute(
            "SELECT DATE(created_at) as day, COUNT(*) as cnt FROM reviews "
            "WHERE deleted_at IS NULL AND DATE(created_at) >= ? GROUP BY day ORDER BY day",
            (since_date,)
        ).fetchall()

        daily_views = conn.execute(
            "SELECT DATE(viewed_at) as day, COUNT(*) as cnt FROM page_views "
            "WHERE DATE(viewed_at) >= ? GROUP BY day ORDER BY day",
            (since_date,)
        ).fetchall()

        store_views = conn.execute(
            "SELECT pv.object_id, o.object_name, COUNT(*) as cnt "
            "FROM page_views pv JOIN objects o ON pv.object_id = o.object_id "
            "GROUP BY pv.object_id ORDER BY cnt DESC LIMIT ?",
            (STORE_VIEWS_RANKING_LIMIT,)
        ).fetchall()

        all_objects = conn.execute(
            "SELECT object_id, object_name, latitude, longitude FROM objects ORDER BY object_id"
        ).fetchall()

        no_review_objects = conn.execute(
            "SELECT o.object_id, o.object_name FROM objects o "
            "LEFT JOIN reviews r ON o.object_id = r.object_id AND r.deleted_at IS NULL "
            "WHERE r.review_id IS NULL ORDER BY o.object_id"
        ).fetchall()

        few_ratings = conn.execute(
            "SELECT o.object_id, o.object_name, COUNT(r.rating) as cnt "
            "FROM objects o LEFT JOIN ratings r ON o.object_id = r.object_id "
            "GROUP BY o.object_id HAVING cnt < ? ORDER BY cnt",
            (FEW_RATINGS_THRESHOLD,)
        ).fetchall()

        inactive_users = conn.execute(
            "SELECT u.user_id, u.username, u.status, MAX(ll.logged_in_at) as last_login "
            "FROM users u LEFT JOIN login_logs ll ON u.user_id = ll.user_id "
            "GROUP BY u.user_id "
            "HAVING last_login < ? OR last_login IS NULL "
            "ORDER BY last_login",
            (inactive_threshold_date,)
        ).fetchall()

        pending_requests = conn.execute(
            "SELECT COUNT(*) as cnt, "
            "AVG(julianday('now') - julianday(created_at)) as avg_days "
            "FROM object_requests WHERE status = 'pending'"
        ).fetchone()

        reported_reviews = conn.execute(
            "SELECT rr.report_id, rr.review_id, rr.reason, rr.created_at, "
            "rv.comment, rv.object_id, u_reporter.username as reporter, u_author.username as author "
            "FROM review_reports rr "
            "JOIN reviews rv ON rr.review_id = rv.review_id AND rv.deleted_at IS NULL "
            "JOIN users u_reporter ON rr.user_id = u_reporter.user_id "
            "JOIN users u_author ON rv.user_id = u_author.user_id "
            "ORDER BY rr.created_at DESC"
        ).fetchall()

    return {
        "new_users": new_users, "active_users": active_users,
        "daily_ratings": daily_ratings, "daily_reviews": daily_reviews,
        "daily_views": daily_views, "store_views": store_views,
        "all_objects": all_objects, "no_review_objects": no_review_objects,
        "few_ratings": few_ratings, "inactive_users": inactive_users,
        "pending_requests": pending_requests, "reported_reviews": reported_reviews,
    }


@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request, period: str = "30"):
    """管理ダッシュボード"""
    now = datetime.datetime.now()
    inactive_threshold_date = (now - datetime.timedelta(days=INACTIVE_DAYS_THRESHOLD)).isoformat()

    period_map = {"1": 1, "7": 7, "30": 30, "90": 90, "180": 180, "365": 365}
    days = period_map.get(period, 30)
    period_label = {"1": "直近1日", "7": "直近1週間", "30": "直近30日",
                    "90": "直近90日", "180": "直近180日", "365": "直近1年間"}.get(period, "直近30日")
    since_date = (now - datetime.timedelta(days=days)).date().isoformat()

    # DB からデータ取得
    data = _dashboard_query_data(since_date, inactive_threshold_date)

    # 写真未登録チェック
    image_map = get_store_image_map()
    no_photo = [r for r in data["all_objects"] if str(r["object_id"]) not in image_map]
    no_location = [r for r in data["all_objects"] if r["latitude"] is None or r["longitude"] is None]

    # チャート生成（キャッシュ利用）
    cache_key = f"dashboard_{period}"
    cached = _get_cached(cache_key)
    if cached:
        chart_activity, chart_content, chart_store_views = cached
    else:
        chart_activity = _make_ts_chart([
            ('Active Users', data["active_users"], '#1565c0'),
            ('Card Views', data["daily_views"], '#2e7d32'),
        ], 'Daily Active Users & Card Views', 'Count')

        chart_content = _make_ts_chart([
            ('Ratings', data["daily_ratings"], '#f59e0b'),
            ('Reviews', data["daily_reviews"], '#e65100'),
            ('New Users', data["new_users"], '#1565c0'),
        ], 'Daily Ratings, Reviews & New Users', 'Count')

        chart_store_views = _make_bar_chart(data["store_views"], 'Store Card Views Ranking')
        _set_cached(cache_key, (chart_activity, chart_content, chart_store_views))

    return templates.TemplateResponse("admin_dashboard.html", {
        "request": request,
        "current_period": period,
        "period_label": period_label,
        "chart_activity": chart_activity,
        "chart_content": chart_content,
        "chart_store_views": chart_store_views,
        "no_photo": [{"object_id": r["object_id"], "object_name": r["object_name"]} for r in no_photo],
        "no_location": [{"object_id": r["object_id"], "object_name": r["object_name"]} for r in no_location],
        "no_review_objects": [{"object_id": r["object_id"], "object_name": r["object_name"]} for r in data["no_review_objects"]],
        "few_ratings": [{"object_id": r["object_id"], "object_name": r["object_name"], "cnt": r["cnt"]} for r in data["few_ratings"]],
        "inactive_users": [
            {"user_id": r["user_id"], "username": r["username"],
             "last_login": r["last_login"][:10] if r["last_login"] else "未ログイン",
             "status": r["status"] or "active"}
            for r in data["inactive_users"]
        ],
        "pending_count": data["pending_requests"]["cnt"] or 0,
        "pending_avg_days": round(data["pending_requests"]["avg_days"] or 0, 1),
        "reported_reviews": [
            {
                "report_id": r["report_id"], "review_id": r["review_id"],
                "reporter": r["reporter"], "author": r["author"],
                "comment": r["comment"][:80], "reason": r["reason"],
                "created_at": r["created_at"][:10], "object_id": r["object_id"],
            }
            for r in data["reported_reviews"]
        ],
    })


# =============================
# バックアップ・リストア（管理者）
# =============================

@app.get("/admin/backup")
async def admin_backup():
    """DB と店舗画像を ZIP にまとめてダウンロード"""
    import zipfile
    import shutil

    today = datetime.datetime.now().strftime("%Y%m%d")
    zip_filename = f"lunchmap_backup_{today}.zip"

    buf = BytesIO()
    db_path = DATA_DIR / "lunchmap.db"

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # DB ファイル（WAL をフラッシュしてから安全にコピー）
        if db_path.exists():
            # VACUUM INTO で一貫性のあるスナップショットを作成
            snapshot_path = DATA_DIR / "lunchmap_backup_tmp.db"
            # パスを resolve して DATA_DIR 配下であることを検証
            resolved = snapshot_path.resolve()
            assert str(resolved).startswith(str(DATA_DIR.resolve())), "Invalid snapshot path"
            try:
                import sqlite3 as _sqlite3
                src = _sqlite3.connect(str(db_path), timeout=30)
                src.execute(f"VACUUM INTO '{resolved}'")
                src.close()
                zf.write(snapshot_path, "lunchmap.db")
                snapshot_path.unlink()
            except Exception:
                # フォールバック: 直接コピー
                zf.write(db_path, "lunchmap.db")
                if snapshot_path.exists():
                    snapshot_path.unlink()

        # 店舗画像
        if STORE_IMAGES_DIR.exists():
            for img_file in STORE_IMAGES_DIR.iterdir():
                if img_file.is_file():
                    zf.write(img_file, f"store_images/{img_file.name}")

        # セッション秘密鍵
        key_file = DATA_DIR / ".session_secret_key"
        if key_file.exists():
            zf.write(key_file, ".session_secret_key")

    buf.seek(0)
    from starlette.responses import Response
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
    )


@app.post("/admin/restore")
async def admin_restore(request: Request, backup_file: UploadFile = File(...)):
    """ZIP バックアップからデータを復元（一時展開→検証→差し替え方式）"""
    import zipfile
    import shutil
    import tempfile
    import sqlite3 as _sqlite3

    content = await backup_file.read()
    MAX_BACKUP_SIZE = 100 * 1024 * 1024  # 100MB
    if len(content) > MAX_BACKUP_SIZE:
        return RedirectResponse(url="/admin/dashboard?restore_error=size", status_code=303)

    buf = BytesIO(content)
    if not zipfile.is_zipfile(buf):
        return RedirectResponse(url="/admin/dashboard?restore_error=format", status_code=303)

    buf.seek(0)
    # 一時ディレクトリに展開して検証
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="lunchmap_restore_"))
    try:
        with zipfile.ZipFile(buf, "r") as zf:
            names = zf.namelist()

            # --- Phase 1: 一時ディレクトリに展開 ---
            tmp_db = None
            if "lunchmap.db" in names:
                tmp_db = tmp_dir / "lunchmap.db"
                with open(tmp_db, "wb") as f:
                    f.write(zf.read("lunchmap.db"))

            tmp_images = []
            for name in names:
                if name.startswith("store_images/") and not name.endswith("/"):
                    img_name = pathlib.Path(name).name
                    if ".." in img_name or "/" in img_name:
                        continue
                    tmp_img = tmp_dir / img_name
                    with open(tmp_img, "wb") as f:
                        f.write(zf.read(name))
                    tmp_images.append((tmp_img, img_name))

            tmp_key = None
            if ".session_secret_key" in names:
                tmp_key = tmp_dir / ".session_secret_key"
                with open(tmp_key, "wb") as f:
                    f.write(zf.read(".session_secret_key"))

        # --- Phase 2: DB の整合性を検証 ---
        if tmp_db and tmp_db.exists():
            try:
                test_conn = _sqlite3.connect(str(tmp_db))
                test_conn.execute("SELECT COUNT(*) FROM users")
                test_conn.execute("PRAGMA integrity_check")
                test_conn.close()
            except Exception as e:
                logging.error(f"Restore aborted: backup DB integrity check failed: {e}")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return RedirectResponse(url="/admin/dashboard?restore_error=corrupt", status_code=303)

        # --- Phase 3: 検証成功、既存データを退避してから差し替え ---
        db_path = DATA_DIR / "lunchmap.db"

        if tmp_db and tmp_db.exists():
            # 既存 DB を退避
            if db_path.exists():
                backup_existing = DATA_DIR / "lunchmap_pre_restore.db"
                shutil.copy2(db_path, backup_existing)

            # WAL/SHM を削除
            for ext in ["-wal", "-shm"]:
                wal_file = DATA_DIR / f"lunchmap.db{ext}"
                if wal_file.exists():
                    wal_file.unlink()

            # 検証済み DB を本番パスに移動
            shutil.move(str(tmp_db), str(db_path))
            logging.info("Restored lunchmap.db from backup (integrity verified)")

        # 店舗画像を復元
        STORE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        for tmp_img, img_name in tmp_images:
            shutil.move(str(tmp_img), str(STORE_IMAGES_DIR / img_name))
            logging.info(f"Restored store image: {img_name}")

        # セッション秘密鍵を復元
        if tmp_key and tmp_key.exists():
            shutil.move(str(tmp_key), str(DATA_DIR / ".session_secret_key"))
            logging.info("Restored session secret key from backup")

        # DB を再初期化（マイグレーション適用）
        from database import init_db
        init_db()

        return RedirectResponse(url="/admin/dashboard?restore_ok=1", status_code=303)
    except Exception as e:
        logging.error(f"Restore failed: {e}")
        return RedirectResponse(url="/admin/dashboard?restore_error=failed", status_code=303)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/report_review")
@limiter.limit("5/minute")
async def report_review(request: Request, review_id: int = Form(...), reason: str = Form("")):
    """口コミを通報"""
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse(url="/", status_code=303)
    with get_db() as conn:
        existing = conn.execute(
            "SELECT 1 FROM review_reports WHERE review_id = ? AND user_id = ?",
            (review_id, int(user_id))
        ).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO review_reports (review_id, user_id, reason, created_at) VALUES (?, ?, ?, ?)",
                (review_id, int(user_id), reason.strip(), datetime.datetime.now().isoformat())
            )
        rv = conn.execute("SELECT object_id FROM reviews WHERE review_id = ?", (review_id,)).fetchone()
    redirect_url = f"/store/{rv['object_id']}?reported=1" if rv else "/recommendations"
    return RedirectResponse(url=redirect_url, status_code=303)


@app.post("/admin/dismiss_report")
async def admin_dismiss_report(report_id: int = Form(...)):
    """管理者による通報の却下（通報データを削除）"""
    with get_db() as conn:
        conn.execute("DELETE FROM review_reports WHERE report_id = ?", (report_id,))
    return RedirectResponse(url="/admin/comments", status_code=303)


@app.post("/admin/dismiss_all_reports")
async def admin_dismiss_all_reports(review_id: int = Form(...)):
    """管理者による口コミへの全通報を一括却下"""
    with get_db() as conn:
        conn.execute("DELETE FROM review_reports WHERE review_id = ?", (review_id,))
    return RedirectResponse(url="/admin/comments", status_code=303)


@app.post("/admin/delete_review")
async def admin_delete_review(review_id: int = Form(...)):
    """管理者による口コミ削除（論理削除＋関連通報を削除）"""
    with get_db() as conn:
        conn.execute("DELETE FROM review_reports WHERE review_id = ?", (review_id,))
        conn.execute(
            "UPDATE reviews SET deleted_at = ?, deleted_by = 'admin' WHERE review_id = ?",
            (datetime.datetime.now().isoformat(), review_id)
        )
    return RedirectResponse(url="/admin/comments", status_code=303)
