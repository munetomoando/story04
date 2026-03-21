"""
SQLite WAL モード データベース管理モジュール
"""
import sqlite3
import pathlib
import os
import logging
import pandas as pd
from contextlib import contextmanager

DATA_DIR = pathlib.Path(os.getenv("DATA_DIR", "/data")).resolve()
DB_PATH = DATA_DIR / "lunchmap.db"


@contextmanager
def get_db():
    """WALモードのSQLite接続を返すコンテキストマネージャー。
    正常終了時に commit、例外時に rollback して close する。
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """テーブルを作成する（初回のみ・べき等）"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS objects (
            object_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            object_name TEXT    UNIQUE NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ratings (
            user_id   INTEGER NOT NULL,
            object_id INTEGER NOT NULL,
            rating    INTEGER NOT NULL,
            PRIMARY KEY (user_id, object_id)
        );
        CREATE TABLE IF NOT EXISTS recommendations (
            user_id              INTEGER NOT NULL,
            object_id            INTEGER NOT NULL,
            recommendation_score REAL    NOT NULL,
            updated_at           TEXT,
            PRIMARY KEY (user_id, object_id)
        );
        CREATE TABLE IF NOT EXISTS object_requests (
            request_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            object_name TEXT    NOT NULL,
            status      TEXT    NOT NULL DEFAULT 'pending',
            created_at  TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reviews (
            review_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            object_id   INTEGER NOT NULL,
            comment     TEXT    NOT NULL,
            created_at  TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS login_logs (
            log_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            logged_in_at TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS page_views (
            view_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            object_id  INTEGER NOT NULL,
            viewed_at  TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS review_reactions (
            review_id     INTEGER NOT NULL,
            user_id       INTEGER NOT NULL,
            reaction_type TEXT    NOT NULL DEFAULT 'heart',
            created_at    TEXT    NOT NULL,
            PRIMARY KEY (review_id, user_id, reaction_type)
        );
        CREATE TABLE IF NOT EXISTS review_reports (
            report_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id  INTEGER NOT NULL,
            user_id    INTEGER NOT NULL,
            reason     TEXT    NOT NULL DEFAULT '',
            created_at TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_login_logs_at ON login_logs(logged_in_at);
        CREATE INDEX IF NOT EXISTS idx_page_views_at ON page_views(viewed_at);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_review_per_user_object ON reviews(user_id, object_id) WHERE deleted_at IS NULL;
        CREATE INDEX IF NOT EXISTS idx_reviews_deleted ON reviews(deleted_at);
        CREATE TABLE IF NOT EXISTS group_codes (
            code       TEXT PRIMARY KEY,
            label      TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS genres (
            genre_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL UNIQUE,
            sort_order INTEGER NOT NULL DEFAULT 0
        );
    """)
    # 既存テーブルへのカラム追加マイグレーション
    migrations = [
        ("objects", "latitude", "REAL"),
        ("objects", "longitude", "REAL"),
        ("users", "created_at", "TEXT"),
        ("ratings", "rated_at", "TEXT"),
        ("objects", "genre", "TEXT"),
        ("users", "status", "TEXT DEFAULT 'active'"),
        ("users", "group_code", "TEXT"),
        ("reviews", "deleted_at", "TEXT"),
        ("reviews", "deleted_by", "TEXT"),
    ]
    for table, col, col_type in migrations:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass  # カラムが既に存在する場合は無視

    # review_likes → review_reactions マイグレーション
    try:
        conn.execute("SELECT 1 FROM review_likes LIMIT 1")
        # 旧テーブルが存在 → データを移行して削除
        conn.execute(
            "INSERT OR IGNORE INTO review_reactions (review_id, user_id, reaction_type, created_at)"
            " SELECT review_id, user_id, 'heart', created_at FROM review_likes"
        )
        conn.execute("DROP TABLE review_likes")
        conn.commit()
        logging.info("Migrated review_likes to review_reactions")
    except sqlite3.OperationalError:
        pass

    # 旧ユニークインデックスを削除（soft-delete 対応のため部分インデックスに移行）
    try:
        conn.execute("DROP INDEX IF EXISTS idx_one_review_per_user_object")
    except sqlite3.OperationalError:
        pass

    # ジャンルの初期データ投入（テーブルが空の場合のみ）
    if conn.execute("SELECT COUNT(*) FROM genres").fetchone()[0] == 0:
        default_genres = [
            "和食", "洋食", "中華", "イタリアン", "カレー", "ラーメン",
            "蕎麦・うどん", "寿司", "焼肉・韓国料理", "定食・食堂",
            "ファストフード", "カフェ", "タイ・エスニック", "居酒屋", "その他",
        ]
        conn.executemany(
            "INSERT INTO genres (name, sort_order) VALUES (?, ?)",
            [(name, i) for i, name in enumerate(default_genres)]
        )
        conn.commit()
        logging.info(f"Inserted {len(default_genres)} default genres")

    conn.close()
    logging.info("SQLite DB initialized (WAL mode)")


def migrate_from_csv(data_dir: pathlib.Path, seed_dir: pathlib.Path):
    """既存 CSV データを SQLite に移行する（テーブルが空の場合のみ）"""
    with get_db() as conn:
        # ---- users ----
        if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            for csv_dir in [data_dir, seed_dir]:
                p = csv_dir / "users.csv"
                if p.exists():
                    try:
                        df = pd.read_csv(str(p), encoding="utf-8-sig", dtype=str, keep_default_na=False)
                        if not df.empty and "username" in df.columns:
                            conn.executemany(
                                "INSERT OR IGNORE INTO users (user_id, username, password_hash) VALUES (?, ?, ?)",
                                [
                                    (int(r["user_id"]), r["username"].strip(), r["password_hash"].strip())
                                    for _, r in df.iterrows()
                                ]
                            )
                            logging.info(f"Migrated {len(df)} users from {p}")
                            break
                    except Exception as e:
                        logging.warning(f"migrate users from {p}: {e}")

        # ---- objects ----
        if conn.execute("SELECT COUNT(*) FROM objects").fetchone()[0] == 0:
            for csv_dir in [data_dir, seed_dir]:
                p = csv_dir / "objects.csv"
                if p.exists():
                    try:
                        df = pd.read_csv(str(p), encoding="utf-8-sig", dtype=str, keep_default_na=False)
                        if not df.empty and "object_name" in df.columns:
                            conn.executemany(
                                "INSERT OR IGNORE INTO objects (object_id, object_name) VALUES (?, ?)",
                                [
                                    (int(r["object_id"]), r["object_name"].strip())
                                    for _, r in df.iterrows()
                                ]
                            )
                            logging.info(f"Migrated {len(df)} objects from {p}")
                            break
                    except Exception as e:
                        logging.warning(f"migrate objects from {p}: {e}")

        # ---- ratings ----
        if conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0] == 0:
            for csv_dir in [data_dir, seed_dir]:
                p = csv_dir / "ratings.csv"
                if p.exists():
                    try:
                        df = pd.read_csv(str(p), encoding="utf-8-sig", dtype=str, keep_default_na=False)
                        if not df.empty and "rating" in df.columns:
                            rows = []
                            for _, r in df.iterrows():
                                try:
                                    rows.append((int(r["user_id"]), int(r["object_id"]), int(float(r["rating"]))))
                                except (ValueError, KeyError):
                                    pass
                            conn.executemany(
                                "INSERT OR IGNORE INTO ratings (user_id, object_id, rating) VALUES (?, ?, ?)",
                                rows
                            )
                            logging.info(f"Migrated {len(rows)} ratings from {p}")
                            break
                    except Exception as e:
                        logging.warning(f"migrate ratings from {p}: {e}")

    logging.info("CSV → SQLite migration done")
