"""
協調フィルタリング推薦エンジン（NumPy行列演算版）
"""
import pandas as pd
import numpy as np
import os
import sqlite3
import logging
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

__all__ = [
    "recommend_for_all_users",
    "recommend_for_single_user",
    "categorize_recommendation",
    "get_username",
    "get_recommendations_for_user",
    "update_user_similarity_from_db",
    "user_object_matrix",
    "user_zscore_matrix",
    "user_info",
    "user_similarity",
]

# ---- DB パス ----
DATA_DIR = Path(os.getenv("DATA_DIR", "/data")).resolve()
DB_PATH = DATA_DIR / "lunchmap.db"

import threading

# グローバル変数
user_object_matrix = pd.DataFrame()
user_zscore_matrix = pd.DataFrame()
user_similarity = pd.DataFrame()
user_info = {}
object_info = {}
_sim_lock = threading.Lock()  # 類似度行列の競合防止


def _get_conn():
    return sqlite3.connect(str(DB_PATH), timeout=30)


def update_user_and_object_info():
    global user_info, object_info
    try:
        conn = _get_conn()
        user_df = pd.read_sql_query(
            "SELECT CAST(user_id AS TEXT) as user_id, username FROM users", conn
        )
        objects_df = pd.read_sql_query(
            "SELECT CAST(object_id AS TEXT) as object_id, object_name FROM objects", conn
        )
        conn.close()
    except Exception as e:
        logging.warning(f"update_user_and_object_info error: {e}")
        return

    user_info = dict(zip(user_df["user_id"].str.strip(), user_df["username"].fillna("").astype(str)))
    object_info = dict(zip(objects_df["object_id"].str.strip(), objects_df["object_name"].fillna("").astype(str)))


def compute_standardized_rating(series):
    """各ユーザーの rating を Z-score 標準化"""
    mean = series.mean()
    std = series.std(ddof=1)
    if pd.isna(std) or std < 0.1:
        return series * 0
    return (series - mean) / std


def update_user_similarity_from_db():
    """DB から評価データを読み込み、類似度行列を更新"""
    global user_object_matrix, user_zscore_matrix, user_similarity

    try:
        conn = _get_conn()
        df = pd.read_sql_query(
            "SELECT CAST(r.user_id AS TEXT) as user_id, CAST(r.object_id AS TEXT) as object_id, r.rating "
            "FROM ratings r JOIN users u ON r.user_id = u.user_id "
            "WHERE u.status IS NULL OR u.status IN ('active', 'warned')",
            conn,
        )
        conn.close()
    except Exception as e:
        logging.warning(f"update_user_similarity_from_db error: {e}")
        with _sim_lock:
            user_object_matrix = pd.DataFrame()
            user_zscore_matrix = pd.DataFrame()
            user_similarity = pd.DataFrame()
        return

    if df.empty:
        with _sim_lock:
            user_object_matrix = pd.DataFrame()
            user_zscore_matrix = pd.DataFrame()
            user_similarity = pd.DataFrame()
        return

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
    df["z_score"] = df.groupby("user_id")["rating"].transform(compute_standardized_rating).fillna(0)

    _uom = df.pivot_table(index="user_id", columns="object_id", values="rating", aggfunc="mean")
    _uzm = df.pivot_table(index="user_id", columns="object_id", values="z_score", aggfunc="mean").fillna(0)

    # 文字列に統一
    _uom.index = _uom.index.astype(str)
    _uom.columns = _uom.columns.astype(str)
    _uzm.index = _uzm.index.astype(str)
    _uzm.columns = _uzm.columns.astype(str)

    # コサイン類似度（行列演算）
    sim_array = cosine_similarity(_uzm)
    _us = pd.DataFrame(sim_array, index=_uzm.index, columns=_uzm.index)

    # ロックで一括代入（読み取り側が中途半端な状態を見ない）
    with _sim_lock:
        user_object_matrix = _uom
        user_zscore_matrix = _uzm
        user_similarity = _us


def recommend_for_all_users(threshold=0.3):
    """全ユーザーの推薦スコアを行列演算で一括計算"""
    global user_object_matrix, user_zscore_matrix, user_similarity

    update_user_similarity_from_db()

    if user_zscore_matrix.empty or user_similarity.empty or user_object_matrix.empty:
        return pd.DataFrame(columns=["user_id", "object_id", "recommendation_score"]), user_similarity

    # 類似度行列から閾値未満をゼロにし、自己類似度もゼロにする
    sim = user_similarity.values.copy()
    np.fill_diagonal(sim, 0)
    sim[sim < threshold] = 0

    # Zスコア行列 (users × objects)
    z = user_zscore_matrix.values  # NaN なし (fillna(0) 済み)

    # 評価済みマスク (True = 評価済み)
    rated_mask = ~user_object_matrix.reindex(
        index=user_zscore_matrix.index, columns=user_zscore_matrix.columns
    ).isna().values

    # 未評価マスク (True = 未評価 → 推薦対象)
    unrated_mask = ~rated_mask

    # 加重和: sim (U×U) @ (z * rated_mask) → (U×O)
    # 各ユーザーに対し、類似ユーザーの「評価済み店舗のZスコア × 類似度」の和
    weighted_sum = sim @ (z * rated_mask)

    # 重み和: sim (U×U) @ rated_mask → (U×O)
    weight_sum = sim @ rated_mask.astype(float)

    # 推薦スコア = 加重和 / 重み和
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = np.where(weight_sum > 0, weighted_sum / weight_sum, 0)

    # 未評価のもののみ結果として抽出
    user_ids = user_zscore_matrix.index.values
    object_ids = user_zscore_matrix.columns.values

    rows = []
    for i, uid in enumerate(user_ids):
        for j, oid in enumerate(object_ids):
            if unrated_mask[i, j]:
                rows.append((uid, oid, round(float(scores[i, j]), 2)))

    result_df = pd.DataFrame(rows, columns=["user_id", "object_id", "recommendation_score"])
    return result_df, user_similarity


def recommend_for_single_user(target_user_id, threshold=0.3):
    """特定ユーザーのみの推薦スコアを行列演算で計算（即時応答用）"""
    global user_object_matrix, user_zscore_matrix, user_similarity

    target_user_id = str(target_user_id)

    if user_zscore_matrix.empty or user_similarity.empty or user_object_matrix.empty:
        return pd.DataFrame(columns=["user_id", "object_id", "recommendation_score"])

    if target_user_id not in user_similarity.index:
        return pd.DataFrame(columns=["user_id", "object_id", "recommendation_score"])

    # 類似度ベクトル (1×U)
    sim_vec = user_similarity.loc[target_user_id].values.copy()
    sim_vec[user_similarity.index.get_loc(target_user_id)] = 0  # 自己を除外
    sim_vec[sim_vec < threshold] = 0

    # Zスコア行列 (U×O)
    z = user_zscore_matrix.values

    # 評価済みマスク
    rated_mask = ~user_object_matrix.reindex(
        index=user_zscore_matrix.index, columns=user_zscore_matrix.columns
    ).isna().values

    # ターゲットユーザーの未評価マスク
    target_idx = user_zscore_matrix.index.get_loc(target_user_id)
    target_unrated = ~rated_mask[target_idx]

    # 加重和: sim_vec (1×U) @ (z * rated_mask) → (1×O)
    weighted_sum = sim_vec @ (z * rated_mask)
    weight_sum = sim_vec @ rated_mask.astype(float)

    with np.errstate(divide='ignore', invalid='ignore'):
        scores = np.where(weight_sum > 0, weighted_sum / weight_sum, 0)

    object_ids = user_zscore_matrix.columns.values
    rows = [
        (target_user_id, oid, round(float(scores[j]), 2))
        for j, oid in enumerate(object_ids)
        if target_unrated[j]
    ]

    return pd.DataFrame(rows, columns=["user_id", "object_id", "recommendation_score"])


def explain_recommendations(target_user_id, object_ids, threshold=0.3):
    """推薦理由を生成する。object_ids は理由を知りたい店舗IDのリスト。
    A: 確信度（人数×類似度）、B: 一致度（評価のばらつき）、
    C: 共通の好み（共通高評価店舗名）、D: スコア補足 を組み合わせて理由文を生成。
    Returns: {object_id: {"reason": str, "sub_reason": str}}
    """
    target_user_id = str(target_user_id)

    if user_similarity.empty or user_object_matrix.empty:
        return {}

    if target_user_id not in user_similarity.index:
        return {}

    # 類似度ベクトル（閾値適用済み）
    sim_series = user_similarity.loc[target_user_id].copy()
    sim_series[target_user_id] = 0
    sim_series[sim_series < threshold] = 0
    similar_users = sim_series[sim_series > 0]

    if similar_users.empty:
        return {}

    # C: 対象ユーザーの高評価店舗（素点4点以上）を事前計算
    target_ratings = user_object_matrix.loc[target_user_id] if target_user_id in user_object_matrix.index else pd.Series(dtype=float)
    target_favorites = set(target_ratings[target_ratings >= 4.0].index) if not target_ratings.empty else set()

    # オブジェクト名の辞書を準備
    update_user_and_object_info()

    results = {}
    for oid in object_ids:
        oid = str(oid)
        if oid not in user_object_matrix.columns:
            continue

        # この店を評価済みの類似ユーザーを抽出
        ratings_col = user_object_matrix[oid]
        rated_similar = similar_users.index.intersection(ratings_col.dropna().index)
        rated_similar = [u for u in rated_similar if similar_users[u] > 0]

        if not rated_similar:
            results[oid] = {"reason": "", "sub_reason": ""}
            continue

        count = len(rated_similar)
        ratings_values = ratings_col[rated_similar]
        avg_rating = float(ratings_values.mean())
        max_sim = float(similar_users[rated_similar].max())
        min_rating = float(ratings_values.min())
        max_rating = float(ratings_values.max())

        # === A: 確信度に基づくメイン理由文 ===
        many = count >= 5
        high_sim = max_sim >= 0.6

        if many and avg_rating >= 4.0:
            reason = f"好みが近い多くのユーザーが高く評価しています（平均 {avg_rating:.1f}点）"
        elif not many and high_sim and avg_rating >= 4.0:
            reason = f"あなたと特に好みが近いユーザーが高く評価しています（平均 {avg_rating:.1f}点）"
        elif many:
            reason = f"好みが近い{count}人のユーザーが評価しています（平均 {avg_rating:.1f}点）"
        elif high_sim:
            reason = f"あなたと特に好みが近い{count}人のユーザーが評価しています（平均 {avg_rating:.1f}点）"
        else:
            if avg_rating >= 4.0:
                reason = f"好みが近い{count}人のユーザーが高く評価しています（平均 {avg_rating:.1f}点）"
            elif avg_rating >= 3.0:
                reason = f"好みが近い{count}人のユーザーが好意的に評価しています（平均 {avg_rating:.1f}点）"
            else:
                reason = f"好みが近い{count}人のユーザーが評価しています（平均 {avg_rating:.1f}点）"

        # === B: 一致度に基づく補足 ===
        sub_reason = ""
        if count >= 2 and min_rating >= 4.0:
            if min_rating == max_rating:
                sub_reason = f"好みが近いユーザー全員が{int(min_rating)}点をつけています"
            else:
                sub_reason = f"好みが近いユーザー全員が4点以上をつけています"
        elif count >= 2 and (max_rating - min_rating) >= 2:
            sub_reason = f"好みが近いユーザーの間で評価が分かれています（{int(min_rating)}〜{int(max_rating)}点）"

        # === C: 共通の好みに基づく補足（Bがない場合のみ） ===
        if not sub_reason and target_favorites:
            # 類似ユーザーの高評価店舗との共通部分を探す
            common_favorites = set()
            for uid in rated_similar:
                if uid in user_object_matrix.index:
                    u_ratings = user_object_matrix.loc[uid]
                    u_favs = set(u_ratings[u_ratings >= 4.0].index)
                    common_favorites |= (u_favs & target_favorites)
            # 推薦対象の店舗自体は除外
            common_favorites.discard(oid)
            if common_favorites:
                # 店舗名に変換して最大2件
                fav_names = [object_info.get(str(fid), "") for fid in list(common_favorites)[:2]]
                fav_names = [n for n in fav_names if n]
                if fav_names:
                    sub_reason = f"{'や'.join(fav_names)}を高評価した人に人気です"

        results[oid] = {"reason": reason, "sub_reason": sub_reason}

    return results


def categorize_recommendation(score, default=None):
    """スコアに基づいて星マークを付与。default は閾値未満時の戻り値"""
    try:
        score_val = float(score)
    except (ValueError, TypeError):
        return default
    if pd.isna(score_val) or score_val < 0.25:
        return default
    elif score_val >= 0.75:
        return "★★★"
    elif score_val >= 0.5:
        return "★★"
    elif score_val >= 0.25:
        return "★"
    return default


def get_username(user_id):
    return user_info.get(user_id, "Unknown")


def get_recommendations_for_user(username: str):
    update_user_and_object_info()

    user_id = None
    for uid, uname in user_info.items():
        if uname == username:
            user_id = uid
            break
    if user_id is None:
        return {}

    rec_df = recommend_for_single_user(user_id)
    if rec_df.empty:
        return {}

    rec_df = rec_df.sort_values("recommendation_score", ascending=False)

    return {
        (object_info.get(str(row["object_id"]).strip()) or "（名称未登録）"): categorize_recommendation(row["recommendation_score"])
        for _, row in rec_df.iterrows()
        if row["recommendation_score"] >= 0.25
    }
