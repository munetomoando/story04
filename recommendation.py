"""
協調フィルタリング推薦エンジン（NumPy行列演算版）＋ジャンル加重
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
    "object_genre_map",
    "user_genre_profile",
]

# ---- DB パス ----
DATA_DIR = Path(os.getenv("DATA_DIR", "/data")).resolve()
DB_PATH = DATA_DIR / "lunchmap.db"

import threading

# ジャンル加重パラメータ
GENRE_SIM_ALPHA = 0.3   # 類似度ブレンド比率（0=店舗のみ, 1=ジャンルのみ）
GENRE_BOOST_BETA = 0.3  # スコアへのジャンル適合度ブースト係数

# グローバル変数
user_object_matrix = pd.DataFrame()
user_zscore_matrix = pd.DataFrame()
user_similarity = pd.DataFrame()
user_info = {}
object_info = {}
object_genre_map = {}      # {object_id(str): genre(str)}
user_genre_profile = {}    # {user_id(str): {genre: avg_rating}}
_genre_sim_matrix = pd.DataFrame()  # ジャンル嗜好ベースの類似度
_sim_lock = threading.Lock()  # 類似度行列の競合防止


def _get_conn():
    return sqlite3.connect(str(DB_PATH), timeout=30)


def update_user_and_object_info():
    global user_info, object_info, object_genre_map
    try:
        conn = _get_conn()
        user_df = pd.read_sql_query(
            "SELECT CAST(user_id AS TEXT) as user_id, username FROM users", conn
        )
        objects_df = pd.read_sql_query(
            "SELECT CAST(object_id AS TEXT) as object_id, object_name, genre FROM objects", conn
        )
        conn.close()
    except Exception as e:
        logging.warning(f"update_user_and_object_info error: {e}")
        return

    user_info = dict(zip(user_df["user_id"].str.strip(), user_df["username"].fillna("").astype(str)))
    object_info = dict(zip(objects_df["object_id"].str.strip(), objects_df["object_name"].fillna("").astype(str)))
    # ジャンルマップ（genre が NULL/空でないもの）
    object_genre_map = {
        str(row["object_id"]).strip(): str(row["genre"])
        for _, row in objects_df.iterrows()
        if row["genre"] and str(row["genre"]).strip()
    }


def compute_standardized_rating(series):
    """各ユーザーの rating を Z-score 標準化"""
    mean = series.mean()
    std = series.std(ddof=1)
    if pd.isna(std) or std < 0.1:
        return series * 0
    return (series - mean) / std


def _build_user_genre_profile(rating_df):
    """評価データとジャンルマップからユーザーごとのジャンル嗜好プロファイルを構築。
    Returns: {user_id: {genre: avg_rating}}, genre_matrix (DataFrame: users × genres)
    """
    global user_genre_profile, object_genre_map

    # オブジェクト情報を最新化（ジャンルマップ更新）
    update_user_and_object_info()

    if not object_genre_map:
        user_genre_profile = {}
        return pd.DataFrame()

    # 評価にジャンル列を付与
    df = rating_df.copy()
    df["genre"] = df["object_id"].map(object_genre_map)
    df = df.dropna(subset=["genre"])

    if df.empty:
        user_genre_profile = {}
        return pd.DataFrame()

    # ユーザー×ジャンルの平均評価
    genre_pivot = df.pivot_table(
        index="user_id", columns="genre", values="rating", aggfunc="mean"
    ).fillna(0)
    genre_pivot.index = genre_pivot.index.astype(str)

    # プロファイル辞書を構築
    profile = {}
    for uid in genre_pivot.index:
        row = genre_pivot.loc[uid]
        rated_genres = row[row > 0]
        if not rated_genres.empty:
            profile[uid] = rated_genres.to_dict()
    user_genre_profile = profile

    return genre_pivot


def _compute_genre_similarity(genre_matrix):
    """ジャンル嗜好ベクトルからユーザー間コサイン類似度を計算"""
    if genre_matrix.empty or len(genre_matrix) < 2:
        return pd.DataFrame()

    sim_array = cosine_similarity(genre_matrix.values)
    return pd.DataFrame(sim_array, index=genre_matrix.index, columns=genre_matrix.index)


def _get_user_genre_affinity(user_id, object_id):
    """ユーザーの当該店舗ジャンルへの適合度を 0〜1 で返す"""
    user_id = str(user_id)
    object_id = str(object_id)

    genre = object_genre_map.get(object_id)
    if not genre or user_id not in user_genre_profile:
        return 0.0

    profile = user_genre_profile[user_id]
    if genre not in profile:
        return 0.0

    # ユーザーの全ジャンル平均評価の中での相対位置（0〜1に正規化）
    vals = list(profile.values())
    min_v, max_v = min(vals), max(vals)
    if max_v <= min_v:
        return 0.5  # 全ジャンル同評価なら中立
    return (profile[genre] - min_v) / (max_v - min_v)


def update_user_similarity_from_db():
    """DB から評価データを読み込み、類似度行列を更新（ジャンル加重付き）"""
    global user_object_matrix, user_zscore_matrix, user_similarity, _genre_sim_matrix

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
            _genre_sim_matrix = pd.DataFrame()
        return

    if df.empty:
        with _sim_lock:
            user_object_matrix = pd.DataFrame()
            user_zscore_matrix = pd.DataFrame()
            user_similarity = pd.DataFrame()
            _genre_sim_matrix = pd.DataFrame()
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

    # 店舗ベースのコサイン類似度
    store_sim_array = cosine_similarity(_uzm)
    _store_sim = pd.DataFrame(store_sim_array, index=_uzm.index, columns=_uzm.index)

    # ジャンル嗜好プロファイル＆ジャンルベースの類似度
    genre_matrix = _build_user_genre_profile(df)
    _gsim = _compute_genre_similarity(genre_matrix)

    # ブレンド類似度: (1-α)×店舗 + α×ジャンル
    if not _gsim.empty:
        # インデックスを揃える（ジャンル未設定ユーザーは店舗のみ）
        common_users = _store_sim.index.intersection(_gsim.index)
        _blended = _store_sim.copy()
        if len(common_users) >= 2:
            store_sub = _store_sim.loc[common_users, common_users].values
            genre_sub = _gsim.loc[common_users, common_users].values
            blended_sub = (1 - GENRE_SIM_ALPHA) * store_sub + GENRE_SIM_ALPHA * genre_sub
            _blended.loc[common_users, common_users] = blended_sub
        _us = _blended
    else:
        _us = _store_sim

    # ロックで一括代入（読み取り側が中途半端な状態を見ない）
    with _sim_lock:
        user_object_matrix = _uom
        user_zscore_matrix = _uzm
        user_similarity = _us
        _genre_sim_matrix = _gsim


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

    # 未評価のもののみ結果として抽出（ジャンル適合度ブースト付き）
    user_ids = user_zscore_matrix.index.values
    object_ids = user_zscore_matrix.columns.values

    rows = []
    for i, uid in enumerate(user_ids):
        for j, oid in enumerate(object_ids):
            if unrated_mask[i, j]:
                base_score = float(scores[i, j])
                affinity = _get_user_genre_affinity(uid, oid)
                boosted = base_score * (1 + GENRE_BOOST_BETA * affinity)
                rows.append((uid, oid, round(boosted, 2)))

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
    rows = []
    for j, oid in enumerate(object_ids):
        if target_unrated[j]:
            base_score = float(scores[j])
            affinity = _get_user_genre_affinity(target_user_id, oid)
            boosted = base_score * (1 + GENRE_BOOST_BETA * affinity)
            rows.append((target_user_id, oid, round(boosted, 2)))

    return pd.DataFrame(rows, columns=["user_id", "object_id", "recommendation_score"])


def explain_recommendations(target_user_id, object_ids, threshold=0.3):
    """推薦理由を生成する。object_ids は理由を知りたい店舗IDのリスト。
    A: 確信度（人数×類似度）、B: 一致度（評価のばらつき）、
    C: 共通の好み（共通高評価店舗名）、D: ジャンル適合 を組み合わせて理由文を生成。
    Returns: {object_id: {"reason": str, "sub_reason": str, "genre": str, "genre_note": str}}
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

    # ジャンル嗜好情報
    target_profile = user_genre_profile.get(target_user_id, {})
    # ユーザーの好きなジャンルTOP3を特定
    if target_profile:
        sorted_genres = sorted(target_profile.items(), key=lambda x: x[1], reverse=True)
        top_genres = {g for g, v in sorted_genres[:3] if v >= 3.5}
    else:
        top_genres = set()

    results = {}
    for oid in object_ids:
        oid = str(oid)
        if oid not in user_object_matrix.columns:
            continue

        # この店のジャンル
        genre = object_genre_map.get(oid, "")
        genre_note = ""

        # この店を評価済みの類似ユーザーを抽出
        ratings_col = user_object_matrix[oid]
        rated_similar = similar_users.index.intersection(ratings_col.dropna().index)
        rated_similar = [u for u in rated_similar if similar_users[u] > 0]

        if not rated_similar:
            # 類似ユーザーの評価がなくてもジャンル情報で補足
            if genre and genre in top_genres:
                avg_g = target_profile.get(genre, 0)
                genre_note = f"あなたが好む{genre}ジャンル（平均{avg_g:.1f}点）のお店です"
            results[oid] = {"reason": "", "sub_reason": "", "genre": genre, "genre_note": genre_note}
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

        # === D: ジャンル適合に基づく補足 ===
        if genre and target_profile:
            affinity = _get_user_genre_affinity(target_user_id, oid)
            avg_g = target_profile.get(genre, 0)
            if affinity >= 0.7 and avg_g >= 3.5:
                genre_note = f"あなたが好む{genre}ジャンル（平均{avg_g:.1f}点）のお店です"
            elif genre in target_profile and avg_g >= 3.0:
                # 同ジャンルの高評価店舗名を取得
                same_genre_favs = []
                for fav_oid in target_favorites:
                    if object_genre_map.get(str(fav_oid)) == genre:
                        name = object_info.get(str(fav_oid), "")
                        if name:
                            same_genre_favs.append(name)
                if same_genre_favs:
                    names = "や".join(same_genre_favs[:2])
                    genre_note = f"あなたが高評価した{names}と同じ{genre}ジャンルです"
                else:
                    genre_note = f"{genre}ジャンルのお店です"
            elif genre:
                genre_note = f"{genre}ジャンルのお店です"

        results[oid] = {"reason": reason, "sub_reason": sub_reason, "genre": genre, "genre_note": genre_note}

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
