import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity


__all__ = [
    "recommend_for_all_users", 
    "categorize_recommendation", 
    "get_username", 
    "get_recommendations_for_user", 
    "user_object_matrix",
    "user_zscore_matrix",  # ✅ 追加
    "user_info", 
    "user_similarity"
]

RATINGS_FILE = "/opt/render/project/src/ratings.csv"
USERS_FILE = "/opt/render/project/src/users.csv"
OBJECTS_FILE = "/opt/render/project/src/objects.csv"

# グローバル変数（既存コードで使われているもの）
user_object_matrix = pd.DataFrame()
user_zscore_matrix = pd.DataFrame()
user_similarity = pd.DataFrame()

def update_user_and_object_info():
    user_df = pd.read_csv(USERS_FILE, encoding="utf-8-sig", dtype={"user_id": str})
    global user_info
    user_info = dict(zip(user_df["user_id"], user_df["username"]))

    objects_df = pd.read_csv(OBJECTS_FILE, encoding="utf-8-sig", dtype={"object_id": str})
    global object_info
    object_info = dict(zip(objects_df["object_id"], objects_df["object_name"]))

def compute_standardized_rating(series):
    """各ユーザーの rating を平均と標準偏差で標準化する。標準偏差が小さい場合は 0.1 を下限とする"""
    mean = series.mean()
    std = series.std(ddof=1)  # 標本標準偏差を計算
    if pd.isna(std) or std < 0.1:  # NaN または標準偏差が小さすぎる場合
        return series * 0  # 標準偏差が 0 ならすべて 0 にする
    return (series - mean) / std


def update_user_similarity_from_csv():
    global user_object_matrix, user_zscore_matrix, user_similarity

    # --- (1) ratings.csv の存在チェック & 読み込み ---
    if not os.path.exists(RATINGS_FILE):
        print("❌ RATINGS_FILEが見つかりません。空のDataFrameを返します。")
        # 空で初期化
        user_object_matrix = pd.DataFrame()
        user_zscore_matrix = pd.DataFrame()
        user_similarity = pd.DataFrame()
        return  # ここで終了

    # CSV読み込み
    df = pd.read_csv(RATINGS_FILE, encoding="utf-8-sig", dtype={"user_id": str, "object_id": str})
    # 数値化
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
    else:
        df["rating"] = 0

    # --- (2) Zスコアの計算 ---
    if not df.empty and "rating" in df.columns:
        # グループごとに transform() を用いて標準化
        df["z_score"] = df.groupby("user_id")["rating"].transform(compute_standardized_rating).fillna(0)
        
        # ✅ デバッグ用にZスコアを表示
        print("🔍 Zスコアの計算結果 (先頭10行):")
        print(df[["user_id", "object_id", "rating", "z_score"]].head(10))
    else:
        # もし空の場合はZスコア計算できないので、そのまま
        df["z_score"] = 0

    # --- (3) ピボットテーブル作成 ---
    if not df.empty:
        # 同一 user_id, object_id が複数ある場合は平均値を採用
        user_object_matrix_local = df.pivot_table(
            index="user_id", columns="object_id", values="rating", aggfunc="mean"
        )
        user_zscore_matrix_local = df.pivot_table(
            index="user_id", columns="object_id", values="z_score", aggfunc="mean"
        ).fillna(0)  # Zスコアは欠損を0とみなす

        # user_id / object_id を文字列に統一
        user_object_matrix_local.index = user_object_matrix_local.index.astype(str)
        user_object_matrix_local.columns = user_object_matrix_local.columns.astype(str)
        user_zscore_matrix_local.index = user_zscore_matrix_local.index.astype(str)
        user_zscore_matrix_local.columns = user_zscore_matrix_local.columns.astype(str)

        # グローバル変数に代入（元コード維持）
        user_object_matrix = user_object_matrix_local
        user_zscore_matrix = user_zscore_matrix_local

        # --- (4) コサイン類似度を計算 ---
        similarity_array = cosine_similarity(user_zscore_matrix)
        new_user_similarity = pd.DataFrame(
            similarity_array,
            index=user_zscore_matrix.index,
            columns=user_zscore_matrix.index
        )
        user_similarity = new_user_similarity

        print("📊 user_zscore_matrix（標準化評価行列）")
        print(user_zscore_matrix.tail())  # 下数行をデバッグ表示

    else:
        # df が空の場合は空DataFrameで初期化
        user_object_matrix = pd.DataFrame()
        user_zscore_matrix = pd.DataFrame()
        user_similarity = pd.DataFrame()

    # 最後に簡単なデバッグ表示（ユーザーインデックスを確認）
    print("📊 user_object_matrix のインデックス:", user_object_matrix.index)
    print("📊 user_similarity のインデックス:", user_similarity.index)




def recommend_for_all_users(threshold=0.3):
    global user_object_matrix, user_zscore_matrix, user_similarity

    update_user_similarity_from_csv()

    # 万が一、まだ行列が用意されていない or 空の場合は、ここで更新しておく手もある
    if user_zscore_matrix.empty or user_similarity.empty:
        print("⚠ user_zscore_matrix / user_similarity が空です。再計算します。")
        update_user_similarity_from_csv()

    # user_object_matrix が空の場合も、計算が不可能なので空のDataFrameを返す
    if user_object_matrix.empty:
        print("⚠ user_object_matrix が空です。推薦スコアを計算できません。")
        return pd.DataFrame(columns=["user_id", "object_id", "recommendation_score"]), user_similarity

    print("📊 user_similarity（ユーザー類似度行列）")
    print(user_similarity.index)
    print("📊 user_object_matrix のインデックス（ユーザーID）")
    print(user_object_matrix.index)

    recommendations = []
    
    # 各ユーザーについて推薦スコアを計算
    for target_user in user_object_matrix.index:
        user_data = user_object_matrix.loc[target_user]
        # 未評価(= NaN)のオブジェクト一覧
        unrated_objects = user_data[user_data.isna()].index
        
        print(f"🔍 ユーザー {target_user} の未評価オブジェクト数: {len(unrated_objects)}")

        # 閾値以上のユーザーを取得（自己は除く）
        similar_users = user_similarity.loc[target_user].drop(target_user, errors="ignore")
        similar_users = similar_users[similar_users >= threshold]

        print(f"🔍 ユーザー {target_user} の類似ユーザー数: {len(similar_users)}")
        
        for object_id in unrated_objects:
            scores = []
            weights = []
            for sim_user, sim_score in similar_users.items():
                # sim_userがobject_idを評価しているか(= user_object_matrixでNaNでない)チェック
                if (object_id in user_object_matrix.columns
                    and not pd.isna(user_object_matrix.loc[sim_user, object_id])):
                    # zスコアを類似度で加重平均
                    standardized_rating = user_zscore_matrix.loc[sim_user, object_id]
                    scores.append(standardized_rating * sim_score)
                    weights.append(sim_score)

            if weights:
                rec_score = np.sum(scores) / np.sum(weights)
            else:
                rec_score = 0  # 類似ユーザーがいない場合は0

            # デバッグ用
            print(f"ユーザー {target_user}, オブジェクト {object_id}")
            print(f"  scores: {scores}")
            print(f"  weights: {weights}")
            print(f"  計算結果: {rec_score}")

            recommendations.append((target_user, object_id, round(rec_score, 2)))

    # user_similarity が array の場合はDataFrameに戻す (念のため)
    if isinstance(user_similarity, np.ndarray):
        user_similarity = pd.DataFrame(
            user_similarity,
            index=user_zscore_matrix.index,
            columns=user_zscore_matrix.index
        )

    # 推薦結果をDataFrameにまとめ
    return pd.DataFrame(recommendations, columns=["user_id", "object_id", "recommendation_score"]), user_similarity


def categorize_recommendation(score):
    """スコアに基づいて星マークを付与"""
    if pd.isna(score) or score < 0.25:  # ⭐ 0.25未満を除外
        return None
    elif score >= 0.75:
        return "★★★"
    elif score >= 0.5:
        return "★★"
    elif score >= 0.25:
        return "★"

def get_username(user_id):
    """user_id から username を取得する"""
    return user_info.get(user_id, "Unknown")

user_df = pd.read_csv(USERS_FILE, encoding="utf-8-sig", dtype={"user_id": str})
user_info = dict(zip(user_df["user_id"], user_df["username"]))

# オブジェクト情報の読み込み
objects_df = pd.read_csv(OBJECTS_FILE, encoding="utf-8-sig", dtype={"object_id": str})
object_info = dict(zip(objects_df["object_id"], objects_df["object_name"]))

def get_recommendations_for_user(username: str):
    update_user_and_object_info()

    user_id = None
    for uid, uname in user_info.items():
        if uname == username:
            user_id = uid
            break
    if user_id is None:
        print(f"⚠ ユーザー名 {username} が user_info に見つかりません。")
        return {}

    # 推薦スコアを取得
    recommendations_df, _ = recommend_for_all_users()
    if recommendations_df.empty:
        print("⚠ recommendations_df が空です。")
        return {}
    
    user_recommendations = recommendations_df[recommendations_df["user_id"] == user_id].copy()
    user_recommendations.sort_values("recommendation_score", ascending=False, inplace=True)

    # オブジェクト名とスコアの辞書を作成（object_id を str に変換）
    recommendations = {
        object_info.get(str(row["object_id"]), "Unknown"):
            categorize_recommendation(row["recommendation_score"])
        for _, row in user_recommendations.iterrows()
        if row["recommendation_score"] >= 0.25  # ⭐ 0.25以上のスコアのみ表示
    }

    return recommendations
