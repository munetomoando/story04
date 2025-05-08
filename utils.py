import pandas as pd

def load_object_names(csv_path="ratings.csv", column_name="object_name"):
    """
    CSVファイルからレストラン名のユニークなリストを取得する。
    
    Parameters:
      csv_path (str): 評価データのCSVファイルのパス
      column_name (str): レストラン名が格納されているカラム名
     
    Returns:
      List[str]: ユニークなレストラン名のリスト
    """
    try:
        df = pd.read_csv(csv_path)
        object_names = df[column_name].dropna().unique().tolist()
        return object_names
    except Exception as e:
        # エラーハンドリング。必要に応じてログ出力なども検討する
        print(f"Error loading CSV: {e}")
        return []