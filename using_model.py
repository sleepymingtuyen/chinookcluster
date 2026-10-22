"""Dùng model đã huấn luyện (trained/*.pkl) để xếp khách hàng vào cụm.
Chạy:  python using_model.py            (demo khách số 5)
       python using_model.py 12         (khách số 12)
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

from connectors.connector import Connector
from dal.interest_dao import InterestDAO
from features.interest_features import build_album_interest, build_track_interest

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent


def load_model(view):
    with open(ROOT / "trained" / f"{view}_cluster_model.pkl", "rb") as f:
        return pickle.load(f)


def predict_cluster(model, matrix_row):
    """matrix_row: DataFrame 1 dòng, đã có đúng các cột thể loại của lúc huấn luyện."""
    X = model["tfidf"].transform(matrix_row.values).toarray()
    return int(model["kmeans"].predict(X)[0])


def cluster_of_customer(customer_id):
    conn = Connector()
    conn.connect()
    purchases = InterestDAO.get_purchases(conn)
    album_genre = InterestDAO.get_album_genre(conn)
    customers = InterestDAO.get_customers(conn)
    conn.disconnect()

    one = purchases[purchases["CustomerId"] == customer_id]
    if one.empty:
        return None
    name = customers.loc[customers["CustomerId"] == customer_id, "FullName"].iloc[0]

    track_model, album_model = load_model("track"), load_model("album")
    track_row = build_track_interest(one, columns=track_model["columns"])
    album_row = build_album_interest(one, album_genre, columns=album_model["columns"])
    return name, predict_cluster(track_model, track_row), predict_cluster(album_model, album_row)


def predict_from_genre_counts(view, genre_counts):
    """Khách MỚI (chưa có trong DB): chỉ cần số track/album theo thể loại,
    ví dụ {"Rock": 10, "Jazz": 4}."""
    model = load_model(view)
    row = pd.DataFrame([genre_counts]).reindex(columns=model["columns"], fill_value=0).fillna(0)
    return predict_cluster(model, row)


if __name__ == "__main__":
    cid = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    res = cluster_of_customer(cid)
    if res is None:
        print("Không tìm thấy khách", cid)
    else:
        name, tc, ac = res
        print(f"Khách {cid} - {name}: cụm track = {tc}, cụm album = {ac}")

    demo = {"Rock": 8, "Metal": 6, "Alternative & Punk": 4}
    print("Khách mới nghe", demo)
    print("  -> cụm track =", predict_from_genre_counts("track", demo))
    print("  -> cụm album =", predict_from_genre_counts("album", demo))
