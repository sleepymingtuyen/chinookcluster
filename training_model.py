"""Gom cụm khách hàng Chinook theo (1) sự quan tâm tới TRACK và (2) sự quan tâm tới ALBUM.

Chạy:  python training_model.py            (tự chọn k)
       python training_model.py --k-track 4 --k-album 4   (ép k)
Kết quả: trained/*.pkl, output/*.csv, output/*.png
"""
import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import adjusted_rand_score

from clustering.model_selection import choose_k, evaluate_k
from clustering.profiling import profile_clusters
from clustering.visualization import plot_genre_heatmap, plot_k_selection, plot_pca
from connectors.connector import Connector
from dal.interest_dao import InterestDAO
from features.interest_features import build_album_interest, build_track_interest

try:  # in tiếng Việt đúng trên console Windows
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
TRAINED = ROOT / "trained"
RANDOM_STATE = 42


def run_view(view, title, matrix, purchases, k_forced=None):
    """Huấn luyện một góc nhìn: TF-IDF (giảm trọng số thể loại ai cũng nghe, vd Rock)
    -> chuẩn hoá L2 (khoảng cách cosine, như test_cosine.py ở bài mẫu) -> KMeans."""
    tfidf = TfidfTransformer()                       # norm='l2' mặc định
    X = tfidf.fit_transform(matrix.values).toarray()

    scores = evaluate_k(X, random_state=RANDOM_STATE)
    scores.to_csv(OUT / f"k_selection_{view}.csv", index=False)
    print(f"\n=== {title}: ma trận {matrix.shape[0]} khách x {matrix.shape[1]} thể loại")
    print(scores.round(3).to_string(index=False))

    k = k_forced or choose_k(scores)
    print(f"-> k = {k}" + (" (ép tay)" if k_forced else " (tự chọn)"))
    km = KMeans(n_clusters=k, n_init=30, random_state=RANDOM_STATE).fit(X)
    labels = pd.Series(km.labels_, index=matrix.index, name=f"{view}_cluster")

    profile, share = profile_clusters(matrix, labels, purchases)
    profile.to_csv(OUT / f"cluster_profile_{view}.csv", index=False, encoding="utf-8-sig")
    print(profile.to_string(index=False))

    plot_k_selection(scores, k, OUT / f"{view}_k_selection.png", title)
    plot_pca(X, labels, OUT / f"{view}_pca.png", title)
    plot_genre_heatmap(share, OUT / f"{view}_genre_heatmap.png", f"{title}: tỉ trọng thể loại theo cụm")

    model = {"view": view, "columns": list(matrix.columns), "tfidf": tfidf,
             "kmeans": km, "k": k, "scores": scores}
    with open(TRAINED / f"{view}_cluster_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-track", type=int, default=None)
    ap.add_argument("--k-album", type=int, default=None)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    TRAINED.mkdir(exist_ok=True)

    conn = Connector()
    conn.connect()
    purchases = InterestDAO.get_purchases(conn)
    customers = InterestDAO.get_customers(conn)
    album_genre = InterestDAO.get_album_genre(conn)
    conn.disconnect()
    print("Số dòng hoá đơn:", len(purchases), "| số khách:", purchases["CustomerId"].nunique())

    track_matrix = build_track_interest(purchases)
    album_matrix = build_album_interest(purchases, album_genre)

    track_labels = run_view("track", "Quan tâm tới TRACK", track_matrix, purchases, args.k_track)
    album_labels = run_view("album", "Quan tâm tới ALBUM", album_matrix, purchases, args.k_album)

    result = customers.set_index("CustomerId").join([track_labels, album_labels]).reset_index()
    result.to_csv(OUT / "customer_clusters.csv", index=False, encoding="utf-8-sig")

    print("\n=== Đối chiếu hai cách gom cụm")
    ari = adjusted_rand_score(result["track_cluster"], result["album_cluster"])
    print(f"Adjusted Rand Index giữa cụm track và cụm album = {ari:.2f} (1 = trùng hẳn, 0 = độc lập)")
    cross = pd.crosstab(result["track_cluster"], result["album_cluster"],
                        rownames=["track_cluster"], colnames=["album_cluster"])
    print(cross)
    cross.to_csv(OUT / "crosstab_track_vs_album.csv")
    print("\nĐã lưu kết quả vào", OUT)


if __name__ == "__main__":
    main()
