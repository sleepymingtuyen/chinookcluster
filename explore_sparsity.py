"""Khảo sát: vì sao KHÔNG gom cụm trực tiếp trên ma trận khách x TrackId / khách x AlbumId.
Chạy:  python explore_sparsity.py
"""
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from connectors.connector import Connector
from dal.interest_dao import InterestDAO

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

conn = Connector()
conn.connect()
p = InterestDAO.get_purchases(conn)
conn.disconnect()

print("Số lần mỗi track được mua (số dòng hoá đơn theo track):")
print(p.groupby("TrackId").size().value_counts().sort_index().to_string())

for col in ["TrackId", "AlbumId"]:
    m = p.pivot_table(index="CustomerId", columns=col, values="Quantity", aggfunc="sum", fill_value=0)
    density = (m.values > 0).mean()
    X = TfidfTransformer().fit_transform(m.values).toarray()
    X = normalize(TruncatedSVD(10, random_state=42).fit_transform(X))
    print(f"\nMa trận khách x {col}: {m.shape}, độ phủ {density:.1%}")
    for k in range(2, 9):
        lab = KMeans(k, n_init=20, random_state=42).fit_predict(X)
        print(f"  k={k}  silhouette={silhouette_score(X, lab, metric='cosine'):.3f}  kích thước cụm={sorted(np.bincount(lab).tolist())}")
print("\nNhận xét: silhouette tăng đều theo k và các cụm vụn dần -> chỉ là hệ quả của độ thưa,"
      " không phải cấu trúc sở thích thật. Vì vậy chuyển sang biểu diễn theo thể loại.")
