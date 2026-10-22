import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score


def stability_ari(X, k, n_rep=10, frac=0.8, n_init=10, random_state=42):
    """Độ ổn định của cụm: lấy mẫu con 80% khách, gom lại, so với cụm gốc bằng Adjusted Rand Index.
    ARI gần 1 = cụm lặp lại được; gần 0 = cụm chỉ là ngẫu nhiên."""
    rng = np.random.default_rng(random_state)
    base = KMeans(n_clusters=k, n_init=30, random_state=random_state).fit_predict(X)
    scores = []
    for r in range(n_rep):
        idx = rng.choice(len(X), size=int(len(X) * frac), replace=False)
        lab = KMeans(n_clusters=k, n_init=n_init, random_state=r + 1).fit_predict(X[idx])
        scores.append(adjusted_rand_score(base[idx], lab))
    return float(np.mean(scores))


def evaluate_k(X, k_values=range(2, 8), random_state=42):
    """Bảng chỉ số cho từng k: inertia (elbow), silhouette (cosine), Davies-Bouldin, độ ổn định, cụm nhỏ nhất."""
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=30, random_state=random_state).fit(X)
        labels = km.labels_
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, labels, metric="cosine"),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "stability_ari": stability_ari(X, k, random_state=random_state),
            "min_cluster_size": int(np.bincount(labels).min()),
        })
    return pd.DataFrame(rows)


def choose_k(scores, min_cluster_size=5, min_stability=0.6):
    """Chọn k có silhouette cao nhất trong các k thoả: không có cụm quá nhỏ và đủ ổn định.
    Nếu không k nào thoả thì chọn k có silhouette cao nhất."""
    ok = scores[(scores["min_cluster_size"] >= min_cluster_size) &
                (scores["stability_ari"] >= min_stability)]
    if ok.empty:
        ok = scores
    return int(ok.sort_values("silhouette", ascending=False).iloc[0]["k"])
