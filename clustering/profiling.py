import pandas as pd


def _top(series, n=3):
    return " | ".join(f"{name}" for name in series.head(n).index)


def profile_clusters(matrix, labels, purchases, top_n=3):
    """Mô tả từng cụm.
    Trả về (profile_df, genre_share) trong đó genre_share = tỉ trọng thể loại của mỗi cụm."""
    grp = matrix.groupby(labels).sum()
    share = grp.div(grp.sum(axis=1), axis=0)                     # tỉ trọng thể loại trong cụm
    overall = matrix.sum() / matrix.values.sum()                 # tỉ trọng thể loại toàn bộ khách
    lift = share.div(overall, axis=1)                            # >1: cụm quan tâm hơn mức chung

    p = purchases.merge(labels.rename("Cluster"), left_on="CustomerId", right_index=True)
    per_cust = p.groupby("CustomerId").agg(spend=("LineTotal", "sum"),
                                           n_tracks=("TrackId", "nunique"),
                                           n_albums=("AlbumId", "nunique"))
    per_cust["tracks_per_album"] = per_cust["n_tracks"] / per_cust["n_albums"]
    per_cust["Cluster"] = labels

    rows = []
    for c in sorted(share.index):
        s = share.loc[c].sort_values(ascending=False)
        # thể loại 'đặc trưng': lift cao nhất trong các thể loại chiếm >= 3% sở thích của cụm
        distinct = lift.loc[c][share.loc[c] >= 0.03].sort_values(ascending=False)
        pc = p[p["Cluster"] == c]
        top_artists = pc.groupby("ArtistName")["CustomerId"].nunique().sort_values(ascending=False)
        top_albums = pc.groupby("AlbumTitle")["CustomerId"].nunique().sort_values(ascending=False)
        pcust = per_cust[per_cust["Cluster"] == c]
        rows.append({
            "Cluster": c,
            "n_customers": int((labels == c).sum()),
            "top_genres": ", ".join(f"{g} ({v:.0%})" for g, v in s.head(top_n).items()),
            "distinctive_genres": ", ".join(f"{g} (x{distinct[g]:.1f})" for g in distinct.head(top_n).index),
            "avg_distinct_albums": round(pcust["n_albums"].mean(), 1),
            "avg_tracks_per_album": round(pcust["tracks_per_album"].mean(), 2),
            "avg_spend": round(pcust["spend"].mean(), 2),
            "top_artists": _top(top_artists, top_n),
            "top_albums": _top(top_albums, top_n),
        })
    return pd.DataFrame(rows), share
