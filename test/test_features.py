import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from connectors.connector import Connector
from dal.interest_dao import InterestDAO
from features.interest_features import build_album_interest, build_track_interest

conn = Connector()
conn.connect()
purchases = InterestDAO.get_purchases(conn)
album_genre = InterestDAO.get_album_genre(conn)
conn.disconnect()

tm = build_track_interest(purchases)
am = build_album_interest(purchases, album_genre)
print("Ma trận track:", tm.shape, "| ma trận album:", am.shape)
assert tm.shape[0] == am.shape[0] == 59
assert tm.values.sum() == purchases["Quantity"].sum()          # không mất dòng mua nào
assert am.values.sum() == purchases.drop_duplicates(["CustomerId", "AlbumId"]).shape[0]
# cố định cột cho khách mới
one = build_track_interest(purchases[purchases["CustomerId"] == 1], columns=list(tm.columns))
assert list(one.columns) == list(tm.columns)
print("test_features OK")
