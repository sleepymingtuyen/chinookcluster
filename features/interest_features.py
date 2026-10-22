"""Xây ma trận 'sự quan tâm' (khách hàng x thể loại) cho hai góc nhìn: track và album.

Vì sao không dùng thẳng ma trận khách x TrackId / khách x AlbumId?
  - Có 59 khách, 1984 track khác nhau được mua, mỗi track nhiều nhất chỉ 2 khách mua
    -> ma trận khách x track có độ phủ ~1.9%, hầu như không có 2 khách nào chung track.
  - Khách x album: độ phủ ~7%, mỗi album tối đa 15 khách mua.
  Với dữ liệu thưa như vậy, khoảng cách giữa các khách gần như bằng nhau và cụm không có ý nghĩa
  (xem explore_sparsity.py). Do đó 'sự quan tâm' được biểu diễn qua thể loại của track/album.
"""


def _align(matrix, columns):
    """Cố định danh sách cột (để dự đoán khách mới dùng đúng không gian đặc trưng đã huấn luyện)."""
    if columns is None:
        return matrix.sort_index(axis=1)
    return matrix.reindex(columns=columns, fill_value=0)


def build_track_interest(purchases, columns=None):
    """Quan tâm tới TRACK: số track (có tính Quantity) khách mua theo từng thể loại."""
    d = purchases.dropna(subset=["GenreName"])
    m = d.pivot_table(index="CustomerId", columns="GenreName",
                      values="Quantity", aggfunc="sum", fill_value=0)
    return _align(m, columns)


def build_album_interest(purchases, album_genre, columns=None):
    """Quan tâm tới ALBUM: số album KHÁC NHAU khách đã mua (>=1 track) theo thể loại chủ đạo của album.
    Mỗi album chỉ tính một lần nên khách mua nhiều track của cùng 1 album không bị 'thổi phồng'."""
    d = purchases.dropna(subset=["AlbumId"]).copy()
    d["AlbumId"] = d["AlbumId"].astype(int)
    d = d.drop_duplicates(["CustomerId", "AlbumId"])
    d["AlbumGenre"] = d["AlbumId"].map(album_genre)
    d = d.dropna(subset=["AlbumGenre"])
    m = d.pivot_table(index="CustomerId", columns="AlbumGenre",
                      values="AlbumId", aggfunc="count", fill_value=0)
    return _align(m, columns)
