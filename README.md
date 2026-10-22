# Gom cụm khách hàng Chinook theo sự quan tâm tới Track và Album

## Cách chạy (Python 3.10+)
```
pip install -r requirements.txt
python explore_sparsity.py     # (tuỳ chọn) chứng minh vì sao không gom trực tiếp theo TrackId/AlbumId
python training_model.py       # huấn luyện 2 mô hình, xuất output/ và trained/
python using_model.py 5        # dùng model: xếp khách số 5 (hoặc khách mới) vào cụm
python test/test_connector.py
python test/test_features.py
```
Chạy các lệnh từ thư mục gốc của project (chứa `training_model.py`).
Có thể ép số cụm: `python training_model.py --k-track 4 --k-album 4`.

## Cấu trúc 
| File | Vai trò |
|---|---|
| `connectors/connector.py` | Kết nối SQLite, `connect()` / `queryDataset()` như Connector ở bài mẫu |
| `dal/interest_dao.py` | Các câu SQL lấy lịch sử mua, khách hàng, thể loại chủ đạo của album |
| `features/interest_features.py` | Tạo ma trận khách x thể loại cho góc nhìn track và album |
| `clustering/model_selection.py` | Chọn k: elbow, silhouette (cosine), Davies-Bouldin, độ ổn định (ARI) |
| `clustering/profiling.py` | Mô tả từng cụm (thể loại nổi bật, nghệ sĩ, album, chi tiêu...) |
| `clustering/visualization.py` | Biểu đồ chọn k, PCA 2D, heatmap thể loại theo cụm |
| `training_model.py` | Huấn luyện + lưu model (`trained/*.pkl`) và kết quả (`output/`) |
| `using_model.py` | Nạp model, dự đoán cụm cho khách có sẵn hoặc khách mới |
| `explore_sparsity.py` | Khảo sát độ thưa của dữ liệu |
| `test/` | Kiểm thử connector và đặc trưng |

## Phương pháp
1. **Dữ liệu**: InvoiceLine -> Invoice (khách) -> Track -> Album/Artist/Genre.
2. **Biểu diễn sự quan tâm**: ma trận khách x TrackId có độ phủ 1.9% (mỗi track tối đa 2 khách mua), khách x AlbumId 7.3%, nên không dùng trực tiếp.
   - *Track*: số track khách mua theo từng thể loại.
   - *Album*: số album khác nhau khách mua theo thể loại chủ đạo của album (mỗi album tính một lần).
3. **Mô hình**: TF-IDF (giảm trọng số thể loại phổ biến như Rock) -> chuẩn hoá L2 (khoảng cách cosine) -> KMeans.
4. **Chọn k**: trong k = 2..7, lấy silhouette cao nhất trong các k có cụm nhỏ nhất >= 5 khách và độ ổn định ARI >= 0.6.

## Kết quả (xem output/)
- `customer_clusters.csv`: cụm track và cụm album của từng khách.
- `cluster_profile_track.csv`, `cluster_profile_album.csv`: mô tả cụm.
- `*_k_selection.png`, `*_pca.png`, `*_genre_heatmap.png`: biểu đồ.
- `training_log.txt`: log đầy đủ khi chạy.

## Hạn chế cần nêu khi báo cáo
- Chỉ 59 khách, mỗi khách ~38 track; silhouette chỉ ~0.25 nên cấu trúc cụm **yếu**, cụm mang tính mô tả hơn là phân tách rõ.
- Nhãn cụm (0, 1, 2) chỉ là số thứ tự, không so sánh trực tiếp cụm track với cụm album (ARI giữa hai cách gom ≈ 0.27).
