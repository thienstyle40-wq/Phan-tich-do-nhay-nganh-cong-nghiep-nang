# Data Dictionary: Stock Beta Analysis (HPG & HSG)

Từ điển này giải thích các biến số (features) và mục tiêu dự báo (target) được sử dụng trong tập dữ liệu sau khi chạy qua module tiền xử lý (`preprocessing.py`).

## 1. Dữ Liệu Gốc (Raw Data)
*(Thu thập từ VNStock API)*
- `time`: Ngày giao dịch.
- `open`, `high`, `low`, `close`: Giá mở cửa, cao nhất, thấp nhất, đóng cửa.
- `volume`: Khối lượng giao dịch trong ngày.

## 2. Mục Tiêu Dự Báo (Target Model)
- `Target_Return`: Biến động % của giá đóng cửa vào NGÀY HÔM SAU (`pct_return.shift(-1)`). Được sử dụng làm nhãn dự báo thực tế đo lường lợi nhuận (Return).

## 3. Các Biến Giải Thích Kỹ Thuật (Technical Features)
Các chỉ báo này được tính toán để đóng vai trò làm Input "dạy" mô hình nhận diện hành vi thị trường:

| Tên Cột | Ý nghĩa / Cách tính (Feynman Style) | Phân nhóm |
| :--- | :--- | :--- |
| `RSI` | **Relative Strength Index (14 ngày)**: Cảm xúc thị trường. >70 là hưng phấn (quá mua), <30 là hoảng loạn (quá bán). | Động lượng (Momentum) |
| `MACD_12_26_9` | **Moving Average Convergence Divergence**: Độ lệch giữa đường trung bình nhanh và chậm. Chỉ ra khi nào xu hướng bị gãy hoặc được củng cố. | Xu hướng (Trend) |
| `ADX_14` | **Average Directional Index**: Sức mạnh của xu hướng. Bất kể giá lên hay xuống, nếu ADX cao tức là xu hướng đó đang rất "hung hãn". | Sức mạnh (Strength) |
| `BBP_20_2.0` | **Bollinger Bands %P**: Vị trí dải băng. Đo lường giá hiện tại đang ở sát trần hay sát sàn của dao động chuẩn. | Biến động (Volatility) |

## 4. Các Biến Quan Hệ Tương Đối & Thể Tích (Relative & Volume Features)
Đây là các biến mang linh hồn của **Beta Analysis** (Độ nhạy):

| Tên Cột | Ý nghĩa |
| :--- | :--- |
| `Vol_Ratio` | Khối lượng khớp lệnh chia cho Trung bình Khối lượng 20 ngày. Nếu cột này = 2.0, nghĩa là tiền đang vào gấp đôi bình thường (Đột biến). |
| `RS` | **Relative Strength (Cổ phiếu vs VN-Index)**: Lấy giá cổ phiếu chia giá VN-Index. Nếu biểu đồ RS vểnh lên, cổ phiếu đang mạnh hơn thị trường chung. |
| `Alpha` | Mức sinh lời Vượt trội = % Lợi nhuận cổ phiếu - % Lợi nhuận VN-Index. Nếu Alpha dương, bạn đang đánh bại thị trường. |
| `Inertia` | Quán tính = khoảng cách từ giá hiện tại đến đường Trung bình 20 ngày. Nếu quá xa, giá sẽ có xu hướng từ tính (Mean Reversion) bị hút ngược về đường SMA20. |
