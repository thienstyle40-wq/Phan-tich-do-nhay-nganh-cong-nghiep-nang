# Báo Cáo Tổng Kết Dự Án: Hệ Thống Dự Báo Biến Động Giá Cổ Phiếu Ngành Thép (HPG, HSG)

## 1. Giới Thiệu Chung
Dự án được thực hiện nhằm xây dựng một hệ thống Machine Learning (học máy) để phân tích, đo lường độ nhạy (Beta) và dự báo biến động tỷ suất sinh lời ngắn hạn (1 ngày) của hai cổ phiếu đầu ngành thép trên thị trường chứng khoán Việt Nam: **Tập đoàn Hòa Phát (HPG)** và **Tập đoàn Hoa Sen (HSG)**.

Hệ thống được thiết kế theo chuẩn MLOps thông qua một **Pipeline tự động hóa** từ khâu thu thập dữ liệu, tiền xử lý, huấn luyện cho đến đánh giá và xuất bản mô hình để đẩy lên môi trường sản xuất (Production).

---

## 2. Quy Trình Kỹ Thuật (Pipeline)

### Bước 1: Thu thập dữ liệu (Data Collection)
*   **Mã nguồn:** `scipts/data_collection.py`
*   **Chi tiết:** Các chỉ số tài chính thô (OHLCV) của `HPG`, `HSG` và `VNINDEX` được lấy tự động thông qua thư viện `vnstock` (Nguồn dữ liệu KBSV).
*   **Giai đoạn dữ liệu:**
    *   Tập huấn luyện (Training): `2020-01-01` đến `2025-12-31`.
    *   Tập kiểm định/Dự báo (Inference): Quý 1 năm `2026` (Đến tháng 4/2026).

### Bước 2: Tiền xử lý & Kỹ thuật đặc trưng (Preprocessing & Feature Engineering)
*   **Mã nguồn:** `scipts/preprocessing.py`
*   **Chi tiết:** 
    *   Làm sạch dữ liệu, xử lý giá trị khuyết thiếu.
    *   Sử dụng thư viện `pandas_ta` để tạo ra các đặc trưng kỹ thuật quan trọng như: Động lượng (RSI), Xướng hướng (MACD, ADX), Độ biến động (Bollinger Bands), Dòng tiền (Volume Ratio) và Quán tính giá (Inertia).
    *   So sánh tương quan để tạo các chỉ số sức mạnh giá tương đối so với VN-Index (Cột `RS` và `Alpha`).
    *   Thực hiện chuẩn hóa dữ liệu bằng `StandardScaler` để đảm bảo các yếu tố không bị chênh lệch quy mô khi đi vào các mô hình học máy.

### Bước 3: Đánh giá mô hình chuẩn (Model Evaluation)
*   **Mã nguồn:** `scipts/model.py`
*   **Phương pháp:** Do đặc thù chuỗi thời gian phân tích tài chính, dự án áp dụng kĩ thuật `TimeSeriesSplit` (với 5 fold) để kiểm định chéo (Cross-Validation), tránh hiện tượng "Look-ahead bias" (Nhìn trộm tương lai).
*   **Mô hình thử nghiệm:** 
    1.  `Ridge Regression`
    2.  `Support Vector Regression (SVR)`
    3.  `Random Forest`
    4.  `XGBoost`
*   **Chỉ số đánh giá:** Mean Absolute Error (MAE) và Root Mean Squared Error (RMSE) tính theo điểm $\%$.

### Bước 4: Tối ưu hóa siêu tham số (Fine-Tuning)
*   **Mã nguồn:** `scipts/fine_tune.py`
*   **Chi tiết:** Sử dụng cơ chế tìm kiếm `GridSearchCV` quy mô lớn quét qua nhiều không gian tham số của Ridge, SVR và XGBoost theo chuỗi thời gian. 
*   **Kết quả:** Chọn mô hình xuất sắc nhất theo từng cổ phiếu vào tiến trình lưu nháp (file `.pkl`).

### Bước 5: Đóng gói và MLOps (MLOps Packaging)
*   **Mã nguồn:** `scipts/mlops_packaging.py`
*   **Chi tiết:** 
    *   Ghi nhớ bộ chuẩn hóa không gian đặc trưng (`scaler.joblib`) nhằm phục vụ việc Scaling nhất quán cho dữ liệu Inference sau này.
    *   Lấy mô hình xuất sắc trực tiếp đưa vào kiểm thử dữ liệu năm 2026. Mô hình chính thức được xuất dưới định dạng nén tối ưu bộ nhớ `.joblib`.
    *   Tự động xuất đồ thị trực quan so sánh Thực Tế và Dự Báo (thư mục `results/visualizations/`) và tự động gen báo cáo chi tiết thành file markdown (thư mục `results/reports/`).
    *   Tất cả tham số và độ lệch Validation được tự động tracking lại vào thư mục `logs`.

---

## 3. Kiến Trúc Thư Mục Hệ Thống
*   `data/`: Chứa dữ liệu (`raw/` cho dữ liệu lấy từ web API, `processed/` cho dữ liệu đã chạy qua pipeline tiền xử lý).
*   `documentation/`: Chứa các báo cáo văn bản, nhật ký lưu trữ.
*   `logs/`: Ghi nhận dữ liệu truy vết về log training và biến động chỉ số.
*   `models/`: Lưu trữ các phiên bản học máy (`.pkl` dùng tam thời và `.joblib` chính thức để triển khai).
*   `results/`: Lưu các báo cáo (`reports/`) và hình ảnh kết quả dự đoán (`visualizations/`).
*   `saved_objects/`: Chứa các object như Pipeline hay Scaler.
*   `scipts/`: Toàn bộ các mô đun hệ thống (`data_collection`, `preprocessing`, `model`, `fine_tune`, `mlops_packaging`).

## 4. Tổng Kết
Hệ thống dự báo Beta cổ phiếu thép đã được module hóa chuẩn chỉnh giúp chuẩn hóa quy trình phân tích tự động. Ưu điểm nổi bật của cấu trúc này nằm ở khả năng có thể dễ dàng chạy lại nguyên vẹn luồng trích xuất, scale, train cho một cổ phiếu bất kỳ với ngày cập nhật mới nhất, đảm bảo tính tái sử dụng và tách bạch không gian kiểm thử. Việc phân tích này hứa hẹn xây dựng được các chỉ báo kỹ thuật vượt trội cho việc lướt sóng T+ trong năm 2026 đối với HPG và HSG.
