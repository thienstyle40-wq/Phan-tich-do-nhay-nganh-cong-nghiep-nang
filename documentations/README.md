# Stock Beta Analysis & Sensitivity ML Pipeline
**Project: Phân tích độ nhạy (Beta) Cổ phiếu ngành Thép (HPG & HSG)**

Dự án này sử dụng mô hình Máy học (SVR - Support Vector Regression) để dự báo tỷ lệ phần trăm biến động từng ngày (`Pct_Return`) của các cổ phiếu biến động mạnh. Bằng cách chắt lọc các quy luật tuyến tính tương tự thuyết Beta sinh lời vượt trội (Alpha & RS) cùng nhiều chỉ báo kỹ thuật khác.

---

## 📂 Tổ Chức Thư Mục (MLOps Standard)

```text
Project root/
│
├── data/
│   ├── raw/                 # Dữ liệu dạng thô tải từ VNStock API (2020 - 2026)
│   └── processed/           # Dữ liệu Features/Target đã qua Scale và Cleaned
│
├── documentation/           # Từ điển định nghĩa dữ liệu (data_dictionary.md) và Hướng dẫn
│
├── logs/                    # Training Logs chứa thông tin quá trình Fine-tune siêu tham số
│
├── models/                  # Các Machine Learning model (Pickle/Joblib) đã học đủ trí khôn
│
├── notebooks/               
│   ├── exploratory/         # (EDA) Nhìn nhận bài toán, Heatmap tự do
│   └── inference/           # Validation. Đánh giá Mô hình trên tệp Dữ liệu Tương lai (Blind Test)
│
├── requirements.txt         # Các thư viện phụ thuộc (vnstock, pandas_ta, xgboost, sklearn, v.v...)
│
├── results/                 
│   ├── visualizations/      # Biểu đồ đánh giá Output dạng hình ảnh (.png)
│   └── reports/             # Báo cáo đánh giá KPI thực tế của mô hình
│
├── saved_objects/           # Lưu lại File Standard Scaler để Inference tương lai ko bị Leak
│
└── scipts/                  # Lõi hệ thống tự động: Collect -> Preprocess -> Model -> FineTune
```

## 🧠 Quy trình 8 Bước (Geron's Workflow)

1. **Look at Big Picture**: Chọn Regression (SVR + Ridge) đo lường biến động lợi nhuận. KPI: MAE < 1.5%.
2. **Get Data**: Kéo API VNStock (Dọc 2020-2025 làm Train, Quý 1-2026 làm Validation).
3. **Discover**: Visualize ma trận tương quan giữa VN-Index và Thép (Xem tại Notebook EDA).
4. **Prepare Data**: Sinh Feature (RSI, RS, ADX, MACD). Cách ly Scale giữa tập Train và Inference để vô hiệu hoá Data Leakage.
5. **Explore Models**: CV chéo giữa (Ridge, RF, XGB, SVR).
6. **Fine-Tune**: Cắm `GridSearchCV` để vắt kiệt tối đa sức mạnh mô hình, chốt KPI SVR là quán quân với MAE dưới 1.5%.
7. & 8. **Launch & Maintain**: Đóng hộp MLOps thành File `.joblib`. Và theo dõi Chart Validation qua `Inference_2026_Validation.ipynb`.

## 🚀 Cách Khởi Chạy
1. Chạy `scipts/data_collection.py` để làm mới Data.
2. Chạy `scipts/preprocessing.py` để đẩy Pipeline xử lý mới.
3. Mở Jupyter Notebook `notebooks/inference/Inference_2026_Validation.ipynb` và nhấn Run All để nhìn tận mắt Biểu đồ tích luỹ vốn thuật toán.
