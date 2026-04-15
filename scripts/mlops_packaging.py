import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Tạo các thư mục vật lý thiết yếu
    dirs_to_make = [
        "saved_objects",
        "models",
        "logs",
        os.path.join("results", "visualizations"),
        os.path.join("results", "reports"),
        "documentation"
    ]
    for d in dirs_to_make:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
        
    print("✅ Đã tạo/kiểm tra sự tồn tại của các thư mục vật lý.")
    
    # Chúng ta đã lưu model ở file fine_tune.py
    # Load model và scaler từ base_dir
    # Note: process_pipeline fitted a scaler, but we didn't save it initially.
    # To save the exact scaler, we have to refit it on the final training columns, 
    # or rely on the actual code. Since the features were already scaled in `train_processed.csv`,
    # let's write a simplified logic to refit the scaler just to dump it.
    
    from sklearn.preprocessing import StandardScaler
    tickers = ["HPG", "HSG"]
    
    log_file_path = os.path.join(base_dir, "logs", "training_log.txt")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(f"--- MLOps Training Log: Thập niên 2026 ---\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
    for ticker in tickers:
        print(f"--- Đóng gói Artifacts cho {ticker} ---")
        
        # A. Saved Object (Scaler)
        raw_train_path = os.path.join(base_dir, "data", "raw", f"{ticker}_raw.csv") # We might not need raw if we re-read processed. 
        # Actually in preprocessing, we scaled features. We can just fit a dummy scaler on the raw features that we need to scale later.
        # However, to be strictly correct, we'll instantiate our Preprocessor if we can import it.
        try:
            import sys
            sys.path.append(os.path.join(base_dir, "scipts")) # scipts instead of scripts
            from preprocessing import FinancialDataPreprocessor
            
            preprocessor = FinancialDataPreprocessor(ticker=ticker)
            # Re-run prepare training to inherently fit the scaler internally
            _ = preprocessor.prepare_training_data(raw_train_path)
            
            scaler_out = os.path.join(base_dir, "saved_objects", f"{ticker}_scaler.joblib")
            joblib.dump(preprocessor.scaler, scaler_out)
            print(f"✅ Đã lưu Scaler tại: {scaler_out}")
            
        except Exception as e:
            print(f"Lỗi lưu Scaler: {e}")
            
        # B. Models & Logs & Visualizations
        old_model_path = os.path.join(base_dir, "models", f"{ticker}_best_model.pkl")
        inf_data_path = os.path.join(base_dir, "data", "processed", f"{ticker}_inference_processed.csv")
        
        if os.path.exists(old_model_path) and os.path.exists(inf_data_path):
            # Load SVR model
            model = joblib.load(old_model_path)
            
            # Save formally as described
            new_model_path = os.path.join(base_dir, "models", f"{ticker}_beta_svr_model.joblib")
            joblib.dump(model, new_model_path)
            print(f"✅ Đã lưu Model tại: {new_model_path}")
            
            # Inference data prep
            df = pd.read_csv(inf_data_path)
            df.columns = [col.lower() for col in df.columns]
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
            df['target'] = df['pct_return'].shift(-1) * 100
            df = df.dropna(subset=['target'])
            
            features = ['rsi', 'macd_12_26_9', 'adx_14', 'vol_ratio', 'rs', 'alpha', 'inertia']
            available_features = [col for col in features if col in df.columns]
            X = df[available_features]
            y_true = df['target']
            
            y_pred = model.predict(X)
            
            mae = mean_absolute_error(y_true, y_pred)
            
            # Log results
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(f"TICKER: {ticker}\n")
                f.write(f"Model Type: SVR (Support Vector Regression)\n")
                f.write(f"Best Hyperparameters: {model.get_params()}\n")
                f.write(f"MAE trên tập Validation (Quý 1/2026): {mae:.3f}%\n")
                f.write(f"----------------------------------------\n")
                
            print(f"✅ Đã log kết quả vào: {log_file_path}")
            
            # Visualizations
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, y_true, label='Thực tế (%)', marker='o', alpha=0.6)
            plt.plot(df.index, y_pred, label='Dự báo (%)', linestyle='dashed', color='red', alpha=0.8)
            plt.axhline(0, color='black', alpha=0.5)
            plt.title(f'Actual vs Predicted - {ticker} (2026)')
            plt.ylabel('% Biến động')
            plt.legend()
            
            plot_path = os.path.join(base_dir, "results", "visualizations", f"{ticker}_actual_vs_predicted.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Đã lưu Biểu đồ tại: {plot_path}")
            
            # Reports
            report_path = os.path.join(base_dir, "results", "reports", f"{ticker}_final_report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"# Final Report Dữ liệu nhạy Beta: {ticker}\n\n")
                f.write(f"- **Mô hình**: Support Vector Regression (SVR)\n")
                f.write(f"- **Tham số tối ưu**: `C={model.C}`, `epsilon={model.epsilon}`\n")
                f.write(f"- **Độ chính xác MAE**: Tín hiệu bắt lệch {mae:.3f}%\n")
                if mae < 1.5:
                    f.write(f"- **Nhận xét KPI**: Đạt yêu cầu (< 1.5%). Sẵn sàng đưa vào giao dịch phái sinh/phòng vệ.\n")
                else:
                    f.write(f"- **Nhận xét KPI**: MAE = {mae:.3f}%. Cao hơn 1.5%. Phù hợp swing trade nhưng rủi ro ngày cao hơn kỳ vọng.\n")
                    
if __name__ == "__main__":
    main()
