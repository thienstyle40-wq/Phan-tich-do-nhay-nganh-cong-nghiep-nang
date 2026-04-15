import os
import pandas as pd
from vnstock import Vnstock

def get_and_save_data(ticker, start_date, end_date, output_dir, source="KBS"):
    """
    Fetch OHLCV data using vnstock logic.
    """
    print(f"Loading data for {ticker} from {start_date} to {end_date}...")
    
    # Sử dụng logic được cung cấp bởi user
    if ticker == "VNINDEX":
        # vnstock coi VNINDEX là index
        # Tùy thuộc vào version vnstock và source, đôi khi truyền thẳng VNINDEX vào stock() vẫn hoạt động với một số nguồn
        stock = Vnstock().stock(symbol=ticker, source=source)
    else:
        stock = Vnstock().stock(symbol=ticker, source=source)
        
    try:
        # Gọi API lịch sử
        df = stock.quote.history(
            start=start_date,
            end=end_date,
            interval="1D"
        )
        
        if df is not None and not df.empty:
            df['time'] = pd.to_datetime(df['time'])
            df = df[df['time'] >= start_date]
            df['Ticker'] = ticker  # Gán nhãn để phân biệt và quản lý
            
            # Đảm bảo columns viết thường để nhất quán
            df.columns = [col.lower() if col != 'Ticker' else col for col in df.columns]

            # Xuất trực tiếp ra thư mục raw
            file_name = f"{ticker}_raw.csv"
            save_path = os.path.join(output_dir, file_name)
            
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"[SUCCESS] Saved {ticker}")
            print(df.head(3))
            print("-" * 50)
            return True
        else:
            print(f"[FAILED] Cannot get data for {ticker}. Dataframe is None or empty.")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to load {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Define parameters
    tickers = ["HSG", "HPG", "VNINDEX"]
    # Theo dự án: Train/Val (2020-01-01 -> 2025-12-31) và Inference (2026-01-01 -> 2026-04-03)
    # Lấy gộp một lần để lưu thô
    start_date = '2020-01-01'
    end_date = '2026-04-03'
    
    # Path settings
    # Lấy thư mục gốc (Project root) dựa trên file hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_raw_dir = os.path.join(project_root, "data", "raw")
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(data_raw_dir, exist_ok=True)
    
    print("=" * 50)
    
    # Vòng lặp kéo dữ liệu
    for ticker in tickers:
        get_and_save_data(ticker, start_date, end_date, output_dir=data_raw_dir)

if __name__ == "__main__":
    main()
