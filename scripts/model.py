import pandas as pd
import numpy as np
import os
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error, root_mean_squared_error

class ModelEvaluator:
    def __init__(self, ticker="HPG", forecast_horizon=1):
        self.ticker = ticker
        self.forecast_horizon = forecast_horizon
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.train_path = os.path.join(self.base_dir, "data", "processed", f"{self.ticker}_train_processed.csv")
        
        # Target column
        self.target_col = 'Target_Return'
        
        # The exact feature columns that were scaled in Step 4
        self.features = ['rsi', 'macd_12_26_9', 'bbp_20_2.0', 'adx_14', 'vol_ratio', 'rs', 'alpha', 'inertia']
        
        # Define the models exactly as agreed in the plan
        self.models = {
            "Ridge Regression": Ridge(alpha=1.0),
            "SVR": SVR(C=1.0, epsilon=0.1),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        }

    def load_and_prepare_data(self):
        """Loads processed training data and creates the forecast target."""
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Training data not found at {self.train_path}. Please run preprocessing first.")
            
        df = pd.read_csv(self.train_path)
        
        # When read from CSV, column names might be lowercase
        df.columns = [col.lower() for col in df.columns]
        
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
        df = df.sort_index()
        
        # Create Target: shifted Pct_Return (predicting the future return)
        # Shift -X means row today gets the value from X days in the future
        df[self.target_col] = df['pct_return'].shift(-self.forecast_horizon)
        
        # Drop rows where target is NaN (the last X days of the dataset)
        df = df.dropna(subset=[self.target_col])
        return df

    def evaluate_models(self):
        df = self.load_and_prepare_data()
        
        # Ensure only existing features are used
        available_features = [col for col in self.features if col in df.columns]
        X = df[available_features]
        # Multiply target by 100 to calculate MAE/RMSE in percentage points (%)
        y = df[self.target_col] * 100 
        
        print(f"--- Đánh giá chéo (Cross-Validation) các Model cho {self.ticker} ---")
        print(f"Dữ liệu: {X.shape[0]} mẫu | Dự báo trước: {self.forecast_horizon} ngày")
        print(f"Features sử dụng: {available_features}\n")
        
        # TimeSeriesSplit is crucial for financial data to prevent look-ahead bias!
        tscv = TimeSeriesSplit(n_splits=5)
        
        scoring = {
            'MAE': make_scorer(mean_absolute_error),
            'RMSE': make_scorer(root_mean_squared_error)
        }
        
        results = []
        
        for name, model in self.models.items():
            print(f"  > Đang huấn luyện {name}...")
            cv_scores = cross_validate(model, X, y, cv=tscv, scoring=scoring, n_jobs=-1)
            
            avg_mae = np.mean(cv_scores['test_MAE'])
            avg_rmse = np.mean(cv_scores['test_RMSE'])
            
            results.append({
                'Model': name,
                'MAE (%)': avg_mae,
                'RMSE (%)': avg_rmse
            })
            
        # Sort by best MAE
        results_df = pd.DataFrame(results).sort_values(by='MAE (%)')
        print(f"\n=== KẾT QUẢ ĐÁNH GIÁ {self.ticker} (Sắp xếp theo MAE) ===")
        print(results_df.to_string(index=False))
        print("="*60 + "\n")
        
        return results_df

if __name__ == "__main__":
    # Test for both HPG and HSG with a 1-day forecast horizon
    for ticker in ["HPG", "HSG"]:
        try:
            evaluator = ModelEvaluator(ticker=ticker, forecast_horizon=1)
            evaluator.evaluate_models()
        except Exception as e:
            print(f"Lỗi khi đánh giá mô hình cho {ticker}: {e}")
