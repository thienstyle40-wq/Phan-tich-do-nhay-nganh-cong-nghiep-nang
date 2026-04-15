import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer

class ModelTuner:
    def __init__(self, ticker="HPG", forecast_horizon=1):
        self.ticker = ticker
        self.forecast_horizon = forecast_horizon
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.train_path = os.path.join(self.base_dir, "data", "processed", f"{self.ticker}_train_processed.csv")
        self.models_dir = os.path.join(self.base_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.target_col = 'Target_Return'
        self.features = ['rsi', 'macd_12_26_9', 'adx_14', 'vol_ratio', 'rs', 'alpha', 'inertia']
        
        # We will tune these 3 models
        self.models_and_params = {
            "Ridge": {
                "model": Ridge(),
                "params": {
                    "alpha": [0.1, 1.0, 10.0, 50.0, 100.0, 500.0]
                }
            },
            "SVR": {
                "model": SVR(),
                "params": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "epsilon": [0.05, 0.1, 0.2, 0.5]
                }
            },
            "XGBoost": {
                "model": xgb.XGBRegressor(random_state=42),
                "params": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.05],
                    "max_depth": [2, 3], # Keep trees extremely shallow to avoid noise fitting
                    "reg_alpha": [1.0, 5.0, 10.0], # L1 regularization
                    "reg_lambda": [1.0, 5.0, 10.0] # L2 regularization
                }
            }
        }

    def load_and_prepare_data(self):
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Missing {self.train_path}")
            
        df = pd.read_csv(self.train_path)
        df.columns = [col.lower() for col in df.columns]
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
        df = df.sort_index()
        
        df[self.target_col] = df['pct_return'].shift(-self.forecast_horizon)
        df = df.dropna(subset=[self.target_col])
        return df

    def tune_models(self):
        df = self.load_and_prepare_data()
        available_features = [col for col in self.features if col in df.columns]
        X = df[available_features]
        y = df[self.target_col] * 100  # Scaling target to % for easier reading
        
        print(f"\n--- Đang tối ưu hóa (Fine-Tuning) cho {self.ticker} ---")
        
        tscv = TimeSeriesSplit(n_splits=5)
        # We want to minimize MAE, but grid_search uses scoring where 'greater is better'.
        # Scikit-learn has 'neg_mean_absolute_error' for this exact purpose.
        
        best_overall_model = None
        best_overall_score = float("inf")
        best_model_name = ""
        
        for name, info in self.models_and_params.items():
            print(f"  > Tìm kiếm tham số tốt nhất cho {name}...")
            grid = GridSearchCV(
                estimator=info["model"],
                param_grid=info["params"],
                cv=tscv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            grid.fit(X, y)
            
            best_mae = -grid.best_score_
            print(f"      + Tốt nhất của {name}: MAE = {best_mae:.3f} % | Params: {grid.best_params_}")
            
            if best_mae < best_overall_score:
                best_overall_score = best_mae
                best_overall_model = grid.best_estimator_
                best_model_name = name

        print(f"\n🏆 QUÁN QUÂN {self.ticker}: {best_model_name} với MAE = {best_overall_score:.3f} %")
        
        # Save the best model
        model_filename = os.path.join(self.models_dir, f"{self.ticker}_best_model.pkl")
        joblib.dump(best_overall_model, model_filename)
        print(f"Đã lưu mô hình quán quân tại: {model_filename}")

if __name__ == "__main__":
    for ticker in ["HPG", "HSG"]:
        tuner = ModelTuner(ticker=ticker, forecast_horizon=1)
        tuner.tune_models()
