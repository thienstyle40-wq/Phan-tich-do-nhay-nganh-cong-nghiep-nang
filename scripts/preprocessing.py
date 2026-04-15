import pandas as pd
import numpy as np
import pandas_ta as ta
import os
from sklearn.preprocessing import StandardScaler

class FinancialDataPreprocessor:
    def __init__(self, ticker="HPG"):
        self.ticker = ticker
        self.scaler = StandardScaler()
        self.vnindex_df = None

    def load_data(self, file_path):
        """Loads OHLCV data from CSV."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        
        df = pd.read_csv(file_path)
        df.columns = [col.lower() for col in df.columns]
        return df

    def clean_data(self, df):
        """Cleans data: datetime conversion, indexing, and missing value handling."""
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
        df = df.set_index('time')
        df = df[~df.index.duplicated(keep='first')]
        
        if 'ticker' in df.columns:
            df = df.drop(columns=['ticker'])
            
        df = df.ffill().bfill()
        return df

    def load_vnindex(self, start_date, end_date):
        """Loads VN-Index data from raw CSV."""
        # Using correct case for path checking or just standard path 
        vni_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "VNINDEX_raw.csv")
        try:
            vni = pd.read_csv(vni_path)
            vni.columns = [col.lower() for col in vni.columns]
            vni['time'] = pd.to_datetime(vni['time'])
            vni = vni.set_index('time')
            vni = vni.sort_index()
            # Just keep close
            self.vnindex_df = vni[['close']].rename(columns={'close': 'vni_close'})
        except Exception as e:
            print(f"Warning: Could not load VN-Index data from {vni_path}: {e}")

    def add_technical_indicators(self, df):
        """Adds all requested technical indicators using pandas_ta."""
        # 1. Momentum: RSI
        df['RSI'] = ta.rsi(df['close'], length=14)
        
        # 2. Trend: MA & MACD
        df['SMA_20'] = ta.sma(df['close'], length=20)
        df['SMA_50'] = ta.sma(df['close'], length=50)
        df['SMA_200'] = ta.sma(df['close'], length=200)
        
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd], axis=1)
        
        # 3. Volatility: Bollinger Bands
        bbands = ta.bbands(df['close'], length=20, std=2)
        df = pd.concat([df, bbands], axis=1)
        
        # 4. Trend Strength: ADX
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        df = pd.concat([df, adx], axis=1)
        
        # ADXR
        df['ADXR_14'] = (df['ADX_14'] + df['ADX_14'].shift(14)) / 2
        
        # 5. Volume Ratio
        df['Vol_SMA_20'] = ta.sma(df['volume'], length=20)
        df['Vol_Ratio'] = df['volume'] / df['Vol_SMA_20']
        
        # 6. Inertia
        df['Inertia'] = (df['close'] - df['SMA_20']) / df['SMA_20'] * 100
        
        # 7. Returns
        df['Pct_Return'] = df['close'].pct_change()
        df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))
        
        return df

    def add_relative_performance(self, df):
        """Adds RS and Alpha relative to VN-Index."""
        if self.vnindex_df is None:
            self.load_vnindex(df.index.min(), df.index.max())
            
        if self.vnindex_df is not None:
            df = df.join(self.vnindex_df, how='left')
            df['vni_close'] = df['vni_close'].ffill()
            df['vni_pct_return'] = df['vni_close'].pct_change()
            df['RS'] = (df['close'] / df['vni_close']) * 100
            df['Alpha'] = df['Pct_Return'] - df['vni_pct_return']
            
        return df

    def scale_features(self, df, feature_cols, is_training=True):
        """Scales selected features using StandardScaler."""
        scaled_df = df.copy()
        
        # Filter to only the columns that exist in the dataframe to avoid errors
        actual_cols = [col for col in feature_cols if col in df.columns]
        
        try:
            if is_training:
                scaled_values = self.scaler.fit_transform(scaled_df[actual_cols])
            else:
                scaled_values = self.scaler.transform(scaled_df[actual_cols])
            scaled_df[actual_cols] = scaled_values
        except Exception as e:
            print(f"Error in scaling: {e}")
            raise
        return scaled_df

    def process_pipeline(self, file_path, start_date=None, end_date=None, is_training=True, feature_cols=None):
        """Runs the full preprocessing pipeline.
        
        If is_training is True, fits the scaler. 
        If is_training is False, uses the fitted scaler.
        """
        print(f"--- Processing {self.ticker} ({'Training' if is_training else 'Inference'}) ---")
        
        df = self.load_data(file_path)
        df = self.clean_data(df)
        
        # Calculate indicators on the FULL dataset to prevent NaN from window sizes 
        # when we slice into inference data
        df = self.add_technical_indicators(df)
        
        print("Calculating relative performance vs VN-Index...")
        df = self.add_relative_performance(df)
        
        # Now filter by dates if provided
        if start_date:
            df = df.loc[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df.loc[df.index <= pd.Timestamp(end_date)]
            
        # Drop rows with NaN 
        df = df.dropna()
        
        # Default features to scale
        if feature_cols is None:
            feature_cols = ['RSI', 'MACD_12_26_9', 'BBP_20_2.0', 'ADX_14', 'Vol_Ratio', 'RS', 'Alpha', 'Inertia']
            
        print(f"Scaling features: {feature_cols}")
        df = self.scale_features(df, feature_cols, is_training=is_training)
        
        return df

    def prepare_training_data(self, file_path, start_date='2020-01-01', end_date='2025-12-31', feature_cols=None):
        return self.process_pipeline(file_path, start_date=start_date, end_date=end_date, is_training=True, feature_cols=feature_cols)
        
    def prepare_inference_data(self, file_path, start_date='2026-01-01', end_date='2026-04-03', feature_cols=None):
        return self.process_pipeline(file_path, start_date=start_date, end_date=end_date, is_training=False, feature_cols=feature_cols)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    tickers = ["HPG", "HSG"]
    
    for ticker in tickers:
        print(f"\n{'='*40}")
        raw_data_path = os.path.join(base_dir, "data", "raw", f"{ticker}_raw.csv")
        preprocessor = FinancialDataPreprocessor(ticker=ticker)
        
        try:
            # 1. Prepare Training Data (2020-2025)
            train_df = preprocessor.prepare_training_data(raw_data_path)
            train_out = os.path.join(output_dir, f"{ticker}_train_processed.csv")
            train_df.to_csv(train_out)
            print(f"Success: Training data saved: {os.path.basename(train_out)} | Shape: {train_df.shape}")
            
            # 2. Prepare Inference Data (2026) using the SAME scaler
            inf_df = preprocessor.prepare_inference_data(raw_data_path)
            inf_out = os.path.join(output_dir, f"{ticker}_inference_processed.csv")
            inf_df.to_csv(inf_out)
            print(f"Success: Inference data saved: {os.path.basename(inf_out)} | Shape: {inf_df.shape}")
            
        except Exception as e:
            print(f"Error processing {ticker}: {str(e).encode('ascii', 'ignore').decode('ascii')}")
