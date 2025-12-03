"""
Data Preprocessing Module
Adds technical indicators, normalizes data, and creates sequences for LSTM
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ta
import config
import os


class DataPreprocessor:
    """Preprocess trading data for LSTM model"""
    
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = []
        
    def add_technical_indicators(self, df):
        """
        Add technical indicators to the dataset
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        
        print("Adding technical indicators...")
        
        # Simple Moving Averages
        for period in config.TECHNICAL_INDICATORS['SMA']:
            df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
        
        # Exponential Moving Averages
        for period in config.TECHNICAL_INDICATORS['EMA']:
            df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
        
        # Relative Strength Index
        for period in config.TECHNICAL_INDICATORS['RSI']:
            df[f'rsi_{period}'] = ta.momentum.rsi(df['close'], window=period)
        
        # MACD
        for params in config.TECHNICAL_INDICATORS['MACD']:
            fast, slow, signal = params
            macd = ta.trend.MACD(df['close'], window_slow=slow, window_fast=fast, window_sign=signal)
            df[f'macd'] = macd.macd()
            df[f'macd_signal'] = macd.macd_signal()
            df[f'macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        for period in config.TECHNICAL_INDICATORS['BB']:
            bb = ta.volatility.BollingerBands(df['close'], window=period, window_dev=2)
            df[f'bb_high_{period}'] = bb.bollinger_hband()
            df[f'bb_mid_{period}'] = bb.bollinger_mavg()
            df[f'bb_low_{period}'] = bb.bollinger_lband()
            df[f'bb_width_{period}'] = bb.bollinger_wband()
        
        # Average True Range
        for period in config.TECHNICAL_INDICATORS['ATR']:
            df[f'atr_{period}'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=period)
        
        # Stochastic Oscillator
        for period in config.TECHNICAL_INDICATORS['STOCH']:
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=period)
            df[f'stoch_{period}'] = stoch.stoch()
            df[f'stoch_signal_{period}'] = stoch.stoch_signal()
        
        # Average Directional Index
        for period in config.TECHNICAL_INDICATORS['ADX']:
            df[f'adx_{period}'] = ta.trend.adx(df['high'], df['low'], df['close'], window=period)
        
        # On-Balance Volume
        if config.TECHNICAL_INDICATORS['OBV']:
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # VWAP approximation (using daily data)
        if config.TECHNICAL_INDICATORS['VWAP']:
            df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # Additional features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['close_open_pct'] = (df['close'] - df['open']) / df['open']
        
        # Volume features
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        print(f"Added {len(df.columns) - 6} technical indicators")
        
        return df
    
    def prepare_features(self, df, target_col='close'):
        """
        Prepare features for model training
        
        Args:
            df: DataFrame with technical indicators
            target_col: Column to predict
        
        Returns:
            Cleaned DataFrame ready for sequence creation
        """
        df = df.copy()
        
        # Drop rows with NaN values (from indicator calculations)
        initial_len = len(df)
        df = df.dropna()
        print(f"Dropped {initial_len - len(df)} rows with NaN values")
        
        # Store feature columns (excluding date and target)
        self.feature_columns = [col for col in df.columns if col not in ['date', target_col]]
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def create_sequences(self, df, target_col='close', sequence_length=60):
        """
        Create sequences for LSTM input
        
        Args:
            df: DataFrame with features
            target_col: Column to predict
            sequence_length: Number of time steps in each sequence
        
        Returns:
            X (sequences), y (targets), dates
        """
        print(f"Creating sequences with length {sequence_length}...")
        
        # Separate features and target
        feature_data = df[self.feature_columns].values
        target_data = df[target_col].values
        dates = df['date'].values
        
        # Normalize features
        feature_data_scaled = self.scaler.fit_transform(feature_data)
        
        X, y, seq_dates = [], [], []
        
        for i in range(sequence_length, len(feature_data_scaled)):
            X.append(feature_data_scaled[i-sequence_length:i])
            y.append(target_data[i])
            seq_dates.append(dates[i])
        
        X = np.array(X)
        y = np.array(y)
        seq_dates = np.array(seq_dates)
        
        print(f"Created {len(X)} sequences")
        print(f"Input shape: {X.shape}")
        print(f"Output shape: {y.shape}")
        
        return X, y, seq_dates
    
    def split_data(self, X, y, dates, train_ratio=0.8):
        """
        Split data into train and test sets
        
        Args:
            X: Input sequences
            y: Target values
            dates: Corresponding dates
            train_ratio: Ratio of training data
        
        Returns:
            X_train, X_test, y_train, y_test, train_dates, test_dates
        """
        split_idx = int(len(X) * train_ratio)
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        train_dates = dates[:split_idx]
        test_dates = dates[split_idx:]
        
        print(f"\nData split:")
        print(f"Training set: {len(X_train)} samples ({train_ratio*100:.1f}%)")
        print(f"Test set: {len(X_test)} samples ({(1-train_ratio)*100:.1f}%)")
        print(f"Training period: {train_dates[0]} to {train_dates[-1]}")
        print(f"Test period: {test_dates[0]} to {test_dates[-1]}")
        
        return X_train, X_test, y_train, y_test, train_dates, test_dates
    
    def process_pair(self, df, pair_name):
        """
        Complete preprocessing pipeline for a trading pair
        
        Args:
            df: Raw DataFrame with OHLCV data
            pair_name: Name of the trading pair
        
        Returns:
            Dictionary with processed data
        """
        print(f"\n{'='*50}")
        print(f"Processing {pair_name}")
        print(f"{'='*50}")
        
        # Add technical indicators
        df_with_indicators = self.add_technical_indicators(df)
        
        # Prepare features
        df_clean = self.prepare_features(df_with_indicators)
        
        # Save processed data
        filename = f"{pair_name}_processed_{config.INTERVAL}.csv"
        filepath = os.path.join(config.DATA_DIR, filename)
        df_clean.to_csv(filepath, index=False)
        print(f"Saved processed data to {filepath}")
        
        # Create sequences
        X, y, dates = self.create_sequences(
            df_clean,
            target_col='close',
            sequence_length=config.SEQUENCE_LENGTH
        )
        
        # Split data
        X_train, X_test, y_train, y_test, train_dates, test_dates = self.split_data(
            X, y, dates,
            train_ratio=config.TRAIN_TEST_SPLIT
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_dates': train_dates,
            'test_dates': test_dates,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'df_clean': df_clean
        }
    
    def inverse_transform_predictions(self, predictions, feature_idx=0):
        """
        Inverse transform scaled predictions
        
        Args:
            predictions: Scaled predictions
            feature_idx: Index of the feature to inverse transform
        
        Returns:
            Unscaled predictions
        """
        # Create a full feature array with zeros
        n_features = len(self.feature_columns)
        dummy = np.zeros((len(predictions), n_features))
        dummy[:, feature_idx] = predictions.flatten()
        
        # Inverse transform
        inv_transformed = self.scaler.inverse_transform(dummy)
        
        return inv_transformed[:, feature_idx]


if __name__ == "__main__":
    # Test the preprocessor
    from data_scraper import DataScraper
    
    scraper = DataScraper()
    preprocessor = DataPreprocessor()
    
    # Load data
    for pair_name in config.TRADING_PAIRS.keys():
        df = scraper.load_data(pair_name, 'raw')
        
        if df is not None:
            processed_data = preprocessor.process_pair(df, pair_name)
            print(f"\nProcessed {pair_name} successfully!")
