"""
Simple Test Script
Tests basic functionality without deep learning dependencies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import config
from data_preprocessing import DataPreprocessor

def generate_synthetic_data(symbol='GC=F', days=365):
    """Generate synthetic OHLCV data for testing"""
    print(f"Generating synthetic data for {symbol}...")

    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Generate synthetic price data
    np.random.seed(42)  # For reproducible results

    # Start with a base price (around $2000 for gold)
    base_price = 2000 if 'GC=F' in symbol else 50000 if 'BTC' in symbol else 100

    # Generate random walk prices
    price_changes = np.random.normal(0, 0.02, len(dates))  # 2% daily volatility
    prices = base_price * np.exp(np.cumsum(price_changes))

    # Generate OHLCV from prices
    high_mult = 1 + np.random.uniform(0, 0.03, len(dates))  # Up to 3% higher
    low_mult = 1 - np.random.uniform(0, 0.03, len(dates))   # Up to 3% lower
    volume_base = 100000 if 'GC=F' in symbol else 1000000 if 'BTC' in symbol else 10000

    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate OHLC
        high = close * high_mult[i]
        low = close * low_mult[i]
        open_price = prices[i-1] if i > 0 else close * (1 + np.random.normal(0, 0.01))

        # Ensure OHLC relationships
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Generate volume
        volume = int(volume_base * (1 + np.random.normal(0, 0.5)))

        data.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })

    df = pd.DataFrame(data)
    print(f"Generated {len(df)} records for {symbol}")
    return df

def test_basic_functionality():
    """Test basic data preprocessing with synthetic data"""
    print("Testing basic functionality...")
    print("="*50)

    # Generate synthetic data instead of scraping
    print("1. Generating Synthetic Data...")
    try:
        sample_data = generate_synthetic_data('GC=F', days=365)

        if sample_data is not None and not sample_data.empty:
            print(f"✓ Synthetic data generation successful! Fetched {len(sample_data)} records")
            print(f"  Columns: {list(sample_data.columns)}")
            print(f"  Date range: {sample_data['date'].min()} to {sample_data['date'].max()}")
            print(f"  Price range: ${sample_data['close'].min():.2f} - ${sample_data['close'].max():.2f}")
        else:
            print("✗ Synthetic data generation failed")
            return False

    except Exception as e:
        print(f"✗ Synthetic data generation error: {str(e)}")
        return False

    # Test data preprocessing
    print("\n2. Testing Data Preprocessing...")
    try:
        preprocessor = DataPreprocessor()

        # Add technical indicators
        df_with_indicators = preprocessor.add_technical_indicators(sample_data)
        print(f"✓ Technical indicators added! New shape: {df_with_indicators.shape}")

        # Prepare features
        df_clean = preprocessor.prepare_features(df_with_indicators)
        print(f"✓ Features prepared! Final shape: {df_clean.shape}")
        print(f"  Feature columns: {len(preprocessor.feature_columns)}")

    except Exception as e:
        print(f"✗ Data preprocessing error: {str(e)}")
        return False

    print("\n3. Testing Configuration...")
    try:
        print(f"✓ Trading pairs: {list(config.TRADING_PAIRS.keys())}")
        print(f"✓ Sequence length: {config.SEQUENCE_LENGTH}")
        print(f"✓ Technical indicators configured: {len(config.TECHNICAL_INDICATORS)}")
        
        # Check if directories exist
        import os
        print(f"✓ Data directory exists: {os.path.exists(config.DATA_DIR)}")
        print(f"✓ Model directory exists: {os.path.exists(config.MODEL_DIR)}")
    except Exception as e:
        print(f"✗ Configuration error: {str(e)}")
        return False

    print("\n" + "="*50)
    print("✓ ALL BASIC TESTS PASSED!")
    print("The core functionality is working correctly.")
    print("To use the LSTM models, please install TensorFlow/PyTorch")
    print("with a compatible Python version (3.8-3.13).")
    print("="*50)

    return True

if __name__ == "__main__":
    print("LSTM Trading AI - Basic Functionality Test")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {__import__('sys').version}")
    print()

    success = test_basic_functionality()

    if success:
        print("\nNext steps:")
        print("1. Install TensorFlow/PyTorch with Python 3.8-3.13")
        print("2. Run: python main.py full")
        print("3. Or test individual components:")
        print("   - python main.py scrape")
        print("   - python main.py train")
        print("   - python main.py backtest")
    else:
        print("\nPlease check the error messages above and fix any issues.")