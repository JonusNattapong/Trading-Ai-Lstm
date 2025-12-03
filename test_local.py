#!/usr/bin/env python3
"""
LSTM Trading AI - Local Test Script
Tests the complete system with CPU-only PyTorch or skips deep learning if not available.
"""

import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_pytorch():
    """Test if PyTorch is available and working."""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} available")
        print(f"  CUDA available: {torch.cuda.is_available()}")

        # Test basic tensor operations
        x = torch.randn(3, 3)
        print(f"  Tensor test: {x.shape} tensor created successfully")
        return True
    except ImportError:
        print("✗ PyTorch not available")
        return False
    except Exception as e:
        print(f"✗ PyTorch error: {e}")
        return False

def test_basic_functionality():
    """Test basic data pipeline without deep learning."""
    print("\n" + "="*50)
    print("Testing Basic Functionality (No Deep Learning)")
    print("="*50)

    try:
        from data_scraper import DataScraper
        from data_preprocessing import DataPreprocessor
        import config

        # Test data scraping
        print("1. Testing Data Scraper...")
        scraper = DataScraper()
        data = scraper.fetch_yahoo_data('GC=F', '2023-01-01', '2024-01-01')
        print(f"✓ Data scraping successful! Fetched {len(data)} records")

        # Test preprocessing
        print("2. Testing Data Preprocessing...")
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.prepare_features(data)
        print(f"✓ Data preprocessing successful! Shape: {processed_data.shape}")
        print(f"  Features: {len(preprocessor.feature_columns)}")

        # Test configuration
        print("3. Testing Configuration...")
        print(f"✓ Trading pairs: {list(config.TRADING_PAIRS.keys())}")
        print(f"✓ Sequence length: {config.SEQUENCE_LENGTH}")

        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_full_system():
    """Test the complete system with deep learning."""
    print("\n" + "="*50)
    print("Testing Full System (With Deep Learning)")
    print("="*50)

    try:
        from data_scraper import DataScraper
        from data_preprocessing import DataPreprocessor
        from lstm_model import create_model
        from train import ModelTrainer
        from predict import TradingPredictor
        import config

        # Data collection
        print("1. Collecting Data...")
        scraper = DataScraper()
        data = scraper.fetch_yahoo_data('GC=F', '2020-01-01', '2024-01-01')
        print(f"✓ Data collected: {len(data)} records")

        # Data preprocessing
        print("2. Preprocessing Data...")
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.prepare_features(data)
        print(f"✓ Data processed: {processed_data.shape}")

        # Create sequences
        print("3. Creating Sequences...")
        X, y = preprocessor.create_sequences(processed_data)
        print(f"✓ Sequences created: X={X.shape}, y={y.shape}")

        # Model creation
        print("4. Creating Model...")
        model = create_model('attention', input_shape=(config.SEQUENCE_LENGTH, len(preprocessor.feature_columns)))
        print("✓ Model created successfully")

        # Model training (quick test)
        print("5. Training Model (Quick Test)...")
        trainer = ModelTrainer('XAUUSD')
        # Train for only 2 epochs for quick test
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Quick training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        model.train()
        for epoch in range(2):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"  Epoch {epoch+1}/2 - Loss: {epoch_loss/len(dataloader):.4f}")

        print("✓ Model training completed")

        # Quick prediction test
        print("6. Testing Predictions...")
        predictor = TradingPredictor('XAUUSD')
        predictor.load_model(model, preprocessor)

        # Get last sequence for prediction
        last_sequence = X_tensor[-1:].unsqueeze(0)  # Add batch dimension
        prediction = predictor.predict_next_price(last_sequence)
        print(f"✓ Prediction test successful: {prediction}")

        print("\n🎉 Full system test completed successfully!")
        return True

    except Exception as e:
        print(f"✗ Full system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("LSTM Trading AI - Local Test Script")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version}")
    print("="*60)

    # Test PyTorch availability
    pytorch_available = test_pytorch()

    # Always test basic functionality
    basic_success = test_basic_functionality()

    # Test full system if PyTorch is available
    if pytorch_available:
        full_success = test_full_system()
    else:
        print("\n⚠️  Skipping full system test (PyTorch not available)")
        print("   To enable full testing:")
        print("   1. Use Python 3.8-3.13")
        print("   2. Install PyTorch: pip install torch torchvision torchaudio")
        print("   3. Or use Google Colab: https://colab.research.google.com/github/JonusNattapong/Trading-Ai-Lstm/blob/main/lstm_trading_ai.ipynb")
        full_success = False

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"PyTorch Available: {'✓' if pytorch_available else '✗'}")
    print(f"Basic Functionality: {'✓' if basic_success else '✗'}")
    print(f"Full System: {'✓' if full_success else '✗'}")

    if basic_success and not full_success:
        print("\n✅ Core data pipeline working! Use Google Colab for full deep learning features.")
    elif full_success:
        print("\n🎉 All tests passed! System is ready for trading.")
    else:
        print("\n❌ Tests failed. Check error messages above.")

    print("\nNext steps:")
    print("1. Run: python main.py scrape   # Collect data")
    print("2. Run: python main.py train    # Train models")
    print("3. Run: python main.py backtest # Test strategies")
    print("4. Or: python main.py full      # Run everything")

if __name__ == "__main__":
    main()