# LSTM Trading AI for XAUUSD & BTCUSD

A comprehensive deep learning trading system using LSTM neural networks with attention mechanisms to predict and trade XAUUSD (Gold) and BTCUSD (Bitcoin).

**🎯 Special Focus: Low Capital Trading ($100-250 to $10,000+)**
*Realistic strategies and risk management for small starting capital*

## ⚠️ Critical Risk Warnings

**TRADING INVOLVES SUBSTANTIAL RISK OF LOSS**

- **Most retail traders lose money** in their first year
- **Past performance does not guarantee future results**
- **Never risk money you cannot afford to lose completely**
- **This system is for educational purposes only**
- **Always paper trade first** to test strategies
- **Start with very small amounts** while learning
- **Have a backup plan** for complete capital loss
- **Consider professional financial advice**
- **Trading can be psychologically addictive** - set strict limits

**Statistical Reality:**
- 70-80% of retail traders lose money
- Only 5-10% become consistently profitable
- Success requires years of education and practice
- No system guarantees profits

**Python 3.14 Compatibility**: This project was developed with Python 3.14.0, but PyTorch/TensorFlow may have compatibility issues. For best results:

### Recommended Setup:
1. **Use Python 3.8-3.13** for full deep learning functionality
2. **Install Visual C++ Redistributables** (required for PyTorch on Windows)
3. **Or use Google Colab** for immediate testing

### Quick Test (No Deep Learning):
```bash
python test_basic.py  # Tests data pipeline only
```

### Comprehensive Local Test:
```bash
python test_local.py  # Tests everything available on your system
```

### Full Setup (Recommended):
```bash
# For Python 3.8-3.13
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python main.py full
```

### Google Colab (Easiest Option):
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JonusNattapong/Trading-Ai-Lstm/blob/main/lstm_trading_ai.ipynb)

**Colab Setup:**
1. Open the notebook link above
2. Run all cells in order
3. No local installation required!

### Low Capital Trading ($100-250 to $10,000+)

**Important Reality Check:**
Growing $100-250 to $10,000+ requires either:
- **Very long timeframes** (2-8 years) with conservative 2-5% monthly returns
- **Extremely high risk tolerance** (50%+ monthly returns - very unrealistic)
- **Perfect execution** (90%+ win rate - statistically impossible)

**Realistic Expectations:**
- **Conservative (2% monthly)**: 8+ years for $200 → $10,000
- **Moderate (5% monthly)**: 4+ years for $200 → $10,000  
- **Aggressive (10% monthly)**: 2.5+ years (very risky, unlikely)
- **Very Aggressive (20% monthly)**: 1.5+ years (extremely risky, statistically improbable)

**Recommended Approach:**
1. **Start with paper trading** for 1-3 months
2. **Use ultra-conservative settings** (1% risk per trade)
3. **Focus on learning** rather than quick profits
4. **Scale gradually** as capital and experience grow
5. **Accept 1-3 year timeline** for significant growth

**Low Capital Configuration:**
- Risk per trade: 1% (vs 2% standard)
- Stop loss: 1.5% (vs 2% standard)
- Take profit: 3% (vs 5% standard)
- Minimum position size: $10
- Daily loss limit: 5% of capital

## 📄 Research Paper

A comprehensive academic research paper documenting the LSTM trading system:

### Compile the Paper
```bash
python compile_paper.py
```

**Paper Highlights:**
- Complete methodology and experimental results
- Performance analysis across market conditions
- Risk management framework validation
- Low capital trading optimization
- Future research directions

**Files:**
- `research_paper.tex` - LaTeX source
- `references.bib` - Bibliography
- `RESEARCH_README.md` - Detailed documentation

## Current Status

### ✅ Working Features (Python 3.8-3.14):
- **Data Collection**: Yahoo Finance integration for XAUUSD and BTCUSD
   - Note: The XAUUSD (Gold) pipeline now prefers the Hugging Face dataset
      `ZombitX64/xauusd-gold-price-historical-data-2004-2025` (daily and multiple
      granularities) when available. The system will attempt to use the Hugging
      Face `datasets` library (lazy import) to load the data, and will fall back
      to direct JSONL HTTP download if `datasets` cannot be imported on your
      system. To change the granularity, modify `config.INTERVAL` (e.g., `1d`,
      `1h`, `30m`, `15m`, `5m`, `1m`, `1w`, `1M`) or pass the `interval` argument
      to `DataScraper.fetch_financial_data`.
- **Technical Indicators**: 40+ indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- **Data Preprocessing**: Feature engineering and sequence creation
- **Configuration System**: Flexible hyperparameters and trading pairs
- **Basic Testing**: Comprehensive test scripts for validation

### ⚠️ Deep Learning Features (Python 3.8-3.13 Required):
- **LSTM Models**: Simple, Bidirectional, and Attention architectures
- **Model Training**: Complete training pipeline with validation
- **Backtesting**: Strategy testing with risk management
- **Predictions**: Real-time price predictions

### 🧪 Testing Options:

**Immediate Testing (No Installation Needed):**
- Use Google Colab notebook for full functionality

**Local Testing:**
- `python test_basic.py` - Tests data pipeline only
- `python test_local.py` - Tests all available features on your system
- `python main.py full` - Full pipeline (requires compatible Python/PyTorch)

### 📊 Test Results:
- ✅ Data scraping: Successfully fetches 250+ records per pair
- ✅ Technical indicators: Adds 27+ features successfully
- ✅ Basic pipeline: All core functionality validated
- ⚠️ Deep learning: Requires Python 3.8-3.13 for PyTorch compatibility

## Project Structure

```
Trading-Ai-Lstm/
├── config.py                  # Standard configuration
├── config_low_capital.py      # Low capital optimized settings
├── data_scraper.py           # Data collection module
├── data_preprocessing.py     # Feature engineering and preprocessing
├── lstm_model.py             # LSTM model architectures
├── train.py                  # Training script
├── predict.py                # Prediction and backtesting
├── main.py                   # Main execution script
├── test_basic.py             # Basic functionality test (no deep learning)
├── test_local.py             # Comprehensive local test script
├── low_capital_strategy.py   # Low capital strategy analysis
├── demo_low_capital.py       # Low capital backtesting demo
├── final_summary.py          # Complete low capital trading guide
├── research_paper.tex       # LaTeX research paper
├── references.bib           # Paper bibliography
├── compile_paper.py         # LaTeX compilation script
├── RESEARCH_README.md       # Research paper documentation
├── lstm_trading_ai.ipynb     # Google Colab notebook
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/                     # Data storage
├── models/                   # Saved models
├── logs/                     # Training logs
├── results/                  # Results and visualizations
└── __pycache__/              # Python cache files
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JonusNattapong/Trading-Ai-Lstm.git
cd Trading-Ai-Lstm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Low Capital Trading Guide
```bash
# Complete guide and reality check for $100-250 trading
python final_summary.py

# Analyze realistic growth scenarios
python low_capital_strategy.py

# Test low capital backtesting simulation
python demo_low_capital.py
```

### Run Full Pipeline
```bash
python main.py full
```

### Individual Steps

1. **Scrape Data**:
```bash
python main.py scrape
```

2. **Train Models**:
```bash
# Train for all pairs with attention model
python main.py train

# Train specific pair with specific model
python main.py train --pairs XAUUSD --models attention

# Train multiple models
python main.py train --pairs XAUUSD BTCUSD --models simple attention multi_head
```

3. **Backtest Models**:
```bash
# Backtest all pairs
python main.py backtest

# Backtest specific pair
python main.py backtest --pairs BTCUSD --models attention
```

4. **Update Data**:
```bash
python main.py update
```

## Configuration

Edit `config.py` to customize:

- **Trading Pairs**: Add or modify trading pairs
- **Data Settings**: Date range, interval (1d, 1h, 15m)
- **Technical Indicators**: Enable/disable specific indicators
- **Model Architecture**: LSTM units, dropout rate, dense layers
- **Training Parameters**: Batch size, epochs, learning rate
- **Strategy Parameters**: Stop-loss, take-profit, risk per trade

## Model Types

1. **Simple LSTM**: Stacked LSTM layers with dropout and batch normalization
2. **Bidirectional LSTM**: Processes sequences in both directions
3. **Attention LSTM**: Adds attention mechanism to focus on important time steps
4. **Multi-Head Attention LSTM**: Uses multiple attention heads for better feature extraction

## Technical Indicators

The system automatically calculates:
- **Trend**: SMA, EMA, MACD, ADX
- **Momentum**: RSI, Stochastic Oscillator
- **Volatility**: Bollinger Bands, ATR
- **Volume**: OBV, VWAP, Volume Ratio
- **Price Features**: Returns, log returns, high-low percentage, close-open percentage

## Strategy

The backtesting strategy includes:
- **Entry**: Signal when predicted price change > threshold
- **Exit**: Stop-loss (2%), Take-profit (5%), or reverse signal
- **Risk Management**: 2% risk per trade
- **Position Types**: Long and short positions

## Output

The system generates:

1. **Trained Models**: Saved in `models/` directory
   - Model weights (.h5)
   - Preprocessor (scaler and features)
   - Model architecture (JSON)
   - Training info (JSON)

2. **Visualizations**: Saved in `results/` directory
   - Training history plots
   - Prediction vs actual charts
   - Backtest results with trade signals
   - Portfolio value over time

3. **Metrics**: 
   - MSE, RMSE, MAE, MAPE
   - Direction accuracy
   - Win rate, profit factor
   - Total return

## Example Usage

```python
from data_scraper import DataScraper
from train import ModelTrainer
from predict import TradingPredictor

# Scrape data
scraper = DataScraper()
all_data = scraper.fetch_all_pairs()

# Train model
trainer = ModelTrainer('XAUUSD', model_type='attention')
metrics = trainer.run_full_training()

# Make predictions and backtest
predictor = TradingPredictor('XAUUSD', model_type='attention')
predictor.load_model()
trades_df, portfolio, results = predictor.backtest_strategy(all_data['XAUUSD'])
```

## Requirements

- **Python 3.8-3.13** (Python 3.14 not yet supported by TensorFlow/PyTorch)
- TensorFlow 2.15+ or PyTorch 2.0+
- pandas
- numpy
- scikit-learn
- yfinance
- matplotlib
- seaborn
- ta (Technical Analysis library)

## Installation

### For Python 3.8-3.13:

1. **Using pip:**
```bash
pip install tensorflow pandas numpy scikit-learn yfinance matplotlib seaborn ta plotly
```

2. **Or using conda:**
```bash
conda install tensorflow pandas numpy scikit-learn yfinance matplotlib seaborn -c conda-forge
```

### For Python 3.14 (Current Issue):

Python 3.14 is very new and TensorFlow/PyTorch don't support it yet. Please use Python 3.11 or 3.12:

**Option 1: Create a new conda environment with Python 3.11:**
```bash
conda create -n trading_ai python=3.11
conda activate trading_ai
pip install -r requirements.txt
```

**Option 2: Use pyenv to install Python 3.11:**
```bash
pyenv install 3.11.8
pyenv local 3.11.8
pip install -r requirements.txt
```

**Current requirements.txt (for Python 3.8-3.13):**
```
tensorflow>=2.15.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
yfinance>=0.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
requests>=2.31.0
beautifulsoup4>=4.12.0
ta>=0.10.0
plotly>=5.15.0
```

## Performance Notes

- Training time varies based on:
  - Dataset size
  - Model complexity
  - Hardware (GPU recommended)
- Expected training time: 10-30 minutes per model on CPU
- GPU acceleration significantly reduces training time

## Tips for Best Results

1. **Data Quality**: Ensure sufficient historical data (2+ years recommended)
2. **Feature Selection**: Experiment with different technical indicators
3. **Model Selection**: Attention-based models generally perform best
4. **Hyperparameter Tuning**: Adjust learning rate, batch size, and sequence length
5. **Risk Management**: Keep stop-loss and position sizing conservative
6. **Backtesting**: Always backtest on out-of-sample data

## Disclaimer

This project is for educational purposes only. Trading financial instruments carries risk. Do not use this system for live trading without thorough testing and understanding of the risks involved. Past performance does not guarantee future results.

## License

MIT License

## Author

JonusNattapong

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on GitHub.