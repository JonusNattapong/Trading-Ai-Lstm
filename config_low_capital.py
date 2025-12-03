"""
Low Capital Configuration for LSTM Trading AI
Optimized settings for $100-250 starting capital
"""

import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, LOGS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Trading pairs
TRADING_PAIRS = {
    'XAUUSD': 'GC=F',  # Gold futures
    'BTCUSD': 'BTC-USD'  # Bitcoin
}

# Data scraping settings
START_DATE = '2020-01-01'
END_DATE = None  # None means today
INTERVAL = '1d'  # 1d, 1h, 15m, etc.

# Feature engineering
TECHNICAL_INDICATORS = {
    'SMA': [7, 14, 21, 50, 200],
    'EMA': [12, 26],
    'RSI': [14],
    'MACD': [(12, 26, 9)],
    'BB': [20],  # Bollinger Bands
    'ATR': [14],
    'STOCH': [14],
    'ADX': [14],
    'OBV': True,
    'VWAP': True
}

# LSTM Model hyperparameters
SEQUENCE_LENGTH = 60  # Number of time steps to look back
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.2

# Model architecture
LSTM_UNITS = [128, 64, 32]  # Units for each LSTM layer
DROPOUT_RATE = 0.3
DENSE_UNITS = [16]
OUTPUT_UNITS = 1  # Predicting next price

# Training parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 7

# 🔧 LOW CAPITAL STRATEGY SETTINGS 🔧
# Ultra-conservative settings for $100-250 capital
STOP_LOSS_PCT = 0.015  # 1.5% (tighter than 2%)
TAKE_PROFIT_PCT = 0.03  # 3% (smaller than 5%)
RISK_PER_TRADE = 0.01  # 1% of capital per trade (smaller than 2%)

# Model naming
MODEL_NAME_PREFIX = 'lstm_trading_low_capital'

# Prediction settings - more sensitive for small moves
PREDICTION_THRESHOLD = 0.005  # 0.5% minimum predicted movement to trade

# Low capital specific settings
MIN_POSITION_SIZE = 10  # Minimum position size in dollars
MAX_DAILY_LOSS = 0.05  # Maximum 5% daily loss
MAX_WEEKLY_LOSS = 0.15  # Maximum 15% weekly loss
COMPOUNDING_FREQUENCY = 'daily'  # Reinvest profits daily