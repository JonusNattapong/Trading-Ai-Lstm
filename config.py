"""
Configuration file for LSTM Trading AI
Contains all hyperparameters and settings
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
    'XAUUSD': 'GLD',  # Gold ETF (more reliable than GC=F)
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

# Strategy parameters
STOP_LOSS_PCT = 0.02  # 2%
TAKE_PROFIT_PCT = 0.05  # 5%
RISK_PER_TRADE = 0.02  # 2% of capital per trade

# Model naming
MODEL_NAME_PREFIX = 'lstm_trading'

# Prediction settings
PREDICTION_THRESHOLD = 0.01  # 1% minimum predicted movement to trade
