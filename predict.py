"""
Prediction and Backtesting Module
Make predictions and backtest trading strategies
Using PyTorch instead of TensorFlow/Keras
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import torch
import os

from data_scraper import DataScraper
from data_preprocessing import DataPreprocessor
import config


class TradingPredictor:
    """Make predictions and backtest strategies"""
    
    def __init__(self, pair_name, model_type='attention'):
        """
        Initialize predictor
        
        Args:
            pair_name: Trading pair (e.g., 'XAUUSD', 'BTCUSD')
            model_type: Type of model ('simple', 'bidirectional', 'attention', 'multi_head')
        """
        self.pair_name = pair_name
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.scaler = None
        self.feature_columns = None
        
    def load_model(self):
        """Load trained model and preprocessor"""
        model_name = f"{config.MODEL_NAME_PREFIX}_{self.pair_name}_{self.model_type}"
        
        # Load model architecture first to recreate the model
        architecture_path = f"{config.MODEL_DIR}/{model_name}_architecture.json"
        if not os.path.exists(architecture_path):
            raise FileNotFoundError(f"Architecture file not found: {architecture_path}")
        
        import json
        with open(architecture_path, 'r') as f:
            architecture = json.load(f)
        
        # Recreate model
        from lstm_model import create_model
        self.model = create_model(self.model_type, architecture['input_size'])
        
        # Load model weights
        model_path = f"{config.MODEL_DIR}/{model_name}.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set to evaluation mode
        print(f"Model loaded from {model_path}")
        
        # Load preprocessor
        preprocessor_path = f"{config.MODEL_DIR}/{model_name}_preprocessor.pkl"
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
        
        preprocessor_data = joblib.load(preprocessor_path)
        self.scaler = preprocessor_data['scaler']
        self.feature_columns = preprocessor_data['feature_columns']
        print(f"Preprocessor loaded from {preprocessor_path}")
        
        return self.model
    
    def prepare_data_for_prediction(self, df):
        """
        Prepare data for making predictions
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Prepared sequences for prediction
        """
        preprocessor = DataPreprocessor()
        
        # Add technical indicators
        df_with_indicators = preprocessor.add_technical_indicators(df)
        
        # Prepare features
        df_clean = preprocessor.prepare_features(df_with_indicators)
        
        # Ensure same features as training
        if not all(col in df_clean.columns for col in self.feature_columns):
            missing = [col for col in self.feature_columns if col not in df_clean.columns]
            raise ValueError(f"Missing features: {missing}")
        
        # Extract features in same order
        feature_data = df_clean[self.feature_columns].values
        
        # Scale features using saved scaler
        feature_data_scaled = self.scaler.transform(feature_data)
        
        return feature_data_scaled, df_clean
    
    def predict_next_price(self, sequence):
        """
        Predict next price from a sequence
        
        Args:
            sequence: Input sequence (sequence_length, n_features)
        
        Returns:
            Predicted price
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        if len(sequence.shape) == 2:
            sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).to(device)
        
        with torch.no_grad():
            prediction = self.model(sequence_tensor)
        
        return prediction.cpu().numpy()[0, 0]
    
    def predict_series(self, df):
        """
        Make predictions for entire dataset
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with predictions
        """
        feature_data_scaled, df_clean = self.prepare_data_for_prediction(df)
        
        predictions = []
        dates = []
        actual_prices = []
        
        # Create sequences and predict
        for i in range(config.SEQUENCE_LENGTH, len(feature_data_scaled)):
            sequence = feature_data_scaled[i-config.SEQUENCE_LENGTH:i]
            pred = self.predict_next_price(sequence)
            
            predictions.append(pred)
            dates.append(df_clean.iloc[i]['date'])
            actual_prices.append(df_clean.iloc[i]['close'])
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'date': dates,
            'actual': actual_prices,
            'predicted': predictions
        })
        
        # Calculate metrics
        results_df['error'] = results_df['actual'] - results_df['predicted']
        results_df['error_pct'] = (results_df['error'] / results_df['actual']) * 100
        results_df['direction'] = np.where(
            (results_df['predicted'].diff() > 0) & (results_df['actual'].diff() > 0), 'correct_up',
            np.where((results_df['predicted'].diff() < 0) & (results_df['actual'].diff() < 0), 'correct_down',
            'incorrect')
        )
        
        return results_df
    
    def backtest_strategy(self, df, initial_capital=10000):
        """
        Backtest trading strategy
        
        Args:
            df: DataFrame with OHLCV data
            initial_capital: Starting capital
        
        Returns:
            Backtest results DataFrame
        """
        print(f"\n{'='*60}")
        print(f"Backtesting Strategy for {self.pair_name}")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"{'='*60}")
        
        # Get predictions
        results_df = self.predict_series(df)
        
        # Initialize backtest variables
        capital = initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        trades = []
        portfolio_values = []
        
        for i in range(1, len(results_df)):
            current_price = results_df.iloc[i]['actual']
            predicted_next = results_df.iloc[i]['predicted']
            current_predicted = results_df.iloc[i-1]['predicted']
            
            # Calculate predicted price change
            predicted_change_pct = ((predicted_next - current_price) / current_price)
            
            # Trading logic
            if position == 0:  # No position
                # Enter long if predicted increase > threshold
                if predicted_change_pct > config.PREDICTION_THRESHOLD:
                    position = 1
                    entry_price = current_price
                    position_size = capital * config.RISK_PER_TRADE
                    
                # Enter short if predicted decrease > threshold
                elif predicted_change_pct < -config.PREDICTION_THRESHOLD:
                    position = -1
                    entry_price = current_price
                    position_size = capital * config.RISK_PER_TRADE
            
            elif position == 1:  # Long position
                # Calculate profit/loss
                price_change_pct = (current_price - entry_price) / entry_price
                
                # Take profit
                if price_change_pct >= config.TAKE_PROFIT_PCT:
                    profit = (current_price - entry_price) / entry_price * position_size
                    capital += profit
                    
                    trades.append({
                        'date': results_df.iloc[i]['date'],
                        'type': 'CLOSE_LONG_TP',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit_pct': price_change_pct * 100,
                        'profit': profit,
                        'capital': capital
                    })
                    
                    position = 0
                    entry_price = 0
                
                # Stop loss
                elif price_change_pct <= -config.STOP_LOSS_PCT:
                    loss = (current_price - entry_price) / entry_price * position_size
                    capital += loss
                    
                    trades.append({
                        'date': results_df.iloc[i]['date'],
                        'type': 'CLOSE_LONG_SL',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit_pct': price_change_pct * 100,
                        'profit': loss,
                        'capital': capital
                    })
                    
                    position = 0
                    entry_price = 0
                
                # Reverse signal
                elif predicted_change_pct < -config.PREDICTION_THRESHOLD:
                    profit = (current_price - entry_price) / entry_price * position_size
                    capital += profit
                    
                    trades.append({
                        'date': results_df.iloc[i]['date'],
                        'type': 'CLOSE_LONG_REVERSE',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit_pct': price_change_pct * 100,
                        'profit': profit,
                        'capital': capital
                    })
                    
                    position = 0
                    entry_price = 0
            
            elif position == -1:  # Short position
                # Calculate profit/loss
                price_change_pct = (entry_price - current_price) / entry_price
                
                # Take profit
                if price_change_pct >= config.TAKE_PROFIT_PCT:
                    profit = (entry_price - current_price) / entry_price * position_size
                    capital += profit
                    
                    trades.append({
                        'date': results_df.iloc[i]['date'],
                        'type': 'CLOSE_SHORT_TP',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit_pct': price_change_pct * 100,
                        'profit': profit,
                        'capital': capital
                    })
                    
                    position = 0
                    entry_price = 0
                
                # Stop loss
                elif price_change_pct <= -config.STOP_LOSS_PCT:
                    loss = (entry_price - current_price) / entry_price * position_size
                    capital += loss
                    
                    trades.append({
                        'date': results_df.iloc[i]['date'],
                        'type': 'CLOSE_SHORT_SL',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit_pct': price_change_pct * 100,
                        'profit': loss,
                        'capital': capital
                    })
                    
                    position = 0
                    entry_price = 0
                
                # Reverse signal
                elif predicted_change_pct > config.PREDICTION_THRESHOLD:
                    profit = (entry_price - current_price) / entry_price * position_size
                    capital += profit
                    
                    trades.append({
                        'date': results_df.iloc[i]['date'],
                        'type': 'CLOSE_SHORT_REVERSE',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit_pct': price_change_pct * 100,
                        'profit': profit,
                        'capital': capital
                    })
                    
                    position = 0
                    entry_price = 0
            
            # Track portfolio value
            portfolio_values.append(capital)
        
        # Create trades DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Calculate metrics
        if len(trades_df) > 0:
            total_return = ((capital - initial_capital) / initial_capital) * 100
            n_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['profit'] > 0])
            losing_trades = len(trades_df[trades_df['profit'] < 0])
            win_rate = (winning_trades / n_trades) * 100 if n_trades > 0 else 0
            
            avg_profit = trades_df[trades_df['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['profit'] < 0]['profit'].mean() if losing_trades > 0 else 0
            profit_factor = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
            
            print(f"\n{'='*60}")
            print(f"BACKTEST RESULTS")
            print(f"{'='*60}")
            print(f"Initial Capital: ${initial_capital:,.2f}")
            print(f"Final Capital: ${capital:,.2f}")
            print(f"Total Return: {total_return:.2f}%")
            print(f"Total Trades: {n_trades}")
            print(f"Winning Trades: {winning_trades}")
            print(f"Losing Trades: {losing_trades}")
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Average Profit: ${avg_profit:.2f}")
            print(f"Average Loss: ${avg_loss:.2f}")
            print(f"Profit Factor: {profit_factor:.2f}")
            print(f"{'='*60}")
        else:
            print("No trades executed")
        
        return trades_df, portfolio_values, results_df
    
    def plot_backtest_results(self, trades_df, portfolio_values, results_df):
        """Plot backtest results"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'Backtest Results - {self.pair_name} - {self.model_type.upper()}', fontsize=16)
        
        # Portfolio value over time
        axes[0].plot(portfolio_values, label='Portfolio Value')
        axes[0].axhline(y=portfolio_values[0], color='r', linestyle='--', alpha=0.5, label='Initial Capital')
        axes[0].set_title('Portfolio Value Over Time')
        axes[0].set_xlabel('Trade Number')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Actual vs Predicted prices
        axes[1].plot(results_df['date'], results_df['actual'], label='Actual', alpha=0.7)
        axes[1].plot(results_df['date'], results_df['predicted'], label='Predicted', alpha=0.7)
        
        # Mark trades
        if len(trades_df) > 0:
            long_closes = trades_df[trades_df['type'].str.contains('LONG')]
            short_closes = trades_df[trades_df['type'].str.contains('SHORT')]
            
            for _, trade in long_closes.iterrows():
                axes[1].scatter(trade['date'], trade['exit_price'], color='green', marker='^', s=100, alpha=0.6)
            
            for _, trade in short_closes.iterrows():
                axes[1].scatter(trade['date'], trade['exit_price'], color='red', marker='v', s=100, alpha=0.6)
        
        axes[1].set_title('Price with Trade Signals')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Price')
        axes[1].legend()
        axes[1].grid(True)
        
        # Trade profit distribution
        if len(trades_df) > 0:
            axes[2].bar(range(len(trades_df)), trades_df['profit'], 
                       color=['green' if p > 0 else 'red' for p in trades_df['profit']])
            axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[2].set_title('Trade Profit Distribution')
            axes[2].set_xlabel('Trade Number')
            axes[2].set_ylabel('Profit ($)')
            axes[2].grid(True)
        else:
            axes[2].text(0.5, 0.5, 'No Trades Executed', ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"{config.RESULTS_DIR}/backtest_{self.pair_name}_{self.model_type}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Backtest plot saved to {filename}")
        
        plt.show()


if __name__ == "__main__":
    # Test predictions and backtesting
    for pair_name in config.TRADING_PAIRS.keys():
        print(f"\n\n{'='*70}")
        print(f"PREDICTING AND BACKTESTING {pair_name}")
        print(f"{'='*70}\n")
        
        try:
            # Load data
            scraper = DataScraper()
            df = scraper.load_data(pair_name, 'raw')
            
            if df is not None:
                # Create predictor
                predictor = TradingPredictor(pair_name, model_type='attention')
                predictor.load_model()
                
                # Backtest
                trades_df, portfolio_values, results_df = predictor.backtest_strategy(df, initial_capital=10000)
                
                # Plot results
                predictor.plot_backtest_results(trades_df, portfolio_values, results_df)
                
                # Save trades
                if len(trades_df) > 0:
                    trades_path = f"{config.RESULTS_DIR}/trades_{pair_name}.csv"
                    trades_df.to_csv(trades_path, index=False)
                    print(f"Trades saved to {trades_path}")
        
        except Exception as e:
            print(f"Error processing {pair_name}: {str(e)}")
