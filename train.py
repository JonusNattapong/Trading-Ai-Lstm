"""
Training Module
Train LSTM models for trading predictions
Using PyTorch instead of TensorFlow/Keras
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import joblib
import json
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_scraper import DataScraper
from data_preprocessing import DataPreprocessor
from lstm_model import create_model, create_optimizer_and_criterion, train_model, create_callbacks
import config


class ModelTrainer:
    """Train and evaluate LSTM trading models"""
    
    def __init__(self, pair_name, model_type='attention'):
        """
        Initialize trainer
        
        Args:
            pair_name: Trading pair to train on (e.g., 'XAUUSD', 'BTCUSD')
            model_type: Type of model ('simple', 'bidirectional', 'attention', 'multi_head')
        """
        self.pair_name = pair_name
        self.model_type = model_type
        self.model_builder = None
        self.history = None
        self.preprocessor = DataPreprocessor()
        self.processed_data = None
        
    def load_and_process_data(self):
        """Load and preprocess data"""
        print(f"\n{'='*60}")
        print(f"Loading data for {self.pair_name}")
        print(f"{'='*60}")
        
        scraper = DataScraper()
        df = scraper.load_data(self.pair_name, 'raw')
        
        if df is None:
            print(f"No data found for {self.pair_name}. Fetching...")
            all_data = scraper.fetch_all_pairs()
            df = all_data.get(self.pair_name)
        
        if df is None:
            raise ValueError(f"Could not load data for {self.pair_name}")
        
        # Process data
        self.processed_data = self.preprocessor.process_pair(df, self.pair_name)
        
        return self.processed_data
    
    def build_model(self):
        """Build the LSTM model"""
        X_train = self.processed_data['X_train']
        input_size = X_train.shape[2]  # Number of features
        
        print(f"\n{'='*60}")
        print(f"Building {self.model_type.upper()} model")
        print(f"Input size: {input_size}")
        print(f"{'='*60}")
        
        self.model = create_model(self.model_type, input_size)
        
        print(f"Model created successfully!")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return self.model
    
    def train(self):
        """Train the model"""
        X_train = self.processed_data['X_train']
        y_train = self.processed_data['y_train']
        X_test = self.processed_data['X_test']
        y_test = self.processed_data['y_test']
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        model_name = f"{config.MODEL_NAME_PREFIX}_{self.pair_name}_{self.model_type}"
        
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"Epochs: {config.EPOCHS}")
        print(f"{'='*60}\n")
        
        # Create optimizer and criterion
        optimizer, criterion = create_optimizer_and_criterion(
            self.model, 
            learning_rate=config.LEARNING_RATE
        )
        
        # Create callbacks (placeholder for PyTorch)
        callbacks = create_callbacks(
            model_name,
            patience_early=config.EARLY_STOPPING_PATIENCE,
            patience_lr=config.REDUCE_LR_PATIENCE
        )
        
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Train model
        self.history = train_model(
            self.model, train_loader, test_loader, optimizer, criterion,
            num_epochs=config.EPOCHS, patience=callbacks['patience_early'], device=device
        )
        
        return self.history
    
    def evaluate(self):
        """Evaluate model performance"""
        X_test = self.processed_data['X_test']
        y_test = self.processed_data['y_test']
        
        # Convert to PyTorch tensors
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()
        
        print(f"\n{'='*60}")
        print(f"Evaluating model on test set")
        print(f"{'='*60}")
        
        # Make predictions
        with torch.no_grad():
            X_test_tensor = X_test_tensor.to(device)
            y_pred_tensor = self.model(X_test_tensor)
            y_pred = y_pred_tensor.cpu().numpy().flatten()
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Direction accuracy (did we predict the right direction?)
        y_test_diff = np.diff(y_test)
        y_pred_diff = np.diff(y_pred)
        direction_accuracy = np.mean((y_test_diff > 0) == (y_pred_diff > 0)) * 100
        
        print(f"\nTest Loss (MSE): {mse:.6f}")
        print(f"Test RMSE: {rmse:.6f}")
        print(f"Test MAE: {mae:.6f}")
        print(f"Test MAPE: {mape:.2f}%")
        print(f"Direction Accuracy: {direction_accuracy:.2f}%")
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'direction_accuracy': float(direction_accuracy)
        }
        
        return metrics, y_pred
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History - {self.pair_name} - {self.model_type.upper()}', fontsize=16)
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss (MSE)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE
        axes[0, 1].plot(self.history.history['mae'], label='Train MAE')
        axes[0, 1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # MAPE
        axes[1, 0].plot(self.history.history['mape'], label='Train MAPE')
        axes[1, 0].plot(self.history.history['val_mape'], label='Val MAPE')
        axes[1, 0].set_title('Mean Absolute Percentage Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAPE (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate (if available)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                           ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"{config.RESULTS_DIR}/training_history_{self.pair_name}_{self.model_type}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {filename}")
        
        plt.show()
    
    def plot_predictions(self, y_pred):
        """Plot predictions vs actual"""
        y_test = self.processed_data['y_test']
        test_dates = self.processed_data['test_dates']
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f'Predictions vs Actual - {self.pair_name} - {self.model_type.upper()}', fontsize=16)
        
        # Full comparison
        axes[0].plot(test_dates, y_test, label='Actual', alpha=0.7)
        axes[0].plot(test_dates, y_pred, label='Predicted', alpha=0.7)
        axes[0].set_title('Full Test Set Predictions')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True)
        
        # Last 100 predictions (zoomed)
        n_show = min(100, len(y_test))
        axes[1].plot(test_dates[-n_show:], y_test[-n_show:], label='Actual', marker='o', alpha=0.7)
        axes[1].plot(test_dates[-n_show:], y_pred[-n_show:], label='Predicted', marker='x', alpha=0.7)
        axes[1].set_title(f'Last {n_show} Predictions (Zoomed)')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Price')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"{config.RESULTS_DIR}/predictions_{self.pair_name}_{self.model_type}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Predictions plot saved to {filename}")
        
        plt.show()
    
    def save_model(self):
        """Save trained model and preprocessor"""
        model_name = f"{config.MODEL_NAME_PREFIX}_{self.pair_name}_{self.model_type}"
        
        # Save model
        model_path = f"{config.MODEL_DIR}/{model_name}.pth"
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Save preprocessor (scaler and feature columns)
        preprocessor_path = f"{config.MODEL_DIR}/{model_name}_preprocessor.pkl"
        preprocessor_data = {
            'scaler': self.preprocessor.scaler,
            'feature_columns': self.preprocessor.feature_columns
        }
        joblib.dump(preprocessor_data, preprocessor_path)
        print(f"Preprocessor saved to {preprocessor_path}")
        
        # Save model architecture as JSON
        architecture_path = f"{config.MODEL_DIR}/{model_name}_architecture.json"
        from lstm_model import save_model_architecture
        save_model_architecture(self.model, architecture_path)
        
        # Save training info
        info = {
            'pair_name': self.pair_name,
            'model_type': self.model_type,
            'sequence_length': config.SEQUENCE_LENGTH,
            'train_test_split': config.TRAIN_TEST_SPLIT,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_features': len(self.preprocessor.feature_columns),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        info_path = f"{config.MODEL_DIR}/{model_name}_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
        print(f"Model info saved to {info_path}")
    
    def run_full_training(self):
        """Run complete training pipeline"""
        print(f"\n{'#'*60}")
        print(f"# FULL TRAINING PIPELINE")
        print(f"# Pair: {self.pair_name}")
        print(f"# Model: {self.model_type.upper()}")
        print(f"{'#'*60}\n")
        
        # Load and process data
        self.load_and_process_data()
        
        # Build model
        self.build_model()
        
        # Train model
        self.train()
        
        # Evaluate model
        metrics, y_pred = self.evaluate()
        
        # Plot results
        self.plot_training_history()
        self.plot_predictions(y_pred)
        
        # Save model
        self.save_model()
        
        # Save metrics
        model_name = f"{config.MODEL_NAME_PREFIX}_{self.pair_name}_{self.model_type}"
        metrics_path = f"{config.RESULTS_DIR}/{model_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {metrics_path}")
        
        print(f"\n{'#'*60}")
        print(f"# TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'#'*60}\n")
        
        return metrics


if __name__ == "__main__":
    # Train models for all pairs
    for pair_name in config.TRADING_PAIRS.keys():
        print(f"\n\n{'='*70}")
        print(f"TRAINING {pair_name}")
        print(f"{'='*70}\n")
        
        # Train attention model (best performance usually)
        trainer = ModelTrainer(pair_name, model_type='attention')
        metrics = trainer.run_full_training()
        
        print(f"\nFinal metrics for {pair_name}:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
