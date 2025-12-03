"""
LSTM Model Architecture
Advanced LSTM model with attention mechanism for trading predictions
Using PyTorch instead of TensorFlow/Keras
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import config
import os


class LSTMTradingModel(nn.Module):
    """Build and compile LSTM model for trading using PyTorch"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3, output_size=1):
        """
        Initialize model
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Number of output units
        """
        super(LSTMTradingModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, output_size)
        
        # Dropout and activation
        self.dropout_layer = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        
        Returns:
            Output tensor
        """
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Take the last time step output
        out = lstm_out[:, -1, :]
        
        # Dense layers
        out = self.dropout_layer(out)
        out = self.relu(self.fc1(out))
        out = self.dropout_layer(out)
        out = self.fc2(out)
        
        return out


class BidirectionalLSTMTradingModel(nn.Module):
    """Bidirectional LSTM model for trading"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3, output_size=1):
        super(BidirectionalLSTMTradingModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Dense layers (input size doubled due to bidirectional)
        self.fc1 = nn.Linear(hidden_size * 2, 16)
        self.fc2 = nn.Linear(16, output_size)
        
        # Dropout and activation
        self.dropout_layer = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        out = lstm_out[:, -1, :]  # Take last time step
        out = self.dropout_layer(out)
        out = self.relu(self.fc1(out))
        out = self.dropout_layer(out)
        out = self.fc2(out)
        return out


class AttentionLSTMTradingModel(nn.Module):
    """LSTM with attention mechanism"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3, output_size=1):
        super(AttentionLSTMTradingModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, output_size)
        
        # Dropout and activation
        self.dropout_layer = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Attention weights
        attention_weights = self.softmax(self.attention(lstm_out))
        
        # Apply attention
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Dense layers
        out = self.dropout_layer(context)
        out = self.relu(self.fc1(out))
        out = self.dropout_layer(out)
        out = self.fc2(out)
        
        return out


def create_model(model_type, input_size):
    """
    Create model based on type
    
    Args:
        model_type: Type of model ('simple', 'bidirectional', 'attention')
        input_size: Number of input features
    
    Returns:
        PyTorch model
    """
    if model_type == 'simple':
        model = LSTMTradingModel(input_size=input_size)
    elif model_type == 'bidirectional':
        model = BidirectionalLSTMTradingModel(input_size=input_size)
    elif model_type == 'attention':
        model = AttentionLSTMTradingModel(input_size=input_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def create_optimizer_and_criterion(model, learning_rate=0.001):
    """
    Create optimizer and loss function
    
    Args:
        model: PyTorch model
        learning_rate: Learning rate
    
    Returns:
        optimizer, criterion
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    return optimizer, criterion


def train_model(model, train_loader, val_loader, optimizer, criterion, 
                num_epochs=100, patience=15, device='cpu'):
    """
    Train the model
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        criterion: Loss function
        num_epochs: Number of epochs
        patience: Early stopping patience
        device: Device to train on
    
    Returns:
        Training history
    """
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    return history


def save_model_architecture(model, filepath):
    """
    Save model architecture as dictionary
    
    Args:
        model: PyTorch model
        filepath: Path to save the architecture
    """
    architecture = {
        'model_type': model.__class__.__name__,
        'input_size': model.input_size,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
        'dropout': model.dropout,
        'output_size': model.output_size
    }
    
    import json
    with open(filepath, 'w') as f:
        json.dump(architecture, f, indent=4)
    
    print(f"Model architecture saved to {filepath}")


if __name__ == "__main__":
    # Test model creation
    print("Testing PyTorch LSTM Model Architecture\n")
    
    # Assume 50 features
    input_size = 50
    
    # Test Simple LSTM
    print("\n" + "="*50)
    model = create_model('simple', input_size)
    print(f"Simple LSTM: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test Bidirectional LSTM
    print("\n" + "="*50)
    model = create_model('bidirectional', input_size)
    print(f"Bidirectional LSTM: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test Attention LSTM
    print("\n" + "="*50)
    model = create_model('attention', input_size)
    print(f"Attention LSTM: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


def create_callbacks(model_name, patience_early=15, patience_lr=7):
    """
    Create training callbacks (placeholder for PyTorch)
    
    Args:
        model_name: Name for saved models
        patience_early: Patience for early stopping
        patience_lr: Patience for learning rate reduction
    
    Returns:
        Dictionary with callback parameters
    """
    callbacks = {
        'model_name': model_name,
        'patience_early': patience_early,
        'patience_lr': patience_lr
    }
    
    return callbacks


if __name__ == "__main__":
    # Test model creation
    print("Testing PyTorch LSTM Model Architecture\n")
    
    # Assume 50 features
    input_size = 50
    
    # Test Simple LSTM
    print("\n" + "="*50)
    model = create_model('simple', input_size)
    print(f"Simple LSTM: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test Bidirectional LSTM
    print("\n" + "="*50)
    model = create_model('bidirectional', input_size)
    print(f"Bidirectional LSTM: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test Attention LSTM
    print("\n" + "="*50)
    model = create_model('attention', input_size)
    print(f"Attention LSTM: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
