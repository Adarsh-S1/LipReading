"""
Training Script for Lipreading Model
===================================

This module loads pre-extracted landmark features and trains a lipreading model.
It supports various model architectures including LSTM, CNN, and Transformer-based models.

Author: Enhanced Lipreading System
Date: 2024
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

from utils_landmark import load_landmark_features, compute_landmark_features, create_landmark_sequence_features


class LandmarkDataset(Dataset):
    """
    Dataset class for landmark features.
    
    Handles loading and preprocessing of landmark features for training.
    """
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, 
                 sequence_length: int = 10, transform=None):
        """
        Initialize dataset.
        
        Args:
            features: Feature matrix of shape (n_samples, n_features)
            labels: Label array of shape (n_samples,)
            sequence_length: Length of sequences for temporal modeling
            transform: Optional transform to apply to features
        """
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Create sequences if needed
        if len(features.shape) == 2:  # Frame-level features
            self.sequences = self._create_sequences()
        else:  # Already sequences
            self.sequences = features
    
    def _create_sequences(self) -> np.ndarray:
        """Create sequences from frame-level features."""
        sequences = []
        for i in range(len(self.features) - self.sequence_length + 1):
            sequence = self.features[i:i + self.sequence_length]
            sequences.append(sequence)
        return np.array(sequences)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
        
        return torch.FloatTensor(sequence), torch.LongTensor([label])


class LipreadingLSTM(nn.Module):
    """
    LSTM-based lipreading model.
    
    Uses LSTM layers to process temporal sequences of landmark features.
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 num_classes: int, dropout: float = 0.1):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(LipreadingLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output
        output = self.classifier(lstm_out[:, -1, :])
        
        return output


class LipreadingCNN(nn.Module):
    """
    CNN-based lipreading model.
    
    Uses 1D convolutions to process landmark features.
    """
    
    def __init__(self, input_size: int, num_classes: int, dropout: float = 0.1):
        """
        Initialize CNN model.
        
        Args:
            input_size: Number of input features
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(LipreadingCNN, self).__init__()
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool1d(2)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        # Transpose for 1D convolution (batch, features, sequence)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        
        # Global average pooling
        x = torch.mean(x, dim=2)
        
        # Classification
        output = self.classifier(x)
        
        return output


class LipreadingTransformer(nn.Module):
    """
    Transformer-based lipreading model.
    
    Uses self-attention to process landmark sequences.
    """
    
    def __init__(self, input_size: int, d_model: int, nhead: int, 
                 num_layers: int, num_classes: int, dropout: float = 0.1):
        """
        Initialize Transformer model.
        
        Args:
            input_size: Number of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(LipreadingTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer forward pass
        x = self.transformer(x)
        
        # Use the last output
        output = self.classifier(x[:, -1, :])
        
        return output


class LipreadingTrainer:
    """
    Trainer class for lipreading models.
    
    Handles training, validation, and model saving.
    """
    
    def __init__(self, model: nn.Module, device: str = "cpu", learning_rate: float = 1e-3):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            device: Device to run training on
            learning_rate: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target.squeeze())
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target.squeeze())
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int, save_path: str = None) -> Dict:
        """
        Train model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            save_path: Path to save best model
            
        Returns:
            Training history dictionary
        """
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc and save_path:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f'  New best model saved to {save_path}')
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


def load_landmark_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load landmark data from directory.
    
    Args:
        data_dir: Directory containing landmark files
        
    Returns:
        Tuple of (features, labels)
    """
    features = []
    labels = []
    
    data_dir = Path(data_dir)
    
    # Load all landmark files
    for landmark_file in data_dir.glob("*_landmarks.npy"):
        data = load_landmark_features(str(landmark_file))
        
        # Extract features
        landmark_features = compute_landmark_features(data['landmarks'])
        features.append(landmark_features)
        
        # Extract label from filename (assuming format: class_name_landmarks.npy)
        label = landmark_file.stem.replace('_landmarks', '')
        labels.extend([label] * len(landmark_features))
    
    # Combine all features
    all_features = np.vstack(features)
    all_labels = np.array(labels)
    
    return all_features, all_labels


def create_model(model_type: str, input_size: int, num_classes: int, **kwargs) -> nn.Module:
    """
    Create model based on type.
    
    Args:
        model_type: Type of model ('lstm', 'cnn', 'transformer')
        input_size: Number of input features
        num_classes: Number of output classes
        **kwargs: Additional model parameters
        
    Returns:
        PyTorch model
    """
    if model_type == 'lstm':
        return LipreadingLSTM(
            input_size=input_size,
            hidden_size=kwargs.get('hidden_size', 128),
            num_layers=kwargs.get('num_layers', 2),
            num_classes=num_classes,
            dropout=kwargs.get('dropout', 0.1)
        )
    elif model_type == 'cnn':
        return LipreadingCNN(
            input_size=input_size,
            num_classes=num_classes,
            dropout=kwargs.get('dropout', 0.1)
        )
    elif model_type == 'transformer':
        return LipreadingTransformer(
            input_size=input_size,
            d_model=kwargs.get('d_model', 128),
            nhead=kwargs.get('nhead', 8),
            num_layers=kwargs.get('num_layers', 4),
            num_classes=num_classes,
            dropout=kwargs.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train lipreading model")
    parser.add_argument("--data_dir", required=True, help="Directory containing landmark features")
    parser.add_argument("--model_type", choices=['lstm', 'cnn', 'transformer'], 
                       default='lstm', help="Type of model to train")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--sequence_length", type=int, default=10, help="Sequence length")
    parser.add_argument("--output_dir", default="trained_models", help="Output directory")
    parser.add_argument("--device", default="cpu", help="Device to use")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    logger.info("Loading landmark data...")
    features, labels = load_landmark_data(args.data_dir)
    
    # Create sequences
    dummy_landmarks_data = []
    for _ in range(len(features)):
        dummy_landmarks_data.append({
            'lip_landmarks': np.random.randn(20, 2),
            'contour_landmarks': np.random.randn(17, 2),
            'eyes_eyebrows_nose_landmarks': np.random.randn(31, 2)
        })
    
    sequences = create_landmark_sequence_features(
        [{'landmarks': dummy_landmarks_data}], 
        args.sequence_length
    )
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = LandmarkDataset(X_train, y_train, args.sequence_length)
    val_dataset = LandmarkDataset(X_val, y_val, args.sequence_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    input_size = features.shape[1]
    num_classes = len(np.unique(labels))
    
    model = create_model(args.model_type, input_size, num_classes)
    logger.info(f"Created {args.model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = LipreadingTrainer(model, args.device, args.learning_rate)
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        train_loader, val_loader, args.epochs, 
        os.path.join(args.output_dir, f"best_{args.model_type}_model.pth")
    )
    
    # Plot training history
    trainer.plot_training_history(
        os.path.join(args.output_dir, f"{args.model_type}_training_history.png")
    )
    
    # Save training history
    with open(os.path.join(args.output_dir, f"{args.model_type}_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training completed. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
