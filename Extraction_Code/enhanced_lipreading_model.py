"""
Enhanced Lipreading Model with Landmark-based Features
Implements the landmark-based approach described in the paper with:
- 68 facial landmarks (20 lips, 17 contour, 31 eyes/eyebrows/nose)
- Angle matrix computation between lip and contour landmarks
- Motion sequence extraction with temporal differences
- Bi-directional GRU for temporal modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import os
import json

from landmark_features import (
    LandmarkGRU,
    create_landmark_model,
    process_landmark_sequence_with_gru,
    extract_enhanced_landmark_features
)


class EnhancedLipreadingModel(nn.Module):
    """
    Enhanced lipreading model that combines visual appearance features with landmark-based features.
    Uses bi-directional GRU for temporal modeling of landmark motion sequences.
    """
    
    def __init__(
        self,
        visual_input_size: int = 224,  # Visual feature dimension
        landmark_input_size: int = 340,  # Landmark feature dimension (20*17)
        landmark_hidden_size: int = 128,
        landmark_num_layers: int = 2,
        visual_hidden_size: int = 256,
        fusion_hidden_size: int = 512,
        num_classes: int = 1000,  # Number of output classes
        dropout: float = 0.1
    ):
        super(EnhancedLipreadingModel, self).__init__()
        
        self.visual_input_size = visual_input_size
        self.landmark_input_size = landmark_input_size
        self.landmark_hidden_size = landmark_hidden_size
        self.num_classes = num_classes
        
        # Landmark processing branch
        self.landmark_gru = LandmarkGRU(
            input_size=landmark_input_size,
            hidden_size=landmark_hidden_size,
            num_layers=landmark_num_layers,
            dropout=dropout
        )
        
        # Visual processing branch (can be replaced with any visual encoder)
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_input_size, visual_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(visual_hidden_size, visual_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(landmark_hidden_size + visual_hidden_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final classification head
        self.classifier = nn.Linear(fusion_hidden_size, num_classes)
        
    def forward(
        self, 
        visual_features: torch.Tensor, 
        landmark_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the enhanced lipreading model.
        
        Args:
            visual_features: Visual features of shape (B, T, visual_input_size)
            landmark_features: Landmark motion features of shape (B, T, landmark_input_size)
        
        Returns:
            Output logits of shape (B, T, num_classes)
        """
        batch_size, seq_len = visual_features.shape[:2]
        
        # Process landmark features through bi-directional GRU
        landmark_processed = self.landmark_gru(landmark_features)  # (B, T, landmark_hidden_size)
        
        # Process visual features
        visual_processed = self.visual_encoder(visual_features)  # (B, T, visual_hidden_size)
        
        # Fuse features
        fused_features = torch.cat([landmark_processed, visual_processed], dim=-1)  # (B, T, landmark_hidden_size + visual_hidden_size)
        fused_features = self.fusion_layer(fused_features)  # (B, T, fusion_hidden_size)
        
        # Generate predictions
        logits = self.classifier(fused_features)  # (B, T, num_classes)
        
        return logits
    
    def extract_landmark_features(
        self, 
        landmarks_list: list, 
        lip_indices: list, 
        contour_indices: list, 
        eyes_eyebrows_nose_indices: list,
        frame_w: int, 
        frame_h: int
    ) -> torch.Tensor:
        """
        Extract landmark features from a sequence of landmarks.
        
        Args:
            landmarks_list: List of landmark objects for each frame
            lip_indices: 20 lip landmark indices
            contour_indices: 17 facial contour landmark indices
            eyes_eyebrows_nose_indices: 31 eyes/eyebrows/nose landmark indices
            frame_w: Frame width
            frame_h: Frame height
        
        Returns:
            Processed landmark features of shape (1, T, landmark_hidden_size)
        """
        # Extract raw landmark features
        raw_features = extract_enhanced_landmark_features(
            landmarks_list, lip_indices, contour_indices, 
            eyes_eyebrows_nose_indices, frame_w, frame_h
        )  # (T, 340)
        
        # Process through GRU
        device = next(self.parameters()).device
        processed_features = process_landmark_sequence_with_gru(
            raw_features, self.landmark_gru, device
        )  # (1, T, landmark_hidden_size)
        
        return processed_features


class LipreadingTrainer:
    """
    Trainer class for the enhanced lipreading model.
    Handles training, validation, and model saving/loading.
    """
    
    def __init__(
        self,
        model: EnhancedLipreadingModel,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_step(
        self, 
        visual_features: torch.Tensor, 
        landmark_features: torch.Tensor, 
        labels: torch.Tensor
    ) -> float:
        """
        Perform one training step.
        
        Args:
            visual_features: Visual features (B, T, visual_input_size)
            landmark_features: Landmark features (B, T, landmark_input_size)
            labels: Ground truth labels (B, T)
        
        Returns:
            Loss value
        """
        # Move inputs to device
        visual_features = visual_features.to(self.device)
        landmark_features = landmark_features.to(self.device)
        labels = labels.to(self.device)
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        logits = self.model(visual_features, landmark_features)
        
        # Compute loss
        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(
        self, 
        visual_features: torch.Tensor, 
        landmark_features: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Perform validation.
        
        Args:
            visual_features: Visual features (B, T, visual_input_size)
            landmark_features: Landmark features (B, T, landmark_input_size)
            labels: Ground truth labels (B, T)
        
        Returns:
            Tuple of (loss, accuracy)
        """
        # Move inputs to device
        visual_features = visual_features.to(self.device)
        landmark_features = landmark_features.to(self.device)
        labels = labels.to(self.device)
        
        self.model.eval()
        
        with torch.no_grad():
            logits = self.model(visual_features, landmark_features)
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Compute accuracy
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == labels).float().mean().item()
            
        return loss.item(), accuracy
    
    def save_model(self, path: str, epoch: int, loss: float, accuracy: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']


def create_enhanced_lipreading_model(
    visual_input_size: int = 224,
    landmark_input_size: int = 340,
    landmark_hidden_size: int = 128,
    visual_hidden_size: int = 256,
    fusion_hidden_size: int = 512,
    num_classes: int = 1000,
    dropout: float = 0.1
) -> EnhancedLipreadingModel:
    """
    Create an enhanced lipreading model with landmark-based features.
    
    Args:
        visual_input_size: Dimension of visual features
        landmark_input_size: Dimension of landmark features (340 for angle matrix)
        landmark_hidden_size: Hidden size for landmark GRU
        visual_hidden_size: Hidden size for visual encoder
        fusion_hidden_size: Hidden size for fusion layer
        num_classes: Number of output classes
        dropout: Dropout rate
    
    Returns:
        Initialized EnhancedLipreadingModel
    """
    model = EnhancedLipreadingModel(
        visual_input_size=visual_input_size,
        landmark_input_size=landmark_input_size,
        landmark_hidden_size=landmark_hidden_size,
        visual_hidden_size=visual_hidden_size,
        fusion_hidden_size=fusion_hidden_size,
        num_classes=num_classes,
        dropout=dropout
    )
    
    return model


def test_enhanced_model():
    """Test the enhanced lipreading model with dummy data."""
    print("Testing Enhanced Lipreading Model...")
    
    # Create model
    model = create_enhanced_lipreading_model()
    
    # Create dummy data
    batch_size, seq_len = 2, 10
    visual_features = torch.randn(batch_size, seq_len, 224, requires_grad=True)
    landmark_features = torch.randn(batch_size, seq_len, 340, requires_grad=True)
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Test forward pass
    with torch.no_grad():
        logits = model(visual_features, landmark_features)
        print(f"Input visual features shape: {visual_features.shape}")
        print(f"Input landmark features shape: {landmark_features.shape}")
        print(f"Output logits shape: {logits.shape}")
        print(f"Expected output shape: ({batch_size}, {seq_len}, 1000)")
        
        # Test trainer
        trainer = LipreadingTrainer(model)
        loss = trainer.train_step(visual_features, landmark_features, labels)
        val_loss, accuracy = trainer.validate(visual_features, landmark_features, labels)
        
        print(f"Training loss: {loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {accuracy:.4f}")
    
    print("Enhanced lipreading model test completed successfully!")
    return model


if __name__ == "__main__":
    # Test the enhanced model
    test_enhanced_model()
