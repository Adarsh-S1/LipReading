"""
Example Usage of Landmark Extraction and Training Pipeline
========================================================

This script demonstrates how to use the landmark extraction and training pipeline
for lipreading models. It shows the complete workflow from raw videos to trained models.

Author: Enhanced Lipreading System
Date: 2024
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract_landmarks import LandmarkExtractor
from utils_landmark import load_landmark_features, compute_landmark_features
from train_model import create_model, LipreadingTrainer, LandmarkDataset
from torch.utils.data import DataLoader


def example_landmark_extraction():
    """Example of extracting landmarks from a video."""
    print("=" * 60)
    print("EXAMPLE: Landmark Extraction from Video")
    print("=" * 60)
    
    # Initialize extractor
    extractor = LandmarkExtractor(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True
    )
    
    # Example video path (replace with your video)
    video_path = "example_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        print("Please provide a valid video file for this example.")
        return None
    
    try:
        # Extract landmarks
        result = extractor.extract_landmarks_from_video(
            video_path=video_path,
            output_dir="example_output",
            save_visualization=True
        )
        
        print(f"Successfully extracted landmarks from {result['valid_frames']} frames")
        print(f"Total frames processed: {result['total_frames']}")
        print(f"Output saved to: example_output/")
        
        return result
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        return None
    
    finally:
        extractor.close()


def example_feature_processing():
    """Example of processing extracted landmarks."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Feature Processing")
    print("=" * 60)
    
    # Load extracted landmarks
    landmark_file = "example_output/example_video_landmarks.npy"
    
    if not os.path.exists(landmark_file):
        print(f"Landmark file not found: {landmark_file}")
        print("Please run landmark extraction first.")
        return None
    
    try:
        # Load data
        data = load_landmark_features(landmark_file)
        print(f"Loaded landmarks from {data['valid_frames']} frames")
        
        # Compute features
        features = compute_landmark_features(data['landmarks'])
        print(f"Computed features shape: {features.shape}")
        
        # Show feature statistics
        print(f"Feature statistics:")
        print(f"  Mean: {np.mean(features, axis=0)[:5]}...")  # First 5 features
        print(f"  Std: {np.std(features, axis=0)[:5]}...")
        print(f"  Min: {np.min(features, axis=0)[:5]}...")
        print(f"  Max: {np.max(features, axis=0)[:5]}...")
        
        return features
        
    except Exception as e:
        print(f"Error during feature processing: {e}")
        return None


def example_model_training():
    """Example of training a lipreading model."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Model Training")
    print("=" * 60)
    
    # Create dummy data for demonstration
    print("Creating dummy training data...")
    
    # Generate dummy features and labels
    n_samples = 100
    n_features = 50
    n_classes = 5
    
    # Create dummy features
    features = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, n_classes, n_samples)
    
    # Create sequences
    sequence_length = 10
    sequences = []
    for i in range(n_samples - sequence_length + 1):
        sequence = features[i:i + sequence_length]
        sequences.append(sequence)
    sequences = np.array(sequences)
    
    # Create labels for sequences
    sequence_labels = labels[:len(sequences)]
    
    print(f"Created {len(sequences)} sequences of length {sequence_length}")
    print(f"Feature shape: {sequences.shape}")
    print(f"Number of classes: {n_classes}")
    
    # Create dataset
    dataset = LandmarkDataset(sequences, sequence_labels, sequence_length)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Create model
    model = create_model(
        model_type='lstm',
        input_size=n_features,
        num_classes=n_classes,
        hidden_size=64,
        num_layers=2
    )
    
    print(f"Created LSTM model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = LipreadingTrainer(model, device="cpu", learning_rate=0.001)
    
    # Train for a few epochs (demonstration)
    print("\nTraining model (demonstration with 5 epochs)...")
    
    for epoch in range(5):
        train_loss, train_acc = trainer.train_epoch(dataloader)
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
    
    print("Training completed!")
    return model, trainer


def example_model_evaluation():
    """Example of evaluating a trained model."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Model Evaluation")
    print("=" * 60)
    
    # Create dummy test data
    n_test = 20
    n_features = 50
    n_classes = 5
    
    test_features = np.random.randn(n_test, n_features)
    test_labels = np.random.randint(0, n_classes, n_test)
    
    # Create sequences
    sequence_length = 10
    test_sequences = []
    for i in range(n_test - sequence_length + 1):
        sequence = test_features[i:i + sequence_length]
        test_sequences.append(sequence)
    test_sequences = np.array(test_sequences)
    test_sequence_labels = test_labels[:len(test_sequences)]
    
    # Create test dataset
    test_dataset = LandmarkDataset(test_sequences, test_sequence_labels, sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Create and train a simple model
    model = create_model('lstm', n_features, n_classes, hidden_size=32, num_layers=1)
    trainer = LipreadingTrainer(model, device="cpu", learning_rate=0.01)
    
    # Train briefly
    for epoch in range(3):
        trainer.train_epoch(test_loader)
    
    # Evaluate
    val_loss, val_acc = trainer.validate(test_loader)
    print(f"Test Loss: {val_loss:.4f}")
    print(f"Test Accuracy: {val_acc:.2f}%")
    
    return val_loss, val_acc


def example_complete_pipeline():
    """Example of the complete pipeline from extraction to training."""
    print("\n" + "=" * 80)
    print("COMPLETE PIPELINE EXAMPLE")
    print("=" * 80)
    
    print("This example demonstrates the complete workflow:")
    print("1. Extract landmarks from video")
    print("2. Process landmarks into features")
    print("3. Train a lipreading model")
    print("4. Evaluate the model")
    
    # Step 1: Landmark extraction
    print("\nStep 1: Landmark Extraction")
    extraction_result = example_landmark_extraction()
    
    if extraction_result is None:
        print("Skipping remaining steps due to extraction failure.")
        return
    
    # Step 2: Feature processing
    print("\nStep 2: Feature Processing")
    features = example_feature_processing()
    
    if features is None:
        print("Skipping remaining steps due to processing failure.")
        return
    
    # Step 3: Model training
    print("\nStep 3: Model Training")
    model, trainer = example_model_training()
    
    # Step 4: Model evaluation
    print("\nStep 4: Model Evaluation")
    val_loss, val_acc = example_model_evaluation()
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("The complete lipreading pipeline has been demonstrated.")
    print("You can now use this system with your own video data.")


def main():
    """Main function to run examples."""
    print("FACIAL LANDMARK EXTRACTION AND LIPREADING TRAINING")
    print("Example Usage and Demonstration")
    print("=" * 80)
    
    try:
        # Run complete pipeline example
        example_complete_pipeline()
        
    except KeyboardInterrupt:
        print("\nExample interrupted by user.")
    except Exception as e:
        print(f"\nError during example execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
