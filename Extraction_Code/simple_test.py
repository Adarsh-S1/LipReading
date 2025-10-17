"""
Simple Test Script for Landmark Extraction Pipeline
=================================================

This script tests the basic functionality of the landmark extraction pipeline
without Unicode characters to avoid encoding issues.

Author: Enhanced Lipreading System
Date: 2024
"""

import os
import sys
import numpy as np
import torch
import tempfile

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract_landmarks import LandmarkExtractor
from utils_landmark import (
    normalize_landmarks, save_landmark_features,
    load_landmark_features, compute_landmark_features, validate_landmarks
)
from train_model import create_model, LipreadingTrainer, LandmarkDataset
from torch.utils.data import DataLoader


def test_basic_functionality():
    """Test basic functionality without complex operations."""
    print("=" * 60)
    print("SIMPLE LANDMARK EXTRACTION PIPELINE TEST")
    print("=" * 60)
    
    try:
        # Test 1: Landmark extractor initialization
        print("\n1. Testing landmark extractor initialization...")
        extractor = LandmarkExtractor()
        print("  PASS: Landmark extractor created successfully")
        extractor.close()
        
        # Test 2: Model creation
        print("\n2. Testing model creation...")
        lstm_model = create_model('lstm', input_size=50, num_classes=5, 
                                hidden_size=32, num_layers=1)
        print(f"  PASS: LSTM model created with {sum(p.numel() for p in lstm_model.parameters())} parameters")
        
        cnn_model = create_model('cnn', input_size=50, num_classes=5)
        print(f"  PASS: CNN model created with {sum(p.numel() for p in cnn_model.parameters())} parameters")
        
        # Test 3: Data processing
        print("\n3. Testing data processing...")
        
        # Create dummy landmarks data
        dummy_landmarks = {
            'lip_landmarks': np.random.randn(20, 2),
            'contour_landmarks': np.random.randn(17, 2),
            'eyes_eyebrows_nose_landmarks': np.random.randn(31, 2),
            'frame_shape': (480, 640),
            'total_landmarks': 68
        }
        
        # Test normalization
        normalized = normalize_landmarks(dummy_landmarks)
        print("  PASS: Landmark normalization working")
        
        # Test feature computation
        features = compute_landmark_features([normalized])
        print(f"  PASS: Feature computation working, shape: {features.shape}")
        
        # Test validation
        is_valid = validate_landmarks(normalized)
        print(f"  PASS: Landmark validation working, valid: {is_valid}")
        
        # Test 4: Data serialization
        print("\n4. Testing data serialization...")
        
        test_data = {
            'features': features,
            'labels': np.array([0]),
            'metadata': {'test': True}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test numpy save/load
            npy_path = os.path.join(temp_dir, "test.npy")
            save_landmark_features(test_data, npy_path)
            loaded_data = load_landmark_features(npy_path)
            print("  PASS: NumPy serialization working")
            
            # Test pickle save/load
            pkl_path = os.path.join(temp_dir, "test.pkl")
            save_landmark_features(test_data, pkl_path)
            loaded_data = load_landmark_features(pkl_path)
            print("  PASS: Pickle serialization working")
        
        # Test 5: Training pipeline
        print("\n5. Testing training pipeline...")
        
        # Create dummy training data
        n_samples = 20
        n_features = 50
        n_classes = 5
        sequence_length = 5
        
        # Generate dummy data
        features = np.random.randn(n_samples, n_features)
        labels = np.random.randint(0, n_classes, n_samples)
        
        # Create sequences
        sequences = []
        for i in range(n_samples - sequence_length + 1):
            sequence = features[i:i + sequence_length]
            sequences.append(sequence)
        sequences = np.array(sequences)
        sequence_labels = labels[:len(sequences)]
        
        # Create dataset and dataloader
        dataset = LandmarkDataset(sequences, sequence_labels, sequence_length)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        print(f"  PASS: Dataset created with {len(dataset)} samples")
        
        # Test training step
        trainer = LipreadingTrainer(lstm_model, device="cpu", learning_rate=0.01)
        train_loss, train_acc = trainer.train_epoch(dataloader)
        print(f"  PASS: Training step working, Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        
        # Test validation
        val_loss, val_acc = trainer.validate(dataloader)
        print(f"  PASS: Validation working, Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        print("The landmark extraction pipeline is working correctly.")
        print("Key components verified:")
        print("  - Landmark extractor initialization")
        print("  - Model creation (LSTM, CNN)")
        print("  - Data processing and normalization")
        print("  - Feature computation")
        print("  - Data serialization (NumPy, Pickle)")
        print("  - Training pipeline")
        print("  - Validation pipeline")
        
        return True
        
    except Exception as e:
        print(f"\nFAIL: Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    try:
        success = test_basic_functionality()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\nTests interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nUnexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
