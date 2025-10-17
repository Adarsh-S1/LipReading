"""
Test Script for Landmark Extraction Pipeline
===========================================

This script tests the complete landmark extraction and training pipeline
to ensure all components work correctly.

Author: Enhanced Lipreading System
Date: 2024
"""

import os
import sys
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract_landmarks import LandmarkExtractor
from utils_landmark import (
    extract_frames_from_video, normalize_landmarks, save_landmark_features,
    load_landmark_features, compute_landmark_features, validate_landmarks
)
from train_model import create_model, LipreadingTrainer, LandmarkDataset
from torch.utils.data import DataLoader


def test_landmark_extraction():
    """Test landmark extraction functionality."""
    print("Testing landmark extraction...")
    
    # Create dummy video frames for testing
    dummy_frames = []
    for i in range(10):
        # Create a dummy frame with a simple pattern
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_frames.append(frame)
    
    # Initialize extractor
    extractor = LandmarkExtractor()
    
    try:
        # Test frame processing
        landmarks_list = []
        for frame in dummy_frames:
            landmarks_data = extractor.extract_landmarks_from_frame(frame)
            if landmarks_data is not None:
                landmarks_list.append(landmarks_data)
        
        print(f"  PASS: Processed {len(landmarks_list)} frames with landmarks")
        
        # Test landmark validation
        valid_count = 0
        for landmarks in landmarks_list:
            if validate_landmarks(landmarks):
                valid_count += 1
        
        print(f"  PASS: {valid_count}/{len(landmarks_list)} landmarks are valid")
        
        return landmarks_list
        
    except Exception as e:
        print(f"  FAIL: Error in landmark extraction: {e}")
        return None
    
    finally:
        extractor.close()


def test_feature_processing(landmarks_list):
    """Test feature processing functionality."""
    print("Testing feature processing...")
    
    if not landmarks_list:
        print("  ✗ No landmarks to process")
        return None
    
    try:
        # Test normalization
        normalized_landmarks = []
        for landmarks in landmarks_list:
            normalized = normalize_landmarks(landmarks)
            normalized_landmarks.append(normalized)
        
        print(f"  ✓ Normalized {len(normalized_landmarks)} landmark sets")
        
        # Test feature computation
        features = compute_landmark_features(normalized_landmarks)
        print(f"  ✓ Computed features shape: {features.shape}")
        
        # Test feature statistics
        print(f"  ✓ Feature statistics:")
        print(f"    Mean: {np.mean(features, axis=0)[:3]}...")
        print(f"    Std: {np.std(features, axis=0)[:3]}...")
        
        return features
        
    except Exception as e:
        print(f"  ✗ Error in feature processing: {e}")
        return None


def test_data_serialization(features):
    """Test data serialization and loading."""
    print("Testing data serialization...")
    
    if features is None:
        print("  ✗ No features to serialize")
        return None
    
    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test data structure
            test_data = {
                'features': features,
                'labels': np.random.randint(0, 5, len(features)),
                'metadata': {'test': True, 'n_samples': len(features)}
            }
            
            # Test saving and loading
            npy_path = os.path.join(temp_dir, "test_features.npy")
            pkl_path = os.path.join(temp_dir, "test_features.pkl")
            
            # Save as numpy
            save_landmark_features(test_data, npy_path)
            print("  ✓ Saved as .npy format")
            
            # Save as pickle
            save_landmark_features(test_data, pkl_path)
            print("  ✓ Saved as .pkl format")
            
            # Load numpy
            loaded_npy = load_landmark_features(npy_path)
            print("  ✓ Loaded from .npy format")
            
            # Load pickle
            loaded_pkl = load_landmark_features(pkl_path)
            print("  ✓ Loaded from .pkl format")
            
            # Verify data integrity
            assert np.array_equal(loaded_npy['features'], features)
            assert np.array_equal(loaded_pkl['features'], features)
            print("  ✓ Data integrity verified")
            
        return True
        
    except Exception as e:
        print(f"  ✗ Error in data serialization: {e}")
        return False


def test_model_creation():
    """Test model creation and basic functionality."""
    print("Testing model creation...")
    
    try:
        # Test different model types
        input_size = 50
        num_classes = 5
        
        # LSTM model
        lstm_model = create_model('lstm', input_size, num_classes, 
                                hidden_size=32, num_layers=1)
        print(f"  ✓ Created LSTM model with {sum(p.numel() for p in lstm_model.parameters())} parameters")
        
        # CNN model
        cnn_model = create_model('cnn', input_size, num_classes)
        print(f"  ✓ Created CNN model with {sum(p.numel() for p in cnn_model.parameters())} parameters")
        
        # Transformer model
        transformer_model = create_model('transformer', input_size, num_classes,
                                       d_model=32, nhead=4, num_layers=2)
        print(f"  ✓ Created Transformer model with {sum(p.numel() for p in transformer_model.parameters())} parameters")
        
        return [lstm_model, cnn_model, transformer_model]
        
    except Exception as e:
        print(f"  ✗ Error in model creation: {e}")
        return None


def test_training_pipeline(models):
    """Test training pipeline."""
    print("Testing training pipeline...")
    
    if not models:
        print("  ✗ No models to test")
        return False
    
    try:
        # Create dummy training data
        n_samples = 50
        n_features = 50
        n_classes = 5
        sequence_length = 10
        
        # Generate dummy features and labels
        features = np.random.randn(n_samples, n_features)
        labels = np.random.randint(0, n_classes, n_samples)
        
        # Create sequences
        sequences = []
        for i in range(n_samples - sequence_length + 1):
            sequence = features[i:i + sequence_length]
            sequences.append(sequence)
        sequences = np.array(sequences)
        sequence_labels = labels[:len(sequences)]
        
        print(f"  ✓ Created {len(sequences)} training sequences")
        
        # Test with each model
        for i, model in enumerate(models):
            model_name = ['LSTM', 'CNN', 'Transformer'][i]
            
            # Create dataset and dataloader
            dataset = LandmarkDataset(sequences, sequence_labels, sequence_length)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
            
            # Create trainer
            trainer = LipreadingTrainer(model, device="cpu", learning_rate=0.01)
            
            # Test training step
            train_loss, train_acc = trainer.train_epoch(dataloader)
            print(f"  ✓ {model_name} training step: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            
            # Test validation
            val_loss, val_acc = trainer.validate(dataloader)
            print(f"  ✓ {model_name} validation: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error in training pipeline: {e}")
        return False


def test_complete_workflow():
    """Test the complete workflow."""
    print("Testing complete workflow...")
    
    try:
        # Step 1: Landmark extraction
        landmarks_list = test_landmark_extraction()
        if not landmarks_list:
            print("  ✗ Landmark extraction failed")
            return False
        
        # Step 2: Feature processing
        features = test_feature_processing(landmarks_list)
        if features is None:
            print("  ✗ Feature processing failed")
            return False
        
        # Step 3: Data serialization
        serialization_success = test_data_serialization(features)
        if not serialization_success:
            print("  ✗ Data serialization failed")
            return False
        
        # Step 4: Model creation
        models = test_model_creation()
        if not models:
            print("  ✗ Model creation failed")
            return False
        
        # Step 5: Training pipeline
        training_success = test_training_pipeline(models)
        if not training_success:
            print("  ✗ Training pipeline failed")
            return False
        
        print("  ✓ Complete workflow test passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Error in complete workflow: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("LANDMARK EXTRACTION PIPELINE TESTS")
    print("=" * 80)
    
    test_results = []
    
    # Test 1: Landmark extraction
    print("\n1. Testing landmark extraction...")
    landmarks_list = test_landmark_extraction()
    test_results.append(landmarks_list is not None)
    
    # Test 2: Feature processing
    print("\n2. Testing feature processing...")
    features = test_feature_processing(landmarks_list)
    test_results.append(features is not None)
    
    # Test 3: Data serialization
    print("\n3. Testing data serialization...")
    serialization_success = test_data_serialization(features)
    test_results.append(serialization_success)
    
    # Test 4: Model creation
    print("\n4. Testing model creation...")
    models = test_model_creation()
    test_results.append(models is not None)
    
    # Test 5: Training pipeline
    print("\n5. Testing training pipeline...")
    training_success = test_training_pipeline(models)
    test_results.append(training_success)
    
    # Test 6: Complete workflow
    print("\n6. Testing complete workflow...")
    workflow_success = test_complete_workflow()
    test_results.append(workflow_success)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    test_names = [
        "Landmark Extraction",
        "Feature Processing", 
        "Data Serialization",
        "Model Creation",
        "Training Pipeline",
        "Complete Workflow"
    ]
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "PASS" if result else "FAIL"
        print(f"{i+1}. {name}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! The pipeline is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


def main():
    """Main test function."""
    try:
        success = run_all_tests()
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
