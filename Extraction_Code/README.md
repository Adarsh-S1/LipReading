# Facial Landmark Extraction for Lipreading

This module provides a complete pipeline for extracting facial landmarks from videos and training lipreading models. The system is designed for **modularity and efficiency** - feature extraction is independent of the training stage.

## 🎯 **Overview**

The pipeline consists of three main components:

1. **`extract_landmarks.py`** - Extracts facial landmarks from videos/images
2. **`utils_landmark.py`** - Helper functions for processing and normalization
3. **`train_model.py`** - Trains lipreading models using pre-extracted features

## 📁 **Project Structure**

```
Extraction_Code/
├── extract_landmarks.py      # Main landmark extraction script
├── utils_landmark.py        # Utility functions
├── train_model.py           # Training script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🚀 **Quick Start**

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Extract Landmarks from Video

```bash
python extract_landmarks.py --input video.mp4 --output extracted_landmarks --save_visualization
```

### 3. Train Lipreading Model

```bash
python train_model.py --data_dir extracted_landmarks --model_type lstm --epochs 50
```

## 📋 **Detailed Usage**

### Landmark Extraction

#### From Video File
```bash
python extract_landmarks.py \
    --input /path/to/video.mp4 \
    --output /path/to/output \
    --min_detection_confidence 0.5 \
    --min_tracking_confidence 0.5 \
    --refine_landmarks \
    --save_visualization
```

#### From Image Sequence
```bash
python extract_landmarks.py \
    --input /path/to/image_directory \
    --output /path/to/output
```

#### Command Line Arguments

- `--input`: Input video file or image directory
- `--output`: Output directory for extracted features
- `--min_detection_confidence`: Minimum confidence for face detection (default: 0.5)
- `--min_tracking_confidence`: Minimum confidence for landmark tracking (default: 0.5)
- `--refine_landmarks`: Use refined landmarks for better accuracy
- `--save_visualization`: Save visualization frames with landmarks drawn

### Model Training

#### LSTM Model
```bash
python train_model.py \
    --data_dir extracted_landmarks \
    --model_type lstm \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --sequence_length 10
```

#### CNN Model
```bash
python train_model.py \
    --data_dir extracted_landmarks \
    --model_type cnn \
    --epochs 100 \
    --batch_size 32
```

#### Transformer Model
```bash
python train_model.py \
    --data_dir extracted_landmarks \
    --model_type transformer \
    --epochs 100 \
    --batch_size 16 \
    --d_model 128 \
    --nhead 8 \
    --num_layers 4
```

#### Training Arguments

- `--data_dir`: Directory containing extracted landmark features
- `--model_type`: Type of model ('lstm', 'cnn', 'transformer')
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate for optimizer
- `--sequence_length`: Length of sequences for temporal modeling
- `--output_dir`: Directory to save trained models
- `--device`: Device to use ('cpu' or 'cuda')

## 🏗️ **Architecture Details**

### Landmark Extraction

The system extracts **68 facial landmarks** using MediaPipe:

- **20 lip landmarks**: Precise lip movement tracking
- **17 facial contour landmarks**: Facial structure reference
- **31 eyes/eyebrows/nose landmarks**: Additional facial context

#### Features Extracted

1. **Raw landmark coordinates** (normalized)
2. **Lip width and height**
3. **Lip aspect ratio**
4. **Distance from lip center to face center**
5. **Lip opening** (vertical distance between upper and lower lip)

### Model Architectures

#### 1. LSTM Model
- **Purpose**: Temporal sequence modeling
- **Architecture**: Multi-layer LSTM with dropout
- **Input**: Sequences of landmark features
- **Output**: Classification predictions

#### 2. CNN Model
- **Purpose**: Spatial feature extraction
- **Architecture**: 1D convolutions with max pooling
- **Input**: Landmark feature sequences
- **Output**: Classification predictions

#### 3. Transformer Model
- **Purpose**: Self-attention based modeling
- **Architecture**: Multi-head attention with positional encoding
- **Input**: Landmark feature sequences
- **Output**: Classification predictions

## 📊 **Output Formats**

### Landmark Features

The system saves features in multiple formats:

1. **`.npy`** - NumPy array format (recommended)
2. **`.pkl`** - Pickle format for Python objects
3. **`.csv`** - CSV format for compatibility
4. **`_metadata.json`** - Metadata about extraction

### Training Outputs

1. **`best_model.pth`** - Best model weights
2. **`training_history.json`** - Training metrics
3. **`training_history.png`** - Training plots

## 🔧 **Customization**

### Adding New Landmark Features

```python
def custom_landmark_feature(landmarks_data):
    """Custom feature extraction function."""
    # Your custom feature extraction logic
    return custom_features
```

### Creating Custom Models

```python
class CustomLipreadingModel(nn.Module):
    """Custom lipreading model."""
    def __init__(self, input_size, num_classes):
        super().__init__()
        # Your custom architecture
        pass
    
    def forward(self, x):
        # Your forward pass
        return output
```

## 📈 **Performance Tips**

### For Landmark Extraction

1. **Use GPU acceleration** when available
2. **Adjust confidence thresholds** for your data
3. **Use refined landmarks** for better accuracy
4. **Process videos in batches** for efficiency

### For Model Training

1. **Use appropriate sequence lengths** (10-20 frames)
2. **Apply data augmentation** for robustness
3. **Use early stopping** to prevent overfitting
4. **Monitor validation metrics** closely

## 🐛 **Troubleshooting**

### Common Issues

1. **No landmarks detected**: Lower confidence thresholds
2. **Memory issues**: Reduce batch size or sequence length
3. **Poor accuracy**: Check data quality and model architecture
4. **Slow training**: Use GPU acceleration

### Debug Mode

```bash
# Enable debug logging
export PYTHONPATH=$PYTHONPATH:.
python extract_landmarks.py --input video.mp4 --output debug_output
```

## 📚 **API Reference**

### LandmarkExtractor Class

```python
extractor = LandmarkExtractor(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True
)

# Extract from video
result = extractor.extract_landmarks_from_video("video.mp4", "output_dir")

# Extract from images
result = extractor.extract_landmarks_from_image_sequence("image_dir", "output_dir")
```

### Utility Functions

```python
# Load features
data = load_landmark_features("features.npy")

# Compute features
features = compute_landmark_features(landmarks_data)

# Create sequences
sequences = create_landmark_sequence_features(landmarks_data, sequence_length=10)
```

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 **Acknowledgments**

- MediaPipe for facial landmark detection
- PyTorch for deep learning framework
- OpenCV for computer vision utilities
