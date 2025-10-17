"""
Utility Functions for Landmark Extraction
========================================

This module contains helper functions for facial landmark extraction,
normalization, and feature processing for lipreading models.

Author: Enhanced Lipreading System
Date: 2024
"""

import os
import cv2
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging


def extract_frames_from_video(video_path: str, 
                            max_frames: Optional[int] = None,
                            frame_skip: int = 1) -> List[np.ndarray]:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract (None for all)
        frame_skip: Extract every nth frame (1 = all frames)
        
    Returns:
        List of frames as numpy arrays
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_skip == 0:
            frames.append(frame.copy())
            
        frame_count += 1
        
        if max_frames is not None and len(frames) >= max_frames:
            break
    
    cap.release()
    return frames


def normalize_landmarks(landmarks_data: Dict) -> Dict:
    """
    Normalize landmark coordinates for consistent training.
    
    Args:
        landmarks_data: Dictionary containing landmark coordinates
        
    Returns:
        Dictionary with normalized landmarks
    """
    # Extract coordinates
    lip_landmarks = landmarks_data['lip_landmarks']
    contour_landmarks = landmarks_data['contour_landmarks']
    eyes_eyebrows_nose_landmarks = landmarks_data['eyes_eyebrows_nose_landmarks']
    frame_height, frame_width = landmarks_data['frame_shape']
    
    # Method 1: Normalize by face bounding box
    all_landmarks = np.vstack([lip_landmarks, contour_landmarks, eyes_eyebrows_nose_landmarks])
    
    if len(all_landmarks) == 0:
        return landmarks_data
    
    # Calculate bounding box
    min_x, min_y = np.min(all_landmarks, axis=0)
    max_x, max_y = np.max(all_landmarks, axis=0)
    
    # Add padding
    padding = 0.1
    width = max_x - min_x
    height = max_y - min_y
    min_x -= width * padding
    min_y -= height * padding
    max_x += width * padding
    max_y += height * padding
    
    # Normalize coordinates
    def normalize_coords(coords):
        if len(coords) == 0:
            return coords
        normalized = coords.copy().astype(np.float32)
        normalized[:, 0] = (coords[:, 0] - min_x) / (max_x - min_x)
        normalized[:, 1] = (coords[:, 1] - min_y) / (max_y - min_y)
        return normalized
    
    # Method 2: Normalize by inter-ocular distance (alternative approach)
    def normalize_by_eye_distance(coords):
        if len(coords) == 0:
            return coords
        # Use distance between eyes as reference
        # This is a simplified approach - in practice, you'd identify eye landmarks
        eye_distance = np.linalg.norm(contour_landmarks[0] - contour_landmarks[1]) if len(contour_landmarks) > 1 else 1.0
        normalized = coords.copy().astype(np.float32)
        normalized = normalized / eye_distance
        return normalized
    
    # Apply normalization
    normalized_lip = normalize_coords(lip_landmarks)
    normalized_contour = normalize_coords(contour_landmarks)
    normalized_eyes_eyebrows_nose = normalize_coords(eyes_eyebrows_nose_landmarks)
    
    return {
        'lip_landmarks': normalized_lip,
        'contour_landmarks': normalized_contour,
        'eyes_eyebrows_nose_landmarks': normalized_eyes_eyebrows_nose,
        'frame_shape': landmarks_data['frame_shape'],
        'total_landmarks': landmarks_data['total_landmarks'],
        'normalization_method': 'bounding_box',
        'bounding_box': {
            'min_x': min_x, 'min_y': min_y,
            'max_x': max_x, 'max_y': max_y
        }
    }


def create_landmark_visualization(frame: np.ndarray, landmarks_data: Dict, 
                                show_indices: bool = False) -> np.ndarray:
    """
    Create visualization of landmarks on frame.
    
    Args:
        frame: Input frame
        landmarks_data: Dictionary containing landmark coordinates
        show_indices: Whether to show landmark indices
        
    Returns:
        Frame with landmarks drawn
    """
    vis_frame = frame.copy()
    
    # Extract landmarks
    lip_landmarks = landmarks_data['lip_landmarks']
    contour_landmarks = landmarks_data['contour_landmarks']
    eyes_eyebrows_nose_landmarks = landmarks_data['eyes_eyebrows_nose_landmarks']
    
    # Draw lip landmarks (red)
    for i, (x, y) in enumerate(lip_landmarks):
        cv2.circle(vis_frame, (int(x), int(y)), 2, (0, 0, 255), -1)
        if show_indices:
            cv2.putText(vis_frame, str(i), (int(x), int(y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Draw contour landmarks (green)
    for i, (x, y) in enumerate(contour_landmarks):
        cv2.circle(vis_frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        if show_indices:
            cv2.putText(vis_frame, str(i), (int(x), int(y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Draw eyes/eyebrows/nose landmarks (blue)
    for i, (x, y) in enumerate(eyes_eyebrows_nose_landmarks):
        cv2.circle(vis_frame, (int(x), int(y)), 2, (255, 0, 0), -1)
        if show_indices:
            cv2.putText(vis_frame, str(i), (int(x), int(y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return vis_frame


def save_landmark_features(data: Dict, filepath: str) -> None:
    """
    Save landmark features to file.
    
    Args:
        data: Dictionary containing landmark data
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.suffix == '.npy':
        # Save as numpy array
        np.save(str(filepath), data)
    elif filepath.suffix == '.pkl':
        # Save as pickle
        with open(str(filepath), 'wb') as f:
            pickle.dump(data, f)
    elif filepath.suffix == '.csv':
        # Save as CSV (flattened landmarks)
        flattened_data = []
        for i, landmarks in enumerate(data['landmarks']):
            row = {'frame': i}
            if 'lip_landmarks' in landmarks:
                for j, (x, y) in enumerate(landmarks['lip_landmarks']):
                    row[f'lip_{j}_x'] = x
                    row[f'lip_{j}_y'] = y
            if 'contour_landmarks' in landmarks:
                for j, (x, y) in enumerate(landmarks['contour_landmarks']):
                    row[f'contour_{j}_x'] = x
                    row[f'contour_{j}_y'] = y
            if 'eyes_eyebrows_nose_landmarks' in landmarks:
                for j, (x, y) in enumerate(landmarks['eyes_eyebrows_nose_landmarks']):
                    row[f'eyes_eyebrows_nose_{j}_x'] = x
                    row[f'eyes_eyebrows_nose_{j}_y'] = y
            flattened_data.append(row)
        
        df = pd.DataFrame(flattened_data)
        df.to_csv(str(filepath), index=False)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def load_landmark_features(filepath: str) -> Dict:
    """
    Load landmark features from file.
    
    Args:
        filepath: Path to feature file
        
    Returns:
        Dictionary containing landmark data
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.npy':
        return np.load(str(filepath), allow_pickle=True).item()
    elif filepath.suffix == '.pkl':
        with open(str(filepath), 'rb') as f:
            return pickle.load(f)
    elif filepath.suffix == '.csv':
        df = pd.read_csv(str(filepath))
        # Convert back to original format
        landmarks = []
        for _, row in df.iterrows():
            frame_data = {}
            # Extract lip landmarks
            lip_coords = []
            for j in range(20):  # Assuming 20 lip landmarks
                if f'lip_{j}_x' in row and f'lip_{j}_y' in row:
                    lip_coords.append([row[f'lip_{j}_x'], row[f'lip_{j}_y']])
            frame_data['lip_landmarks'] = np.array(lip_coords)
            
            # Extract contour landmarks
            contour_coords = []
            for j in range(17):  # Assuming 17 contour landmarks
                if f'contour_{j}_x' in row and f'contour_{j}_y' in row:
                    contour_coords.append([row[f'contour_{j}_x'], row[f'contour_{j}_y']])
            frame_data['contour_landmarks'] = np.array(contour_coords)
            
            # Extract eyes/eyebrows/nose landmarks
            eyes_eyebrows_nose_coords = []
            for j in range(31):  # Assuming 31 eyes/eyebrows/nose landmarks
                if f'eyes_eyebrows_nose_{j}_x' in row and f'eyes_eyebrows_nose_{j}_y' in row:
                    eyes_eyebrows_nose_coords.append([row[f'eyes_eyebrows_nose_{j}_x'], row[f'eyes_eyebrows_nose_{j}_y']])
            frame_data['eyes_eyebrows_nose_landmarks'] = np.array(eyes_eyebrows_nose_coords)
            
            landmarks.append(frame_data)
        
        return {
            'landmarks': landmarks,
            'total_frames': len(landmarks)
        }
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def compute_landmark_features(landmarks_data: List[Dict]) -> np.ndarray:
    """
    Compute derived features from landmarks for lipreading.
    
    Args:
        landmarks_data: List of landmark dictionaries
        
    Returns:
        Feature matrix of shape (n_frames, n_features)
    """
    features = []
    
    for landmarks in landmarks_data:
        frame_features = []
        
        # Extract coordinates
        lip_landmarks = landmarks['lip_landmarks']
        contour_landmarks = landmarks['contour_landmarks']
        eyes_eyebrows_nose_landmarks = landmarks['eyes_eyebrows_nose_landmarks']
        
        # Feature 1: Lip landmark coordinates (flattened)
        if len(lip_landmarks) > 0:
            frame_features.extend(lip_landmarks.flatten())
        
        # Feature 2: Lip width and height
        if len(lip_landmarks) > 0:
            lip_width = np.max(lip_landmarks[:, 0]) - np.min(lip_landmarks[:, 0])
            lip_height = np.max(lip_landmarks[:, 1]) - np.min(lip_landmarks[:, 1])
            frame_features.extend([lip_width, lip_height])
        
        # Feature 3: Lip aspect ratio
        if len(lip_landmarks) > 0:
            lip_width = np.max(lip_landmarks[:, 0]) - np.min(lip_landmarks[:, 0])
            lip_height = np.max(lip_landmarks[:, 1]) - np.min(lip_landmarks[:, 1])
            aspect_ratio = lip_width / (lip_height + 1e-8)
            frame_features.append(aspect_ratio)
        
        # Feature 4: Distance from lip center to face center
        if len(lip_landmarks) > 0 and len(contour_landmarks) > 0:
            lip_center = np.mean(lip_landmarks, axis=0)
            face_center = np.mean(contour_landmarks, axis=0)
            distance = np.linalg.norm(lip_center - face_center)
            frame_features.append(distance)
        
        # Feature 5: Lip opening (vertical distance between upper and lower lip)
        if len(lip_landmarks) > 0:
            upper_lip_y = np.min(lip_landmarks[:, 1])
            lower_lip_y = np.max(lip_landmarks[:, 1])
            lip_opening = lower_lip_y - upper_lip_y
            frame_features.append(lip_opening)
        
        features.append(frame_features)
    
    return np.array(features)


def create_landmark_sequence_features(landmarks_data: List[Dict], 
                                    sequence_length: int = 10) -> np.ndarray:
    """
    Create sequence features from landmarks for temporal modeling.
    
    Args:
        landmarks_data: List of landmark dictionaries
        sequence_length: Length of sequences to create
        
    Returns:
        Feature sequences of shape (n_sequences, sequence_length, n_features)
    """
    # Compute frame-level features
    frame_features = compute_landmark_features(landmarks_data)
    
    # Create sequences
    sequences = []
    for i in range(len(frame_features) - sequence_length + 1):
        sequence = frame_features[i:i + sequence_length]
        sequences.append(sequence)
    
    return np.array(sequences)


def validate_landmarks(landmarks_data: Dict) -> bool:
    """
    Validate landmark data quality.
    
    Args:
        landmarks_data: Dictionary containing landmark data
        
    Returns:
        True if landmarks are valid, False otherwise
    """
    try:
        # Check if required keys exist
        required_keys = ['lip_landmarks', 'contour_landmarks', 'eyes_eyebrows_nose_landmarks']
        for key in required_keys:
            if key not in landmarks_data:
                return False
        
        # Check if landmarks are not empty
        for key in required_keys:
            if len(landmarks_data[key]) == 0:
                return False
        
        # Check if coordinates are valid (not NaN or infinite)
        for key in required_keys:
            coords = landmarks_data[key]
            if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
                return False
        
        return True
        
    except Exception:
        return False


def get_landmark_statistics(landmarks_data: List[Dict]) -> Dict:
    """
    Compute statistics about landmark data.
    
    Args:
        landmarks_data: List of landmark dictionaries
        
    Returns:
        Dictionary containing statistics
    """
    stats = {
        'total_frames': len(landmarks_data),
        'valid_frames': 0,
        'lip_landmark_counts': [],
        'contour_landmark_counts': [],
        'eyes_eyebrows_nose_landmark_counts': []
    }
    
    for landmarks in landmarks_data:
        if validate_landmarks(landmarks):
            stats['valid_frames'] += 1
            stats['lip_landmark_counts'].append(len(landmarks['lip_landmarks']))
            stats['contour_landmark_counts'].append(len(landmarks['contour_landmarks']))
            stats['eyes_eyebrows_nose_landmark_counts'].append(len(landmarks['eyes_eyebrows_nose_landmarks']))
    
    # Compute averages
    if stats['lip_landmark_counts']:
        stats['avg_lip_landmarks'] = np.mean(stats['lip_landmark_counts'])
        stats['avg_contour_landmarks'] = np.mean(stats['contour_landmark_counts'])
        stats['avg_eyes_eyebrows_nose_landmarks'] = np.mean(stats['eyes_eyebrows_nose_landmark_counts'])
    
    return stats
