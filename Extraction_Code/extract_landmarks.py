"""
Facial Landmark Extraction for Lipreading Model (Batch Processing)
==================================================================

This module handles the extraction of facial landmarks from multiple video files
in a directory for lipreading model training.

This version is modified to be directly compatible with the LipFormer model
which expects a combined landmark array of shape (T, 37, 2).

Author: Enhanced Lipreading System
Date: 2024
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import logging

# --- Utility Functions ---
# NOTE: These are now functional replacements for the previous placeholders.

def extract_frames_from_video(video_path: str) -> List[np.ndarray]:
    """Extracts all frames from a video file."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def create_landmark_visualization(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Draws the extracted landmarks on a frame for visualization."""
    vis_frame = frame.copy()
    for (x, y) in landmarks:
        cv2.circle(vis_frame, (x, y), 2, (0, 255, 0), -1)
    return vis_frame

# -----------------------------------------

class LandmarkExtractor:
    """
    Main class for extracting facial landmarks from videos for the LipFormer model.
    
    Features:
    - Extracts 37 specific landmarks (20 lip + 17 contour) using MediaPipe.
    - Outputs a single NumPy array of shape (T, 37, 2) per video.
    """
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 refine_landmarks: bool = True):
        """
        Initialize the landmark extractor.
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.refine_landmarks = refine_landmarks
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Define landmark indices required by the LipFormer model
        self.model_specific_indices = self._get_model_specific_indices()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _get_model_specific_indices(self) -> List[int]:
        """
        Get the 37 landmark indices (20 lip + 17 contour) required by my_model.py.
        NOTE: These indices are based on the Dlib 68-point convention mapped to MediaPipe.
        Adjust if your model's "20 lip" and "17 contour" follow a different convention.
        """
        # Facial Contour Indices (17 points)
        contour_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400
        ]
        
        # Lip Indices (20 points)
        lip_indices = [
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
            76, 185, 40, 39, 37, 0, 267, 269, 270
        ]
        
        # Combine and return the 37 required indices
        return contour_indices + lip_indices
    
    def extract_landmarks_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the 37 required landmarks from a single frame.
        Returns a NumPy array of shape (37, 2) or None if no face is detected.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark
        frame_height, frame_width = frame.shape[:2]
        
        coords = []
        for idx in self.model_specific_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                x = int(lm.x * frame_width)
                y = int(lm.y * frame_height)
                coords.append([x, y])
        
        if len(coords) != 37:
            self.logger.warning(f"Expected 37 landmarks but found {len(coords)}. Skipping frame.")
            return None

        return np.array(coords)
    
    def extract_landmarks_from_video(self, video_path: str, 
                                   output_dir: str,
                                   save_visualization: bool = True) -> Optional[Dict]:
        """
        Extract landmarks from an entire video file and save them as a single .npy file.
        """
        self.logger.info(f"Processing video: {video_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        try:
            frames = extract_frames_from_video(video_path)
            if not frames:
                self.logger.error(f"Could not extract any frames from {video_path}. Skipping.")
                return None
        except Exception as e:
            self.logger.error(f"Error reading video file {video_path}: {e}")
            return None

        self.logger.info(f"Extracted {len(frames)} frames from video")
        
        all_landmarks = []
        video_name = Path(video_path).stem
        vis_output_dir = os.path.join(output_dir, f"{video_name}_visualization")
        
        if save_visualization:
            os.makedirs(vis_output_dir, exist_ok=True)

        for i, frame in enumerate(frames):
            landmarks_coords = self.extract_landmarks_from_frame(frame)
            
            if landmarks_coords is not None:
                all_landmarks.append(landmarks_coords)
                
                if save_visualization:
                    vis_frame = create_landmark_visualization(frame, landmarks_coords)
                    vis_path = os.path.join(vis_output_dir, f"frame_{i:06d}_landmarks.jpg")
                    cv2.imwrite(vis_path, vis_frame)
        
        if not all_landmarks:
            self.logger.warning(f"No landmarks could be extracted from video: {video_name}")
            return None
            
        # Stack landmarks into a single NumPy array: (T, 37, 2)
        final_landmarks_array = np.stack(all_landmarks, axis=0)
        self.logger.info(f"Successfully extracted landmarks from {final_landmarks_array.shape[0]} frames.")
        
        # Save the final array
        output_path = os.path.join(output_dir, f"{video_name}_landmarks.npy")
        np.save(output_path, final_landmarks_array)
        
        # Save metadata for reference
        metadata = {
            'video_name': video_name,
            'video_path': str(video_path),
            'total_frames_in_video': len(frames),
            'frames_with_landmarks': final_landmarks_array.shape[0],
            'output_shape': final_landmarks_array.shape,
            'output_path': output_path,
            'extraction_params': {
                'min_detection_confidence': self.min_detection_confidence,
                'min_tracking_confidence': self.min_tracking_confidence,
                'refine_landmarks': self.refine_landmarks
            }
        }
        
        metadata_path = os.path.join(output_dir, f"{video_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved landmarks for '{video_name}' to {output_path}")
        return metadata
    
    def close(self):
        """Close the MediaPipe face mesh."""
        self.face_mesh.close()

def process_videos_from_directory(input_dir: str, output_dir: str, config: Dict):
    """
    Finds all videos in an input directory and processes them one by one.
    """
    extractor = LandmarkExtractor(
        min_detection_confidence=config.get("min_detection_confidence", 0.5),
        min_tracking_confidence=config.get("min_tracking_confidence", 0.5),
        refine_landmarks=config.get("refine_landmarks", True)
    )
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [p for p in Path(input_dir).rglob('*') if p.suffix.lower() in video_extensions]
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    print(f"Found {len(video_files)} videos to process.")
    
    try:
        for video_path in video_files:
            print("-" * 50)
            extractor.extract_landmarks_from_video(
                video_path=str(video_path),
                output_dir=output_dir,
                save_visualization=config.get("save_visualization", True)
            )
        print("-" * 50)
        print("Batch processing complete.")
        
    finally:
        extractor.close()
        print("Landmark extractor closed.")

if __name__ == "__main__":
    INPUT_DIR = "D:/ADARSH/test_dataset"
    OUTPUT_DIR = "extracted_landmarks_model_ready"

    extractor_config = {
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
        "refine_landmarks": True,
        "save_visualization": True
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    process_videos_from_directory(INPUT_DIR, OUTPUT_DIR, extractor_config)