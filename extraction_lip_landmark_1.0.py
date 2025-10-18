"""
Combined and Parallelized Data Extraction for Lip-Reading
=========================================================

This script integrates the functionalities of lip ROI (Region of Interest)
extraction and facial landmark extraction into a single, efficient pipeline.

Features:
- Processes each video file only ONCE to extract both data types.
- Uses multiprocessing to process multiple videos in parallel for significant speed-up.
- Extracts 37 specific landmarks (20 lip + 17 contour) for the LipFormer model.
- Extracts cropped lip ROI images for the visual stream.
- All settings are managed from a central CONFIG dictionary.

Author: Combined from original scripts
Date: 2024
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# --- 1. Main Configuration ---
CONFIG = {
    # --- Path Settings ---
    "input_dir": "D:/ADARSH/processed_clips",
    "output_lip_roi_dir": "D:/ADARSH/extracted_lip_crosssection",
    "output_landmarks_dir": "D:/ADARSH/extracted_landmarks_model_ready",

    # --- ROI Settings ---
    "roi_width": 160,       # Target width for the lip crop (for my_model.py)
    "roi_height": 80,       # Target height for the lip crop (for my_model.py)
    "roi_padding": 9,      # Padding in pixels around detected lips

    # --- MediaPipe Settings ---
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,

    # --- Processing Settings ---
    "save_visualization": False, # Set to True to save frames with landmarks drawn on them
    # Use all available CPU cores minus one, or set to a specific number e.g., 4
    "num_workers": max(1, cpu_count() - 1),
}


# --- 2. Landmark Definitions ---

# Indices for the 37 landmarks (17 contour + 20 lip) required by my_model.py
MODEL_LANDMARK_INDICES = [
    # Facial Contour (17 points)
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400,
    # Lips (20 points)
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    76, 185, 40, 39, 37, 0, 267, 269, 270
]

# Indices for the lip outline to calculate the bounding box for the ROI crop
LIP_OUTLINE_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311,
    312, 13, 82, 81, 42, 183, 78
]


def process_video(video_path: Path, config: Dict):
    """
    Worker function to process a single video file.
    This function is executed by each process in the multiprocessing pool.
    """
    video_name = video_path.stem
    logging.info(f"[{video_name}] Starting processing...")

    # --- Setup output directories for this specific video ---
    lip_roi_output_dir = Path(config["output_lip_roi_dir"]) / video_name
    landmark_output_dir = Path(config["output_landmarks_dir"])
    vis_output_dir = landmark_output_dir / f"{video_name}_visualization"

    os.makedirs(lip_roi_output_dir, exist_ok=True)
    os.makedirs(landmark_output_dir, exist_ok=True)
    if config["save_visualization"]:
        os.makedirs(vis_output_dir, exist_ok=True)

    # --- Initialize MediaPipe ---
    # Each process needs its own instance of FaceMesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=config["min_detection_confidence"],
        min_tracking_confidence=config["min_tracking_confidence"]
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"[{video_name}] Error: Cannot open video.")
        return

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    all_model_landmarks = []
    frame_idx = 0
    frames_processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            frames_processed += 1
            face_landmarks = results.multi_face_landmarks[0].landmark

            # --- Task 1: Extract and store the 37 model-specific landmarks ---
            model_coords = []
            for idx in MODEL_LANDMARK_INDICES:
                lm = face_landmarks[idx]
                x, y = int(lm.x * frame_width), int(lm.y * frame_height)
                model_coords.append([x, y])
            all_model_landmarks.append(np.array(model_coords))

            # --- Task 2: Extract and save the lip ROI ---
            lip_outline_coords = []
            for idx in LIP_OUTLINE_INDICES:
                lm = face_landmarks[idx]
                x, y = int(lm.x * frame_width), int(lm.y * frame_height)
                lip_outline_coords.append((x, y))

            xs = [p[0] for p in lip_outline_coords]
            ys = [p[1] for p in lip_outline_coords]
            pad = config["roi_padding"]
            x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
            x2, y2 = min(frame_width, max(xs) + pad), min(frame_height, max(ys) + pad)

            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                roi_resized = cv2.resize(
                    roi,
                    (config["roi_width"], config["roi_height"]),
                    interpolation=cv2.INTER_LINEAR
                )
                roi_path = lip_roi_output_dir / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(roi_path), roi_resized)

            # --- Task 3 (Optional): Save visualization ---
            if config["save_visualization"]:
                vis_frame = frame.copy()
                for (x, y) in model_coords:
                    cv2.circle(vis_frame, (x, y), 2, (0, 255, 0), -1)
                vis_path = vis_output_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(vis_path), vis_frame)

        frame_idx += 1

    # --- Finalize and save landmark data for the video ---
    if not all_model_landmarks:
        logging.warning(f"[{video_name}] No landmarks were extracted.")
    else:
        final_landmarks_array = np.stack(all_model_landmarks, axis=0)
        output_path = landmark_output_dir / f"{video_name}_landmarks.npy"
        np.save(output_path, final_landmarks_array)
        logging.info(f"[{video_name}] Saved landmarks array with shape {final_landmarks_array.shape} to {output_path}")

        # Save metadata
        metadata = {
            'video_name': video_name,
            'total_frames_in_video': frame_idx,
            'frames_with_landmarks': final_landmarks_array.shape[0],
            'output_shape': final_landmarks_array.shape,
            'output_path': str(output_path)
        }
        metadata_path = landmark_output_dir / f"{video_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    cap.release()
    face_mesh.close()
    return f"Finished processing {video_name}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Find all video files ---
    input_dir = Path(CONFIG["input_dir"])
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [p for p in input_dir.rglob('*') if p.suffix.lower() in video_extensions]

    if not video_files:
        logging.error(f"No video files found in {input_dir}")
    else:
        logging.info(f"Found {len(video_files)} videos to process using {CONFIG['num_workers']} workers.")

        # --- Create argument list for the multiprocessing pool ---
        tasks = [(path, CONFIG) for path in video_files]

        # --- Run the processing pool ---
        # starmap applies arguments from the 'tasks' list to the 'process_video' function
        with Pool(processes=CONFIG["num_workers"]) as pool:
            results = list(tqdm(pool.starmap(process_video, tasks), total=len(tasks), desc="Processing Videos"))
        
        logging.info("\n" + "="*50)
        logging.info("Batch processing complete.")
        logging.info("="*50)