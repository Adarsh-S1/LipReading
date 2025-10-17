import os
import cv2
import numpy as np
import mediapipe as mp
import glob

# Subset of MediaPipe FaceMesh indices that outline the lips region
LIP_LANDMARKS = [
	61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
	14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311,
	312, 13, 82, 81, 42, 183, 78
]


def ensure_dir(path):
	"""Ensures that a directory exists, creating it if necessary."""
	if path and not os.path.isdir(path):
		os.makedirs(path, exist_ok=True)


def landmarks_to_xy(landmarks, width, height):
	"""Converts normalized MediaPipe landmark coordinates to pixel coordinates."""
	pts = []
	for lm in landmarks:
		pts.append((int(lm.x * width), int(lm.y * height)))
	return pts


def compute_bbox_around_points(points, pad_px, frame_w, frame_h):
	"""Computes a bounding box around a list of points with padding."""
	xs = [p[0] for p in points]
	ys = [p[1] for p in points]
	x1 = max(0, min(xs) - pad_px)
	y1 = max(0, min(ys) - pad_px)
	x2 = min(frame_w - 1, max(xs) + pad_px)
	y2 = min(frame_h - 1, max(ys) + pad_px)
	return x1, y1, x2, y2


def process_single_video(input_video, out_dir, roi_size=128, pad_px=10, min_conf=0.5):
	"""
	Processes a single video file to extract lip ROIs.
	This was the original 'run' function.
	"""
	cap = cv2.VideoCapture(input_video)
	if not cap.isOpened():
		print(f"Error: Cannot open video: {input_video}")
		return

	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	ensure_dir(out_dir)
	
	mp_face_mesh = mp.solutions.face_mesh
	mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=min_conf, min_tracking_confidence=min_conf)

	idx = 0
	while True:
		ret, frame = cap.read()
		if not ret:
			break

		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		res = mesh.process(frame_rgb)
		if not res.multi_face_landmarks:
			continue

		landmarks = res.multi_face_landmarks[0].landmark
		lip_pts = [landmarks[i] for i in LIP_LANDMARKS]
		lip_xy = landmarks_to_xy(lip_pts, width, height)
		x1, y1, x2, y2 = compute_bbox_around_points(lip_xy, pad_px, width, height)
		roi = frame[y1:y2, x1:x2]
		
		if roi.size == 0:
			continue
			
		roi_resized = cv2.resize(roi, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)

		# Save image sequence
		out_path = os.path.join(out_dir, f"frame_{idx:06d}.png")
		cv2.imwrite(out_path, roi_resized)
		idx += 1

	cap.release()
	mesh.close()
	print(f"✅ Finished processing {os.path.basename(input_video)}. Frames saved to {out_dir}")


if __name__ == "__main__":
	# --- 1. SET YOUR FOLDERS HERE ---
	# Directory containing all your input videos
	INPUT_VIDEO_DIR = "D:/ADARSH/test_dataset" 
	
	# Directory where the output folders will be saved
	OUTPUT_FRAMES_DIR = "extracted_lip_crosssection"

	# --- 2. SET YOUR PARAMETERS HERE ---
	ROI_SIZE = 128   # The size of the output cropped images (e.g., 128x128 pixels)
	PADDING = 10     # Padding in pixels around the detected lips
	CONFIDENCE = 0.5 # Detection confidence for MediaPipe

	# --- 3. SCRIPT EXECUTION ---
	# List of common video file extensions to look for
	video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
	
	video_paths = []
	for ext in video_extensions:
		video_paths.extend(glob.glob(os.path.join(INPUT_VIDEO_DIR, ext)))

	if not video_paths:
		print(f"❌ No videos found in the specified directory: {INPUT_VIDEO_DIR}")
	else:
		print(f"Found {len(video_paths)} videos to process.")
		
	# Process each video found
	for video_path in video_paths:
		video_name = os.path.basename(video_path)
		video_name_without_ext = os.path.splitext(video_name)[0]
		
		# Create a specific output folder for this video's frames
		output_subdirectory = os.path.join(OUTPUT_FRAMES_DIR, video_name_without_ext)
		
		print(f"\nProcessing video: {video_name}...")
		
		process_single_video(
			input_video=video_path,
			out_dir=output_subdirectory,
			roi_size=ROI_SIZE,
			pad_px=PADDING,
			min_conf=CONFIDENCE
		)
		
	print("\nAll videos have been processed.")