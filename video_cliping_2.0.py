# import os
# import torch
# import cv2
# import mediapipe as mp
# from pyannote.audio import Pipeline
# from huggingface_hub import login
# import subprocess
# from dotenv import load_dotenv
# import time
# from tqdm import tqdm # For a nice progress bar
# import warnings
# warnings.filterwarnings('ignore') 

# load_dotenv()

# # --- Configuration ---
# OUTPUT_CLIPS_DIR = "D:/ADARSH/New folder (2)/clips"
# TEMP_AUDIO_DIR = "temp_audio/"
# VIDEO_DIRECTORY = "D:/ADARSH/New folder (2)/video"

# # Load environment variables
# HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# # --- Constants ---
# MIN_CLIP_DURATION = 0.5
# FACE_DETECTION_INTERVAL = 0.5  # Check for a face every 0.5 seconds

# # --- OPTIMIZATION: Batch process the video to find all timestamps with faces ---
# def get_face_timestamps(video_path, face_detector):
#     """
#     Scans the entire video once to find all timestamps where faces are present.
#     Returns a set of timestamps for fast lookups.
#     """
#     face_timestamps = set()
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error opening video file: {video_path}")
#         return face_timestamps

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     if fps == 0:
#         print(f"Could not determine FPS for {video_path}. Assuming 30.")
#         fps = 30 # Provide a default fallback
        
#     frame_interval = int(fps * FACE_DETECTION_INTERVAL)
#     frame_count = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         if frame_count % frame_interval == 0:
#             try:
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results = face_detector.process(frame_rgb)
#                 if results.detections:
#                     current_time_sec = frame_count / fps
#                     # Mark the current timestamp as having a face
#                     face_timestamps.add(round(current_time_sec, 2))
#             except Exception as e:
#                 print(f"Error during face detection on frame {frame_count}: {e}")

#         frame_count += 1
        
#     cap.release()
#     return face_timestamps

# # --- OPTIMIZATION: Use direct FFmpeg for fast clipping ---
# def clip_with_ffmpeg(input_video, output_video, start_sec, end_sec):
#     """
#     Uses a direct FFmpeg command for fast, reliable clipping.
#     """
#     command = [
#         "ffmpeg",
#         "-i", input_video,
#         "-ss", str(start_sec),
#         "-to", str(end_sec),
#         "-c:v", "libx24", # Re-encode video for clean cuts
#         "-c:a", "aac",    # Re-encode audio
#         "-y",             # Overwrite output file if it exists
#         "-loglevel", "error", # Suppress verbose output
#         output_video
#     ]
#     try:
#         subprocess.run(command, check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"FFmpeg clipping failed for {output_video}. Error: {e}")

# def extract_audio_with_ffmpeg(video_path, temp_audio_path):
#     """Uses FFmpeg to extract audio into a WAV file."""
#     command = [
#         "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
#         "-ar", "16000", "-ac", "1", "-y", "-loglevel", "error", temp_audio_path
#     ]
#     try:
#         subprocess.run(command, check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"FFmpeg Error! Could not extract audio from {video_path}.")
#         raise

# # --- Main processing logic for a single video ---
# def process_video(video_path, vad_pipeline, face_detector):
#     base_filename = os.path.basename(video_path)
#     print(f"\n--- Processing: {base_filename} ---")
    
#     # --- 1. Voice Activity Detection ---
#     temp_audio_path = os.path.join(TEMP_AUDIO_DIR, f"temp_{base_filename}.wav")
#     try:
#         extract_audio_with_ffmpeg(video_path, temp_audio_path)
        
#         # --- FIX STARTS HERE ---
#         # The pipeline returns an 'Annotation' object, not a list
#         annotation = vad_pipeline(temp_audio_path)
#         # You must call .itersegments() to get the speech segments
#         speech_timestamps = [(seg.start, seg.end) for seg in annotation.itersegments()]
#         # --- FIX ENDS HERE ---

#     except Exception as e:
#         print(f"Could not perform VAD on {base_filename}. Error: {e}")
#         return 0
#     finally:
#         if os.path.exists(temp_audio_path):
#             os.remove(temp_audio_path)
            
#     if not speech_timestamps:
#         print(f"No speech found in {base_filename}.")
#         return 0

#     # --- 2. Batched Face Detection ---
#     print("Scanning video for face timestamps...")
#     face_timestamps = get_face_timestamps(video_path, face_detector)
#     if not face_timestamps:
#         print(f"No faces found in {base_filename}.")
#         return 0
        
#     # --- 3. Find Overlapping Segments & Clip ---
#     clip_count = 0
#     print("Checking for segments with both speech and faces...")
#     for i, (start_sec, end_sec) in enumerate(speech_timestamps):
#         duration = end_sec - start_sec
#         if duration < MIN_CLIP_DURATION:
#             continue

#         # Check for overlap: Is any point in the speech segment close to a known face timestamp?
#         has_overlap = False
#         for t in face_timestamps:
#             if start_sec - FACE_DETECTION_INTERVAL <= t <= end_sec + FACE_DETECTION_INTERVAL:
#                 has_overlap = True
#                 break
        
#         if has_overlap:
#             clip_count += 1
#             output_filename = os.path.join(
#                 OUTPUT_CLIPS_DIR,
#                 f"{os.path.splitext(base_filename)[0]}_clip_{clip_count:03d}.mp4"
#             )
#             clip_with_ffmpeg(video_path, output_filename, start_sec, end_sec)

#     print(f"--- Finished {base_filename}. Saved {clip_count} clips. ---")
#     return clip_count

# if __name__ == "__main__":
#     if not HUGGING_FACE_TOKEN:
#         print("Hugging Face token not found. Please set HUGGING_FACE_TOKEN in your .env file.")
#         exit()

#     os.makedirs(OUTPUT_CLIPS_DIR, exist_ok=True)
#     os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
    
#     # --- OPTIMIZATION 1: Load all models ONCE at the start ---
#     print("Initializing AI models... (this may take a moment)")
#     try:
#         login(token=HUGGING_FACE_TOKEN)
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         vad_pipeline = Pipeline.from_pretrained(
#             "pyannote/voice-activity-detection", use_auth_token=HUGGING_FACE_TOKEN
#         ).to(torch.device(device))
        
#         mp_face_detection = mp.solutions.face_detection
#         face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.6)
#         print(f"✅ Models loaded successfully on device: {device}")
#     except Exception as e:
#         print(f"Fatal error: Could not load models. Error: {e}")
#         exit()

#     video_files = [
#         os.path.join(VIDEO_DIRECTORY, f) for f in os.listdir(VIDEO_DIRECTORY)
#         if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
#     ]

#     if not video_files:
#         print(f"No video files found in '{VIDEO_DIRECTORY}'")
#         exit()
        
#     total_clips_saved = 0
#     start_time = time.time()
    
#     # --- Process files sequentially for maximum stability ---
#     for video_path in tqdm(video_files, desc="Processing Videos"):
#         # Pass the pre-loaded models to the processing function
#         clips_saved = process_video(video_path, vad_pipeline, face_detector)
#         total_clips_saved += clips_saved
        
#     end_time = time.time()
    
#     # --- Clean up at the very end ---
#     face_detector.close()

#     print("\n" + "="*50)
#     print("✅ All Videos Processed!")
#     print(f"Total clips saved: {total_clips_saved}")
#     print(f"Total execution time: {end_time - start_time:.2f} seconds")
#     print("="*50)



import os
import torch
import cv2
import mediapipe as mp
import subprocess
from dotenv import load_dotenv
import time
from tqdm import tqdm
import multiprocessing

load_dotenv()

# --- Configuration ---
OUTPUT_CLIPS_DIR = "D:/ADARSH/New folder (2)/clips"
TEMP_AUDIO_DIR = "temp_audio/"
VIDEO_DIRECTORY = "D:/ADARSH/New folder (2)/video"

# --- Constants ---
MIN_CLIP_DURATION = 0.5
FACE_DETECTION_INTERVAL = 0.5  # Check for a face every 0.5 seconds
VAD_SAMPLING_RATE = 16000      # Silero VAD expects 16000Hz audio

# --- Helper Functions (largely unchanged) ---

def get_face_timestamps(video_path, face_detector):
    """
    Scans the entire video once to find all timestamps where faces are present.
    Returns a set of timestamps for fast lookups.
    """
    face_timestamps = set()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return face_timestamps

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Could not determine FPS for {video_path}. Assuming 30.")
        fps = 30 # Provide a default fallback
        
    frame_interval = int(fps * FACE_DETECTION_INTERVAL)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detector.process(frame_rgb)
                if results.detections:
                    current_time_sec = frame_count / fps
                    face_timestamps.add(round(current_time_sec, 2))
            except Exception as e:
                print(f"Error during face detection on frame {frame_count}: {e}")

        frame_count += 1
        
    cap.release()
    return face_timestamps

def clip_with_ffmpeg(input_video, output_video, start_sec, end_sec):
    """
    Uses a direct FFmpeg command with the 'ultrafast' preset for speed.
    """
    command = [
        "ffmpeg",
        "-i", input_video,
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        "-y",
        "-loglevel", "error",
        output_video
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg clipping failed for {output_video}. Error: {e}")

def extract_audio_with_ffmpeg(video_path, temp_audio_path):
    """Uses FFmpeg to extract audio into a WAV file at the required sample rate."""
    command = [
        "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
        "-ar", str(VAD_SAMPLING_RATE), "-ac", "1", "-y", "-loglevel", "error", temp_audio_path
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error! Could not extract audio from {video_path}.")
        raise

# --- Main processing logic for a single video ---

def process_video(video_path, silero_model, silero_utils, face_detector):
    """
    Processes a single video file using Silero VAD for voice detection.
    """
    base_filename = os.path.basename(video_path)
    
    # --- 1. Voice Activity Detection with Silero VAD ---
    temp_audio_path = os.path.join(TEMP_AUDIO_DIR, f"temp_{base_filename}.wav")
    speech_timestamps = []
    try:
        extract_audio_with_ffmpeg(video_path, temp_audio_path)
        
        # Unpack the Silero utility functions
        (get_speech_timestamps, _, read_audio, *_) = silero_utils

        # Read the audio file
        wav = read_audio(temp_audio_path, sampling_rate=VAD_SAMPLING_RATE)
        
        # Get speech segments from Silero VAD
        # The output is a list of dicts with 'start' and 'end' in samples
        speech_segments_samples = get_speech_timestamps(wav, silero_model, sampling_rate=VAD_SAMPLING_rate)
        
        # Convert samples to seconds for the rest of the script
        for seg in speech_segments_samples:
            start_sec = seg['start'] / VAD_SAMPLING_RATE
            end_sec = seg['end'] / VAD_SAMPLING_RATE
            speech_timestamps.append((start_sec, end_sec))

    except Exception as e:
        print(f"Could not perform VAD on {base_filename}. Error: {e}")
        return 0
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            
    if not speech_timestamps:
        print(f"No speech found in {base_filename}.")
        return 0

    # --- 2. Batched Face Detection ---
    face_timestamps = get_face_timestamps(video_path, face_detector)
    if not face_timestamps:
        print(f"No faces found in {base_filename}.")
        return 0
        
    # --- 3. Find Overlapping Segments & Clip ---
    clip_count = 0
    for i, (start_sec, end_sec) in enumerate(speech_timestamps):
        duration = end_sec - start_sec
        if duration < MIN_CLIP_DURATION:
            continue

        has_overlap = False
        for t in face_timestamps:
            if start_sec - FACE_DETECTION_INTERVAL <= t <= end_sec + FACE_DETECTION_INTERVAL:
                has_overlap = True
                break
        
        if has_overlap:
            clip_count += 1
            output_filename = os.path.join(
                OUTPUT_CLIPS_DIR,
                f"{os.path.splitext(base_filename)[0]}_clip_{clip_count:03d}.mp4"
            )
            clip_with_ffmpeg(video_path, output_filename, start_sec, end_sec)

    print(f"--- Processed {base_filename}. Saved {clip_count} clips. ---")
    return clip_count

def worker(video_path):
    """
    A self-contained worker function that initializes its own models
    and processes a single video file.
    """
    # Each process initializes its own models to avoid conflicts.
    # Initialize Silero VAD
    try:
        silero_model, silero_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=True
        )
    except Exception as e:
        print(f"Failed to load Silero VAD model: {e}")
        return 0
    
    # Initialize Face Detector
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.6)
    
    # Process the video
    clips_saved = process_video(video_path, silero_model, silero_utils, face_detector)
    
    # Clean up
    face_detector.close()
    
    return clips_saved

# --- Main execution block with Parallel Processing ---

if __name__ == "__main__":
    os.makedirs(OUTPUT_CLIPS_DIR, exist_ok=True)
    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
    
    # Set the multiprocessing start method for compatibility (especially on Windows/macOS)
    multiprocessing.set_start_method('spawn', force=True)

    video_files = [
        os.path.join(VIDEO_DIRECTORY, f) for f in os.listdir(VIDEO_DIRECTORY)
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
    ]

    if not video_files:
        print(f"No video files found in '{VIDEO_DIRECTORY}'")
        exit()
        
    try:
        num_cores = os.cpu_count()
        num_workers = max(1, min(num_cores // 2, 4)) 
    except NotImplementedError:
        num_workers = 2
    
    print(f"\n🚀 Starting parallel processing with {num_workers} workers...")
    
    start_time = time.time()
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(worker, video_files), total=len(video_files), desc="Processing Videos"))

    total_clips_saved = sum(results)
    end_time = time.time()
    
    print("\n" + "="*50)
    print("✅ All Videos Processed!")
    print(f"Total clips saved: {total_clips_saved}")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print("="*50)