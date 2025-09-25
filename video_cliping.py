import os
import torch
from moviepy.editor import VideoFileClip
import cv2
import mediapipe as mp
from pyannote.audio import Pipeline
from huggingface_hub import login
import subprocess
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
OUTPUT_CLIPS_DIR = "D:/ADARSH/New folder (2)/clips"
TEMP_AUDIO_DIR = "temp_audio/"
video_directory = "D:/ADARSH/New folder (2)/video"

# Load environment variables (recommended for secrets)
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# --- Constants ---
MIN_CLIP_DURATION = 0.5

# --- MediaPipe Initialization ---
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

# --- Functions ---
def has_face(video_clip, start_sec, end_sec):
    """Checks if a face is present in a few sample frames of a video segment."""
    safe_end_sec = max(start_sec, end_sec - 0.1)
    timestamps_to_check = [start_sec, (start_sec + safe_end_sec) / 2, safe_end_sec]

    for t in timestamps_to_check:
        try:
            frame = video_clip.get_frame(t)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(frame_rgb)
            if not results.detections:
                print(f"  -> No face found at timestamp {t:.2f}s. Discarding segment.")
                return False
        except Exception as e:
            print(f"  -> Error processing frame at {t:.2f}s: {e}. Discarding segment.")
            return False

    print(f"  -> Face confirmed for segment {start_sec:.2f}s - {end_sec:.2f}s.")
    return True

def extract_audio_with_ffmpeg(video_path, temp_audio_path):
    """Uses a direct FFmpeg command to extract audio into a WAV file."""
    print("Extracting audio with FFmpeg...")
    command = [
        "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", "-y", temp_audio_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print("Audio extraction successful.")
    except subprocess.CalledProcessError as e:
        print("FFmpeg Error! Could not extract audio.")
        print(f"Stderr: {e.stderr.decode()}")
        raise

# --- Main Logic ---
def main(input_video_path):
    """Main function to process a single video."""
    if not HUGGING_FACE_TOKEN:
        print("Hugging Face token not found. Please set the HUGGING_FACE_TOKEN environment variable.")
        return

    os.makedirs(OUTPUT_CLIPS_DIR, exist_ok=True)
    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
    
    base_filename = os.path.basename(input_video_path)
    temp_audio_path = os.path.join(TEMP_AUDIO_DIR, f"temp_{base_filename}.wav")

    all_segments = []
    try:
        try:
            login(token=HUGGING_FACE_TOKEN)
            print("Successfully logged into Hugging Face Hub.")
        except Exception as e:
            print(f"Hugging Face login failed. Error: {e}")
            return

        print("Initializing Voice Activity Detection pipeline...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection", use_auth_token=HUGGING_FACE_TOKEN
        ).to(torch.device(device))
        
        extract_audio_with_ffmpeg(input_video_path, temp_audio_path)

        print("Running VAD on the extracted audio...")
        speech_activity = pipeline(temp_audio_path)
        print("VAD complete. Found potential speech segments.")
        
        all_segments = list(speech_activity.itersegments())
        
    finally:
        if os.path.exists(temp_audio_path):
            print(f"Cleaning up temporary file: {temp_audio_path}")
            os.remove(temp_audio_path)
            
    if not all_segments:
        print("No speech segments found in the video.")
        return

    total_segments = len(all_segments)
    print(f"Found {total_segments} speech segments. Now processing and clipping...")
    clip_count = 0

    with VideoFileClip(input_video_path) as video:
        for i, segment in enumerate(all_segments):
            start_sec, end_sec = segment.start, segment.end
            duration = end_sec - start_sec

            print(f"\nProcessing segment {i+1}/{total_segments}: {start_sec:.2f}s to {end_sec:.2f}s (Duration: {duration:.2f}s)")

            if duration < MIN_CLIP_DURATION:
                print("  -> Segment is too short. Skipping.")
                continue
            
            if has_face(video, start_sec, end_sec):
                clip_count += 1
                
                # --- START OF FIX ---
                # Create a video-only subclip first
                new_clip = video.subclip(start_sec, end_sec)
                # Then create a separate, independent audio subclip and attach it
                new_clip.audio = video.audio.subclip(start_sec, end_sec)

                output_filename = os.path.join(
                    OUTPUT_CLIPS_DIR,
                    f"clip_{clip_count:04d}_{start_sec:.2f}s_to_{end_sec:.2f}s.mp4"
                )

                print(f"  -> SAVING CLIP: {output_filename}")
                # Write the new, self-contained clip to a file
                new_clip.write_videofile(output_filename, codec="libx264", audio_codec="aac", logger=None, threads=4)

                # Manually close the subclip to free up memory
                #new_clip.close()
                # --- END OF FIX ---

    print(f"\n--- Processing Complete for {base_filename} ---")
    print(f"Successfully saved {clip_count} valid clips to '{OUTPUT_CLIPS_DIR}'.")

if __name__ == "__main__":
    if not os.path.isdir(video_directory):
        print(f"Error: Directory not found at '{video_directory}'")
    else:
        for filename in os.listdir(video_directory):
            # FIX: Construct the full path to the video file
            full_video_path = os.path.join(video_directory, filename)
            
            # Optional but recommended: Check if it's a video file
            if full_video_path.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                print(f"\n{'='*50}\n--- Starting processing for: {filename} ---\n{'='*50}")
                main(full_video_path)
            else:
                print(f"\nSkipping non-video file: {filename}")
    
    # Clean up at the very end
    face_detector.close()