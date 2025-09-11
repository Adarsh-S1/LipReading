import whisperx
#import gc
import torch
import os
import subprocess
import json
import math
import glob
#import secrets as sc
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ### NEW ###: Specify input and output directories
input_video_dir = "D:/ADARSH/dataset/"  # <--- IMPORTANT: Change this to your folder of videos!
output_transcript_dir = "transcriptss/" # <--- IMPORTANT: Change this to where you want transcripts saved!

batch_size = 16  # reduce if low on GPU mem
compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)

# IMPORTANT: Replace YOUR_HF_TOKEN with your actual Hugging Face token for diarization
# You can get one from https://huggingface.co/settings/tokens
YOUR_HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# --- Function to convert video to audio using FFmpeg ---
def convert_video_to_audio(video_input_path, audio_output_path):
    """
    Converts a video file to an MP3 audio file using FFmpeg.
    """
    if not os.path.exists(video_input_path):
        print(f"Error: Video file not found at '{video_input_path}'")
        return None
    command = [
        "ffmpeg", "-i", video_input_path, "-y", "-vn",
        "-acodec", "libmp3lame", "-q:a", "2", audio_output_path
    ]
    print(f"Converting '{os.path.basename(video_input_path)}' to audio...")
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully converted to '{os.path.basename(audio_output_path)}'")
        return audio_output_path
    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg conversion for {os.path.basename(video_input_path)}:")
        print(f"STDERR: {e.stderr.decode()}")
        return None
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please install FFmpeg and ensure it's in your system's PATH.")
        return None

# --- Function to get video FPS using ffprobe ---
def get_video_fps(video_path):
    """
    Gets the frame rate of a video file using ffprobe.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return None
    try:
        command = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate", "-of", "json", video_path
        ]
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        data = json.loads(result.stdout)
        frame_rate_str = data["streams"][0]["r_frame_rate"]
        num, den = map(int, frame_rate_str.split('/'))
        fps = num / den
        print(f"Detected FPS: {fps}")
        return fps
    except Exception as e:
        print(f"Error getting FPS from video '{os.path.basename(video_path)}': {e}")
        return None

# --- Function to save transcript with frame numbers ---
def save_transcript_with_frames(transcription_result, fps, output_path):
    """
    Saves the transcript with start and end frames for each word.
    """
    print(f"Saving frame-by-frame transcript to '{os.path.basename(output_path)}'...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment in transcription_result.get("segments", []):
            for word_info in segment.get("words", []):
                if 'start' in word_info and 'end' in word_info:
                    start_time = word_info['start']
                    end_time = word_info['end']
                    word = word_info['word']
                    start_frame = int(start_time * fps)
                    end_frame = int(math.ceil(end_time * fps))
                    f.write(f"{start_frame} {end_frame} {word.strip()}\n")
    print("Successfully saved transcript.")

# --- Main processing logic for a single video ---
def process_video_and_transcribe(video_path, audio_output_path, transcript_output_path):
    """
    Runs the full transcription pipeline for a single video file.
    """
    # Get video FPS first
    fps = get_video_fps(video_path)
    if not fps:
        print(f"Could not determine video FPS for {os.path.basename(video_path)}. Skipping file.")
        return

    # Convert video to audio
    audio_file_path = convert_video_to_audio(video_path, audio_output_path)
    if not audio_file_path:
        print(f"Audio conversion failed for {os.path.basename(video_path)}, skipping.")
        return

    # 1. Transcribe with original whisper (batched)
    print("\n--- Step 1: Loading Whisper model and transcribing ---")
    # Load model only once if possible, but for simplicity in a loop, we load it here.
    # For large batches, consider loading the model outside the main loop.
    model = whisperx.load_model("kurianbenoy/vegam-whisper-medium-ml", device, compute_type=compute_type)
    #model=WhisperModel("D:/vegam-whisper-medium-ml", device="cuda", compute_type="float16")
    audio = whisperx.load_audio(audio_file_path)
    result = model.transcribe(audio, batch_size=batch_size)
    del model
    #if device == "cuda": gc.collect(); torch.cuda.empty_cache()

    # 2. Align whisper output
    print("\n--- Step 2: Loading alignment model and aligning segments ---")
    try:
        model_a, metadata = whisperx.load_align_model(language_code="ml", device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        del model_a
        #if device == "cuda": gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error during alignment for {os.path.basename(video_path)}: {e}")
        if os.path.exists(audio_file_path): os.remove(audio_file_path)
        return

    # Save the frame-based transcript
    save_transcript_with_frames(result, fps, transcript_output_path)

# --- ### NEW ###: Main execution block to loop through videos ---
if __name__ == "__main__":
    # Create the output directory if it doesn't exist
    os.makedirs(output_transcript_dir, exist_ok=True)

    # Find all video files in the input directory
    video_files = []
    supported_formats = ["*.mp4", "*.mov", "*.avi", "*.mkv"]
    for fmt in supported_formats:
        video_files.extend(glob.glob(os.path.join(input_video_dir, fmt)))

    if not video_files:
        print(f"No video files found in '{input_video_dir}'. Please check the path.")
        exit()

    print(f"Found {len(video_files)} video(s) to process.")
    print("-" * 50)

    # Process each video file
    for i, video_path in enumerate(video_files):
        print(f"\nProcessing video {i+1} of {len(video_files)}: {os.path.basename(video_path)}")
        print("=" * 50)

        # Define output paths for the current video
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        transcript_path = os.path.join(output_transcript_dir, f"{base_name}.txt")
        temp_audio_path = os.path.join(output_transcript_dir, f"{base_name}_temp_audio.mp3")

        # Run the full process for the single video
        process_video_and_transcribe(video_path, temp_audio_path, transcript_path)
        print("-" * 50)

    print("\nAll videos have been processed.")
