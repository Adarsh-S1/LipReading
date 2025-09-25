import whisperx
import torch
import os
import subprocess
import json
import math
import glob
from dotenv import load_dotenv
import multiprocessing  # <<< NEW: Import multiprocessing

load_dotenv()

# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ### NEW ###: Specify input and output directories
input_video_dir = "D:/ADARSH/processed_clips/"
output_transcript_dir = "D:/ADARSH/transcripts/"

batch_size = 16
compute_type = "float16"

# <<< NEW >>>: Configure the number of parallel CPU workers
# A good starting point is the number of CPU cores minus one for the main process.
try:
    NUM_CPU_WORKERS = os.cpu_count() - 1
except:
    NUM_CPU_WORKERS = 3 # Fallback for safety

YOUR_HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")


# --- Helper functions (largely unchanged) ---

def convert_video_to_audio(video_input_path, audio_output_path):
    command = [
        "ffmpeg", "-i", video_input_path, "-y", "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return audio_output_path
    except Exception:
        return None

def get_video_fps(video_path):
    if not os.path.exists(video_path): return None
    try:
        command = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate", "-of", "json", video_path
        ]
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        data = json.loads(result.stdout)
        frame_rate_str = data["streams"][0]["r_frame_rate"]
        num, den = map(int, frame_rate_str.split('/'))
        return num / den
    except Exception:
        return None

def save_transcript_with_frames(transcription_result, fps, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment in transcription_result.get("segments", []):
            for word_info in segment.get("words", []):
                if 'start' in word_info and 'end' in word_info:
                    start_time, end_time, word = word_info['start'], word_info['end'], word_info['word']
                    start_frame = int(start_time * fps)
                    end_frame = int(math.ceil(end_time * fps))
                    f.write(f"{start_frame} {end_frame} {word.strip()}\n")

# --- <<< NEW >>>: CPU Worker Function ---
def cpu_prepare_audio(args):
    """
    This function runs in a separate CPU process.
    It handles file I/O and CPU-bound conversion.
    """
    video_path, temp_audio_path, transcript_path = args
    print(f"[CPU Worker] Processing: {os.path.basename(video_path)}")

    fps = get_video_fps(video_path)
    if not fps:
        print(f"[CPU Worker] Failed to get FPS for {os.path.basename(video_path)}")
        return None

    if not convert_video_to_audio(video_path, temp_audio_path):
        print(f"[CPU Worker] Failed to convert {os.path.basename(video_path)} to audio.")
        return None

    print(f"[CPU Worker] Finished preparing audio for: {os.path.basename(video_path)}")
    # Return all necessary info for the GPU process
    return (temp_audio_path, transcript_path, fps)

# --- <<< NEW >>>: GPU Consumer Function ---
def gpu_transcribe_and_align(audio_path, transcript_path, fps, model, align_model, align_metadata):
    """
    This function runs in the main process and utilizes the GPU.
    """
    print(f"[GPU Consumer] Transcribing: {os.path.basename(audio_path)}")
    try:
        # 1. Transcribe
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=batch_size)

        # 2. Align
        result = whisperx.align(result["segments"], align_model, align_metadata, audio, device, return_char_alignments=False)

        # 3. Save
        save_transcript_with_frames(result, fps, transcript_path)
        print(f"[GPU Consumer] Successfully saved transcript for: {os.path.basename(transcript_path)}")

    except Exception as e:
        print(f"[GPU Consumer] Error processing {os.path.basename(audio_path)}: {e}")
    finally:
        # 4. Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)

# --- Main execution block ---
if __name__ == "__main__":
    # Ensure the script can be run as a standalone executable
    multiprocessing.freeze_support()

    os.makedirs(output_transcript_dir, exist_ok=True)

    # 1. Load models ONCE in the main process
    print("--- Loading models into GPU memory... ---")
    transcribe_model = whisperx.load_model("kurianbenoy/vegam-whisper-medium-ml", device, compute_type=compute_type)
    align_model, align_metadata = whisperx.load_align_model(language_code="ml", device=device)
    print("--- Models loaded. ---")

    # 2. Find all videos and prepare arguments for CPU workers
    video_files = []
    supported_formats = ["*.mp4", "*.mov", "*.avi", "*.mkv"]
    for fmt in supported_formats:
        video_files.extend(glob.glob(os.path.join(input_video_dir, fmt)))

    if not video_files:
        print(f"No video files found in '{input_video_dir}'.")
        exit()

    tasks = []
    for video_path in video_files:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        transcript_path = os.path.join(output_transcript_dir, f"{base_name}.txt")
        temp_audio_path = os.path.join(output_transcript_dir, f"{base_name}_temp_audio.wav")
        tasks.append((video_path, temp_audio_path, transcript_path))

    print(f"\nFound {len(tasks)} videos. Starting parallel processing with {NUM_CPU_WORKERS} CPU workers.")
    print("-" * 50)

    # 3. Run the pipeline
    # We use a multiprocessing Pool to manage the worker processes.
    with multiprocessing.Pool(processes=NUM_CPU_WORKERS) as pool:
        # pool.imap_unordered processes tasks as they are submitted and yields results as they complete.
        # This is efficient because the GPU can start working on the first video that finishes converting.
        for result in pool.imap_unordered(cpu_prepare_audio, tasks):
            if result:
                # As soon as a CPU worker is done, the main process gets the result
                # and runs the GPU-intensive part.
                temp_audio_path, transcript_path, fps = result
                gpu_transcribe_and_align(
                    temp_audio_path,
                    transcript_path,
                    fps,
                    transcribe_model,
                    align_model,
                    align_metadata
                )
    print("-" * 50)
    print("\nAll videos have been processed.")