import whisperx
import torch
import os
import subprocess
import json
import math
import glob

# --- Configuration ---
# This is the directory where the leftover .wav files are, and where the final .txt files will be saved.
TRANSCRIPT_DIR = "D:/ADARSH/transcripts/"
# This is the directory where your original videos are. The script will look here to find the correct FPS.
ORIGINAL_VIDEO_DIR = "D:/ADARSH/processed_clips/"
# If the original video can't be found, use this FPS as a fallback.
DEFAULT_FPS = 30

# --- Model and Device Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
compute_type = "float16" # Use "float32" if you get errors with float16

print(f"Using device: {device}")
print("-" * 50)


# --- Helper Functions (from your original script) ---

def get_video_fps(video_path):
    """Probes a video file to get its frames per second (FPS)."""
    if not os.path.exists(video_path):
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
        return num / den
    except Exception as e:
        print(f"  [Warning] Could not get FPS for {os.path.basename(video_path)}. Error: {e}")
        return None

def save_transcript_with_frames(transcription_result, fps, output_path):
    """Saves the aligned transcription with start and end frame numbers."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment in transcription_result.get("segments", []):
            for word_info in segment.get("words", []):
                if 'start' in word_info and 'end' in word_info:
                    start_time, end_time, word = word_info['start'], word_info['end'], word_info['word']
                    start_frame = int(start_time * fps)
                    end_frame = int(math.ceil(end_time * fps))
                    f.write(f"{start_frame} {end_frame} {word.strip()}\n")


# --- Main execution block ---
if __name__ == "__main__":
    # 1. Load models ONCE
    print("--- Loading models into GPU memory... ---")
    # Using the same custom model you specified
    transcribe_model = whisperx.load_model("kurianbenoy/vegam-whisper-medium-ml", device, compute_type=compute_type)
    align_model, align_metadata = whisperx.load_align_model(language_code="ml", device=device)
    print("--- Models loaded successfully. ---\n")

    # 2. Find leftover temporary audio files
    search_pattern = os.path.join(TRANSCRIPT_DIR, "*_temp_audio.wav")
    audio_files = glob.glob(search_pattern)

    if not audio_files:
        print(f"No leftover audio files ('*_temp_audio.wav') found in '{TRANSCRIPT_DIR}'.")
        exit()

    print(f"Found {len(audio_files)} leftover audio files to process.")
    print("-" * 50)

    # 3. Process each audio file
    for audio_path in audio_files:
        base_name = os.path.basename(audio_path).replace("_temp_audio.wav", "")
        transcript_path = os.path.join(TRANSCRIPT_DIR, f"{base_name}.txt")

        print(f"Processing: {base_name}.wav")

        # Skip if the final transcript already exists
        if os.path.exists(transcript_path):
            print(f"  [Skipping] Transcript '{os.path.basename(transcript_path)}' already exists.")
            # Clean up the lingering audio file
            os.remove(audio_path)
            print(f"  [Cleaned] Removed '{os.path.basename(audio_path)}'.")
            continue

        # Try to find the original video to get the accurate FPS
        found_video = None
        supported_formats = [".mp4", ".mov", ".avi", ".mkv"]
        for fmt in supported_formats:
            potential_video_path = os.path.join(ORIGINAL_VIDEO_DIR, base_name + fmt)
            if os.path.exists(potential_video_path):
                found_video = potential_video_path
                break

        fps = DEFAULT_FPS
        if found_video:
            fps = get_video_fps(found_video) or DEFAULT_FPS
            print(f"  -> Found original video. Using FPS: {fps:.2f}")
        else:
            print(f"  -> Original video not found. Using default FPS: {DEFAULT_FPS}")


        try:
            # 4. Transcribe, Align, and Save
            print("  -> Transcribing and aligning...")
            audio = whisperx.load_audio(audio_path)
            result = transcribe_model.transcribe(audio, batch_size=batch_size)
            result = whisperx.align(result["segments"], align_model, align_metadata, audio, device, return_char_alignments=False)

            save_transcript_with_frames(result, fps, transcript_path)
            print(f"  -> Successfully saved transcript to '{os.path.basename(transcript_path)}'")

        except Exception as e:
            print(f"  [ERROR] Failed to process {os.path.basename(audio_path)}: {e}")
            # If there's an error, we'll skip deleting the audio file for manual review
            continue
        finally:
            # 5. Cleanup the temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"  -> Cleaned up temporary file: '{os.path.basename(audio_path)}'\n")

    print("-" * 50)
    print("Cleanup complete. All leftover audio files have been processed.")