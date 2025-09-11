import whisperx
import gc
import torch
import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Path to your input video file
video_file_path = "dataset/clip_3.mp4" # <--- IMPORTANT: Change this to your video file path!
output_audio_file = "output_audio.mp3" # Output audio file name

batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

# IMPORTANT: Replace YOUR_HF_TOKEN with your actual Hugging Face token for diarization
# You can get one from https://huggingface.co/settings/tokens
YOUR_HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# --- Function to convert video to audio using FFmpeg ---
def convert_video_to_audio(video_input_path, audio_output_path):
    """
    Converts a video file to an MP3 audio file using FFmpeg.

    Args:
        video_input_path (str): The path to the input video file.
        audio_output_path (str): The desired path for the output MP3 audio file.

    Returns:
        str: The path to the generated audio file if successful, None otherwise.
    """
    if not os.path.exists(video_input_path):
        print(f"Error: Video file not found at '{video_input_path}'")
        return None

    # FFmpeg command: -i input_video, -vn no video, -acodec audio codec (libmp3lame for mp3),
    # -q:a audio quality (2 is VBR quality, higher is worse, 0 is best)
    command = [
        "ffmpeg",
        "-i", video_input_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-q:a", "2",
        audio_output_path
    ]

    print(f"Converting '{video_input_path}' to audio...")
    try:
        # Run the ffmpeg command
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully converted '{video_input_path}' to '{audio_output_path}'")
        return audio_output_path
    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg conversion:")
        print(f"STDOUT: {e.stdout.decode()}")
        print(f"STDERR: {e.stderr.decode()}")
        return None
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please install FFmpeg and ensure it's in your system's PATH.")
        print("You can download FFmpeg from: https://ffmpeg.org/download.html")
        return None

# --- Main processing logic ---
def process_video_and_transcribe(video_path):
    # Convert video to audio
    audio_file_path = convert_video_to_audio(video_path, output_audio_file)

    if not audio_file_path:
        print("Audio conversion failed, exiting.")
        return

    # 1. Transcribe with original whisper (batched)
    print("\n--- Step 1: Loading Whisper model and transcribing ---")
    model = whisperx.load_model("kurianbenoy/vegam-whisper-medium-ml", device, compute_type=compute_type)

    audio = whisperx.load_audio(audio_file_path)
    result = model.transcribe(audio, batch_size=batch_size)
    print("Segments before alignment:")
    for segment in result["segments"]:
        print(segment)

    # Clean up model to free GPU memory if needed
    del model
    if device == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    # 2. Align whisper output
    print("\n--- Step 2: Loading alignment model and aligning segments ---")
    # Ensure language_code matches the language of your audio. "ml" is for Malayalam.
    model_a, metadata = whisperx.load_align_model(language_code="ml", device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    print("Segments after alignment:")
    for segment in result["segments"]:
        print(segment)

    # Clean up alignment model
    del model_a
    if device == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Assign speaker labels (Diarization)
    print("\n--- Step 3: Performing speaker diarization and assigning labels ---")
    if YOUR_HF_TOKEN == "YOUR_HF_TOKEN" or not YOUR_HF_TOKEN:
        print("WARNING: Hugging Face token not provided or is default. Diarization will not work.")
        print("Please replace 'YOUR_HF_TOKEN' with your actual token from https://huggingface.co/settings/tokens")
        diarize_segments = [] # Initialize empty to avoid error
    else:
        try:
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)
            # add min/max number of speakers if known
            diarize_segments = diarize_model(audio)
            # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
            print("Diarization segments:")
            print(diarize_segments)

            result = whisperx.assign_word_speakers(diarize_segments, result)
            print("Segments with speaker IDs:")
            for segment in result["segments"]:
                print(segment)
        except Exception as e:
            print(f"Error during diarization: {e}")
            print("Diarization may require specific dependencies or a valid Hugging Face token.")

    # Clean up generated audio file
    if os.path.exists(audio_file_path):
        try:
            os.remove(audio_file_path)
            print(f"Cleaned up temporary audio file: '{audio_file_path}'")
        except OSError as e:
            print(f"Error removing temporary audio file {audio_file_path}: {e}")

# Call the main function with your video file
if __name__ == "__main__":
    process_video_and_transcribe(video_file_path)

