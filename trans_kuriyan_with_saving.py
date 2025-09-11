import whisperx
import gc
import torch
import os
import subprocess
import json
import math
import secrets as sc

# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Path to your input video file
video_file_path = "dataset/clip_3.mp4" # <--- IMPORTANT: Change this to your video file path!
output_audio_file = "output_audio.mp3" # Temporary audio file name
output_transcript_file = "frame_transcript.txt" # ### NEW ### Output file for the frame-based transcript

batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

# IMPORTANT: Replace YOUR_HF_TOKEN with your actual Hugging Face token for diarization
# You can get one from https://huggingface.co/settings/tokens
YOUR_HF_TOKEN = sc.API

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
        "-y", # Overwrite output file if it exists
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

### NEW ###
def get_video_fps(video_path):
    """
    Gets the frame rate of a video file using ffprobe.

    Args:
        video_path (str): The path to the video file.

    Returns:
        float: The frame rate of the video, or None if it cannot be determined.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return None
    try:
        command = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "json",
            video_path
        ]
        print(f"Getting FPS for '{video_path}'...")
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        data = json.loads(result.stdout)
        frame_rate_str = data["streams"][0]["r_frame_rate"]
        num, den = map(int, frame_rate_str.split('/'))
        fps = num / den
        print(f"Detected FPS: {fps}")
        return fps
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error getting FPS from video: {e}")
        return None
    except FileNotFoundError:
        print("Error: ffprobe not found. Please install FFmpeg and ensure it's in your system's PATH.")
        return None

### NEW ###
def save_transcript_with_frames(transcription_result, fps, output_path):
    """
    Saves the transcript with start and end frames for each word.

    Args:
        transcription_result (dict): The result from whisperx alignment.
        fps (float): The frames per second of the source video.
        output_path (str): The path to save the output file.
    """
    print(f"Saving frame-by-frame transcript to '{output_path}'...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment in transcription_result.get("segments", []):
            for word_info in segment.get("words", []):
                if 'start' in word_info and 'end' in word_info:
                    start_time = word_info['start']
                    end_time = word_info['end']
                    word = word_info['word']

                    # Calculate frame numbers
                    start_frame = int(start_time * fps)
                    # Use math.ceil to ensure the frame duration is fully captured
                    end_frame = int(math.ceil(end_time * fps))

                    f.write(f"{start_frame} {end_frame} {word.strip()}\n")
    print("Successfully saved transcript.")


# --- Main processing logic ---
def process_video_and_transcribe(video_path):
    # ### NEW ### Get video FPS first
    fps = get_video_fps(video_path)
    if not fps:
        print("Could not determine video FPS. Aborting.")
        return

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
    
    # Clean up model to free GPU memory if needed
    del model
    if device == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    # 2. Align whisper output
    print("\n--- Step 2: Loading alignment model and aligning segments ---")
    # Ensure language_code matches the language of your audio. "ml" is for Malayalam.
    try:
        model_a, metadata = whisperx.load_align_model(language_code="ml", device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    except Exception as e:
        print(f"Error during alignment: {e}")
        print("Proceeding without alignment. Frame-level timestamps will not be available.")
        # Clean up and exit if alignment fails, as we can't get word timestamps
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
        return

    # ### NEW ### Save the frame-based transcript here
    # We do this after alignment, as that's when we get word-level timestamps.
    save_transcript_with_frames(result, fps, output_transcript_file)

    # Clean up alignment model
    del model_a
    if device == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Assign speaker labels (Diarization) - This is optional and can be run after saving the main transcript
    # print("\n--- Step 3: Performing speaker diarization and assigning labels ---")
    # if YOUR_HF_TOKEN == "YOUR_HF_TOKEN" or not YOUR_HF_TOKEN:
    #     print("WARNING: Hugging Face token not provided or is default. Diarization will be skipped.")
    # else:
    #     try:
    #         diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)
    #         diarize_segments = diarize_model(audio)
    #         result_with_speakers = whisperx.assign_word_speakers(diarize_segments, result)
            
    #         # You could save another file with speaker info here if you wish
    #         print("Segments with speaker IDs:")
    #         # Limiting print output to first few segments for brevity
    #         for i, segment in enumerate(result_with_speakers.get("segments", [])):
    #             if i < 3:
    #                 print(segment)

    #     except Exception as e:
    #         print(f"Error during diarization: {e}")
    #         print("Diarization may require specific dependencies or a valid Hugging Face token.")

    # Clean up generated audio file
    if os.path.exists(audio_file_path):
        try:
            os.remove(audio_file_path)
            print(f"\nCleaned up temporary audio file: '{audio_file_path}'")
        except OSError as e:
            print(f"Error removing temporary audio file {audio_file_path}: {e}")

# Call the main function with your video file
if __name__ == "__main__":
    process_video_and_transcribe(video_file_path)
