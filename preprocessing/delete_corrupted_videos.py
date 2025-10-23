import os
import glob

# --- Configuration ---
deleted_log_path = "D:/ADARSH/deleted_files.txt"
video_dir = "D:/ADARSH/processed_clips/"

# Supported video extensions
video_extensions = [".mp4", ".mov", ".mkv", ".avi"]

def main():
    if not os.path.exists(deleted_log_path):
        print(f"❌ Deleted files log not found: {deleted_log_path}")
        return

    # 1️⃣ Read deleted file names from log
    with open(deleted_log_path, "r", encoding="utf-8") as f:
        deleted_lines = f.readlines()

    # Extract just the base file name (without extension)
    deleted_basenames = []
    for line in deleted_lines:
        file_name = line.split(":")[0].strip()
        base_name = os.path.splitext(file_name)[0]
        deleted_basenames.append(base_name)

    # 2️⃣ Find matching video files and delete them
    deleted_videos = []
    for base in deleted_basenames:
        for ext in video_extensions:
            pattern = os.path.join(video_dir, f"{base}{ext}")
            for file_path in glob.glob(pattern):
                try:
                    os.remove(file_path)
                    deleted_videos.append(os.path.basename(file_path))
                except Exception as e:
                    print(f"⚠️ Could not delete {file_path}: {e}")

    # 3️⃣ Summary
    print("\n--- Video Deletion Summary ---")
    print(f"Total corrupted transcripts listed: {len(deleted_basenames)}")
    print(f"Videos deleted successfully: {len(deleted_videos)}")

    if deleted_videos:
        print("\nDeleted video files:")
        for v in deleted_videos:
            print(f" - {v}")
    else:
        print("No matching video files found for deletion.")

if __name__ == "__main__":
    main()
