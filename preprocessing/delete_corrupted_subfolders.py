import os
import shutil

# --- Configuration ---
deleted_log_path = "D:/ADARSH/deleted_files.txt"
target_root_dir = "D:/ADARSH/extracted_lip_crosssection"  # Folder containing subfolders for clips

def main():
    if not os.path.exists(deleted_log_path):
        print(f"❌ Deleted files log not found: {deleted_log_path}")
        return

    # 1️⃣ Read base names from log
    with open(deleted_log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    deleted_basenames = []
    for line in lines:
        filename = line.split(":")[0].strip()
        base_name = os.path.splitext(filename)[0]
        deleted_basenames.append(base_name)

    # 2️⃣ Check and delete matching subfolders
    deleted_folders = []
    for folder_name in os.listdir(target_root_dir):
        folder_path = os.path.join(target_root_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue  # skip files

        if folder_name in deleted_basenames:
            try:
                shutil.rmtree(folder_path)
                deleted_folders.append(folder_name)
            except Exception as e:
                print(f"⚠️ Could not delete folder {folder_name}: {e}")

    # 3️⃣ Summary
    print("\n--- Subfolder Deletion Summary ---")
    print(f"Total corrupted transcripts listed: {len(deleted_basenames)}")
    print(f"Folders deleted successfully: {len(deleted_folders)}")

    if deleted_folders:
        print("\nDeleted folders:")
        for name in deleted_folders:
            print(f" - {name}")
    else:
        print("No matching subfolders found for deletion.")

if __name__ == "__main__":
    main()
