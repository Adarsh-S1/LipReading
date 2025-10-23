import os
import shutil

def delete_small_folders(base_folder, min_files=3, delete=True):
    folder_file_counts = {}

    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_folder):
        if root == base_folder:
            continue
        folder_file_counts[root] = len(files)

    # Filter folders with fewer than `min_files` files
    small_folders = [folder for folder, count in folder_file_counts.items() if count < min_files]

    if not small_folders:
        print("No folders found with fewer than", min_files, "files.")
        return

    print(f"Found {len(small_folders)} folders with fewer than {min_files} files:\n")
    for folder in small_folders:
        print("🗂️", folder)

    if delete:
        print("\nDeleting these folders...")
        for folder in small_folders:
            try:
                shutil.rmtree(folder)
                print(f"✅ Deleted: {folder}")
            except Exception as e:
                print(f"❌ Could not delete {folder}: {e}")
    else:
        print("\n(delete=False) — No folders deleted. Set delete=True to remove them.")

# Example usage
base_path = 'D:/ADARSH/extracted_lip_crosssection'  # change to your path
delete_small_folders(base_path, min_files=3, delete=True)
