
# COde to check least number of files in subfolder

import os

def get_top5_least_files(base_folder):
    folder_file_counts = {}

    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_folder):
        # Skip the base folder itself (optional)
        if root == base_folder:
            continue
        folder_file_counts[root] = len(files)

    # Sort by number of files (ascending)
    sorted_folders = sorted(folder_file_counts.items(), key=lambda x: x[1])

    # Get top 5 with least number of files
    top5 = sorted_folders[:20]

    # Display results
    print("Top 5 subfolders with the least number of files:\n")
    for folder, count in top5:
        print(f"{folder} — {count} files")

    return top5

# Example usage
base_path = "D:/ADARSH/extracted_lip_crosssection"  # change this to your folder path
get_top5_least_files(base_path)
