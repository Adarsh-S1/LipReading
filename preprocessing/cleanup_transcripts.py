import os
import re

# --- Configuration ---
transcript_dir = "D:/ADARSH/transcripts/"
deleted_log_path = os.path.join(transcript_dir, "deleted_files.txt")

# Malayalam and pattern definitions
malayalam_pattern = re.compile(r'[\u0D00-\u0D7F]')
english_pattern = re.compile(r'[A-Za-z]')
unknown_pattern = re.compile(r'[�]')

def process_file(file_path):
    """Remove unknown chars or delete file if invalid."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return "Deleted", "Empty file"

    valid_malayalam_found = False
    english_found = False
    new_lines = []

    for line in lines:
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3:
            continue

        word = parts[2]

        # Track language
        if english_pattern.search(word):
            english_found = True
        if malayalam_pattern.search(word):
            valid_malayalam_found = True

        # Remove unknown characters only
        clean_word = unknown_pattern.sub("", word)
        new_lines.append(f"{parts[0]} {parts[1]} {clean_word}\n")

    # Decide action
    if english_found or not valid_malayalam_found:
        os.remove(file_path)
        return "Deleted", "Contains English or not Malayalam"

    # Overwrite cleaned file
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    return "Cleaned", "Removed unknown characters (if any)"

def main():
    deleted_files = []
    cleaned_count = 0

    for file_name in os.listdir(transcript_dir):
        if not file_name.endswith(".txt"):
            continue
        file_path = os.path.join(transcript_dir, file_name)

        try:
            status, reason = process_file(file_path)
            if status == "Deleted":
                deleted_files.append((file_name, reason))
            elif status == "Cleaned":
                cleaned_count += 1
        except Exception as e:
            deleted_files.append((file_name, f"Error: {e}"))

    # Write deleted file log
    if deleted_files:
        with open(deleted_log_path, "w", encoding="utf-8") as log:
            for name, reason in deleted_files:
                log.write(f"{name}: {reason}\n")

    # Summary
    print("\n--- Cleanup Summary ---")
    print(f"Files cleaned (unknown chars removed): {cleaned_count}")
    print(f"Files deleted: {len(deleted_files)}")

    if deleted_files:
        print(f"Deleted file names saved to: {deleted_log_path}")

if __name__ == "__main__":
    main()
