import os
import re
from collections import Counter

# --- Configuration ---
transcript_dir = "D:/ADARSH/transcripts/"  # Folder containing transcript files

# Malayalam Unicode range and patterns
malayalam_pattern = re.compile(r'[\u0D00-\u0D7F]')
english_pattern = re.compile(r'[A-Za-z]')
unknown_pattern = re.compile(r'[�]')

def check_word(word):
    """Check if a word is valid Malayalam."""
    if unknown_pattern.search(word):
        return False, "Unknown character"
    if english_pattern.search(word):
        return False, "Contains English"
    if not malayalam_pattern.search(word):
        return False, "Not Malayalam"
    return True, "OK"

def validate_file(file_path):
    """Validate one transcript file line by line."""
    reasons = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not lines:
        return False, ["Empty file"]

    for line in lines:
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3:
            reasons.append("Malformed line")
            continue

        word = parts[2]
        valid, reason = check_word(word)
        if not valid:
            reasons.append(reason)

    if reasons:
        return False, list(set(reasons))
    return True, ["OK"]

def main():
    files = [f for f in os.listdir(transcript_dir) if f.endswith('.txt')]
    total_files = len(files)
    corrupt_files = []
    error_counter = Counter()

    for file_name in files:
        file_path = os.path.join(transcript_dir, file_name)
        try:
            valid, reasons = validate_file(file_path)
            if not valid:
                corrupt_files.append((file_name, ", ".join(reasons)))
                for reason in reasons:
                    error_counter[reason] += 1
        except Exception as e:
            corrupt_files.append((file_name, f"Read Error: {e}"))
            error_counter["Read Error"] += 1

    print("\n--- Transcript Validation Report ---")
    if corrupt_files:
        for name, reason in corrupt_files:
            print(f"[❌] {name}: {reason}")
    else:
        print("✅ All transcripts are valid Malayalam text.")

    print("\n--- Summary ---")
    print(f"Total files: {total_files}")
    print(f"Valid files: {total_files - len(corrupt_files)}")
    print(f"Corrupt files: {len(corrupt_files)}")

    if error_counter:
        print("\n--- Error Type Counts ---")
        for err, count in error_counter.items():
            print(f"{err}: {count}")

if __name__ == "__main__":
    main()
