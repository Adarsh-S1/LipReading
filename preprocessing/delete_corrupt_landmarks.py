import os

# The name of your input file
list_file_path = 'D:/ADARSH/deleted_files.txt'
path = 'D:/ADARSH/extracted_landmarks_model_ready/'

# --- Safety Check ---
if not os.path.exists(list_file_path):
    print(f"Error: The file '{list_file_path}' was not found.")
    print("Please make sure the script is in the same directory as deleted_files.txt")
else:
    print(f"--- Starting deletion process based on {list_file_path} ---")

    # Open the file and read it line by line
    with open(list_file_path, 'r') as f:
        for line in f:
            # 1. Clean up the line and skip empty ones
            line = line.strip()
            if not line:
                continue
            
            # 2. Get only the filename part (before the ':')
            #    line.partition(':')[0] splits the line at the first ':'
            #    and gives us the part before it.
            original_txt_file = line.partition(':')[0].strip()

            if not original_txt_file:
                print(f"Skipping malformed line: {line}")
                continue

            # 3. Split the filename into its base and extension
            #    e.g., "clip_0007_55.63s_to_56.65s.txt"
            #    becomes base_name = "clip_0007_55.63s_to_56.65s"
            #    and ext = ".txt"
            try:
                base_name, ext = os.path.splitext(original_txt_file)
            except Exception as e:
                print(f"Could not process filename '{original_txt_file}': {e}")
                continue

            # 4. Create the names of the files to delete
            landmarks_file = f"{base_name}_landmarks.npy"
            metadata_file = f"{base_name}_metadata.json"

            # 5. Attempt to delete the files, handling errors if they don't exist
            
            # --- Delete landmarks file ---
            try:
                os.remove(path+landmarks_file)
                print(f"[DELETED] {landmarks_file}")
            except FileNotFoundError:
                print(f"[NOT FOUND] {landmarks_file}")
            except Exception as e:
                print(f"[ERROR] Could not delete {landmarks_file}: {e}")

            # --- Delete metadata file ---
            try:
                os.remove(path+metadata_file)
                print(f"[DELETED] {metadata_file}")
            except FileNotFoundError:
                print(f"[NOT FOUND] {metadata_file}")
            except Exception as e:
                print(f"[ERROR] Could not delete {metadata_file}: {e}")

    print("--- Deletion process complete ---")