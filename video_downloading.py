# import os
# import yt_dlp


# LINKS_FILE = "links-1.txt"

# OUTPUT_FOLDER = "downloads"

# # --- Main Script ---

# def download_videos_from_file(file_path, output_dir):
#     """
#     Reads a file line-by-line and downloads the YouTube video from each URL.

#     Args:
#         file_path (str): The path to the text file containing YouTube URLs.
#         output_dir (str): The directory where videos will be saved.
#     """
#     # Create the output directory if it doesn't already exist
#     if not os.path.exists(output_dir):
#         print(f"Creating output directory: {output_dir}")
#         os.makedirs(output_dir)

#     # yt-dlp options
#     # - 'outtmpl': Specifies the output template for filenames.
#     #   '%(title)s.%(ext)s' will save files as "Video Title.mp4".
#     # - 'paths': Specifies the output directory.
#     ydl_opts = {
#         'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
#         'outtmpl': '%(title)s.%(ext)s',
#         'paths': {'home': output_dir}
#     }

#     # Read the links from the file
#     try:
#         with open(file_path, 'r') as f:
#             urls = f.readlines()
#     except FileNotFoundError:
#         print(f"Error: The file '{file_path}' was not found.")
#         print("Please create this file and add your YouTube links to it.")
#         return

#     if not urls:
#         print(f"The file '{file_path}' is empty. No videos to download.")
#         return

#     print(f"Found {len(urls)} links in '{file_path}'. Starting download process...")

#     # Loop through each URL and download the video
#     for i, url in enumerate(urls):
#         # Remove any leading/trailing whitespace (like newlines)
#         clean_url = url.strip()

#         if not clean_url:
#             # Skip empty lines
#             continue

#         print(f"\n--- Downloading video {i+1} of {len(urls)} ---")
#         print(f"URL: {clean_url}")

#         try:
#             # The main download command
#             with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#                 ydl.download([clean_url])
#             print(f"Successfully downloaded video from: {clean_url}")
#         except Exception as e:
#             print(f"Error downloading {clean_url}. Reason: {e}")
#             print("Skipping to the next video.")
        
#         break

#     print("\n--- All downloads complete! ---")


# if __name__ == "__main__":
#     # Check for yt-dlp installation before running
#     try:
#         import yt_dlp
#     except ImportError:
#         print("Error: 'yt-dlp' is not installed.")
#         print("Please install it by running: pip install yt-dlp")
#     else:
#         download_videos_from_file(LINKS_FILE, OUTPUT_FOLDER)


import os
import sys # Import sys to handle exit
import yt_dlp


LINKS_FILE = "playlist_links.txt"

OUTPUT_FOLDER = "downloads"

# --- Main Script ---

def download_videos_from_file(file_path, output_dir):
    """
    Reads a file line-by-line and downloads the YouTube video from each URL.

    Args:
        file_path (str): The path to the text file containing YouTube URLs.
        output_dir (str): The directory where videos will be saved.
    """
    # Create the output directory if it doesn't already exist
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    # yt-dlp options
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': '%(title)s.%(ext)s',
        'paths': {'home': output_dir}
    }

    # Read the links from the file
    try:
        with open(file_path, 'r') as f:
            urls = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please create this file and add your YouTube links to it.")
        return

    if not urls:
        print(f"The file '{file_path}' is empty. No videos to download.")
        return

    print(f"Found {len(urls)} links in '{file_path}'. Starting download process...")
    print("Press Ctrl+C at any time to stop safely.")

    # --- START OF MODIFICATION ---
    try:
        # Loop through each URL and download the video
        limit = 0
        for i, url in enumerate(urls):
            # Remove any leading/trailing whitespace (like newlines)
            clean_url = url.strip()

            if not clean_url:
                # Skip empty lines
                continue

            print(f"\n--- Downloading video {i+1} of {len(urls)} ---")
            print(f"URL: {clean_url}")

            try:
                # The main download command
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([clean_url])
                print(f"Successfully downloaded video from: {clean_url}")
            except Exception as e:
                # This inner try/except handles errors for a single video
                print(f"Error downloading {clean_url}. Reason: {e}")
                print("Skipping to the next video.")

            if limit == 100:
                break

    except KeyboardInterrupt:
        # This outer try/except handles the user pressing Ctrl+C
        print("\n\nDownload process interrupted by user. Exiting gracefully.")
        sys.exit(0)
    # --- END OF MODIFICATION ---

    print("\n--- All downloads complete! ---")


if __name__ == "__main__":
    # Check for yt-dlp installation before running
    try:
        import yt_dlp
    except ImportError:
        print("Error: 'yt-dlp' is not installed.")
        print("Please install it by running: pip install yt-dlp")
    else:
        download_videos_from_file(LINKS_FILE, OUTPUT_FOLDER)