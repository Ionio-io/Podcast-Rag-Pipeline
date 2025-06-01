import os
import ffmpeg
from pathlib import Path
from tqdm import tqdm

def extract_audio(video_path, audio_path):
    """Extract audio from video file using ffmpeg."""
    try:
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, audio_path)
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        return True
    except ffmpeg.Error as e:
        print(f"Error extracting audio: {e.stderr.decode()}")
        return False

def main():
    # Create necessary directories if they don't exist
    Path("videos").mkdir(exist_ok=True)
    Path("audio").mkdir(exist_ok=True)
    
    # Get list of video files
    video_files = [f for f in os.listdir("videos") if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print("No video files found in the 'videos' directory.")
        return
    
    print(f"Found {len(video_files)} video files to process.")
    
    # Process each video file
    for video_file in tqdm(video_files, desc="Converting videos to audio"):
        video_path = os.path.join("videos", video_file)
        audio_path = os.path.join("audio", f"{os.path.splitext(video_file)[0]}.wav")
        
        # Skip if audio already exists
        if os.path.exists(audio_path):
            print(f"Audio already exists for {video_file}, skipping...")
            continue
        
        print(f"\nProcessing {video_file}...")
        
        # Extract audio
        print("Extracting audio...")
        if extract_audio(video_path, audio_path):
            print(f"Audio saved to {audio_path}")
        else:
            print(f"Failed to extract audio from {video_file}")

if __name__ == "__main__":
    main() 