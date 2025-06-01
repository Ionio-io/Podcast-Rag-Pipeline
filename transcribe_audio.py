import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
import whisper
from pathlib import Path
from tqdm import tqdm

def transcribe_audio(audio_path, model):
    """Transcribe audio file using Whisper model."""
    try:
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        print(f"Error transcribing audio: {str(e)}")
        return None

def main():
    # Create necessary directories if they don't exist
    Path("audio").mkdir(exist_ok=True)
    Path("transcripts").mkdir(exist_ok=True)
    
    # Load Whisper model
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    
    # Get list of audio files
    audio_files = [f for f in os.listdir("audio") if f.endswith('.wav')]
    
    if not audio_files:
        print("No audio files found in the 'audio' directory.")
        return
    
    print(f"Found {len(audio_files)} audio files to process.")
    
    # Process each audio file
    for audio_file in tqdm(audio_files, desc="Transcribing audio files"):
        audio_path = os.path.join("audio", audio_file)
        transcript_path = os.path.join("transcripts", f"{os.path.splitext(audio_file)[0]}.txt")
        
        # Skip if transcript already exists
        if os.path.exists(transcript_path):
            print(f"Transcript already exists for {audio_file}, skipping...")
            continue
        
        print(f"\nProcessing {audio_file}...")
        
        # Transcribe audio
        print("Transcribing audio...")
        transcript = transcribe_audio(audio_path, model)
        
        if transcript:
            # Save transcript
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            print(f"Transcript saved to {transcript_path}")

if __name__ == "__main__":
    main() 