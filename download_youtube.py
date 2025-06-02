import os
import sys
import yt_dlp
from tqdm import tqdm

DEFAULT_CHANNEL_ID = "UCAQO8LS1Hmxv2nodVyyQ6Dw"
OUTPUT_DIR = "downloaded_videos"


def get_video_urls(channel_url):
    """Get all video URLs from a YouTube channel using yt-dlp."""
    
    try:
        print(f"Extracting video list from channel...")
        
        # Extract channel ID from URL if needed
        channel_id = channel_url.split('/')[-1]
        
        # Use the more efficient playlist URL approach that yt-dlp suggests
        playlist_url = f"https://www.youtube.com/playlist?list=UU{channel_id[2:]}"
        
        ydl_opts = {
            'quiet': False,
            'extract_flat': True,
            'playlistend': None,  # No limit
        }
        
        print(f"Using playlist URL: {playlist_url}")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)
            
            if 'entries' in info:
                video_urls = []
                total_entries = len(info['entries'])
                print(f"Found {total_entries} videos in playlist")
                
                for entry in info['entries']:
                    if entry and entry.get('id'):
                        video_urls.append(f"https://www.youtube.com/watch?v={entry['id']}")
                
                print(f"Total videos to download: {len(video_urls)}")
                return video_urls
            else:
                print("No entries found in playlist")
                return []
                
    except Exception as e:
        print(f"Error fetching channel videos: {e}")
        print("Falling back to channel URL approach...")
        
        # Fallback to original approach
        ydl_opts = {
            'quiet': False,
            'extract_flat': True,
            'playlistend': None,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(channel_url, download=False)
                
                if 'entries' in info:
                    video_urls = [f"https://www.youtube.com/watch?v={entry['id']}" 
                                 for entry in info['entries'] if entry and entry.get('id')]
                    print(f"Total videos found (fallback): {len(video_urls)}")
                    return video_urls
                else:
                    return []
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return []


def download_video(video_url, output_dir):
    """Download a single video using yt-dlp."""
    ydl_opts = {
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'format': 'best[height<=720]',  # Download best quality up to 720p
        'quiet': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            title = info.get('title', 'Unknown')
            print(f"Downloading: {title}")
            ydl.download([video_url])
            print(f"Downloaded: {title}")
    except Exception as e:
        print(f"Error downloading {video_url}: {e}")


def main():
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage: python download_youtube.py [CHANNEL_ID]")
            print("Examples:")
            print("  python download_youtube.py")
            print("  python download_youtube.py UCAQO8LS1Hmxv2nodVyyQ6Dw")
            print("  python download_youtube.py UCQvW8oQwHk3W1cQK6y6kU1A")
            return
        
        channel_id = sys.argv[1]
    else:
        channel_id = DEFAULT_CHANNEL_ID
    
    channel_url = f"https://www.youtube.com/channel/{channel_id}"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Channel ID: {channel_id}")
    print(f"Fetching videos from: {channel_url}")
    
    video_urls = get_video_urls(channel_url)
    print(f"Found {len(video_urls)} videos.")
    
    if not video_urls:
        print("No videos found. This could be due to:")
        print("1. The channel is private")
        print("2. The channel has no videos")
        print("3. The channel URL is incorrect")
        return
    
    for url in tqdm(video_urls, desc="Downloading videos"):
        download_video(url, OUTPUT_DIR)


if __name__ == "__main__":
    main() 