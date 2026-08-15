"""
Script 3F: Download Single Play Videos for Training
Searches for specific plays on YouTube and downloads short clips.
Ideal for building a dataset of specific events (penalties, deep passes, etc.)
"""

import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import yt_dlp

# Config
DATA_DIR = Path("data")
VIDEOS_DIR = DATA_DIR / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

class SinglePlayDownloader:
    def __init__(self):
        self.ydl_opts = {
            'format': 'best[ext=mp4][height<=720]',  # 720p is enough for training
            'quiet': True,
            'no_warnings': True,
            'max_filesize': 50 * 1024 * 1024, # Max 50MB (single plays are small)
            'match_filter': self.filter_video_length # Only download short videos
        }
    
    def filter_video_length(self, info, *args, **kwargs):
        """Filter out long videos (we only want single plays < 60s)."""
        duration = info.get('duration')
        if duration and duration > 120: # Skip if longer than 2 minutes
            return 'Video too long'
        return None

    def search_and_download(self, query, out_filename):
        """Search YouTube for a specific play and download the best match."""
        out_path = VIDEOS_DIR / out_filename
        
        if out_path.exists() and out_path.stat().st_size > 100000:
            return True, "Exists"

        # Search specifically for short clips
        search_query = f"ytsearch3:{query}" # Get top 3 results
        
        self.ydl_opts['outtmpl'] = str(out_path.with_suffix('')) # yt-dlp adds extension
        
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                # 1. Extract info first to check durations
                info = ydl.extract_info(search_query, download=False)
                
                if not info or 'entries' not in info or not info['entries']:
                    return False, "No results found"
                
                # 2. Find the best short clip (under 60s is ideal for single play)
                best_video = None
                for entry in info['entries']:
                    duration = entry.get('duration', 0)
                    # Ideal play clip is 5-60 seconds
                    if 5 < duration < 90: 
                        best_video = entry
                        break
                
                if not best_video:
                    # Fallback: try the first result if it's not super long
                    first_entry = info['entries'][0]
                    if first_entry.get('duration', 999) < 180:
                        best_video = first_entry
                    else:
                        return False, "No suitable short clip found"

                # 3. Download the selected video
                ydl.download([best_video['webpage_url']])
                
                return True, "Downloaded"
                
        except Exception as e:
            return False, str(e)

def process_dataset(json_path, max_videos=50, penalty_only=False):
    print(f"Loading dataset: {json_path}")
    with open(json_path, 'r') as f:
        plays = json.load(f)
    
    # Filter for penalties if requested
    if penalty_only:
        plays = [p for p in plays if p.get('has_penalty')]
        print(f"Filtered to {len(plays)} penalty plays.")
    
    # Shuffle or select specific plays
    # Let's prioritize plays with clear descriptions
    target_plays = [p for p in plays if len(p.get('play_description', '')) > 20][:max_videos]
    
    downloader = SinglePlayDownloader()
    success = 0
    
    print(f"Searching and downloading {len(target_plays)} single plays...")
    
    for play in tqdm(target_plays):
        game_id = play.get('game_id')
        play_id = play.get('play_id')
        desc = play.get('play_description')
        
        # Clean description for search query
        # Remove specialized stats, keep the "action"
        # "P.Mahomes pass short right to T.Kelce for 13 yards" -> "Mahomes pass Kelce 13 yards"
        clean_desc = desc.split('(')[0] # Remove penalties details often in parens
        
        # Construct search query
        query = f"NFL {clean_desc} highlight"
        if play.get('has_penalty'):
            query += f" {play.get('penalty_type')} penalty"
            
        out_name = f"{game_id}_{play_id}.mp4"
        
        ok, status = downloader.search_and_download(query, out_name)
        
        if ok:
            success += 1
        
        # Sleep to avoid rate limits
        time.sleep(1)

    print(f"\nDone. Downloaded {success}/{len(target_plays)} plays.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=20, help="Max plays to download")
    parser.add_argument("--penalties", action="store_true", help="Only download penalty plays")
    args = parser.parse_args()
    
    train_file = Path("data/processed/train_dataset.json")
    if train_file.exists():
        process_dataset(train_file, max_videos=args.max, penalty_only=args.penalties)
    else:
        print("Dataset not found.")
