"""
Script 3D: Download NFL Videos via Public Page Scraping (No API Key Required)
Replaces deprecated feeds.nfl.com with direct page parsing.
"""

import os
import json
import re
import time
import requests
import argparse
from pathlib import Path
from tqdm import tqdm
from urllib.parse import urljoin

# Config
DATA_DIR = Path("data")
VIDEOS_DIR = DATA_DIR / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

class NFLVideoScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
    
    def find_mp4_in_page(self, url):
        """Scrape a web page looking for .mp4 or .m3u8 links."""
        try:
            r = self.session.get(url, timeout=10)
            if r.status_code != 200:
                return None
            
            # 1. Look for direct MP4 links in source
            mp4_matches = re.findall(r'https?://[^"\']+\.mp4', r.text)
            
            # Filter for likely high-quality game clips (exclude ads/thumbnails)
            valid_mp4s = [m for m in mp4_matches if "nfl" in m.lower() and "highlight" in m.lower()]
            
            if valid_mp4s:
                # Return the longest URL (heuristic: often higher resolution/bitrate)
                return sorted(valid_mp4s, key=len, reverse=True)[0]
                
            # 2. Look for m3u8 (HLS) streams if no MP4
            m3u8_matches = re.findall(r'https?://[^"\']+\.m3u8', r.text)
            if m3u8_matches:
                return m3u8_matches[0] # Return first HLS stream found
                
            return None
        except Exception:
            return None

    def get_espn_video_url(self, game_id, play_id):
        """Try to find video on ESPN's public endpoints."""
        # Common pattern for ESPN video clips
        patterns = [
            f"https://www.espn.com/video/clip?id={game_id}",
            f"https://www.espn.com/nfl/game/_/gameId/{game_id}"
        ]
        
        for url in patterns:
            video_url = self.find_mp4_in_page(url)
            if video_url:
                return video_url
        return None

    def get_nfl_com_video_url(self, game_id, play_id):
        """Try to find video on NFL.com game center pages."""
        # NFL.com game center URL pattern
        url = f"https://www.nfl.com/games/{game_id}" # Note: Game ID format might need adjustment based on year
        return self.find_mp4_in_page(url)

    def download_video(self, url, out_path):
        """Download video content to file."""
        try:
            # If it's an m3u8, we need ffmpeg to download it (skip for now to keep it simple, or use external tool)
            if ".m3u8" in url:
                # Basic HLS handling: usually requires ffmpeg. 
                # For this script, we stick to MP4s or try to find an MP4 fallback.
                return False, "HLS stream detected (needs ffmpeg)"
            
            with self.session.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                # Filter out tiny files (likely errors/pixels)
                if total_size > 0 and total_size < 50000: 
                    return False, "File too small (<50KB)"

                with open(out_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
            return True, "OK"
        except Exception as e:
            if out_path.exists(): out_path.unlink()
            return False, str(e)

def process_dataset(json_path, max_videos=50):
    print(f"Loading dataset: {json_path}")
    with open(json_path, 'r') as f:
        plays = json.load(f)
    
    scraper = NFLVideoScraper()
    success_count = 0
    
    # Shuffle or sample to get variety if needed
    target_plays = plays[:max_videos]
    
    print(f"Attempting to download {len(target_plays)} videos...")
    
    for play in tqdm(target_plays):
        game_id = play.get('game_id')
        play_id = play.get('play_id')
        
        if not game_id: continue
        
        out_filename = f"{game_id}_{play_id}.mp4"
        out_path = VIDEOS_DIR / out_filename
        
        if out_path.exists() and out_path.stat().st_size > 50000:
            success_count += 1
            continue
            
        # Strategy 1: ESPN
        video_url = scraper.get_espn_video_url(game_id, play_id)
        
        # Strategy 2: NFL.com (Fallback)
        if not video_url:
            video_url = scraper.get_nfl_com_video_url(game_id, play_id)
            
        if video_url:
            ok, msg = scraper.download_video(video_url, out_path)
            if ok:
                success_count += 1
            else:
                pass # Download failed
        
        time.sleep(0.5) # Be nice to servers

    print(f"\nDownload complete. Success: {success_count}/{len(target_plays)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=50, help="Max videos to download")
    args = parser.parse_args()
    
    train_file = Path("data/processed/train_dataset.json")
    if train_file.exists():
        process_dataset(train_file, max_videos=args.max)
    else:
        print("Error: train_dataset.json not found. Run processing script first.")
