import os
import requests
from tqdm import tqdm
import pandas as pd

# Configuration
START_YEAR = 1999
END_YEAR = 2025  # Includes the 2025 season (current as of Jan 2026)
DATA_DIR = "nfl_pbp_data"
BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet"

def download_file(url, filepath):
    """Downloads a file with a progress bar."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kilobyte
        
        with open(filepath, "wb") as file, tqdm(
            desc=os.path.basename(filepath),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                file.write(data)
        return True
    else:
        print(f"Error: Could not download {url} (Status: {response.status_code})")
        return False

def main():
    # 1. Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    print(f"Starting download for NFL Play-by-Play data ({START_YEAR}-{END_YEAR})...")

    # 2. Loop through years and download
    for year in range(START_YEAR, END_YEAR + 1):
        filename = f"play_by_play_{year}.parquet"
        filepath = os.path.join(DATA_DIR, filename)
        url = BASE_URL.format(year=year)

        if os.path.exists(filepath):
            print(f"Skipping {year}: File already exists at {filepath}")
            continue

        print(f"Downloading {year} season...")
        success = download_file(url, filepath)
        
        if success:
            # Optional: Verify file integrity by trying to read the schema
            try:
                pd.read_parquet(filepath, columns=['play_id']) # Read only one column to be fast
            except Exception as e:
                print(f"Warning: File for {year} may be corrupted. Error: {e}")

    print("\n--- Download Complete ---")
    print(f"Files saved to: {os.path.abspath(DATA_DIR)}")
    print("You can now load specific seasons using pandas:")
    print(f"  df = pd.read_parquet('{DATA_DIR}/play_by_play_2024.parquet')")

if __name__ == "__main__":
    main()
