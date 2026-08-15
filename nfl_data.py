"""
Script 1: Download all NFL data sources
- Play-by-play (1999-2024) from nflverse
- NFL Big Data Bowl (2023) from Kaggle
- NFL Rulebook (PDF) from NFL Operations
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# Configuration
START_YEAR = 1999
END_YEAR = 2025
DATA_DIR = Path("data")
PBP_DIR = DATA_DIR / "pbp"
BDB_DIR = DATA_DIR / "bdb"
RULEBOOK_DIR = DATA_DIR / "rulebook"

# URLs
PBP_URL_TEMPLATE = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet"
RULEBOOK_URL = "https://operations.nfl.com/media/24emxacq/2024-nfl-rulebook.pdf"

# Kaggle datasets
BDB_DATASETS = {
    "2023": "nflverse/nfl-big-data-bowl-2023",  # Linemen & blocking
    "2024": "nflverse/nfl-big-data-bowl-2024",  # Tackling
}

def create_directories():
    """Create necessary data directories."""
    for dir_path in [PBP_DIR, BDB_DIR, RULEBOOK_DIR, DATA_DIR / "videos", DATA_DIR / "processed"]:
        dir_path.mkdir(parents=True, exist_ok=True)
    print("✓ Directories created")

def download_file(url, filepath, description=""):
    """Download a file with progress bar."""
    if os.path.exists(filepath):
        print(f"⊘ {description} already exists, skipping")
        return True
    
    try:
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print(f"✗ Failed to download {url} (Status: {response.status_code})")
            return False
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB chunks
        
        with open(filepath, "wb") as f, tqdm(
            desc=description,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            leave=False
        ) as pbar:
            for chunk in response.iter_content(block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✓ {description}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {description}: {e}")
        return False

def download_pbp_data():
    """Download Play-by-Play data from nflverse (1999-2025)."""
    print("\n" + "="*60)
    print("Downloading Play-by-Play Data (1999-2025)")
    print("="*60)
    
    successful = 0
    failed = 0
    
    for year in range(START_YEAR, END_YEAR + 1):
        filename = f"play_by_play_{year}.parquet"
        filepath = PBP_DIR / filename
        url = PBP_URL_TEMPLATE.format(year=year)
        
        if download_file(url, filepath, f"PBP {year}"):
            successful += 1
        else:
            failed += 1
    
    print(f"\nPBP Download Summary: {successful} successful, {failed} failed")
    return successful > 0

def download_rulebook():
    """Download NFL Rulebook PDF."""
    print("\n" + "="*60)
    print("Downloading NFL Rulebook")
    print("="*60)
    
    filepath = RULEBOOK_DIR / "2024_nfl_rulebook.pdf"
    return download_file(RULEBOOK_URL, filepath, "NFL Rulebook (PDF)")

def download_bdb_data():
    """Download Big Data Bowl datasets from Kaggle."""
    print("\n" + "="*60)
    print("Downloading Big Data Bowl (Kaggle)")
    print("="*60)
    
    try:
        api = KaggleApi()
        api.authenticate()
        print("✓ Kaggle authenticated")
    except Exception as e:
        print(f"✗ Kaggle authentication failed: {e}")
        print("  Tip: Download kaggle.json from https://www.kaggle.com/settings/account")
        return False
    
    for year, dataset in BDB_DATASETS.items():
        try:
            output_path = BDB_DIR / f"bdb_{year}"
            if output_path.exists() and len(list(output_path.glob("**/*.parquet"))) > 0:
                print(f"⊘ BDB {year} already downloaded, skipping")
                continue
            
            print(f"Downloading BDB {year} ({dataset})...")
            api.dataset_download_files(dataset, path=output_path, unzip=True)
            print(f"✓ BDB {year} downloaded to {output_path}")
        except Exception as e:
            print(f"✗ Failed to download BDB {year}: {e}")
    
    return True

def verify_downloads():
    """Verify that all downloads completed successfully."""
    print("\n" + "="*60)
    print("Verifying Downloads")
    print("="*60)
    
    # Check PBP files
    pbp_files = list(PBP_DIR.glob("*.parquet"))
    print(f"PBP files: {len(pbp_files)}")
    
    # Check rulebook
    rulebook_exists = (RULEBOOK_DIR / "2024_nfl_rulebook.pdf").exists()
    print(f"Rulebook: {'✓' if rulebook_exists else '✗'}")
    
    # Check BDB
    bdb_files = list(BDB_DIR.glob("**/*.parquet"))
    print(f"BDB files: {len(bdb_files)}")
    
    return len(pbp_files) > 20 and rulebook_exists

def main():
    print("NFL VLM Data Download Pipeline")
    print("="*60)
    
    # Create directories
    create_directories()
    
    # Download all data
    pbp_ok = download_pbp_data()
    rulebook_ok = download_rulebook()
    bdb_ok = download_bdb_data()
    
    # Verify
    all_ok = verify_downloads()
    
    print("\n" + "="*60)
    if all_ok:
        print("✓ All downloads completed successfully!")
        print(f"\nNext step: python scripts/02_process_vlm_dataset.py")
    else:
        print("✗ Some downloads failed. Check errors above.")
    print("="*60)

if __name__ == "__main__":
    main()