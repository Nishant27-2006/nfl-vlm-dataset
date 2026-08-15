"""
Script 2 (ENHANCED - FIXED v2): Process and align data into VLM training format
- Attempts to load BDB tracking data from GitHub
- Falls back to derived contextual features from PBP
- Creates balanced penalty/no-penalty splits
- Generates training dataset JSON with metadata
- FIXED: Handles column name variations from nflverse
- FIXED v2: Proper handling of string game_id and play_id
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("data")
PBP_DIR = DATA_DIR / "pbp"
BDB_DIR = DATA_DIR / "bdb"
PROCESSED_DIR = DATA_DIR / "processed"

PENALTY_PRIORITY = [
    "Holding",
    "False Start",
    "Offensive Holding",
    "Defensive Holding",
    "Pass Interference",
    "Illegal Block",
    "Illegal Block Below the Waist",
    "Offsides",
    "Encroachment",
    "Roughing the Passer",
]

MIN_PENALTY_SAMPLES = 10
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# BDB GitHub URLs
BDB_GITHUB_BASE = "https://github.com/nflverse/nfl-big-data-bowl-2023/releases/download"
BDB_2023_TRACKING_URLS = [
    f"{BDB_GITHUB_BASE}/tracking/tracking_week_{week}.parquet" for week in range(1, 10)
]

def load_pbp_data():
    """Load and concatenate all PBP files."""
    print("\nLoading PBP data...")
    dfs = []
    
    pbp_files = sorted(PBP_DIR.glob("*.parquet"))
    for f in tqdm(pbp_files, desc="Loading PBP"):
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"⚠ Error loading {f}: {e}")
    
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"✓ Loaded {len(full_df):,} plays across {len(pbp_files)} seasons")
    return full_df

def load_bdb_tracking_data(max_weeks=None):
    """
    Attempt to load BDB tracking data from GitHub.
    Falls back gracefully if unavailable.
    """
    print("\nAttempting to load Big Data Bowl tracking data...")
    tracking_dfs = []
    weeks_loaded = 0
    
    urls_to_try = BDB_2023_TRACKING_URLS if max_weeks is None else BDB_2023_TRACKING_URLS[:max_weeks]
    
    for url in tqdm(urls_to_try, desc="Loading BDB Tracking"):
        try:
            df = pd.read_parquet(url)
            tracking_dfs.append(df)
            weeks_loaded += 1
        except Exception as e:
            # Silent fail - expected if GitHub is blocked
            continue
    
    if weeks_loaded > 0:
        print(f"✓ Loaded BDB tracking data for {weeks_loaded} weeks")
        return pd.concat(tracking_dfs, ignore_index=True)
    else:
        print("⚠ BDB tracking data unavailable (will use derived features instead)")
        return None

def normalize_column_names(pbp_df):
    """
    Normalize column names from nflverse PBP data.
    Handles variations in column naming across years.
    """
    # Common nflverse column name variations
    col_mapping = {
        'ydstogo': 'yards_to_go',
        'yds_to_go': 'yards_to_go',
        'ydsaft': 'yards_after_catch',
        'yards_aft_catch': 'yards_after_catch',
        'air_yds': 'air_yards',
        'yrdln_100': 'yardline_100',
        'yardline': 'yardline_100',
        'receiver': 'receiver_player_name',
        'receiver_player': 'receiver_player_name',
        'rusher': 'rusher_player_name',
        'rusher_player': 'rusher_player_name',
        'desc': 'play_description',
        'penalty_code': 'penalty',
        'posteam_id': 'posteam',
        'defteam_id': 'defteam',
    }
    
    pbp_df.columns = pbp_df.columns.str.lower()
    
    for old_name, new_name in col_mapping.items():
        if old_name in pbp_df.columns and new_name not in pbp_df.columns:
            pbp_df.rename(columns={old_name: new_name}, inplace=True)
    
    return pbp_df

def enrich_pbp_with_contextual_features(pbp_df):
    """
    Enrich PBP data with derived contextual features.
    These proxy for spatial context when BDB tracking is unavailable.
    """
    print("\nEnriching PBP with contextual features...")
    
    # Normalize columns first
    pbp_df = normalize_column_names(pbp_df)
    
    # Ensure required columns exist with safe defaults
    required_cols = ['down', 'yards_to_go', 'quarter', 'game_seconds_remaining', 
                     'air_yards', 'yards_after_catch', 'play_type', 'yardline_100']
    for col in required_cols:
        if col not in pbp_df.columns:
            pbp_df[col] = 0
    
    # 1. GAME SITUATION FEATURES
    pbp_df['down_situation'] = (
        pbp_df['down'].fillna(0).astype(str) + '&' + 
        pbp_df['yards_to_go'].fillna(0).astype(str)
    )
    
    pbp_df['game_clock_state'] = (
        pbp_df['quarter'].fillna(0).astype(int) * 100 + 
        (pbp_df['game_seconds_remaining'].fillna(0) // 300).astype(int)
    )
    
    # 2. PASS/PLAY DEPTH FEATURES
    pbp_df['pass_depth'] = pd.cut(
        pbp_df['air_yards'].fillna(0),
        bins=[-np.inf, -1, 0, 5, 10, 20, np.inf],
        labels=['behind_los', 'at_los', 'short', 'intermediate', 'deep', 'bomb'],
        include_lowest=True
    ).astype(str)
    
    pbp_df['yards_after_catch_cat'] = pd.cut(
        pbp_df['yards_after_catch'].fillna(0),
        bins=[-np.inf, 0, 5, 10, 20, np.inf],
        labels=['no_gain', 'short_gain', 'medium_gain', 'long_gain', 'explosive'],
        include_lowest=True
    ).astype(str)
    
    # 3. OFFENSIVE/DEFENSIVE CONTEXT
    pbp_df['formation_context'] = pbp_df['play_type'].fillna('unknown').astype(str)
    
    # 4. PLAYER INVOLVEMENT PROXIES
    pbp_df['receiver_group'] = np.where(
        pbp_df.get('receiver_player_name', pd.Series([None]*len(pbp_df), index=pbp_df.index)).notna(),
        'pass_catcher',
        'non_receiver'
    )
    
    pbp_df['rusher_present'] = np.where(
        pbp_df.get('rusher_player_name', pd.Series([None]*len(pbp_df), index=pbp_df.index)).notna(),
        'rush_play',
        'non_rush'
    )
    
    # 5. CONTACT LIKELIHOOD FEATURES
    pbp_df['qb_pressure_situation'] = (
        (pbp_df['down'].fillna(0) <= 2) & 
        (pbp_df['yards_to_go'].fillna(0) >= 7)
    ).astype(int)
    
    pbp_df['goal_line_situation'] = (
        pbp_df['yardline_100'].fillna(100) <= 10
    ).astype(int)
    
    pbp_df['red_zone'] = (
        pbp_df['yardline_100'].fillna(100) <= 20
    ).astype(int)
    
    # 6. EXPECTED CONTACT LEVEL
    pbp_df['contact_likelihood'] = (
        pbp_df['qb_pressure_situation'] * 0.3 +
        pbp_df['goal_line_situation'] * 0.4 +
        pbp_df['red_zone'] * 0.2 +
        (pbp_df['down'].fillna(0) == 3).astype(int) * 0.1
    ).round(2)
    
    print("✓ Added 11 contextual features (pass depth, down/distance, zone, etc.)")
    return pbp_df

def merge_pbp_with_bdb(pbp_df, bdb_tracking_df=None):
    """Merge PBP with BDB tracking data if available."""
    if bdb_tracking_df is not None and not bdb_tracking_df.empty:
        print("\nMerging PBP with BDB tracking data...")
        try:
            bdb_play_stats = bdb_tracking_df.groupby(['gameId', 'playId']).agg({
                'x': ['mean', 'std', 'min', 'max'],
                'y': ['mean', 'std', 'min', 'max'],
                's': ['mean', 'max'],
                'a': ['mean', 'max'],
            }).reset_index()
            
            bdb_play_stats.columns = ['_'.join(col).strip('_') for col in bdb_play_stats.columns.values]
            bdb_play_stats.rename(columns={
                'gameId': 'game_id',
                'playId': 'play_id'
            }, inplace=True)
            
            pbp_df = pbp_df.merge(
                bdb_play_stats,
                on=['game_id', 'play_id'],
                how='left'
            )
            print(f"✓ Merged with BDB tracking data ({len(bdb_play_stats)} plays)")
            return pbp_df
        
        except Exception as e:
            print(f"⚠ Error merging BDB data: {e}")
            print("  Proceeding with derived features only")
            return pbp_df
    else:
        print("✓ Using derived contextual features (BDB not available)")
        return pbp_df

def extract_penalty_features(pbp_df):
    """Extract key penalty features from PBP data."""
    print("\nExtracting penalty features...")
    
    # Normalize column names again (safety check)
    pbp_df = normalize_column_names(pbp_df)
    
    # Create label column
    pbp_df['has_penalty'] = pbp_df['penalty'].notna() & (pbp_df['penalty'] != 0)
    pbp_df['penalty_type'] = pbp_df.get('penalty_type', 'No Penalty').fillna('No Penalty')
    
    # Filter for key features
    feature_cols = [
        'game_id', 'play_id', 'week', 'season',
        'home_team', 'away_team', 'posteam', 'defteam',
        'qtr', 'quarter', 'game_seconds_remaining',
        'play_description', 'penalty', 'penalty_type', 'has_penalty',
        'down', 'yards_to_go', 'yardline_100',
        'air_yards', 'yards_after_catch', 'pass_length',
        'epa', 'wpa', 'play_type',
        'down_situation', 'game_clock_state', 'pass_depth', 
        'yards_after_catch_cat', 'formation_context',
        'receiver_group', 'rusher_present', 'qb_pressure_situation',
        'goal_line_situation', 'red_zone', 'contact_likelihood'
    ]
    
    # Only keep columns that exist
    available_cols = [col for col in feature_cols if col in pbp_df.columns]
    pbp_df = pbp_df[available_cols]
    
    # Fill NaNs in numeric columns
    numeric_cols = pbp_df.select_dtypes(include=[np.number]).columns
    pbp_df[numeric_cols] = pbp_df[numeric_cols].fillna(0)
    
    return pbp_df

def filter_valid_penalties(pbp_df):
    """Filter for penalties with sufficient samples and high quality."""
    print("\nFiltering penalties...")
    
    penalty_counts = pbp_df[pbp_df['has_penalty']]['penalty_type'].value_counts()
    print(f"\nPenalty distribution (top 20):")
    print(penalty_counts.head(20))
    
    # Keep only penalties with min samples
    valid_penalties = penalty_counts[penalty_counts >= MIN_PENALTY_SAMPLES].index.tolist()
    
    # Create stratified dataset
    penalty_plays = pbp_df[pbp_df['penalty_type'].isin(valid_penalties)].copy()
    no_penalty_plays = pbp_df[pbp_df['penalty_type'] == 'No Penalty'].copy()
    
    # Balance classes
    n_penalty = len(penalty_plays)
    n_no_penalty = min(len(no_penalty_plays), n_penalty * 2)
    
    no_penalty_sample = no_penalty_plays.sample(n=n_no_penalty, random_state=42)
    
    balanced_df = pd.concat([penalty_plays, no_penalty_sample], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✓ Penalty plays: {len(penalty_plays):,}")
    print(f"✓ No-penalty plays: {len(no_penalty_sample):,}")
    print(f"✓ Balanced dataset: {len(balanced_df):,}")
    
    return balanced_df, valid_penalties

def stratified_train_test_split(df, penalty_col='penalty_type'):
    """Stratified split maintaining class distribution."""
    print("\nPerforming stratified split...")
    
    unique_penalties = df[penalty_col].unique()
    train_indices = []
    val_indices = []
    test_indices = []
    
    for penalty in unique_penalties:
        mask = df[penalty_col] == penalty
        indices = df[mask].index.tolist()
        np.random.seed(42)
        np.random.shuffle(indices)
        
        n = len(indices)
        n_train = int(n * TRAIN_SPLIT)
        n_val = int(n * VAL_SPLIT)
        
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])
    
    train_df = df.loc[train_indices].reset_index(drop=True)
    val_df = df.loc[val_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)
    
    print(f"✓ Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"✓ Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"✓ Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df

def convert_to_training_format(df, split_name):
    """Convert DataFrame to VLM training format with contextual metadata."""
    training_data = []
    
    for idx, row in df.iterrows():
        # Safe conversion of game_id and play_id (may be strings or ints)
        game_id = str(row['game_id']).strip()
        play_id = str(row['play_id']).strip()
        
        # Build contextual string
        context_str = f"""Down {int(row.get('down', 0))}-{int(row.get('yards_to_go', 0))}
Zone: {row.get('red_zone', 0) == 1 and 'RedZone' or (row.get('goal_line_situation', 0) == 1 and 'GoalLine' or 'Field')}
Situation: {row.get('down_situation', 'unknown')}
Formation: {row.get('formation_context', 'unknown')}""".strip()
        
        sample = {
            "id": f"{game_id}_{play_id}",
            "game_id": game_id,
            "play_id": play_id,
            "season": int(row['season']) if pd.notna(row.get('season')) else 0,
            "week": int(row['week']) if pd.notna(row.get('week')) else 0,
            "home_team": str(row.get('home_team', '')),
            "away_team": str(row.get('away_team', '')),
            "posteam": str(row.get('posteam', '')),
            "defteam": str(row.get('defteam', '')),
            "qtr": int(row.get('qtr', 0)) if pd.notna(row.get('qtr')) else 0,
            "game_seconds_remaining": int(row.get('game_seconds_remaining', 0)) if pd.notna(row.get('game_seconds_remaining')) else 0,
            
            # Visual & Context
            "play_description": str(row.get('play_description', '')),
            "video_filename": f"{game_id}_{play_id}.mp4",
            "context_metadata": context_str,
            
            # Labels
            "has_penalty": bool(row['has_penalty']),
            "penalty_type": str(row['penalty_type']),
            
            # Context Features
            "down": int(row.get('down', 0)) if pd.notna(row.get('down')) else 0,
            "yards_to_go": int(row.get('yards_to_go', 0)) if pd.notna(row.get('yards_to_go')) else 0,
            "pass_depth": str(row.get('pass_depth', 'unknown')),
            "qb_pressure": bool(row.get('qb_pressure_situation', 0)),
            "red_zone": bool(row.get('red_zone', 0)),
            "contact_likelihood": float(row.get('contact_likelihood', 0.0)) if pd.notna(row.get('contact_likelihood')) else 0.0,
            
            # Game Context (EPA/WPA)
            "epa": float(row.get('epa', 0.0)) if pd.notna(row.get('epa')) else 0.0,
            "wpa": float(row.get('wpa', 0.0)) if pd.notna(row.get('wpa')) else 0.0,
            "play_type": str(row.get('play_type', '')),
        }
        training_data.append(sample)
    
    return training_data

def save_dataset_splits(train_df, val_df, test_df, valid_penalties):
    """Save splits and metadata."""
    print("\nSaving dataset splits...")
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet for efficient loading
    train_df.to_parquet(PROCESSED_DIR / "train_split.parquet", index=False)
    val_df.to_parquet(PROCESSED_DIR / "val_split.parquet", index=False)
    test_df.to_parquet(PROCESSED_DIR / "test_split.parquet", index=False)
    
    # Convert to training format and save JSON
    print("Converting to training format (this may take a few minutes)...")
    train_data = convert_to_training_format(train_df, "train")
    val_data = convert_to_training_format(val_df, "val")
    test_data = convert_to_training_format(test_df, "test")
    
    with open(PROCESSED_DIR / "train_dataset.json", "w") as f:
        json.dump(train_data, f, indent=2)
    
    with open(PROCESSED_DIR / "val_dataset.json", "w") as f:
        json.dump(val_data, f, indent=2)
    
    with open(PROCESSED_DIR / "test_dataset.json", "w") as f:
        json.dump(test_data, f, indent=2)
    
    print(f"✓ Splits saved to {PROCESSED_DIR}/")
    
    # Generate statistics
    stats = {
        "total_plays": len(train_df) + len(val_df) + len(test_df),
        "train_plays": len(train_df),
        "val_plays": len(val_df),
        "test_plays": len(test_df),
        "penalty_types": len(valid_penalties),
        "penalty_types_list": valid_penalties,
        "penalty_distribution": train_df['penalty_type'].value_counts().to_dict(),
        "contextual_features": [
            "down_situation", "game_clock_state", "pass_depth",
            "yards_after_catch_cat", "formation_context", "receiver_group",
            "rusher_present", "qb_pressure_situation", "goal_line_situation",
            "red_zone", "contact_likelihood"
        ],
        "data_source": {
            "pbp": "nflverse (1999-2025)",
            "bdb": "GitHub releases (if available) OR derived features",
            "version": "2.2_fixed_game_id_handling"
        }
    }
    
    with open(PROCESSED_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset Statistics:")
    print(f"  Total Plays: {stats['total_plays']:,}")
    print(f"  Penalty Types: {stats['penalty_types']}")
    print(f"  Top 5 Penalties: {list(stats['penalty_distribution'].items())[:5]}")
    print(f"  Contextual Features: {len(stats['contextual_features'])} added")

def main():
    print("="*70)
    print("NFL VLM Dataset Processing Pipeline (Enhanced with Contextual Features)")
    print("="*70)
    
    # Load data
    pbp_df = load_pbp_data()
    
    # Attempt to load BDB tracking data (graceful fallback)
    bdb_df = load_bdb_tracking_data(max_weeks=5)
    
    # Enrich PBP with contextual features (always)
    pbp_df = enrich_pbp_with_contextual_features(pbp_df)
    
    # Merge with BDB if available
    pbp_df = merge_pbp_with_bdb(pbp_df, bdb_df)
    
    # Extract features
    pbp_df = extract_penalty_features(pbp_df)
    
    # Filter valid penalties
    balanced_df, valid_penalties = filter_valid_penalties(pbp_df)
    
    # Split
    train_df, val_df, test_df = stratified_train_test_split(balanced_df)
    
    # Save
    save_dataset_splits(train_df, val_df, test_df, valid_penalties)
    
    print("\n" + "="*70)
    print("✓ Dataset processing complete!")
    print(f"\nNext step: python scripts/05_train_vlm.py")
    print("="*70)

if __name__ == "__main__":
    main()