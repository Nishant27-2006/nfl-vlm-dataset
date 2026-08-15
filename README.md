# nfl-vlm-dataset

A data pipeline that assembles an NFL play-level dataset for **vision-language model** training — pairing play footage with the play-by-play context and rulebook text needed to reason about what happened, with a particular focus on penalty recognition.

## Pipeline

**1. Acquire sources** — `nfl_data.py`

- Play-by-play, 1999–2024, from [nflverse](https://github.com/nflverse)
- NFL Big Data Bowl (2023) tracking data from Kaggle
- The NFL Rulebook (PDF) from NFL Operations

**2. Align into VLM training format** — `nflp2.py`

Joins Big Data Bowl tracking data to play-by-play where available, falling back to contextual features derived from PBP when it isn't. Builds balanced penalty / no-penalty splits and emits a training JSON with metadata. Handles nflverse column-name drift and the string `game_id` / `play_id` typing that breaks naive joins.

**3. Fetch play video** — `nflp3.py`, `scripts/03e_download_from_youtube.py`, `scripts/03f_download_single_plays.py`

Resolves individual plays to video via public page parsing (the old `feeds.nfl.com` API is deprecated), with YouTube-based fallbacks for single plays.

**Bulk PBP download** — `download_nfl.py` pulls seasons 1999–2025 into `nfl_pbp_data/`.

## Layout

```
nfl_data.py       # script 1 - source acquisition
nflp2.py          # script 2 - alignment into VLM training format
nflp3.py          # script 3 - video retrieval via page scraping
download_nfl.py   # bulk play-by-play download, 1999-2025
scripts/          # per-play and YouTube download helpers
data/rulebook/    # NFL rulebook PDF
```

## Regenerating the data

Generated artifacts are gitignored — `data/processed/`, `data/raw/`, `nfl_pbp_data/`, and `*.parquet` total several GB. Rebuild them with:

```bash
pip install -r requirements.txt   # requests, pandas, tqdm
python nfl_data.py
python nflp2.py
```

## Note on footage

NFL game video is copyrighted by the NFL and its broadcast partners. The download scripts are provided for research use; clips should be treated as fetched-at-runtime artifacts rather than redistributed.
