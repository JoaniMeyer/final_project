#!/usr/bin/env python3
"""
prepare_dashboard_data.py

Purpose:
  Build/refresh a single normalized dataset at:
    processed_data/integrated_comments.parquet

Notes:
  • The file reads raw CSV/Parquet files (or an already-combined file),
    normalizes columns, de-duplicates, and writes the Parquet that
    `run_enhanced_topics.py` expects.

Input options:
  --inputs <path ...>   Files or folders (CSV/Parquet). Folders are scanned
                        recursively for *.csv and *.parquet.

Output:
  processed_data/integrated_comments.parquet  (configurable via --out)

Required columns in the final output:
  - text:            str (comment text)
  - creator_id OR source:  str (who the comments belong to; either is fine)
Optional columns (carried through if present):
  - platform, lang, predicted_label, created_at, comment_id, video_id, post_id
"""

from __future__ import annotations
import os
import sys
import argparse
import glob
from typing import List, Dict, Any, Iterable, Optional

import pandas as pd


# --------------------------- helpers --------------------------- #

RAW_TEXT_CANDIDATES = [
    "text", "comment_text", "body", "content", "message"
]

CREATOR_CANDIDATES = [
    "creator_id", "source", "creator", "channel_id", "channel",
    "author_id", "author", "username", "user", "account"
]

PLATFORM_CANDIDATES = ["platform", "site", "source_platform"]
LANG_CANDIDATES = ["lang", "language", "detected_language"]
LABEL_CANDIDATES = ["predicted_label", "label", "class"]
CREATED_AT_CANDIDATES = ["created_at", "timestamp", "publish_time", "time", "createTimeISO"]
COMMENT_ID_CANDIDATES = ["comment_id", "id", "commentId"]
VIDEO_ID_CANDIDATES = ["video_id", "videoId"]
POST_ID_CANDIDATES = ["post_id", "postId", "status_id"]


def _first_present(cols: Iterable[str], candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in cols}
    for name in candidates:
        if name in cols_lower:
            return cols_lower[name]
    return None


def _list_files(inputs: List[str]) -> List[str]:
    files: List[str] = []
    for path in inputs:
        if os.path.isdir(path):
            for ext in ("*.csv", "*.parquet"):
                files.extend(glob.glob(os.path.join(path, "**", ext), recursive=True))
        elif os.path.isfile(path):
            files.append(path)
    # keep order stable but unique
    seen = set()
    uniq = []
    for p in files:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq


def _read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    # try parquet then csv
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path)


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    orig_cols = list(df.columns)

    # text
    text_col = _first_present(orig_cols, RAW_TEXT_CANDIDATES)
    if not text_col:
        raise ValueError("No text-like column found in input. Expected one of: " + ", ".join(RAW_TEXT_CANDIDATES))
    df.rename(columns={text_col: "text"}, inplace=True)

    # creator_id / source (we keep whichever is present; runner accepts either)
    creator_col = _first_present(orig_cols, CREATOR_CANDIDATES)
    if not creator_col:
        raise ValueError("No creator/source column found. Expected one of: " + ", ".join(CREATOR_CANDIDATES))
    # Prefer a stable 'creator_id' if the chosen name isn't already that
    if creator_col != "creator_id":
        # If the chosen is 'source', keep it as 'source'; else standardize to 'creator_id'
        if creator_col.lower() == "source":
            df.rename(columns={creator_col: "source"}, inplace=True)
        else:
            df.rename(columns={creator_col: "creator_id"}, inplace=True)

    # Optional keepers
    opt_maps: Dict[str, List[str]] = {
        "platform": PLATFORM_CANDIDATES,
        "lang": LANG_CANDIDATES,
        "moral_label": LABEL_CANDIDATES,  # Map to 'moral_label' for dashboard compatibility
        "timestamp": CREATED_AT_CANDIDATES,  # Map to 'timestamp' for dashboard compatibility
        "comment_id": COMMENT_ID_CANDIDATES,
        "video_id": VIDEO_ID_CANDIDATES,
        "post_id": POST_ID_CANDIDATES,
    }
    for out_name, cands in opt_maps.items():
        col = _first_present(orig_cols, cands)
        if col and out_name not in df.columns:
            df.rename(columns={col: out_name}, inplace=True)
    
    # Extract video_id from videoWebUrl if video_id doesn't exist but videoWebUrl does
    if "video_id" not in df.columns and "videoWebUrl" in df.columns:
        # Extract video ID from URL pattern like: /video/7543283321607048468
        df["video_id"] = df["videoWebUrl"].str.extract(r'/video/(\d+)')[0]
        # Fill any missing values with 'unknown_video'
        df["video_id"] = df["video_id"].fillna('unknown_video')
    
    # Create timestamp column from createTimeISO if timestamp doesn't exist
    if "timestamp" not in df.columns and "createTimeISO" in df.columns:
        df["timestamp"] = pd.to_datetime(df["createTimeISO"], errors="coerce")
        print("Created timestamp column from createTimeISO")

    # Minimal cleaning
    df["text"] = df["text"].astype(str).str.strip()
    df = df.loc[df["text"].str.len() > 0].copy()

    # Basic type normalization
    for maybe in ("creator_id", "source", "platform", "lang", "predicted_label",
                  "comment_id", "video_id", "post_id"):
        if maybe in df.columns:
            df[maybe] = df[maybe].astype(str)

    if "created_at" in df.columns:
        # Safe parse; keep original if parsing fails
        try:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        except Exception:
            pass

    # De-dup: by (creator, text) if creator present, else by text only
    if "creator_id" in df.columns:
        df = df.drop_duplicates(subset=["creator_id", "text"])
    elif "source" in df.columns:
        df = df.drop_duplicates(subset=["source", "text"])
    else:
        df = df.drop_duplicates(subset=["text"])

    return df


# --------------------------- main logic --------------------------- #

def build_integrated_comments(inputs: List[str], out_path: str) -> pd.DataFrame:
    if not inputs:
        raise ValueError("Please provide at least one input file or folder via --inputs")

    files = _list_files(inputs)
    if not files:
        raise ValueError("No CSV/Parquet files found in the provided --inputs")

    frames: List[pd.DataFrame] = []
    for p in files:
        try:
            df = _read_any(p)
            df = _normalize_frame(df)
            frames.append(df)
            print(f"✓ Loaded {len(df):,} rows from {p}")
        except Exception as e:
            print(f"Skipping {p}: {e}")

    if not frames:
        raise RuntimeError("All inputs failed to load/normalize; nothing to write.")

    combined = pd.concat(frames, ignore_index=True, copy=False)

    # If both 'creator_id' and 'source' exist, prefer 'creator_id' and keep 'source' as-is.
    # Runner accepts either one; having both is fine.
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined.to_parquet(out_path, index=False)
    print(f"\n Wrote {len(combined):,} rows to {out_path}")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize raw comment files into a single Parquet for topic modeling")
    parser.add_argument("--inputs", nargs="+", 
                        default=["../moral_landscape_app/data/processed/scores_new_20250914_103639/part-time-data-20250914_103639.parquet"],
                        help="Files or folders (CSV/Parquet). Folders scanned recursively. Default: latest classified time data only")
    parser.add_argument("--out", default=os.path.join("processed_data", "integrated_comments.parquet"),
                        help="Output Parquet path (default: processed_data/integrated_comments.parquet)")
    args = parser.parse_args()

    build_integrated_comments(args.inputs, args.out)


if __name__ == "__main__":
    main()
