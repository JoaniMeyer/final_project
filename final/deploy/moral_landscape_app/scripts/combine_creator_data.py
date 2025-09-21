#!/usr/bin/env python3
"""
Combine all creator data from combined_clean CSV files into one all_creators.parquet
"""

import os
import pandas as pd
import glob
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_combined_clean_files(data_dir: str = "../../../../data") -> List[str]:
    """Find all combined_clean.csv files"""
    pattern = os.path.join(data_dir, "**", "*_combined_clean.csv")
    files = glob.glob(pattern, recursive=True)
    logger.info(f"Found {len(files)} combined_clean.csv files")
    return files

def load_and_clean_csv(file_path: str) -> pd.DataFrame:
    """Load and clean a single CSV file"""
    logger.info(f"Loading: {file_path}")
    
    # Extract creator name from path
    path_parts = file_path.split('/')
    creator_name = None
    for i, part in enumerate(path_parts):
        if part in ['ubuntu', 'middle', 'chaos']:
            if i + 1 < len(path_parts):
                creator_name = path_parts[i + 1]
                break
    
    # Load CSV
    df = pd.read_csv(file_path)
    
    # Add creator metadata
    df['creator_name'] = creator_name
    df['creator_category'] = path_parts[-3] if len(path_parts) >= 3 else 'unknown'
    
    # Clean up empty rows
    df = df.dropna(subset=['original_text'])
    df = df[df['original_text'].str.strip() != '']
    
    # Generate anon_user_id if not present
    if 'anon_user_id' not in df.columns:
        df['anon_user_id'] = f"user_{creator_name}_{df.index}"
    
    # Add lang column if not present
    if 'lang' not in df.columns:
        df['lang'] = 'en'  # Default to English
    
    # Ensure required columns exist
    required_columns = [
        'video_id', 'comment_id', 'original_text', 'emoji_only', 
        'anon_user_id', 'lang', 'creator_name', 'creator_category'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            if col == 'emoji_only':
                df[col] = False
            elif col == 'lang':
                df[col] = 'en'
            else:
                df[col] = ''
    
    logger.info(f"Loaded {len(df)} rows from {creator_name}")
    result_df = df.copy()
    assert isinstance(result_df, pd.DataFrame)
    return result_df

def create_thread_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Create thread structure from reply_to_comment_id"""
    logger.info("Creating thread structure...")
    
    # Create thread_id based on reply chains
    df = df.copy()
    
    # Initialize thread_id as comment_id for top-level comments
    df['thread_id'] = df['comment_id']
    
    # For replies, use the parent comment's thread_id
    reply_mask = df['reply_to_comment_id'].notna() & (df['reply_to_comment_id'] != 'None')
    
    if reply_mask.sum() > 0:
        # Create mapping from comment_id to thread_id
        thread_mapping = dict(zip(df['comment_id'], df['thread_id']))
        
        # Update thread_id for replies
        for idx in df[reply_mask].index:
            reply_to_id = df.loc[idx, 'reply_to_comment_id']
            if reply_to_id in thread_mapping:
                df.loc[idx, 'thread_id'] = thread_mapping[reply_to_id]
    
    # Add comment order within threads
    df['comment_order'] = df.groupby('thread_id').cumcount()
    
    # Add reply order for replies
    df['reply_order'] = 0
    reply_mask = df['reply_to_comment_id'].notna() & (df['reply_to_comment_id'] != 'None')
    if reply_mask.sum() > 0:
        df.loc[reply_mask, 'reply_order'] = df[reply_mask].groupby('reply_to_comment_id').cumcount() + 1
    
    logger.info(f"Created {df['thread_id'].nunique()} threads")
    return df

def prepare_text_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare text for inference by combining original and context"""
    logger.info("Preparing text for inference...")
    
    df = df.copy()
    
    # Create text_used_for_inference column
    df['text_used_for_inference'] = df['original_text']
    
    # For replies, add context from the comment being replied to
    reply_mask = df['reply_to_comment_id'].notna() & (df['reply_to_comment_id'] != 'None')
    
    if reply_mask.sum() > 0:
        # Create mapping from comment_id to original_text
        text_mapping = dict(zip(df['comment_id'], df['original_text']))
        
        for idx in df[reply_mask].index:
            reply_to_id = df.loc[idx, 'reply_to_comment_id']
            if reply_to_id in text_mapping:
                context_text = text_mapping[reply_to_id]
                current_text = df.loc[idx, 'original_text']
                # Combine context and current text
                df.loc[idx, 'text_used_for_inference'] = f"Context: {context_text} | Reply: {current_text}"
    
    return df

def main():
    """Main function to combine all creator data"""
    logger.info("Starting to combine creator data...")
    
    # Find all combined_clean files
    files = find_combined_clean_files()
    
    if not files:
        logger.error("No combined_clean.csv files found!")
        return
    
    # Load and combine all files
    dfs = []
    for file_path in files:
        try:
            df = load_and_clean_csv(file_path)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            continue
    
    if not dfs:
        logger.error("No data loaded!")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined {len(combined_df)} total rows")
    
    # Create thread structure
    combined_df = create_thread_structure(combined_df)
    
    # Prepare text for inference
    combined_df = prepare_text_for_inference(combined_df)
    
    # Remove duplicates based on comment_id
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['comment_id'], keep='first')
    final_count = len(combined_df)
    
    logger.info(f"Removed {initial_count - final_count} duplicate comments")
    
    # Select and reorder columns
    columns = [
        'video_id', 'comment_id', 'anon_user_id', 'original_text', 
        'text_used_for_inference', 'emoji_only', 'lang', 'creator_name', 
        'creator_category', 'thread_id', 'comment_order', 'reply_order',
        'reply_to_comment_id', 'reply_to_original_text'
    ]
    
    # Only include columns that exist
    existing_columns = [col for col in columns if col in combined_df.columns]
    combined_df = combined_df[existing_columns]
    
    # Ensure comment_id and reply_to_comment_id are strings
    if 'comment_id' in combined_df.columns:
        combined_df['comment_id'] = combined_df['comment_id'].astype(str)
    if 'reply_to_comment_id' in combined_df.columns:
        combined_df['reply_to_comment_id'] = combined_df['reply_to_comment_id'].astype(str)
    if 'thread_id' in combined_df.columns:
        combined_df['thread_id'] = combined_df['thread_id'].astype(str)
    
    # Save to parquet
    output_dir = "../data/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "all_creators.parquet")
    
    combined_df.to_parquet(output_file, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("CREATOR DATA COMBINATION SUMMARY")
    print("="*60)
    print(f"Total Comments: {len(combined_df)}")
    print(f"Total Threads: {combined_df['thread_id'].nunique()}")  # type: ignore
    print(f"Total Videos: {combined_df['video_id'].nunique()}")  # type: ignore
    print(f"Total Creators: {combined_df['creator_name'].nunique()}")  # type: ignore
    print(f"Output File: {output_file}")
    print(f"File Size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    
    print("\nCreator Categories:")
    creator_cats = combined_df['creator_category'].value_counts()  # type: ignore
    print(creator_cats)
    
    print("\nCreator Names:")
    creator_names = combined_df['creator_name'].value_counts().head(10)  # type: ignore
    print(creator_names)
    
    print("\nThread Statistics:")
    thread_sizes = combined_df.groupby('thread_id').size()  # type: ignore
    print(f"  Average thread size: {float(thread_sizes.mean()):.2f}")
    print(f"  Max thread size: {int(thread_sizes.max())}")
    print(f"  Threads with replies: {int((thread_sizes > 1).sum())}")
    
    print("="*60)

if __name__ == "__main__":
    main()
