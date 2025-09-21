#!/usr/bin/env python3
"""
Load thread data from the time folders and create thread structure
"""

import pandas as pd
import json
import os
from pathlib import Path

def load_creator_thread_data(creator_path, creator_name=None):
    """Load thread data for a specific creator"""
    if creator_name is None:
        creator_name = creator_path.split('/')[-2]
    csv_file = os.path.join(creator_path, f"tiktok-comments-scraper_{creator_name}.csv")
    
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return None
    
    # Load the CSV
    df = pd.read_csv(csv_file)
    
    # Rename columns to match thread builder expectations
    df = df.rename(columns={
        'cid': 'comment_id',
        'repliesToId': 'reply_to_comment_id',
        'text': 'text',
        'videoWebUrl': 'video_id',
        'createTimeISO': 'timestamp',
        'diggCount': 'digg_count',
        'replyCommentTotal': 'reply_count'
    })
    
    # Clean up the data
    df['comment_id'] = df['comment_id'].astype(str)
    
    # Handle reply_to_comment_id carefully
    df['reply_to_comment_id'] = df['reply_to_comment_id'].fillna('')
    df['reply_to_comment_id'] = df['reply_to_comment_id'].astype(str)
    df['reply_to_comment_id'] = df['reply_to_comment_id'].replace('nan', '')
    df['reply_to_comment_id'] = df['reply_to_comment_id'].replace('None', '')
    df['reply_to_comment_id'] = df['reply_to_comment_id'].replace('', None)
    
    # Ensure text is string
    df['text'] = df['text'].astype(str)
    
    # Add creator info
    creator_name = creator_path.split('/')[-2]
    category = creator_path.split('/')[-3]
    df['creator'] = creator_name
    df['category'] = category
    
    return df

def build_threads_for_creator(df):
    """Build thread structure for a creator's data using current approach"""
    try:
        # Create thread structure using the current approach
        df = df.copy()
        
        # Create thread_id based on reply chains
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
        
        # Create thread structure
        threads = []
        for thread_id, thread_group in df.groupby('thread_id'):
            # Get the parent comment (first comment in thread)
            parent_comments = thread_group[thread_group['reply_to_comment_id'].isna()]
            
            # Skip if no parent comment found
            if len(parent_comments) == 0:
                continue
                
            parent_comment = parent_comments.iloc[0]
            
            # Get all replies in this thread
            replies = thread_group[thread_group['reply_to_comment_id'].notna()].sort_values('comment_order')
            
            thread = {
                'thread_id': thread_id,
                'video_id': parent_comment['video_id'],
                'parent_comment': parent_comment['text'],
                'parent_comment_id': parent_comment['comment_id'],
                'replies': replies[['comment_id', 'text', 'reply_to_comment_id', 'comment_order']].to_dict('records'),
                'total_replies': len(replies),
                'creator': parent_comment['creator'],
                'category': parent_comment['category']
            }
            threads.append(thread)
        
        # Flatten for display
        flattened_threads = []
        for thread in threads:
            # Add parent comment
            flattened_threads.append({
                'thread_id': thread['thread_id'],
                'video_id': thread['video_id'],
                'comment_id': thread['parent_comment_id'],
                'text': thread['parent_comment'],
                'depth': 0,
                'parent_id': 'ROOT',
                'comment_order': 0,
                'creator': thread['creator'],
                'category': thread['category']
            })
            
            # Add replies
            for reply in thread['replies']:
                flattened_threads.append({
                    'thread_id': thread['thread_id'],
                    'video_id': thread['video_id'],
                    'comment_id': reply['comment_id'],
                    'text': reply['text'],
                    'depth': 1,
                    'parent_id': reply['reply_to_comment_id'],
                    'comment_order': reply['comment_order'],
                    'creator': thread['creator'],
                    'category': thread['category']
                })
        
        return {
            'threads': threads,
            'flattened': flattened_threads,
            'total_threads': len(threads),
            'total_comments': len(flattened_threads)
        }
    except Exception as e:
        print(f"Error building threads: {e}")
        return None

def main():
    """Load all creator thread data"""
    
    # Get the project root directory (3 levels up from this script)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    
    # Define the paths with correct creator names using relative paths
    creators = [
        (project_root / 'data' / 'chaos' / 'drphillips' / 'time', 'drphillips'),
        (project_root / 'data' / 'middle' / 'pieter' / 'time', 'thepieterkriel'), 
        (project_root / 'data' / 'ubuntu' / 'dodo' / 'time', 'dodo')
    ]
    
    all_thread_data = {}
    
    for creator_path, creator_name in creators:
        print(f"Loading thread data for {creator_name}...")
        
        # Load the data
        df = load_creator_thread_data(str(creator_path), creator_name)
        if df is None:
            continue
        
        print(f"   Loaded {len(df)} comments")
        
        # Build thread structure
        thread_data = build_threads_for_creator(df)
        if thread_data is None:
            continue
        
        print(f"   Built {thread_data['total_threads']} threads")
        
        # Store the data
        all_thread_data[creator_name] = {
            'raw_data': df.to_dict('records'),
            'threads': thread_data['threads'],
            'flattened': thread_data['flattened'],
            'total_threads': thread_data['total_threads'],
            'total_comments': thread_data['total_comments']
        }
    
    # Create output directory if it doesn't exist
    output_dir = 'processed_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to JSON
    output_file = os.path.join(output_dir, 'thread_data.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_thread_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"Thread data saved to {output_file}")
    
    # Print summary
    for creator, data in all_thread_data.items():
        print(f"\n{creator}:")
        print(f"   Total comments: {data['total_comments']}")
        print(f"   Total threads: {data['total_threads']}")

if __name__ == "__main__":
    main()
