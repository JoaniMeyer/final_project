import os
# Force Transformers to ignore TensorFlow completely
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pandas as pd
import json
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

def load_time_data():
    """Load all new time data from time folders"""
    time_folders = [
        '../../../data/chaos/drphillips/time/tiktok-comments-scraper_drphillips.csv',
        '../../../data/chaos/drphillips/time/tiktok-scraper_drphillips.csv',
        '../../../data/middle/pieter/time/tiktok-comments-scraper_thepieterkriel.csv',
        '../../../data/middle/pieter/time/tiktok-scraper_thepieterkriel.csv',
        '../../../data/ubuntu/dodo/time/tiktok-comments-scraper_dodo.csv',
        '../../../data/ubuntu/dodo/time/tiktok-scraper_dodo.csv'
    ]
    
    all_time_data = []
    
    for folder in time_folders:
        try:
            df = pd.read_csv(folder)
            # Add source column based on folder name
            if 'thepieterkriel' in folder:
                source = 'thepieterkriel'
            elif 'dodo' in folder:
                source = 'dodo'
            elif 'drphillips' in folder:
                source = 'drphillips'
            else:
                source = folder.split('/')[-2]  # Fallback
            df['source'] = source
            all_time_data.append(df)
            print(f"Loaded {len(df)} comments from {source}")
        except FileNotFoundError:
            print(f"File not found: {folder}")
        except Exception as e:
            print(f"Error loading {folder}: {e}")
    
    if all_time_data:
        combined = pd.concat(all_time_data, ignore_index=True)
        print(f"\nTotal time data loaded: {len(combined)} comments")
        return combined
    else:
        print("No time data found!")
        return pd.DataFrame()

def load_existing_scored_data():
    """Load existing V4 scored data to avoid duplicates"""
    try:
        scored = pd.read_parquet('../moral_landscape_app/data/processed/scores/part-2e9ef066-5f9f-478c-aae7-a5b0f0613a18.parquet')
        
        # Add source column if it doesn't exist (for existing data)
        if 'source' not in scored.columns:
            # Try to infer source from other columns or set a default
            if 'creator_id' in scored.columns:
                scored['source'] = scored['creator_id']
            else:
                # For existing data without source info, we'll need to map it differently
                # For now, mark as 'existing_data' so we can filter it out later
                scored['source'] = 'existing_data'
        
        print(f"Loaded {len(scored)} existing scored comments")
        return scored
    except Exception as e:
        print(f"Error loading existing scored data: {e}")
        return pd.DataFrame()

def filter_new_comments(time_data, existing_scored):
    """Filter out comments that are already classified"""
    if existing_scored.empty:
        return time_data
    
    # Create a set of existing texts for fast lookup
    existing_texts = set(existing_scored['text'].str.lower().str.strip())
    
    # Filter time data to only include new comments
    new_comments = []
    for _, row in time_data.iterrows():
        if pd.notna(row['text']):
            text_lower = str(row['text']).lower().strip()
            if text_lower not in existing_texts:
                new_comments.append(row)
    
    filtered_df = pd.DataFrame(new_comments)
    print(f"Filtered to {len(filtered_df)} new comments (removed {len(time_data) - len(filtered_df)} duplicates)")
    
    # Extract video_id from videoWebUrl if available
    if 'videoWebUrl' in filtered_df.columns and 'video_id' not in filtered_df.columns:
        # Extract video ID from URL pattern like: https://www.tiktok.com/@username/video/1234567890
        filtered_df['video_id'] = filtered_df['videoWebUrl'].str.extract(r'/video/(\d+)')[0]
        # Fill any missing values with 'unknown_video'
        filtered_df['video_id'] = filtered_df['video_id'].fillna('unknown_video')
        print(f"Extracted video_id from videoWebUrl")
    
    return filtered_df

def load_cultural_dictionary():
    """Load South African cultural dictionary for enhanced classification"""
    try:
        # Try improved dictionary first
        dict_path = "config/sa_cultural_dict_improved.json"
        if os.path.exists(dict_path):
            with open(dict_path, "r") as f:
                cultural_dict = json.load(f)
            print(f"Loaded improved cultural dictionary with {len(cultural_dict)} terms")
            return cultural_dict
        else:
            # Fallback to old dictionary
            dict_path = "../moral_landscape_app/dictionary/sa_cultural_dict.json"
            if os.path.exists(dict_path):
                with open(dict_path, "r") as f:
                    cultural_dict = json.load(f)
                print(f" Loaded cultural dictionary with {len(cultural_dict)} terms")
                return cultural_dict
            else:
                print(" Cultural dictionary not found")
                return []
    except Exception as e:
        print(f" Could not load cultural dictionary: {e}")
        return []

def enhance_text_with_cultural_context(text, cultural_dict):
    """Enhance text with cultural context for better classification using improved dictionary"""
    if not cultural_dict:
        return text
    
    enhanced_text = text.lower()
    cultural_terms_found = []
    cultural_categories = []
    
    # Find cultural terms in the text using improved dictionary structure
    for item in cultural_dict:
        if isinstance(item, dict) and 'term' in item:
            term = item['term'].lower().strip()
            if term in enhanced_text:
                cultural_terms_found.append(term)
                
                # Extract categories for better context
                if 'category' in item and isinstance(item['category'], list):
                    cultural_categories.extend(item['category'])
                
                # Also check variants if they exist
                if 'variants' in item and isinstance(item['variants'], list):
                    for variant in item['variants']:
                        if variant.lower().strip() in enhanced_text:
                            cultural_terms_found.append(f"{term}({variant})")
    
    # Add enhanced cultural context if terms found
    if cultural_terms_found:
        # Remove duplicates and sort for consistency
        unique_terms = sorted(list(set(cultural_terms_found)))
        unique_categories = sorted(list(set(cultural_categories)))
        
        cultural_context = f" [CULTURAL_TERMS: {', '.join(unique_terms)}]"
        if unique_categories:
            cultural_context += f" [CULTURAL_CATEGORIES: {', '.join(unique_categories)}]"
        
        enhanced_text += cultural_context
    
    return enhanced_text

def classify_new_comments(new_comments):
    """Classify new comments using V4 ensemble model with cultural context"""
    if new_comments.empty:
        print("No new comments to classify")
        return pd.DataFrame()
    
    print(f"\nClassifying {len(new_comments)} new comments with V4 ensemble model...")
    
    # Load cultural dictionary for enhanced classification
    cultural_dict = load_cultural_dictionary()
    
    # Load V4 ensemble model files
    model_base_path = '../moral_landscape_app/african_moral_classifier_V4'
    
    try:
        # Load model info
        with open(f'{model_base_path}/model_info.json', 'r') as f:
            model_info = json.load(f)
        
        # Load checkpoint paths
        with open(f'{model_base_path}/checkpoint_paths.json', 'r') as f:
            checkpoint_info = json.load(f)
        
        print(f"V4 ensemble model loaded: {model_info['model_type']}")
        print(f"Ensemble size: {model_info['ensemble_size']}")
        print(f"Base model: {model_info['base_model']}")
        
        # Load just one model for now (simpler approach)
        model_path = f"{model_base_path}/{checkpoint_info['checkpoint_paths'][0]}"
        print(f"Loading single model: {model_path}")
        
        # Load tokenizer and model with specific settings to avoid TensorFlow
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_info['base_model'], use_fast=True)
        print("Tokenizer loaded")
        
        print("Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map=None
        )
        model.eval()
        print("Model loaded")
        
        # Use M1 optimized settings
        if torch.backends.mps.is_available():
            print("Using M1 Metal Performance Shaders (MPS)")
            model = model.to('mps')
        else:
            print("Using CPU (MPS not available)")
            model = model.cpu()
        
        print("Single model loaded successfully")
        
        # Label mapping
        label_map = {"Ubuntu": 0, "Middle": 1, "Chaos": 2}
        reverse_label_map = {0: "Ubuntu", 1: "Middle", 2: "Chaos"}
        
        # Prepare texts with cultural context enhancement
        original_texts = new_comments['text'].fillna('').astype(str).tolist()
        enhanced_texts = []
        
        print("Enhancing texts with cultural context...")
        for text in original_texts:
            enhanced_text = enhance_text_with_cultural_context(text, cultural_dict)
            enhanced_texts.append(enhanced_text)
        
        # Classify in batches
        batch_size = 16  # Smaller batch size for ensemble
        all_results = []
        
        for i in range(0, len(enhanced_texts), batch_size):
            batch_texts = enhanced_texts[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(enhanced_texts) + batch_size - 1)//batch_size}")
            
            try:
                # Get predictions from single model
                # Tokenize enhanced texts
                inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                                 max_length=model_info['max_length'], return_tensors="pt")
                
                # Move inputs to same device as model
                if torch.backends.mps.is_available():
                    inputs = {k: v.to('mps') for k, v in inputs.items()}
                
                # Get predictions
                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)
                
                # Process results
                for j, (enhanced_text, pred, probs) in enumerate(zip(batch_texts, predictions, probabilities)):
                    idx = i + j
                    if idx < len(new_comments):
                        row = new_comments.iloc[idx].copy()
                        row['label'] = reverse_label_map[pred.item()]  # type: ignore
                        row['proba_Ubuntu'] = probs[0].item()  # type: ignore
                        row['proba_Middle'] = probs[1].item()  # type: ignore
                        row['proba_Chaos'] = probs[2].item()  # type: ignore
                        row['model_type'] = 'realistic_ensemble_transformer_v4'
                        
                        # Add cultural context info (enhanced format)
                        if '[CULTURAL_TERMS:' in enhanced_text:
                            cultural_terms = enhanced_text.split('[CULTURAL_TERMS:')[1].split(']')[0]
                            row['cultural_terms'] = cultural_terms
                            
                            # Also extract categories if present
                            if '[CULTURAL_CATEGORIES:' in enhanced_text:
                                cultural_categories = enhanced_text.split('[CULTURAL_CATEGORIES:')[1].split(']')[0]
                                row['cultural_categories'] = cultural_categories
                            else:
                                row['cultural_categories'] = ''
                        else:
                            row['cultural_terms'] = ''
                            row['cultural_categories'] = ''
                        
                        all_results.append(row)
            
            except Exception as e:
                print(f"Error in batch {i//batch_size + 1}: {e}")
                # Skip failed batches - no fallback
                continue
        
        results_df = pd.DataFrame(all_results)
        print(f"Successfully classified {len(results_df)} comments with V4 ensemble")
        return results_df
        
    except Exception as e:
        print(f"Error loading V4 ensemble model: {e}")
        print("Cannot proceed without real V4 classifications")
        return pd.DataFrame()  # Return empty DataFrame - no fallback

def save_results(new_scored_data, existing_scored_data):
    """Save new results with timestamp to avoid overwriting"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create new scores directory
    new_scores_dir = f'../moral_landscape_app/data/processed/scores_new_{timestamp}'
    os.makedirs(new_scores_dir, exist_ok=True)
    
    # Clean problematic columns for parquet save
    columns_to_keep = ['text', 'source', 'label', 'proba_Ubuntu', 'proba_Middle', 'proba_Chaos', 'model_type', 'cultural_terms', 'cultural_categories']
    
    # Essential columns
    essential_cols = ['createTimeISO', 'diggCount', 'replyCommentTotal', 'videoWebUrl', 'video_id']
    for col in essential_cols:
        if col in new_scored_data.columns:
            columns_to_keep.append(col)
    
    # Reply structure columns
    reply_cols = ['repliesToId', 'reply_to_comment_id', 'replyToId', 'cid', 'comment_id', 'id', 'uid', 'uniqueId']
    for col in reply_cols:
        if col in new_scored_data.columns:
            columns_to_keep.append(col)
    
    # Creator interaction columns
    creator_cols = ['likedByAuthor', 'pinnedByAuthor', 'is_creator_comment', 'creator_name', 'creator_id']
    for col in creator_cols:
        if col in new_scored_data.columns:
            columns_to_keep.append(col)
    
    # Additional useful columns
    additional_cols = ['input', 'lang', 'created_at']
    for col in additional_cols:
        if col in new_scored_data.columns:
            columns_to_keep.append(col)
    
    # Filter to only keep safe columns
    safe_data = new_scored_data[columns_to_keep].copy()
    
    # Save new scored data
    new_file = f'{new_scores_dir}/part-time-data-{timestamp}.parquet'
    safe_data.to_parquet(new_file, index=False)
    print(f"Saved new scored data: {new_file}")
    
    # Combine with existing data
    if not existing_scored_data.empty:
        # Clean existing data too - check which columns actually exist
        existing_columns = ['text', 'label', 'proba_Ubuntu', 'proba_Middle', 'proba_Chaos', 'model_type']
        if 'source' in existing_scored_data.columns:
            existing_columns.append('source')
        if 'createTimeISO' in existing_scored_data.columns:
            existing_columns.append('createTimeISO')
        if 'diggCount' in existing_scored_data.columns:
            existing_columns.append('diggCount')
        if 'replyCommentTotal' in existing_scored_data.columns:
            existing_columns.append('replyCommentTotal')
        
        existing_safe = existing_scored_data[existing_columns].copy()
        combined_data = pd.concat([existing_safe, safe_data], ignore_index=True)
        combined_file = f'{new_scores_dir}/part-combined-{timestamp}.parquet'
        combined_data.to_parquet(combined_file, index=False)
        print(f"Saved combined data: {combined_file}")
        
        # Save summary
        summary = {
            'timestamp': timestamp,
            'new_comments_classified': len(new_scored_data),
            'existing_comments': len(existing_scored_data),
            'total_comments': len(combined_data),
            'new_file': new_file,
            'combined_file': combined_file,
            'moral_distribution': new_scored_data['label'].value_counts().to_dict()
        }
        
        summary_file = f'{new_scores_dir}/classification_summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary: {summary_file}")
        
        return combined_file
    else:
        return new_file

def main():
    print("Starting V4 Classification of New Time Data")
    print("=" * 50)
    
    # Step 1: Load time data
    print("\n1. Loading time data...")
    time_data = load_time_data()
    
    if time_data.empty:
        print("No time data found. Exiting.")
        return
    
    # Step 2: Load existing scored data
    print("\n2. Loading existing scored data...")
    existing_scored = load_existing_scored_data()
    
    # Step 3: Filter new comments
    print("\n3. Filtering new comments...")
    new_comments = filter_new_comments(time_data, existing_scored)
    
    if new_comments.empty:
        print("No new comments to classify. Exiting.")
        return
    
    # Step 4: Classify new comments
    print("\n4. Classifying new comments...")
    new_scored_data = classify_new_comments(new_comments)
    
    if new_scored_data.empty:
        print("Classification failed. Exiting.")
        return
    
    # Step 5: Save results
    print("\n5. Saving results...")
    combined_file = save_results(new_scored_data, existing_scored)
    
    print("\nClassification Complete!")
    print(f"New comments classified: {len(new_scored_data)}")
    print(f"Moral distribution: {new_scored_data['label'].value_counts().to_dict()}")
    print(f"Combined data saved: {combined_file}")
    
    print("\nNext steps:")
    print("1. Review the new classifications")
    print("2. Update the dashboard data integration if satisfied")
    print("3. Test the updated dashboard")

if __name__ == "__main__":
    main()
