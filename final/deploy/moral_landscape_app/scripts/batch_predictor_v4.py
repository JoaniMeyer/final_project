#!/usr/bin/env python3
"""
Batch Predictor V4 - Ensemble Inference for African Moral Classification
Scores every comment with the V4 ensemble (3 checkpoints)
"""

import os
import json
import uuid
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.calibration import CalibratedClassifierCV

# Set environment variables for optimization and TensorFlow compatibility
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA for TensorFlow
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnsembleInferenceLoader:
    """Loads and manages the V4 ensemble for inference"""
    
    def __init__(self, model_dir: str = "../african_moral_classifier_V4"):
        self.model_dir = model_dir
        self.device = self._get_device()
        self.models: List[Any] = []
        self.tokenizer: Optional[Any] = None
        self.model_info: Optional[Dict[str, Any]] = None
        self.calibration_params: Optional[Dict[str, Any]] = None
        self.checkpoint_paths: Optional[List[str]] = None
        
        logger.info(f"Initializing ensemble on device: {self.device}")
        self._load_metadata()
        self._load_ensemble()
    
    def _get_device(self) -> torch.device:
        """Choose the best available device"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _load_metadata(self):
        """Load metadata files"""
        # Load checkpoint paths
        checkpoint_paths_file = os.path.join(self.model_dir, "checkpoint_paths.json")
        with open(checkpoint_paths_file, 'r') as f:
            data = json.load(f)
            self.checkpoint_paths = data["checkpoint_paths"]
        
        # Load model info
        model_info_file = os.path.join(self.model_dir, "model_info.json")
        with open(model_info_file, 'r') as f:
            self.model_info = json.load(f)
        
        # Load calibration parameters
        calibration_file = os.path.join(self.model_dir, "calibration_params.json")
        with open(calibration_file, 'r') as f:
            self.calibration_params = json.load(f)
        
        if self.checkpoint_paths is None:
            raise ValueError("checkpoint_paths is None")
        if self.model_info is None:
            raise ValueError("model_info is None")
        if self.calibration_params is None:
            raise ValueError("calibration_params is None")
        
        if self.checkpoint_paths is None or self.model_info is None:
            raise ValueError("Failed to load metadata files")
            
        logger.info(f"Loaded {len(self.checkpoint_paths)} checkpoint paths")
        logger.info(f"Model type: {self.model_info['model_type']}")
        logger.info(f"Labels: {self.model_info['labels']}")
    
    def _load_ensemble(self):
        """Load all ensemble models"""
        # Load tokenizer from model root directory (not checkpoint)
        if self.checkpoint_paths is None or len(self.checkpoint_paths) == 0:
            raise ValueError("No checkpoint paths available")
        
        # Use the model root directory for tokenizer
        model_root = os.path.join(self.model_dir, "model_0")
        logger.info(f"Loading tokenizer from: {model_root}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_root)
        
        # Load all models with PyTorch-specific settings
        if self.checkpoint_paths is None or self.model_info is None:
            raise ValueError("checkpoint_paths or model_info is None")
            
        for i, checkpoint_path in enumerate(self.checkpoint_paths):
            # Construct full path to checkpoint
            full_checkpoint_path = os.path.join(self.model_dir, checkpoint_path)
            logger.info(f"Loading model {i+1}/{len(self.checkpoint_paths)} from: {full_checkpoint_path}")
            
            # Force PyTorch loading and avoid TensorFlow
            model = AutoModelForSequenceClassification.from_pretrained(
                full_checkpoint_path,
                num_labels=self.model_info['num_labels'],
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            model = model.to(self.device)
            model.eval()  # Set to evaluation mode
            self.models.append(model)
        
        logger.info(f"Successfully loaded {len(self.models)} models")
    
    def predict_batch(self, texts: List[str], batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions on a batch of texts
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        all_logits = []
        
        # Get predictions from each model
        if self.tokenizer is None or self.model_info is None:
            raise ValueError("Tokenizer or model_info not initialized")
            
        for i, model in enumerate(self.models):
            logger.info(f"Computing predictions from model {i+1}/{len(self.models)}")
            
            model_logits = []
            
            # Process in batches
            for j in range(0, len(texts), batch_size):
                batch_texts = texts[j:j + batch_size]
                
                # Tokenize
                max_length = 256
                if self.model_info is not None and isinstance(self.model_info, dict):
                    max_length = self.model_info.get('max_length', 256)
                inputs = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get logits
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    model_logits.append(logits.cpu().numpy())
            
            # Concatenate all batches for this model
            model_logits = np.concatenate(model_logits, axis=0)
            all_logits.append(model_logits)
        
        # Ensemble: average logits across models
        ensemble_logits = np.mean(all_logits, axis=0)
        
        # Apply temperature scaling if specified
        temperature = 1.0
        if self.calibration_params is not None and isinstance(self.calibration_params, dict):
            temperature = self.calibration_params.get('temperature', 1.0)
        if temperature != 1.0:
            ensemble_logits = ensemble_logits / temperature
        
        # Convert to probabilities
        probabilities = F.softmax(torch.tensor(ensemble_logits), dim=-1).numpy()
        predictions = np.argmax(probabilities, axis=-1)
        
        return predictions, probabilities

class BatchScorer:
    """Handles batch scoring of the dataset"""
    
    def __init__(self, ensemble_loader: EnsembleInferenceLoader):
        self.ensemble_loader = ensemble_loader
        self.run_id = str(uuid.uuid4())
        self.ts_scored = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"Initialized batch scorer with run_id: {self.run_id}")
    
    def score_dataset(self, input_file: str, output_dir: str = "data/processed/scores", 
                     batch_size: int = 64) -> Dict[str, Any]:
        """
        Score the entire dataset and write results to parquet shards
        
        Args:
            input_file: Path to all_creators.parquet
            output_dir: Directory to write output shards
            batch_size: Batch size for inference
        
        Returns:
            Dictionary with scoring statistics
        """
        logger.info(f"Loading dataset from: {input_file}")
        
        # Load data
        df = pd.read_parquet(input_file)  # type: ignore
        return self.score_dataframe(df, output_dir, batch_size)  # type: ignore
    
    def score_dataframe(self, df: pd.DataFrame, output_dir: str = "data/processed/scores", 
                       batch_size: int = 64) -> Dict[str, Any]:
        """
        Score a DataFrame and write results to parquet shards
        
        Args:
            df: DataFrame to score
            output_dir: Directory to write output shards
            batch_size: Batch size for inference
        
        Returns:
            Dictionary with scoring statistics
        """
        n_input = len(df)
        logger.info(f"Processing {n_input} rows from DataFrame")
        
        # Select text for inference
        text_column = 'text_used_for_inference'
        if text_column not in df.columns:
            logger.warning(f"Column '{text_column}' not found, using 'text_comment'")
            text_column = 'text_comment'
        # Filter out empty or whitespace-only texts, ensuring the result is a DataFrame
        mask_valid_text = df[text_column].notna() & (df[text_column].astype(str).str.strip() != '')
        df = df.loc[mask_valid_text].copy()
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Filtered result is not a DataFrame. Please check the input data.")
        n_scored = len(df)
        n_skipped_empty_text = n_input - n_scored

        logger.info(f"Scoring {n_scored} rows (skipped {n_skipped_empty_text} empty texts)")
        
        if n_scored == 0:
            logger.warning("No valid texts to score!")
            return {
                'n_input': n_input,
                'n_scored': 0,
                'n_skipped_empty_text': n_skipped_empty_text,
                'run_id': self.run_id
            }
        
        # Get texts for inference
        texts = df[text_column].tolist()
        
        # Make predictions
        logger.info("Starting batch inference...")
        start_time = time.time()
        
        predictions, probabilities = self.ensemble_loader.predict_batch(texts, batch_size)
        
        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.2f} seconds")
        
        # Create results dataframe
        if isinstance(df, pd.DataFrame):
            results_df = self._create_results_dataframe(df, predictions, probabilities)
        else:
            raise ValueError("df must be a pandas DataFrame")
        
        # Write to parquet
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"part-{self.run_id}.parquet")
        
        logger.info(f"Writing results to: {output_file}")
        results_df.to_parquet(output_file, index=False)
        
        # Calculate statistics
        stats = self._calculate_statistics(results_df, n_input, n_scored, n_skipped_empty_text)
        
        # Print summary
        self._print_summary(stats)
        
        return stats
    
    def _create_results_dataframe(self, df: pd.DataFrame, predictions: np.ndarray, 
                                 probabilities: np.ndarray) -> pd.DataFrame:
        """Create the results dataframe with all required columns"""
        
        # Map predictions to labels
        if self.ensemble_loader.model_info is None:
            raise ValueError("model_info is None")
        labels = self.ensemble_loader.model_info['labels']
        predicted_labels = [labels[pred] for pred in predictions]
        
        # Create results dataframe
        results = pd.DataFrame({
            # Join keys
            'video_id': df['video_id'],
            'comment_id': df['comment_id'],
            'anon_user_id': df['anon_user_id'],
            
            # Context
            'lang': df.get('lang', 'unknown'),
            'emoji_only': df.get('emoji_only', False),
            
            # Text used for inference
            'text': df['text_used_for_inference'] if 'text_used_for_inference' in df.columns else df['original_text'],
            
            # Predictions
            'label': predicted_labels,
            
            # Probabilities
            'proba_Ubuntu': probabilities[:, 0].astype(np.float32),
            'proba_Middle': probabilities[:, 1].astype(np.float32),
            'proba_Chaos': probabilities[:, 2].astype(np.float32),
            
            # Run metadata
            'run_id': self.run_id,
            'model_type': self.ensemble_loader.model_info['model_type'],
            'ts_scored': self.ts_scored
        })
        
        # Add thread/order fields if present
        for col in ['thread_id', 'comment_order', 'reply_order']:
            if col in df.columns:
                results[col] = df[col]
        
        return results
    
    def _calculate_statistics(self, results_df: pd.DataFrame, n_input: int, 
                            n_scored: int, n_skipped_empty_text: int) -> Dict[str, Any]:
        """Calculate scoring statistics"""
        
        # Label distribution
        label_counts = results_df['label'].value_counts().to_dict()
        
        # Probability statistics
        prob_cols = ['proba_Ubuntu', 'proba_Middle', 'proba_Chaos']
        prob_sums = results_df[prob_cols].sum(axis=1)
        prob_sum_stats = {
            'mean': float(prob_sums.mean()),
            'std': float(prob_sums.std()),
            'min': float(prob_sums.min()),
            'max': float(prob_sums.max())
        }
        
        model_type = "unknown"
        checkpoints_used = 0
        if self.ensemble_loader.model_info is not None:
            model_type = self.ensemble_loader.model_info['model_type']
        if self.ensemble_loader.checkpoint_paths is not None:
            checkpoints_used = len(self.ensemble_loader.checkpoint_paths)
            
        return {
            'n_input': n_input,
            'n_scored': n_scored,
            'n_skipped_empty_text': n_skipped_empty_text,
            'run_id': self.run_id,
            'model_type': model_type,
            'device': str(self.ensemble_loader.device),
            'checkpoints_used': checkpoints_used,
            'label_distribution': label_counts,
            'probability_sum_stats': prob_sum_stats,
            'ts_scored': self.ts_scored
        }
    
    def _print_summary(self, stats: Dict[str, Any]):
        """Print a summary of the scoring results"""
        print("\n" + "="*60)
        print("BATCH SCORING SUMMARY - V4 ENSEMBLE")
        print("="*60)
        print(f"Device: {stats['device']}")
        print(f"Model Type: {stats['model_type']}")
        print(f"Checkpoints Used: {stats['checkpoints_used']}")
        print(f"Run ID: {stats['run_id']}")
        print(f"Timestamp: {stats['ts_scored']}")
        print()
        print(f"Input Rows: {stats['n_input']}")
        print(f"Scored Rows: {stats['n_scored']}")
        print(f"Skipped (Empty Text): {stats['n_skipped_empty_text']}")
        print()
        print("Label Distribution:")
        for label, count in stats['label_distribution'].items():
            print(f"  {label}: {count}")
        print()
        print("Probability Sum Statistics:")
        prob_stats = stats['probability_sum_stats']
        print(f"  Mean: {prob_stats['mean']:.6f}")
        print(f"  Std: {prob_stats['std']:.6f}")
        print(f"  Range: [{prob_stats['min']:.6f}, {prob_stats['max']:.6f}]")
        print("="*60)

def load_time_folder_data():
    """Load and combine data from time folders"""
    logger.info("Loading data from time folders...")
    
    # Define file paths for all data sources
    data_sources = [
        ('drphillips', f"../../../../data/chaos/drphillips/time/tiktok-scraper_drphillips.csv", 
         f"../../../../data/chaos/drphillips/time/tiktok-comments-scraper_drphillips.csv"),
        ('thepieterkriel', f"../../../../data/middle/pieter/time/tiktok-scraper_thepieterkriel.csv", 
         f"../../../../data/middle/pieter/time/tiktok-comments-scraper_thepieterkriel.csv"),
        ('dodo', f"../../../../data/ubuntu/dodo/time/tiktok-scraper_dodo.csv", 
         f"../../../../data/ubuntu/dodo/time/tiktok-comments-scraper_dodo.csv")
    ]
    
    all_data = []
    
    # Load data from all sources
    for source_name, video_file, comment_file in data_sources:
        try:
            # Load video data
            df_videos = pd.read_csv(video_file)
            df_videos['data_type'] = 'video'
            
            # Load comment data
            df_comments = pd.read_csv(comment_file)
            df_comments['data_type'] = 'comment'
            
            # Extract video ID from comment videoWebUrl for matching
            df_comments['video_id'] = df_comments['videoWebUrl'].str.extract(r'/video/(\d+)')
            
            # Convert video_id to int64 to match the video id column
            df_comments['video_id'] = pd.to_numeric(df_comments['video_id'], errors='coerce')
            df_videos['id'] = pd.to_numeric(df_videos['id'], errors='coerce')
            
            # Merge video and comment data
            df_merged = pd.merge(
                df_comments, 
                df_videos[['id', 'playCount', 'shareCount', 'collectCount', 'text']], 
                left_on='video_id', 
                right_on='id', 
                how='left',
                suffixes=('_comment', '_video')
            )
            
            # Select relevant columns for analysis
            columns_to_keep = [
                'createTimeISO', 'text_comment', 'data_type',
                'diggCount', 'replyCommentTotal', 'uid', 'uniqueId',
                'videoWebUrl', 'likedByAuthor', 'pinnedByAuthor',
                'playCount', 'shareCount', 'collectCount', 'text_video'
            ]
            
            # Only keep columns that exist
            existing_columns = [col for col in columns_to_keep if col in df_merged.columns]
            df_filtered = df_merged[existing_columns].copy()
            
            # Clean and process data
            df_filtered = df_filtered.dropna(subset=['createTimeISO', 'text_comment'])  # type: ignore
            
            # Handle string accessor error by ensuring text column is string
            df_filtered['text_comment'] = df_filtered['text_comment'].astype(str)
            df_filtered = df_filtered[df_filtered['text_comment'].str.strip() != '']
            
            # Convert timestamp
            df_filtered['createTimeISO'] = pd.to_datetime(df_filtered['createTimeISO'])
            
            # Fill missing values with 0 for numeric columns
            numeric_columns = ['diggCount', 'replyCommentTotal', 'playCount', 'shareCount', 'collectCount']
            for col in numeric_columns:
                if col in df_filtered.columns:
                    df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce').fillna(0)  # type: ignore
            
            all_data.append(df_filtered)
            logger.info(f"✅ Loaded {len(df_filtered)} comments from {source_name}")
            
        except Exception as e:
            logger.error(f"❌ Error loading data from {source_name}: {e}")
    
    if not all_data:
        raise ValueError("No data loaded from time folders")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"✅ Combined {len(combined_df)} total comments from time folders")
    
    # Prepare for batch scoring (match expected format)
    combined_df['text_used_for_inference'] = combined_df['text_comment']
    combined_df['comment_id'] = combined_df['uid']
    combined_df['anon_user_id'] = combined_df['uniqueId']
    combined_df['lang'] = 'en'  # Default language
    combined_df['emoji_only'] = False  # Default value
    
    return combined_df

def main():
    """Main function for batch prediction"""
    
    # Set random seeds for determinism
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    model_dir = "../african_moral_classifier_V4"
    output_dir = "../data/processed/scores"
    batch_size = 64
    
    logger.info("Starting V4 batch prediction on NEW TIME FOLDER DATA (simplified)")
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Batch size: {batch_size}")
    
    try:
        # Load new time folder data
        df = load_time_folder_data()
        
        # Use simplified single model approach to avoid TensorFlow issues
        logger.info("Loading single model to avoid TensorFlow compatibility issues...")
        
        # Load just one model from the ensemble
        model_path = os.path.join(model_dir, "model_0")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Set up device
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully on device: {device}")
        
        # Create a simple scorer
        class SimpleScorer:
            def __init__(self, model, tokenizer, device):
                self.model = model
                self.tokenizer = tokenizer
                self.device = device
                self.run_id = str(uuid.uuid4())
                self.ts_scored = datetime.now(timezone.utc).isoformat()
            
            def score_dataframe(self, df, output_dir, batch_size=64):
                n_input = len(df)
                logger.info(f"Processing {n_input} rows from DataFrame")
                
                # Select text for inference
                text_column = 'text_comment'
                if text_column not in df.columns:
                    logger.warning(f"Column '{text_column}' not found")
                    return {'error': 'text_comment column not found'}
                
                # Filter out empty texts
                df = df[df[text_column].notna() & (df[text_column].str.strip() != '')]
                n_scored = len(df)
                n_skipped_empty_text = n_input - n_scored
                
                logger.info(f"Scoring {n_scored} rows (skipped {n_skipped_empty_text} empty texts)")
                
                if n_scored == 0:
                    logger.warning("No valid texts to score!")
                    return {'n_scored': 0}
                
                # Get texts for inference
                texts = df[text_column].tolist()
                
                # Make predictions
                logger.info("Starting batch inference...")
                start_time = time.time()
                
                predictions = []
                probabilities = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    inputs = self.tokenizer(
                        batch_texts,
                        truncation=True,
                        padding=True,
                        max_length=256,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=-1)
                        preds = torch.argmax(probs, dim=-1)
                        
                        predictions.extend(preds.cpu().numpy())
                        probabilities.extend(probs.cpu().numpy())
                
                inference_time = time.time() - start_time
                logger.info(f"Inference completed in {inference_time:.2f} seconds")
                
                # Create results dataframe
                labels = ['Ubuntu', 'Middle', 'Chaos']
                predicted_labels = [labels[pred] for pred in predictions]
                
                results_df = pd.DataFrame({
                    'video_id': df.get('video_id', range(len(df))),
                    'comment_id': df.get('uid', range(len(df))),
                    'anon_user_id': df.get('uniqueId', range(len(df))),
                    'lang': 'en',
                    'emoji_only': False,
                    'text': df[text_column],
                    'label': predicted_labels,
                    'proba_Ubuntu': [prob[0] for prob in probabilities],
                    'proba_Middle': [prob[1] for prob in probabilities],
                    'proba_Chaos': [prob[2] for prob in probabilities],
                    'run_id': self.run_id,
                    'model_type': 'simplified_v4',
                    'ts_scored': self.ts_scored
                })
                
                # Write to parquet
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"part-{self.run_id}.parquet")
                
                logger.info(f"Writing results to: {output_file}")
                results_df.to_parquet(output_file, index=False)
                
                # Calculate statistics
                label_counts = results_df['label'].value_counts().to_dict()
                
                stats = {
                    'n_input': n_input,
                    'n_scored': n_scored,
                    'n_skipped_empty_text': n_skipped_empty_text,
                    'run_id': self.run_id,
                    'model_type': 'simplified_v4',
                    'device': str(self.device),
                    'checkpoints_used': 1,
                    'label_distribution': label_counts,
                    'ts_scored': self.ts_scored
                }
                
                # Print summary
                print("\n" + "="*60)
                print("BATCH SCORING SUMMARY - SIMPLIFIED V4")
                print("="*60)
                print(f"Device: {stats['device']}")
                print(f"Model Type: {stats['model_type']}")
                print(f"Checkpoints Used: {stats['checkpoints_used']}")
                print(f"Run ID: {stats['run_id']}")
                print(f"Timestamp: {stats['ts_scored']}")
                print()
                print(f"Input Rows: {stats['n_input']}")
                print(f"Scored Rows: {stats['n_scored']}")
                print(f"Skipped (Empty Text): {stats['n_skipped_empty_text']}")
                print()
                print("Label Distribution:")
                for label, count in stats['label_distribution'].items():
                    print(f"  {label}: {count}")
                print("="*60)
                
                return stats
        
        # Initialize simple scorer
        scorer = SimpleScorer(model, tokenizer, device)
        
        # Score dataset
        stats = scorer.score_dataframe(df, output_dir, batch_size)
        
        logger.info("Batch prediction completed successfully on NEW DATA!")
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise

if __name__ == "__main__":
    main()
