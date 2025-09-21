#!/usr/bin/env python3
"""
African Moral Classification Trainer V4 - REALISTIC 85-90% TARGET
Conservative improvements with proper checkpoint management
"""

import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set environment variables for M1 Mac optimization
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class AfricanMoralTrainerV4:
    """Realistic trainer targeting 85-90% accuracy"""
    
    def __init__(self):
        self.tokenizer = None
        self.models = []
        self.trainers = []
        self.label_map = {"Ubuntu": 0, "Middle": 1, "Chaos": 2}
        self.reverse_label_map = {0: "Ubuntu", 1: "Middle", 2: "Chaos"}
        
        # Conservative ensemble (3 models)
        self.model_paths = [
            "Davlan/afro-xlmr-base",
            "Davlan/afro-xlmr-base",  # Same model, different seeds
            "Davlan/afro-xlmr-base"   # Same model, different seeds
        ]
        
        # Realistic configuration for 85-90% target
        self.ensemble_size = 3
        self.epochs = 8  # Moderate epochs
        
    def moderate_text_augmentation(self, text: str, label: int) -> List[str]:
        """Moderate text augmentation for realistic performance"""
        augmented = [text]  # Always include original text as baseline
        
        words = text.split()
        if len(words) > 3:
            # Single augmentation pass with low probability to maintain quality
            if random.random() < 0.3:  # 30% chance of augmentation (conservative)
                # Simple word replacement
                for _ in range(min(1, len(words) // 6)):  # Very few replacements
                    if random.random() < 0.2: # 20% chance per word (very conservative)
                        idx = random.randint(0, len(words) - 1) # Select random word index for replacement
                        words[idx] = f"[MASK]{words[idx]}[MASK]" # Apply simple masking pattern (preserves word structure)
                augmented.append(" ".join(words))
        
        return augmented
    
    def load_data(self, csv_path: str = "../benchmark_manual_labeled_balanced.csv") -> Tuple[List[str], List[int]]:
        """Load and moderately augment data"""
        print(f"Loading data from {csv_path}...")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} rows")
        
        texts = []
        labels = []
        
        for idx, row in df.iterrows():
            text = row.get('original_text', '')
            manual_label = row.get('manual_label_new', '')
            
            if text and manual_label in self.label_map:
                cleaned_text = self._clean_text(str(text).strip())
                if cleaned_text:
                    # Moderate augmentation
                    augmented_texts = self.moderate_text_augmentation(cleaned_text, self.label_map[manual_label])
                    
                    for aug_text in augmented_texts:
                        texts.append(aug_text)
                        labels.append(self.label_map[manual_label])
        
        print(f"Loaded {len(texts)} examples (with moderate augmentation)")
        print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
        
        if len(texts) == 0:
            raise ValueError("No valid examples found in the dataset")
        
        return texts, labels
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text or text.strip() == '':
            return ''
        
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove excessive punctuation but keep important ones
        text = re.sub(r'[^\w\s\'\u1F600-\u1F64F\u1F300-\u1F5FF\u1F680-\u1F6FF\u1F1E0-\u1F1FF\u2600-\u26FF\u2700-\u27BF]', ' ', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def prepare_datasets(self, texts: List[str], labels: List[int], 
                        test_size: float = 0.2, random_state: int = 42) -> Tuple[List[CustomDataset], List[CustomDataset]]:
        """Prepare datasets with proper train/val split"""
        print("Preparing datasets...")
        
        # Stratified split
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels
        )
        
        print(f"Training set: {len(X_train)} examples")
        print(f"Validation set: {len(X_val)} examples")
        
        # Load tokenizer
        print("Loading AfroXLM-R tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_paths[0],
            use_fast=True,
            model_max_length=256
        )
        print("Tokenizer loaded successfully")
        
        # Tokenize datasets
        print("Tokenizing datasets...")
        train_encodings = self.tokenizer(
            X_train, 
            truncation=True, 
            padding=True, 
            max_length=256,
            return_tensors="pt"
        )
        
        val_encodings = self.tokenizer(
            X_val, 
            truncation=True, 
            padding=True, 
            max_length=256,
            return_tensors="pt"
        )
        
        # Create datasets
        train_dataset = CustomDataset(train_encodings, y_train)
        val_dataset = CustomDataset(val_encodings, y_val)
        
        print("Datasets prepared successfully")
        return [train_dataset], [val_dataset]
    
    def initialize_models(self) -> None:
        """Initialize ensemble models with different seeds"""
        print("Initializing ensemble models...")
        
        # Set device
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading models on {device}...")
        
        for i in range(self.ensemble_size):
            # Set different seed for each model
            set_seed(42 + i)
            
            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_paths[i],
                num_labels=3,
                problem_type="single_label_classification"
            )
            
            model = model.to(device)
            self.models.append(model)
            print(f"Model {i+1} loaded successfully on {device}")
    
    def compute_metrics(self, pred) -> Dict[str, float]:
        """Compute evaluation metrics"""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        
        return {
            'accuracy': float(acc),
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall)
        }
    
    def train_ensemble(self, train_datasets: List[CustomDataset], 
                      val_datasets: List[CustomDataset], 
                      output_dir: str = "./african_moral_classifier_V4") -> None:
        """Train ensemble with proper checkpoint management"""
        print("Starting realistic ensemble training targeting 85-90%...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for model_idx in range(self.ensemble_size):
            print(f"\nTraining Model {model_idx + 1}/{self.ensemble_size}")
            
            # Realistic training arguments
            training_args = TrainingArguments(
                output_dir=f"{output_dir}/model_{model_idx}",
                num_train_epochs=self.epochs,
                per_device_train_batch_size=4,  # Moderate batch size
                per_device_eval_batch_size=4,
                learning_rate=3e-5,  # Conservative learning rate
                warmup_ratio=0.1,    # Moderate warmup
                weight_decay=0.01,   # Standard weight decay
                logging_dir=f"{output_dir}/model_{model_idx}/logs",
                logging_steps=10,    # Reasonable logging
                eval_strategy="steps",
                eval_steps=50,       # Regular evaluation
                save_strategy="steps",
                save_steps=100,      # Regular saving
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,
                save_total_limit=3,  # Keep few checkpoints
                dataloader_num_workers=2,
                remove_unused_columns=False,
                report_to=None,
                gradient_accumulation_steps=4,  # Moderate accumulation
                gradient_checkpointing=False,
                optim="adamw_torch",
                lr_scheduler_type="linear",  # Simple scheduler
                logging_first_step=True,
                fp16=False,
                dataloader_pin_memory=False,
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.models[model_idx],
                args=training_args,
                train_dataset=train_datasets[0],
                eval_dataset=val_datasets[0],
                compute_metrics=self.compute_metrics,
            )
            
            self.trainers.append(trainer)
            
            # Train the model
            print(f"Training Model {model_idx + 1}...")
            train_result = trainer.train()
            
            # Save model
            trainer.save_model()
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(f"{output_dir}/model_{model_idx}")
            
            # Save training results
            with open(f"{output_dir}/model_{model_idx}/training_results.json", 'w') as f:
                json.dump(train_result.metrics, f, indent=2)
            
            print(f"Model {model_idx + 1} training completed!")
        
        print("Ensemble training completed!")
    
    def ensemble_predict(self, val_dataset: CustomDataset) -> Tuple[np.ndarray, np.ndarray]:
        """Make ensemble predictions"""
        print("Making ensemble predictions...")
        
        all_predictions = []
        all_probabilities = []
        
        for i, trainer in enumerate(self.trainers):
            predictions = trainer.predict(val_dataset)
            preds = predictions.predictions.argmax(-1)
            probs = F.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
            
            all_predictions.append(preds)
            all_probabilities.append(probs)
        
        # Ensemble voting (soft voting with probabilities)
        ensemble_probs = np.mean(all_probabilities, axis=0)
        ensemble_preds = ensemble_probs.argmax(axis=-1)
        
        return ensemble_preds, ensemble_probs
    
    def evaluate_ensemble(self, val_dataset: CustomDataset) -> Dict[str, Any]:
        """Evaluate the ensemble model"""
        print("Evaluating ensemble model...")
        
        # Get ensemble predictions
        preds, probs = self.ensemble_predict(val_dataset)
        labels = val_dataset.labels
        
        # Calculate metrics
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        
        # Per-class metrics
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            labels, preds, average=None
        )
        
        # Convert to numpy arrays for safe indexing
        class_precision = np.array(class_precision)
        class_recall = np.array(class_recall)
        class_f1 = np.array(class_f1)
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        
        # Classification report
        report = classification_report(
            labels, preds, 
            target_names=['Ubuntu', 'Middle', 'Chaos'],
            output_dict=True
        )
        
        # Calculate ensemble confidence
        confidence_scores = np.max(probs, axis=-1)
        avg_confidence = np.mean(confidence_scores)
        
        results = {
            'overall': {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'ensemble_confidence': avg_confidence
            },
            'per_class': {
                'Ubuntu': {
                    'precision': float(class_precision[0]),
                    'recall': float(class_recall[0]),
                    'f1': float(class_f1[0])
                },
                'Middle': {
                    'precision': float(class_precision[1]),
                    'recall': float(class_recall[1]),
                    'f1': float(class_f1[1])
                },
                'Chaos': {
                    'precision': float(class_precision[2]),
                    'recall': float(class_recall[2]),
                    'f1': float(class_f1[2])
                }
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'ensemble_info': {
                'num_models': self.ensemble_size,
                'confidence_scores': confidence_scores.tolist()
            }
        }
        
        # Save evaluation results
        os.makedirs("./african_moral_classifier_V4", exist_ok=True)
        with open("./african_moral_classifier_V4/evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print results
        print("\n" + "="*60)
        print("ENSEMBLE EVALUATION RESULTS - V4 REALISTIC")
        print("="*60)
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Overall F1-Score: {f1:.4f}")
        print(f"Overall Precision: {precision:.4f}")
        print(f"Overall Recall: {recall:.4f}")
        print(f"Ensemble Confidence: {avg_confidence:.4f}")
        
        print("\nPer-Class Performance:")
        class_names = ['Ubuntu', 'Middle', 'Chaos']
        for i, class_name in enumerate(class_names):
            print(f"{class_name}:")
            print(f"  Precision: {float(class_precision[i]):.4f}")
            print(f"  Recall: {float(class_recall[i]):.4f}")
            print(f"  F1-Score: {float(class_f1[i]):.4f}")
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = "./african_moral_classifier_V4/confusion_matrix.png"):
        """Plot confusion matrix"""
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            plt.figure(figsize=(8, 6))
            class_labels = ['Ubuntu', 'Middle', 'Chaos']
            ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            ax.set_xticklabels(class_labels)
            ax.set_yticklabels(class_labels)
            plt.title('Ensemble Confusion Matrix - African Moral Classification V4\n(Realistic 85-90% Target)', fontsize=14, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Confusion matrix saved to {save_path}")
        except Exception as e:
            print(f"Warning: Could not save confusion matrix plot: {e}")
    
    def save_model_info(self, output_dir: str = "./african_moral_classifier_V4", texts: Optional[List[str]] = None):
        """Save model information"""
        model_info = {
            'model_type': 'realistic_ensemble_transformer_v4',
            'base_model': 'Davlan/afro-xlmr-base',
            'num_labels': 3,
            'label_mapping': self.label_map,
            'training_data': {
                'source': '../benchmark_manual_labeled_balanced.csv',
                'total_examples': len(texts) if texts else 'unknown',
                'augmented_examples': len(texts) if texts else 'unknown'
            },
            'realistic_config': {
                'max_length': 256,
                'batch_size': 4,
                'effective_batch_size': 16,
                'epochs': self.epochs,
                'ensemble_size': self.ensemble_size,
                'learning_rate': '3e-5',
                'warmup_ratio': '0.1',
                'scheduler': 'linear',
                'mixed_precision': False,
                'gradient_accumulation': 4,
                'realistic_features': [
                    '3-Model Ensemble Learning',
                    'Moderate Data Augmentation',
                    'Conservative Hyperparameters',
                    'Proper Checkpoint Management',
                    'Soft Voting Ensemble',
                    'Balanced Training Strategy',
                    'Realistic Target: 85-90%'
                ],
                'improvements_over_v2': [
                    'Ensemble learning for robustness',
                    'Moderate augmentation for diversity',
                    'Better hyperparameter tuning',
                    'Proper checkpoint handling',
                    'Conservative approach to avoid overfitting'
                ]
            }
        }
        
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Model information saved to {output_dir}/model_info.json")

def main():
    """Main training pipeline V4 - REALISTIC 85-90% TARGET"""
    start_time = time.time()
    print(f"Starting REALISTIC African Moral Classification Training V4")
    print("="*80)
    print("TARGET: 85-90% ACCURACY (REALISTIC)")
    print("FEATURES: 3-Model Ensemble + Moderate Augmentation + Proper Checkpoints")
    print("="*80)
    
    # Initialize trainer
    trainer = AfricanMoralTrainerV4()
    
    # Load data
    texts, labels = trainer.load_data()
    
    # Prepare datasets
    train_datasets, val_datasets = trainer.prepare_datasets(texts, labels)
    
    # Initialize models
    trainer.initialize_models()
    
    # Train ensemble
    trainer.train_ensemble(train_datasets, val_datasets)
    
    # Evaluate ensemble
    results = trainer.evaluate_ensemble(val_datasets[0])
    
    # Plot confusion matrix
    if results and 'confusion_matrix' in results:
        cm = np.array(results['confusion_matrix'])
        trainer.plot_confusion_matrix(cm)
    
    # Save model information
    trainer.save_model_info(texts=texts)
    
    total_elapsed = time.time() - start_time
    print(f"\nREALISTIC Training V4 completed successfully!")
    print(f"Total time: {timedelta(seconds=int(total_elapsed))}")
    print("Ensemble models saved to: ./african_moral_classifier_V4/")
    print("Evaluation results saved to: ./african_moral_classifier_V4/evaluation_results.json")
    print("TARGETING 85-90% PERFORMANCE (REALISTIC)!")

if __name__ == "__main__":
    main()
