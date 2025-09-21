#!/usr/bin/env python3
"""
African Moral Classification Trainer
Clean implementation using downloaded AfroXLM-R models
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
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Prevent deadlocks
os.environ['OMP_NUM_THREADS'] = '8'  # Use threads for M1
os.environ['MKL_NUM_THREADS'] = '8'  # Optimize for M1
os.environ['OPENBLAS_NUM_THREADS'] = '8'  # Optimize for M1

# Import PyTorch first
import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Import other dependencies
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

class AfricanMoralTrainer:
    """Clean trainer for African moral classification using AfroXLM-R"""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.label_map = {"Ubuntu": 0, "Middle": 1, "Chaos": 2}
        self.reverse_label_map = {0: "Ubuntu", 1: "Middle", 2: "Chaos"}
        
        # Model paths - use HuggingFace model directly
        self.model_path = "Davlan/afro-xlmr-base"
        
    def load_data(self, csv_path: str = "../benchmark_manual_labeled_balanced.csv") -> Tuple[List[str], List[int]]:
        """Load balanced manual labels from CSV"""
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
                    texts.append(cleaned_text)
                    labels.append(self.label_map[manual_label])
        
        print(f"Loaded {len(texts)} examples")
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
                        test_size: float = 0.2, random_state: int = 42) -> Tuple[CustomDataset, CustomDataset]:
        """Prepare training and validation datasets"""
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
            self.model_path,
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
        return train_dataset, val_dataset
    
    def initialize_model(self) -> None:
        """Initialize the AfroXLM-R model"""
        print("Loading AfroXLM-R model...")
        
        # Set device
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"📥 Loading model on {device}...")
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=3,
            problem_type="single_label_classification"
        )
        
        self.model = self.model.to(device)
        print(f"Model loaded successfully on {device}")
    
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
    
    def train_model(self, train_dataset: CustomDataset, val_dataset: CustomDataset, 
                   output_dir: str = "./african_moral_classifier") -> None:
        """Train the model"""
        print("Starting model training...")
        
        # Training arguments optimized for M1 Mac
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,  # Increased for M1
            per_device_eval_batch_size=8,   # Increased for M1
            learning_rate=2e-5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=3,
            dataloader_num_workers=4,  # Increased for M1
            remove_unused_columns=False,
            report_to=None,
            gradient_accumulation_steps=2,  # Reduced since batch size increased
            gradient_checkpointing=False,   # Disabled for M1 (not needed)
            optim="adamw_torch",
            lr_scheduler_type="linear",
            logging_first_step=True,
            fp16=False,  # Disable mixed precision for M1
            dataloader_pin_memory=False,  # Disable for M1
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        # Train the model
        print("Training model (3 epochs)...")
        train_result = self.trainer.train()
        
        # Save model and tokenizer
        self.trainer.save_model()
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        
        # Save training results
        with open(f"{output_dir}/training_results.json", 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        print("Training completed!")
        print(f"Training metrics: {train_result.metrics}")
    

    
    def evaluate_model(self, val_dataset: CustomDataset) -> Dict[str, Any]:
        """Evaluate the trained model"""
        print("Evaluating model...")
        
        if self.trainer is None:
            print("Trainer is None, cannot evaluate")
            return {}
            
        predictions = self.trainer.predict(val_dataset)
        
        # Extract predictions and labels from PredictionOutput
        preds = predictions.predictions.argmax(-1)  # type: ignore
        labels = predictions.label_ids
        
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
        
        results = {
            'overall': {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
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
            'classification_report': report
        }
        
        # Save evaluation results
        os.makedirs("./african_moral_classifier", exist_ok=True)
        with open("./african_moral_classifier/evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Overall F1-Score: {f1:.4f}")
        print(f"Overall Precision: {precision:.4f}")
        print(f"Overall Recall: {recall:.4f}")
        
        print("\nPer-Class Performance:")
        class_names = ['Ubuntu', 'Middle', 'Chaos']
        for i, class_name in enumerate(class_names):
            print(f"{class_name}:")
            print(f"  Precision: {float(class_precision[i]):.4f}")
            print(f"  Recall: {float(class_recall[i]):.4f}")
            print(f"  F1-Score: {float(class_f1[i]):.4f}")
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = "./african_moral_classifier/confusion_matrix.png"):
        """Plot confusion matrix"""
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            plt.figure(figsize=(8, 6))
            class_labels = ['Ubuntu', 'Middle', 'Chaos']
            ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            ax.set_xticklabels(class_labels)
            ax.set_yticklabels(class_labels)
            plt.title('Confusion Matrix - African Moral Classification')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Confusion matrix saved to {save_path}")
        except Exception as e:
            print(f"Warning: Could not save confusion matrix plot: {e}")
    
    def save_model_info(self, output_dir: str = "./african_moral_classifier", texts: Optional[List[str]] = None):
        """Save model information"""
        model_info = {
            'model_type': 'transformer',
            'base_model': 'Davlan/afro-xlmr-base',
            'num_labels': 3,
            'label_mapping': self.label_map,
            'training_data': {
                'source': '../benchmark_manual_labeled_balanced.csv',
                'total_examples': len(texts) if texts else 'unknown'
            },
            'model_config': {
                'max_length': 256,
                'batch_size': 8,
                'effective_batch_size': 16,
                'epochs': 3,
                'learning_rate': '2e-5',
                'warmup_ratio': '0.1',
                'mixed_precision': False
            }
        }
        
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Model information saved to {output_dir}/model_info.json")

def main():
    """Main training pipeline"""
    start_time = time.time()
    print(f"Starting African Moral Classification Training")
    print("="*70)
    
    # Initialize trainer
    trainer = AfricanMoralTrainer()
    
    # Load data
    texts, labels = trainer.load_data()
    
    # Prepare datasets
    train_dataset, val_dataset = trainer.prepare_datasets(texts, labels)
    
    # Initialize model
    trainer.initialize_model()
    
    # Train model
    trainer.train_model(train_dataset, val_dataset)
    
    # Evaluate model
    results = trainer.evaluate_model(val_dataset)
    
    # Plot confusion matrix
    if results and 'confusion_matrix' in results:
        cm = np.array(results['confusion_matrix'])
        trainer.plot_confusion_matrix(cm)
    
    # Save model information
    trainer.save_model_info(texts=texts)
    
    total_elapsed = time.time() - start_time
    print(f"\nTraining completed successfully!")
    print(f"Total time: {timedelta(seconds=int(total_elapsed))}")
    print("Model saved to: ./african_moral_classifier/")
    print("Evaluation results saved to: ./african_moral_classifier/evaluation_results.json")

if __name__ == "__main__":
    main()
