#!/usr/bin/env python3
"""
Install and cache AfroXLM-R model locally
"""

import os
import sys
from pathlib import Path

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'  # 10 minutes timeout

def install_african_model():
    """Install AfroXLM-R model locally"""
    print("Installing AfroXLM-R model locally...")
    
    # Create model directory
    model_dir = Path("./african_model_cache")
    model_dir.mkdir(exist_ok=True)
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        model_name = "Davlan/afro-xlmr-base"
        
        print(f"Downloading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(model_dir),
            local_files_only=False
        )
        
        print(f"Downloading model: {model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=str(model_dir),
            local_files_only=False,
            num_labels=3
        )
        
        # Save to local directory
        local_path = model_dir / "afro-xlmr-base"
        local_path.mkdir(exist_ok=True)
        
        print(f"Saving model to: {local_path}")
        tokenizer.save_pretrained(str(local_path))
        model.save_pretrained(str(local_path))
        
        print("AfroXLM-R model installed successfully!")
        print(f"Model location: {local_path}")
        
        return str(local_path)
        
    except Exception as e:
        print(f"Installation failed: {e}")
        return None

if __name__ == "__main__":
    install_african_model()
