#!/usr/bin/env python3
"""
Test script for batch predictor V4
"""

import os
import sys
import pandas as pd
import numpy as np

# Import using dynamic import
import importlib.util
spec = importlib.util.spec_from_file_location("batch_predictor_v4", "batch_predictor_v4.py")
if spec is not None and spec.loader is not None:
    batch_predictor_v4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch_predictor_v4)
    EnsembleInferenceLoader = batch_predictor_v4.EnsembleInferenceLoader
    BatchScorer = batch_predictor_v4.BatchScorer
else:
    raise ImportError("Could not load batch_predictor_v4 module")

def create_test_data():
    """Create a small test dataset"""
    test_data = {
        'video_id': ['vid1', 'vid2', 'vid3', 'vid4', 'vid5'],
        'comment_id': ['com1', 'com2', 'com3', 'com4', 'com5'],
        'anon_user_id': ['user1', 'user2', 'user3', 'user4', 'user5'],
        'original_text': [
            'This is a great video, very helpful!',
            'I disagree with everything you said',
            'Thanks for sharing this information',
            'This is completely wrong and misleading',
            'Interesting perspective, thanks for explaining'
        ],
        'text_used_for_inference': [
            'This is a great video, very helpful!',
            'I disagree with everything you said',
            'Thanks for sharing this information',
            'This is completely wrong and misleading',
            'Interesting perspective, thanks for explaining'
        ],
        'lang': ['en', 'en', 'en', 'en', 'en'],
        'emoji_only': [False, False, False, False, False]
    }
    
    df = pd.DataFrame(test_data)
    
    # Create directories
    os.makedirs('../data/processed', exist_ok=True)
    
    # Save test data
    df.to_parquet('../data/processed/all_creators.parquet', index=False)
    print("Created test dataset with 5 samples")
    
    return df

def test_ensemble_loader():
    """Test the ensemble loader"""
    print("\nTesting EnsembleInferenceLoader...")
    
    try:
        loader = EnsembleInferenceLoader("../african_moral_classifier_V4")
        print("✅ EnsembleInferenceLoader initialized successfully")
        print(f"   Device: {loader.device}")
        print(f"   Models loaded: {len(loader.models)}")
        print(f"   Labels: {loader.model_info['labels']}")
        return loader
    except Exception as e:
        print(f"❌ Error initializing EnsembleInferenceLoader: {e}")
        return None

def test_batch_scoring(loader):
    """Test batch scoring with test data"""
    print("\nTesting BatchScorer...")
    
    try:
        scorer = BatchScorer(loader)
        print("✅ BatchScorer initialized successfully")
        
        # Test with small batch
        test_texts = [
            "This is a positive comment",
            "This is a negative comment", 
            "This is a neutral comment"
        ]
        
        predictions, probabilities = loader.predict_batch(test_texts, batch_size=2)
        print("✅ Batch prediction test successful")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Probabilities shape: {probabilities.shape}")
        
        return scorer
    except Exception as e:
        print(f"❌ Error in batch scoring test: {e}")
        return None

def main():
    """Main test function"""
    print("🧪 Testing Batch Predictor V4")
    print("="*50)
    
    # Check if model directory exists
    if not os.path.exists("../african_moral_classifier_V4"):
        print("❌ Model directory not found: ../african_moral_classifier_V4")
        print("   Please ensure the V4 models are trained and available")
        return
    
    # Check if metadata files exist
    required_files = [
        "checkpoint_paths.json",
        "model_info.json", 
        "calibration_params.json"
    ]
    
    for file in required_files:
        path = f"../african_moral_classifier_V4/{file}"
        if not os.path.exists(path):
            print(f"❌ Required file not found: {path}")
            return
        else:
            print(f"✅ Found: {path}")
    
    # Create test data
    create_test_data()
    
    # Update paths for new structure
    os.makedirs("../data/processed", exist_ok=True)
    
    # Test ensemble loader
    loader = test_ensemble_loader()
    if loader is None:
        return
    
    # Test batch scoring
    scorer = test_batch_scoring(loader)
    if scorer is None:
        return
    
    print("\n🎉 All tests passed!")
    print("The batch predictor is ready to use.")
    print("\nTo run full batch prediction:")
    print("python batch_predictor_v4.py")

if __name__ == "__main__":
    main()
