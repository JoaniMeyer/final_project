# Moral Landscape Streamlit App

A new Streamlit application for visualizing the moral dynamics of South African TikTok comments through an interactive 3D landscape.

## Project Structure

```
moral_landscape_app/
├── README.md                           # This file
├── requirements.txt                    # Main requirements
├── african_moral_classifier_V4/       # V4 ensemble models
│   ├── checkpoint_paths.json          # Model checkpoint paths
│   ├── model_info.json               # Model metadata
│   ├── calibration_params.json       # Calibration parameters
│   └── model_0/, model_1/, model_2/  # Ensemble models
├── scripts/                           # Processing scripts
│   ├── batch_predictor_v4.py         # Step 2: Batch predictions
│   ├── embeddings_3d_mapper.py       # Step 3: Embeddings + 3D mapping
│   ├── test_batch_predictor.py       # Test batch prediction
│   └── test_embeddings_3d.py         # Test embeddings system
├── requirements/                      # Requirements files
│   ├── requirements_batch_predictor.txt
│   └── requirements_embeddings.txt
└── data/                             # Data directory (created during processing)
    ├── processed/
    │   ├── scores/                   # Step 2 output
    │   └── embeddings/               # Step 3 output
    └── embeddings_3d.parquet         # Final 3D data for Streamlit
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install specific requirements:**
   ```bash
   pip install -r requirements/requirements_batch_predictor.txt
   pip install -r requirements/requirements_embeddings.txt
   ```

## Usage

### Step 2: Batch Predictions
```bash
cd scripts
python batch_predictor_v4.py
```
- Input: `data/processed/all_creators.parquet`
- Output: `data/processed/scores/part-*.parquet`

### Step 3: Embeddings + 3D Mapping
```bash
cd scripts
python embeddings_3d_mapper.py
```
- Input: `data/processed/scores/part-*.parquet`
- Output: `data/embeddings_3d.parquet` (for Streamlit)

### Testing
```bash
cd scripts
python test_batch_predictor.py
python test_embeddings_3d.py
```

## Streamlit App

The Streamlit app will be created in the next step and will use:
- `data/embeddings_3d.parquet` for the 3D Moral Landscape visualization
- Interactive features: rotate, zoom, hover, filters
- Color coding by moral class (Ubuntu/Middle/Chaos) or topic clusters

## Model Information

- **Base Model**: AfroXLM-R (`Davlan/afro-xlmr-base`)
- **Ensemble**: 3 models with different seeds
- **Labels**: Ubuntu, Middle, Chaos
- **Performance**: ~80% accuracy on balanced dataset
- **Training**: 8 epochs, conservative hyperparameters
