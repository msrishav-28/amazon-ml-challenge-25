# Amazon ML Price Prediction

A portfolio-quality multimodal machine learning system for the Amazon ML Challenge 2025. This system predicts product prices using text (product descriptions), images, and tabular features.

## Features

- **Multimodal Architecture**: Combines DeBERTa-small (text), EfficientNet-B2 (images), and tabular features
- **Memory Efficient**: Optimized for 6GB VRAM using LoRA fine-tuning, gradient checkpointing, and mixed precision
- **Resumable Training**: Comprehensive checkpoint system allows training to stop/resume at any point
- **Ensemble Learning**: 2-level stacking with LightGBM, XGBoost, and CatBoost
- **Target Performance**: <9% SMAPE on validation set

## Project Structure

```
.
├── data/                   # Data files (CSV, images, features)
├── models/                 # Trained model files
├── checkpoints/            # Training checkpoints
├── logs/                   # Training logs
├── predictions/            # Model predictions and submissions
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architectures and losses
│   ├── training/          # Training scripts
│   └── utils/             # Utilities (metrics, visualization, checkpoints)
├── scripts/               # Execution scripts for pipeline stages
├── tests/                 # Unit and property-based tests
├── config.py              # Centralized configuration
├── requirements.txt       # Python dependencies
└── setup.py              # Package installation
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd amazon-ml-price-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Hardware Requirements

- **GPU**: NVIDIA RTX 3050 6GB or better (CUDA 11.8+)
- **CPU**: AMD Ryzen 7 5800H or equivalent (8 cores)
- **RAM**: 16GB minimum
- **Storage**: 50GB free space

## Configuration

All configuration parameters are centralized in `config.py`:
- File paths for data, models, and outputs
- Model hyperparameters (learning rate, batch size, etc.)
- Training settings (epochs, gradient accumulation, etc.)
- Feature engineering parameters (IPQ patterns, TF-IDF settings)

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run property-based tests:
```bash
pytest tests/test_config_properties.py -v
```

## Pipeline Stages

The training pipeline consists of 6 stages:

1. **Stage 1**: Data download and verification
2. **Stage 2**: Feature engineering and preprocessing
3. **Stage 3**: Neural network training
4. **Stage 4**: GBDT model training
5. **Stage 5**: Ensemble training and calibration
6. **Stage 6**: Final prediction generation and submission

Each stage saves a checkpoint, allowing the pipeline to resume from any point.

## License

MIT License

## Author

Your Name <msrishav28@example.com>
