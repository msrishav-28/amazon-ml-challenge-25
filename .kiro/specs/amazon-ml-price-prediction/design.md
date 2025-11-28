# Design Document

## Overview

This design document specifies the architecture and implementation details for a multimodal machine learning system that predicts product prices for the Amazon ML Challenge 2025. The system combines three modalities (text, images, and tabular features) through a sophisticated neural network architecture, then ensembles these predictions with gradient boosted decision trees (GBDT) models through 2-level stacking.

The design prioritizes:
- **Memory efficiency**: Optimized for 6GB VRAM using LoRA fine-tuning, gradient checkpointing, and mixed precision
- **Resumability**: Comprehensive checkpoint system allowing training to stop/resume at any point
- **Accuracy**: Target <9% SMAPE through multimodal fusion and ensemble methods
- **Production quality**: Type hints, comprehensive testing, logging, and documentation

## Architecture

### High-Level System Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA ACQUISITION                          │
│  • Download 150K images with resume capability              │
│  • Load train/test CSV files                                │
│  • Create train/validation split (stratified by price)      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                         │
│  • IPQ extraction (quantities, units)                       │
│  • Text statistics (length, word count, etc.)               │
│  • Keyword features (quality, discount indicators)          │
│  • Brand extraction                                         │
│  • TF-IDF vectorization (100 features)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              MULTIMODAL NEURAL NETWORK                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Text Encoder │  │Image Encoder │  │   Tabular    │     │
│  │ DeBERTa-small│  │EfficientNet-B2│  │  Features    │     │
│  │  + LoRA      │  │ (frozen early)│  │  Projection  │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │             │
│         └──────────┬───────┘                  │             │
│                    │                          │             │
│         ┌──────────▼──────────┐               │             │
│         │ Cross-Modal Attention│               │             │
│         │  (Bidirectional)    │               │             │
│         └──────────┬──────────┘               │             │
│                    │                          │             │
│                    └──────────┬───────────────┘             │
│                               │                             │
│                    ┌──────────▼──────────┐                  │
│                    │   Regression Head   │                  │
│                    │   (3-layer MLP)     │                  │
│                    └──────────┬──────────┘                  │
│                               │                             │
│                          Predictions                        │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    GBDT MODELS                               │
│  • LightGBM (custom SMAPE objective)                        │
│  • XGBoost (custom SMAPE objective)                         │
│  • CatBoost (built-in regression)                           │
│  • Optuna hyperparameter optimization                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              2-LEVEL STACKING ENSEMBLE                       │
│  Level 1: Ridge + ElasticNet + Shallow LightGBM            │
│  Level 2: Optimized weighted combination                    │
│  Calibration: Isotonic regression                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
                  Final Predictions
                   (Target: <9% SMAPE)
```

### Training Pipeline Stages

The system is designed for incremental training in 30-minute sessions:

1. **Stage 1 (0-4 hours)**: Data download and verification
2. **Stage 2 (4-8 hours)**: Feature engineering and preprocessing
3. **Stage 3 (8-16 hours)**: Neural network training (12 sessions × 30 min)
4. **Stage 4 (16-20 hours)**: GBDT model training (3 models)
5. **Stage 5 (20-21 hours)**: Ensemble training and calibration
6. **Stage 6 (21-22 hours)**: Final prediction generation and submission

Each stage saves a checkpoint, allowing the pipeline to resume from any point.

## Components and Interfaces

### 1. Data Download Module (`src/data/downloader.py`)

**Purpose**: Download product images with resume capability and parallel processing.

**Key Classes**:
- `ResumableImageDownloader`: Manages batch downloads with progress tracking

**Key Methods**:
```python
def download_batch(df: pd.DataFrame, batch_size: int = 5000, max_workers: int = 40) -> Dict
    """Download images in batches with parallel workers"""

def _download_single(sample_id: str, url: str, timeout: int = 10, max_retries: int = 3) -> Tuple[str, bool]
    """Download single image with retry logic"""

def _save_progress() -> None
    """Save download progress to JSON file"""

def _load_progress() -> Dict
    """Load previous download progress"""
```

**Interface Contract**:
- Input: DataFrame with `sample_id` and `image_link` columns
- Output: Dictionary with download statistics (success, failed, skipped)
- Side Effects: Creates image files, updates progress JSON

### 2. Feature Engineering Module (`src/data/feature_engineering.py`)

**Purpose**: Extract and transform features from raw product data.

**Key Classes**:
- `FeatureEngineer`: Orchestrates all feature extraction

**Key Methods**:
```python
def extract_ipq_features(text: str) -> Dict
    """Extract Item Pack Quantity features using regex"""

def extract_text_statistics(text: str) -> Dict
    """Compute text length, word count, character distributions"""

def extract_keyword_features(text: str) -> Dict
    """Identify quality and discount keywords"""

def extract_brand_features(text: str) -> Dict
    """Extract potential brand names"""

def fit_tfidf(texts: pd.Series) -> None
    """Fit TF-IDF vectorizer on training texts"""

def transform_tfidf(texts: pd.Series) -> np.ndarray
    """Transform texts to TF-IDF features"""

def engineer_features(df: pd.DataFrame, fit_tfidf: bool = False) -> pd.DataFrame
    """Engineer all features from DataFrame"""
```

**Interface Contract**:
- Input: DataFrame with `catalog_content` column
- Output: DataFrame with ~180 engineered features
- Side Effects: Fits TF-IDF vectorizer (if fit_tfidf=True)

### 3. PyTorch Dataset Module (`src/data/dataset.py`)

**Purpose**: Provide efficient data loading for neural network training.

**Key Classes**:
- `AmazonMLDataset`: PyTorch Dataset for multimodal data

**Key Methods**:
```python
def __getitem__(idx: int) -> Dict[str, torch.Tensor]
    """Load and preprocess a single sample"""

def get_dataloader(batch_size: int, shuffle: bool = True, num_workers: int = 4) -> DataLoader
    """Create DataLoader with optimized settings"""
```

**Interface Contract**:
- Input: DataFrame, features, image directory, tokenizer, config
- Output: Dictionary with `input_ids`, `attention_mask`, `image`, `tabular`, `target`, `sample_id`
- Handles: Missing images (creates blank placeholder), NaN values (fills with 0)

### 4. Multimodal Model Module (`src/models/multimodal.py`)

**Purpose**: Define the neural network architecture for multimodal fusion.

**Key Classes**:
- `OptimizedMultimodalModel`: Main model combining text, image, and tabular inputs
- `CrossModalAttention`: Bidirectional attention between modalities
- `GatedFusion`: Alternative simpler fusion mechanism

**Architecture Details**:
```python
class OptimizedMultimodalModel(nn.Module):
    # Text Encoder: DeBERTa-small (44M params)
    self.text_encoder = AutoModel.from_pretrained('microsoft/deberta-v3-small')
    
    # Image Encoder: EfficientNet-B2 (9M params, early layers frozen)
    self.image_encoder = timm.create_model('efficientnet_b2', pretrained=True)
    
    # Projection layers to common dimension (512)
    self.text_proj = nn.Linear(768, 512)
    self.image_proj = nn.Linear(1408, 512)
    
    # Cross-modal attention (bidirectional)
    self.text_to_image = CrossModalAttention(512, num_heads=8)
    self.image_to_text = CrossModalAttention(512, num_heads=8)
    
    # Tabular projection
    self.tabular_proj = nn.Sequential(
        nn.Linear(180, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Dropout(0.1)
    )
    
    # Regression head
    self.regressor = nn.Sequential(
        nn.Linear(1024 + 128, 256),  # fused (512*2) + tabular (128)
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Dropout(0.15),
        nn.Linear(256, 128),
        nn.LayerNorm(128),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(128, 1)
    )
```

**Memory Optimizations**:
- LoRA fine-tuning on text encoder (only 1.6% parameters trainable)
- Gradient checkpointing (40% VRAM savings)
- Mixed precision FP16 training
- Frozen early layers in image encoder

### 5. Loss Functions Module (`src/models/losses.py`)

**Purpose**: Provide custom loss functions optimized for SMAPE metric.

**Key Classes**:
- `HuberSMAPELoss`: Huber-smoothed SMAPE for robustness
- `FocalSMAPELoss`: Focal-weighted SMAPE for handling outliers

**Key Functions**:
```python
def lgb_smape_objective(y_pred, y_train) -> Tuple[np.ndarray, np.ndarray]
    """Custom SMAPE objective for LightGBM (gradient, hessian)"""

def lgb_smape_eval(y_pred, y_train) -> Tuple[str, float, bool]
    """SMAPE evaluation metric for LightGBM"""

def xgb_smape_objective(y_pred, y_train) -> Tuple[np.ndarray, np.ndarray]
    """Custom SMAPE objective for XGBoost"""
```

### 6. Checkpoint Manager Module (`src/utils/checkpoint.py`)

**Purpose**: Manage training checkpoints for resumability.

**Key Classes**:
- `CheckpointManager`: Universal checkpoint manager for all training stages

**Key Methods**:
```python
def save_checkpoint(state: Dict, stage: str, metric: Optional[float] = None, 
                   checkpoint_type: str = 'quick') -> Path
    """Save checkpoint with automatic cleanup"""

def load_checkpoint(checkpoint_path: Path) -> Dict
    """Load checkpoint and return state"""

def get_latest_checkpoint(stage: Optional[str] = None) -> Optional[Path]
    """Get most recent checkpoint for a stage"""
```

**Checkpoint Types**:
- `quick`: ~500MB, saves every 30 minutes
- `full`: ~2GB, saves at epoch boundaries
- `minimal`: ~100MB, saves only essential state

### 7. Training Modules

#### Neural Network Training (`src/training/train_neural_net.py`)

**Key Functions**:
```python
def setup_lora(model, config) -> nn.Module
    """Apply LoRA to text encoder"""

def train_neural_network(config, train_loader, val_loader, test_loader, 
                        resume_from=None) -> Tuple
    """Train neural network with LoRA fine-tuning"""

def predict(model, dataloader, device, config) -> np.ndarray
    """Generate predictions"""

def predict_with_tta(model, dataloader, device, config, n_tta=3) -> np.ndarray
    """Predict with test-time augmentation"""
```

#### GBDT Training (`src/training/train_gbdt.py`)

**Key Functions**:
```python
def optimize_lightgbm(X_train, y_train, X_val, y_val, config) -> Dict
    """Optimize LightGBM hyperparameters with Optuna"""

def train_lightgbm(X_train, y_train, X_val, y_val, config, optimize=True) -> lgb.Booster
    """Train LightGBM with custom SMAPE loss"""

def train_xgboost(X_train, y_train, X_val, y_val, config, optimize=True) -> xgb.Booster
    """Train XGBoost with custom SMAPE loss"""

def train_catboost(X_train, y_train, X_val, y_val, config) -> CatBoostRegressor
    """Train CatBoost with GPU optimization"""

def train_gbdt_models(X_train, y_train, X_val, y_val, X_test, config, 
                     optimize=True) -> Dict
    """Train all GBDT models and return predictions"""
```

#### Ensemble Training (`src/training/train_ensemble.py`)

**Key Functions**:
```python
def train_ensemble(train_preds_dict, val_preds_dict, test_preds_dict,
                  y_train, y_val, config) -> Dict
    """Train 2-level stacking ensemble with isotonic calibration"""
```

**Ensemble Architecture**:
```
Level 0 (Base Models):
├── Neural Network
├── LightGBM
├── XGBoost
└── CatBoost

Level 1 (Meta-Learners):
├── Ridge Regression (CV)
├── ElasticNet (CV)
└── Shallow LightGBM

Level 2 (Final Ensemble):
├── Optimized weighted combination
└── Isotonic regression calibration
```

### 8. Metrics Module (`src/utils/metrics.py`)

**Key Functions**:
```python
def calculate_smape(y_true, y_pred, epsilon=1e-10) -> float
    """Calculate SMAPE metric"""

def smape_scorer(y_true, y_pred) -> float
    """Scikit-learn compatible SMAPE scorer (negative for maximization)"""

def calculate_metrics_by_quantile(y_true, y_pred, n_quantiles=5) -> pd.DataFrame
    """Calculate SMAPE by price quantiles for error analysis"""

def evaluate_predictions(y_true, y_pred, split_name='validation') -> Dict
    """Comprehensive evaluation with SMAPE, MAE, RMSE, MAPE, R²"""
```

### 9. Visualization Module (`src/utils/visualization.py`)

**Key Functions**:
```python
def plot_training_curves(train_losses, val_losses, val_smapes, save_path=None)
    """Plot training and validation curves"""

def plot_predictions(y_true, y_pred, split_name='Validation', save_path=None)
    """Plot predicted vs actual values with residuals"""

def plot_error_distribution(y_true, y_pred, save_path=None)
    """Plot error distribution histograms"""
```

## Data Models

### 1. Raw Data Schema

**train.csv / test.csv**:
```python
{
    'sample_id': str,           # Unique identifier
    'catalog_content': str,     # Product title + description
    'image_link': str,          # URL to product image
    'price': float              # Target variable (train only)
}
```

### 2. Engineered Features Schema

**Engineered features DataFrame** (~180 features):
```python
{
    # IPQ Features
    'ipq_value': float,              # Extracted quantity value
    'ipq_unit': str,                 # Unit (kg, ml, count, etc.)
    'ipq_normalized': float,         # Normalized to standard units
    'ipq_confidence': float,         # Extraction confidence [0-1]
    'has_ipq': bool,                 # Whether IPQ was found
    
    # Text Statistics
    'text_length': int,              # Character count
    'word_count': int,               # Word count
    'digit_count': int,              # Number of digits
    'special_char_count': int,       # Special characters
    'uppercase_ratio': float,        # Ratio of uppercase letters
    'avg_word_length': float,        # Average word length
    
    # Keyword Features
    'has_quality_keywords': bool,    # Contains quality indicators
    'quality_keyword_count': int,    # Number of quality keywords
    'has_discount_keywords': bool,   # Contains discount indicators
    'discount_keyword_count': int,   # Number of discount keywords
    
    # Brand Features
    'has_brand': bool,               # Brand name detected
    'brand_position': float,         # Normalized position in text
    'potential_brand': str,          # Extracted brand name
    
    # Derived Features
    'price_per_unit': float,         # Price / IPQ (if available)
    
    # TF-IDF Features
    'tfidf_0' ... 'tfidf_99': float  # 100 TF-IDF features
}
```

### 3. Model Input/Output Schema

**Neural Network Input**:
```python
{
    'input_ids': torch.Tensor,        # Shape: (batch, seq_len), dtype: long
    'attention_mask': torch.Tensor,   # Shape: (batch, seq_len), dtype: long
    'image': torch.Tensor,            # Shape: (batch, 3, 224, 224), dtype: float32
    'tabular': torch.Tensor,          # Shape: (batch, 180), dtype: float32
    'target': torch.Tensor,           # Shape: (batch,), dtype: float32 (log space)
    'sample_id': str                  # For tracking
}
```

**Neural Network Output**:
```python
torch.Tensor  # Shape: (batch,), dtype: float32, log-space predictions
```

### 4. Checkpoint Schema

**Training Checkpoint**:
```python
{
    'state': {
        'model': OrderedDict,         # Model state dict
        'optimizer': Dict,            # Optimizer state dict
        'epoch': int,                 # Current epoch
        'step': int,                  # Current step within epoch
        'loss': float,                # Current loss value
        'best_smape': float,          # Best validation SMAPE so far
        'ema_state': OrderedDict      # EMA model state (optional)
    },
    'stage': str,                     # Stage identifier
    'metric': float,                  # Validation metric
    'timestamp': str,                 # ISO format timestamp
    'checkpoint_type': str            # 'quick', 'full', or 'minimal'
}
```

**Stage Checkpoint**:
```python
{
    'train_df': pd.DataFrame,         # Training data
    'val_df': pd.DataFrame,           # Validation data
    'test_df': pd.DataFrame,          # Test data
    'train_features': pd.DataFrame,   # Engineered train features
    'val_features': pd.DataFrame,     # Engineered val features
    'test_features': pd.DataFrame,    # Engineered test features
    'feature_engineer': FeatureEngineer,  # Fitted feature engineer
    'metadata': {
        'stage': str,
        'elapsed_hours': float,
        'num_features': int,
        'train_size': int,
        'val_size': int,
        'test_size': int
    }
}
```

### 5. Ensemble Artifacts Schema

```python
{
    'meta_learners': {
        'ridge': RidgeCV,
        'elasticnet': ElasticNetCV,
        'lightgbm': lgb.LGBMRegressor
    },
    'level2_weights': np.ndarray,     # Shape: (3,), optimized weights
    'isotonic': IsotonicRegression,   # Calibration model
    'model_names': List[str],         # Base model names
    'val_smape': float                # Final validation SMAPE
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Data Download Properties

**Property 1: Batch download consistency**
*For any* DataFrame with image URLs and any batch size, downloading in batches should result in the same set of successfully downloaded images as downloading all at once (order-independent).
**Validates: Requirements 1.1**

**Property 2: Resume from interruption**
*For any* download progress state, resuming a download should skip already-downloaded images and only download remaining images, resulting in the complete set without duplicates.
**Validates: Requirements 1.2**

**Property 3: Retry with exponential backoff**
*For any* failed image download, the system should retry exactly 3 times with increasing delays, and the delay between retry N and N+1 should be greater than the delay between N-1 and N.
**Validates: Requirements 1.3**

**Property 4: Corrupted image detection**
*For any* set of downloaded images, running verification should correctly identify all images that cannot be opened by PIL.Image and report them as corrupted.
**Validates: Requirements 1.4**

### Feature Engineering Properties

**Property 5: IPQ extraction determinism**
*For any* product text, extracting IPQ features multiple times should produce identical results (deterministic extraction).
**Validates: Requirements 2.1**

**Property 6: Text statistics correctness**
*For any* text string, the computed word_count should equal len(text.split()), and text_length should equal len(text).
**Validates: Requirements 2.2**

**Property 7: Keyword detection completeness**
*For any* text containing a quality keyword from the predefined list, has_quality_keywords should be True and quality_keyword_count should be at least 1.
**Validates: Requirements 2.3**

**Property 8: TF-IDF transformation consistency**
*For any* fitted TF-IDF vectorizer and any text, transforming the same text multiple times should produce identical feature vectors.
**Validates: Requirements 2.6**

**Property 9: Unit normalization reversibility**
*For any* IPQ value with a known unit, normalizing to standard units and then denormalizing should recover the original value within numerical precision.
**Validates: Requirements 2.7**

**Property 10: Feature serialization round-trip**
*For any* engineered features DataFrame, saving to pickle and loading should produce a DataFrame with identical values and schema.
**Validates: Requirements 2.8**

### Checkpoint System Properties

**Property 11: Checkpoint completeness**
*For any* saved checkpoint, loading it should provide all required fields: model state, optimizer state, epoch, step, and loss.
**Validates: Requirements 3.2**

**Property 12: Training resumption continuity**
*For any* training state, saving a checkpoint at step N and resuming should continue training from step N+1 with the same model weights and optimizer state.
**Validates: Requirements 3.3**

**Property 13: Checkpoint cleanup policy**
*For any* sequence of checkpoint saves, after saving N checkpoints (N > 3), only the most recent 3 non-best checkpoints should exist on disk.
**Validates: Requirements 3.4**

### Model Architecture Properties

**Property 14: LoRA parameter efficiency**
*For any* model with LoRA applied, the ratio of trainable parameters to total parameters should be between 1.5% and 2.0%.
**Validates: Requirements 4.8**

**Property 15: Multimodal forward pass shape consistency**
*For any* batch of inputs with batch_size B, the model output should have shape (B,) and dtype float32.
**Validates: Requirements 4.5**

**Property 16: Tabular feature projection**
*For any* tabular input of shape (B, 180), the tabular projection should output shape (B, 128).
**Validates: Requirements 4.4**

### Training Properties

**Property 17: Learning rate schedule monotonicity**
*For any* training run with linear warmup and decay, the learning rate should increase monotonically during warmup and decrease monotonically after warmup.
**Validates: Requirements 5.3**

**Property 18: Gradient accumulation correctness**
*For any* training loop with gradient accumulation steps = 4, optimizer.step() should be called exactly once every 4 forward passes.
**Validates: Requirements 5.4**

**Property 19: Gradient clipping enforcement**
*For any* batch during training, after gradient clipping, the global gradient norm should not exceed 1.0.
**Validates: Requirements 5.5**

**Property 20: EMA update consistency**
*For any* training step, after EMA update, the EMA parameters should be a weighted average of the previous EMA parameters and current model parameters with decay factor 0.9999.
**Validates: Requirements 5.6**

**Property 21: Best model selection**
*For any* training run, the saved "best model" should correspond to the epoch with the lowest validation SMAPE.
**Validates: Requirements 5.8**

### GBDT Properties

**Property 22: GBDT model serialization**
*For any* trained LightGBM/XGBoost/CatBoost model, saving to disk and loading should produce a model that generates identical predictions on the same input.
**Validates: Requirements 6.5**

**Property 23: GBDT prediction completeness**
*For any* trained GBDT model, predictions should be generated for all samples in train, validation, and test sets with no missing values.
**Validates: Requirements 6.6**

### Ensemble Properties

**Property 24: Meta-feature stacking correctness**
*For any* set of N base model predictions, stacking should produce a meta-feature matrix with N columns and the same number of rows as the input predictions.
**Validates: Requirements 7.1**

**Property 25: Level-2 weight normalization**
*For any* optimized level-2 weights, they should all be non-negative and sum to 1.0 within numerical precision.
**Validates: Requirements 7.3**

**Property 26: Isotonic calibration monotonicity**
*For any* fitted isotonic regression model, for any two predictions p1 < p2, the calibrated predictions should satisfy calibrated(p1) ≤ calibrated(p2).
**Validates: Requirements 7.5**

**Property 27: Ensemble artifact completeness**
*For any* saved ensemble artifacts, loading should provide all meta-learners, level-2 weights, and isotonic calibration model.
**Validates: Requirements 7.6**

### Dataset Properties

**Property 28: Data merge correctness**
*For any* raw DataFrame and features DataFrame, merging on sample_id should preserve all rows from the raw DataFrame and add all feature columns.
**Validates: Requirements 8.1**

**Property 29: Tokenization length constraint**
*For any* text input, after tokenization with max_length=128, the output input_ids should have exactly length 128 (padded or truncated).
**Validates: Requirements 8.2**

**Property 30: Image transformation shape**
*For any* input image, after transformation, the output tensor should have shape (3, 224, 224) and values in range approximately [-2.5, 2.5] (after normalization).
**Validates: Requirements 8.3**

**Property 31: Data augmentation conditional application**
*For any* dataset in training mode, applying augmentation should produce different outputs for the same image on different calls; in evaluation mode, outputs should be identical.
**Validates: Requirements 8.4**

**Property 32: NaN filling completeness**
*For any* tabular features with NaN values, after loading through the dataset, all NaN values should be replaced with 0.0.
**Validates: Requirements 8.5**

**Property 33: Image loading fallback**
*For any* missing or corrupted image file, the dataset should return a blank gray image of shape (3, 224, 224) instead of raising an error.
**Validates: Requirements 8.7**

### Metrics Properties

**Property 34: SMAPE formula correctness**
*For any* predictions and targets, SMAPE should equal 100 * mean(|pred - true| / ((|true| + |pred|) / 2 + epsilon)).
**Validates: Requirements 9.1**

**Property 35: Log space conversion**
*For any* predictions in log space, calculating SMAPE should first apply expm1 to convert to original space.
**Validates: Requirements 9.2**

**Property 36: Metrics completeness**
*For any* evaluation call, the returned dictionary should contain keys: 'smape', 'mae', 'rmse', 'mape', 'r2'.
**Validates: Requirements 9.3**

**Property 37: Quantile SMAPE stratification**
*For any* predictions and targets with n_quantiles=5, calculating SMAPE by quantiles should produce exactly 5 groups with non-overlapping price ranges.
**Validates: Requirements 9.4**

### Configuration Properties

**Property 38: Configuration path completeness**
*For any* initialized config object, it should provide valid Path objects for: DATA_DIR, IMAGE_DIR, MODEL_DIR, CHECKPOINT_DIR, PRED_DIR.
**Validates: Requirements 10.2**

**Property 39: Configuration hyperparameter completeness**
*For any* initialized config object, it should provide all required hyperparameters: LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, LORA_R, LORA_ALPHA.
**Validates: Requirements 10.3, 10.4**

**Property 40: Feature engineering config completeness**
*For any* initialized config object, it should provide IPQ_PATTERNS (list of regex), UNIT_CONVERSIONS (dict), and TFIDF_MAX_FEATURES (int).
**Validates: Requirements 10.5**

### Submission Properties

**Property 41: Submission format correctness**
*For any* generated submission CSV, it should have exactly 2 columns named 'sample_id' and 'predicted_price'.
**Validates: Requirements 12.3**

**Property 42: Submission completeness**
*For any* test set with N samples, the submission CSV should have exactly N rows with no missing predictions.
**Validates: Requirements 12.4**

**Property 43: Submission log space conversion**
*For any* submission generation, predictions should be converted from log space using expm1 before writing to CSV.
**Validates: Requirements 12.2**

## Error Handling

### Download Errors
- **Network failures**: Retry with exponential backoff (3 attempts)
- **Invalid URLs**: Log and skip, continue with remaining downloads
- **Timeout**: Configurable timeout (default 10s), retry on timeout
- **Disk space**: Check available space before batch download, fail gracefully

### Data Loading Errors
- **Missing images**: Create blank placeholder (gray image)
- **Corrupted images**: Create blank placeholder, log warning
- **Missing features**: Fill NaN with 0, log warning
- **Invalid text**: Use empty string, log warning

### Training Errors
- **OOM (Out of Memory)**: 
  - Reduce batch size automatically
  - Enable gradient checkpointing
  - Clear CUDA cache between batches
- **NaN loss**: 
  - Log batch that caused NaN
  - Skip batch and continue
  - If persistent, reduce learning rate
- **Checkpoint load failure**:
  - Try previous checkpoint
  - If all fail, start from scratch with warning

### Model Errors
- **Invalid predictions**: Clip to reasonable range [0, 10000]
- **Model load failure**: Provide clear error message with path
- **Incompatible checkpoint**: Validate checkpoint version, provide migration path

## Testing Strategy

### Unit Testing

Unit tests verify specific functionality and edge cases:

**Data Processing Tests**:
- Test IPQ extraction with various formats (e.g., "500ml", "2 kg", "pack of 12")
- Test text statistics with edge cases (empty string, very long text, special characters)
- Test keyword detection with mixed case and partial matches
- Test image loading with missing files, corrupted files, various formats

**Model Tests**:
- Test model forward pass with various batch sizes
- Test LoRA application and parameter counting
- Test cross-modal attention with different sequence lengths
- Test loss function computation with edge cases (zero predictions, equal values)

**Checkpoint Tests**:
- Test checkpoint save/load round-trip
- Test checkpoint cleanup with various numbers of checkpoints
- Test resume from different training stages

**Metrics Tests**:
- Test SMAPE calculation with known values
- Test log space conversion
- Test quantile stratification with edge cases

### Property-Based Testing

Property-based tests verify universal properties across many randomly generated inputs using **Hypothesis** (Python PBT library).

**Configuration**:
- Minimum 100 iterations per property test
- Use appropriate strategies for generating test data (text, images, numerical values)
- Shrink failing examples to minimal reproducible cases

**Test Organization**:
- Each correctness property from the design document should have one corresponding property-based test
- Tests should be tagged with comments referencing the property number
- Example tag format: `# Feature: amazon-ml-price-prediction, Property 10: Feature serialization round-trip`

**Key Property Tests**:

1. **Download resumption** (Property 2):
   - Generate random download states
   - Simulate interruption at random points
   - Verify resume continues correctly

2. **Feature extraction determinism** (Property 5):
   - Generate random product texts
   - Extract features multiple times
   - Verify identical results

3. **TF-IDF consistency** (Property 8):
   - Generate random text corpus
   - Fit TF-IDF and transform same text multiple times
   - Verify identical outputs

4. **Checkpoint round-trip** (Property 12):
   - Generate random model states
   - Save and load checkpoints
   - Verify state preservation

5. **LoRA parameter ratio** (Property 14):
   - Apply LoRA to models
   - Count trainable vs total parameters
   - Verify ratio in expected range

6. **Gradient clipping** (Property 19):
   - Generate random gradients
   - Apply clipping
   - Verify norm constraint

7. **Weight normalization** (Property 25):
   - Generate random weight vectors
   - Optimize and normalize
   - Verify sum to 1.0

8. **SMAPE calculation** (Property 34):
   - Generate random predictions and targets
   - Calculate SMAPE
   - Verify formula correctness

**Test Utilities**:
- Fixtures for creating sample data (images, text, features)
- Generators for random but valid inputs
- Assertion helpers for numerical comparisons with tolerance
- Mock objects for external dependencies (file I/O, network)

### Integration Testing

Integration tests verify end-to-end workflows:

1. **Full pipeline test** (small dataset):
   - Download 100 images
   - Engineer features
   - Train neural network for 1 epoch
   - Train one GBDT model
   - Create ensemble
   - Generate submission
   - Verify SMAPE < 50% (sanity check)

2. **Checkpoint resume test**:
   - Start training
   - Save checkpoint after 10 steps
   - Stop training
   - Resume from checkpoint
   - Verify training continues correctly

3. **Multi-stage pipeline test**:
   - Run Stage 1 (download)
   - Save stage checkpoint
   - Run Stage 2 (features) from Stage 1 checkpoint
   - Verify data flows correctly between stages

### Performance Testing

Performance tests verify efficiency constraints:

1. **Memory usage test**:
   - Train with batch_size=12 on 6GB GPU
   - Monitor VRAM usage
   - Verify stays under 5.5GB (safety margin)

2. **Checkpoint size test**:
   - Save quick checkpoint
   - Verify size < 600MB
   - Save full checkpoint
   - Verify size < 2.5GB

3. **Download speed test**:
   - Download 1000 images with 40 workers
   - Verify completes in < 5 minutes

4. **Feature engineering speed test**:
   - Engineer features for 10K samples
   - Verify completes in < 10 minutes

## Deployment Considerations

This is a development/research project, not a production deployment. However, the following considerations apply:

### Hardware Requirements
- **GPU**: NVIDIA RTX 3050 6GB or better (CUDA 11.8+)
- **CPU**: AMD Ryzen 7 5800H or equivalent (8 cores)
- **RAM**: 16GB minimum
- **Storage**: 50GB free space (20GB for images, 10GB for models, 20GB for checkpoints)

### Software Dependencies
- Python 3.10+
- PyTorch 2.1.0 with CUDA support
- Transformers 4.36.0
- LightGBM 4.1.0, XGBoost 2.0.3, CatBoost 1.2.2
- See requirements.txt for complete list

### Execution Environment
- Windows 10/11 or Linux
- Conda or venv for environment isolation
- Jupyter for exploratory analysis (optional)

### Monitoring and Logging
- All training logs saved to `logs/` directory
- Checkpoint metadata includes timestamps and metrics
- Progress bars for long-running operations (tqdm)
- Validation metrics logged after each epoch
- Final metrics printed in formatted tables

### Reproducibility
- Fixed random seeds (RANDOM_SEED=42)
- Deterministic algorithms where possible
- Version pinning for all dependencies
- Checkpoint system allows exact reproduction of any training state
