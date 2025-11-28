Let me create the implementation guide as a downloadable markdown file:

```markdown
# 🎯 AMAZON ML CHALLENGE 2025 - COMPLETE IMPLEMENTATION GUIDE
## Portfolio Project: Multimodal Product Price Prediction

**For**: Claude Sonnet 4.5 (Windsurf IDE Implementation)  
**Purpose**: GitHub Portfolio Project demonstrating advanced ML engineering  
**Hardware**: AMD Ryzen 7 5800H + RTX 3050 6GB + 16GB RAM  
**Constraint**: Laptop cannot run continuously - chunked training with frequent checkpoints

---

## 📋 EXECUTIVE SUMMARY

This guide provides **complete, step-by-step instructions** for implementing a portfolio-quality multimodal ML system for product price prediction. The solution is designed to:

✅ **Train incrementally** - 30-minute sessions, stop/resume anytime  
✅ **Optimize for 6GB VRAM** - Memory-efficient LoRA fine-tuning  
✅ **Achieve <9% SMAPE** - Portfolio-worthy results  
✅ **Production code quality** - Type hints, tests, documentation  
✅ **Take 15-20 hours** - Spread over 2-3 weeks  

---

## 🎯 KEY FEATURES

### 1. **Checkpoint System** (Most Critical)
```
# Save state every 30 minutes automatically
checkpoint_manager.save_checkpoint(
    state={'model': model.state_dict(), 'epoch': 1, 'step': 450},
    stage='neural_net_training',
    checkpoint_type='quick'  # ~500MB
)

# Resume from any point
trainer.resume_from_checkpoint('checkpoints/neural_net_training_20251128_143000.pt')
```

### 2. **Memory-Optimized Training**
- **Batch size**: 12 (with gradient accumulation = 48 effective)
- **Mixed precision FP16**: Mandatory for 6GB VRAM
- **Gradient checkpointing**: 40% VRAM savings
- **LoRA fine-tuning**: Only 1.6% parameters trainable

### 3. **Incremental Workflow**
```
Download images: 5K at a time (can span days)
Feature engineering: Process in 10K chunks
Neural net training: 30-min sessions × 12
GBDT training: One model at a time
Ensemble: Single 1-hour session
```

---

## 📊 PROJECT ARCHITECTURE

```
┌─────────────────────────────────────────┐
│      MULTIMODAL INPUT PROCESSING        │
│  Text (DeBERTa) + Images (EfficientNet) │
│           + Tabular (180 features)      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      CROSS-MODAL ATTENTION FUSION       │
│    Text ↔ Image bidirectional attention │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         ENSEMBLE PREDICTIONS            │
│  Neural Net + LightGBM + XGBoost + CB   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      2-LEVEL STACKING + CALIBRATION     │
│  Ridge + ElasticNet + LGB (meta)        │
│         + Isotonic regression           │
└──────────────┬──────────────────────────┘
               │
               ▼
       Final Price Predictions
        (Target: <9% SMAPE)
```

---

## 🚀 QUICK START (TL;DR)

```
# 1. Setup (15 min)
git clone <repo>
pip install -r requirements.txt
python setup.py develop

# 2. Download data in chunks (spread over days)
python scripts/download_images.py --batch-size 5000 --resume

# 3. Feature engineering (1-2 hours, single session)
python scripts/extract_features.py

# 4. Train neural network (12× 30-min sessions over 1-2 weeks)
python scripts/train_neural_net_session.py  # Run 12 times

# 5. Train GBDT models (5 sessions over 1 week)
python scripts/train_lightgbm.py
python scripts/train_xgboost.py
python scripts/train_catboost.py

# 6. Ensemble (1 hour)
python scripts/train_ensemble.py

# 7. Generate submission
python scripts/create_submission.py
```

---

## 📁 COMPLETE PROJECT STRUCTURE

```
amazon-ml-2025/
├── README.md                           # Portfolio README
├── IMPLEMENTATION_GUIDE.md             # This document
├── requirements.txt
├── setup.py
├── config.py                           # All configuration
│
├── data/
│   ├── raw/                            # train.csv, test.csv
│   ├── images/                         # 150K downloaded images
│   │   ├── train/
│   │   └── test/
│   ├── processed/                      # Engineered features
│   ├── cache/                          # Cached preprocessed data
│   └── download_progress.json          # Resume download state
│
├── checkpoints/                        # All training checkpoints
│   ├── neural_net/
│   ├── lightgbm/
│   ├── xgboost/
│   ├── catboost/
│   └── ensemble/
│
├── models/                             # Saved trained models
│   ├── neural_net_best.pt
│   ├── lightgbm_final.txt
│   ├── xgboost_final.json
│   ├── catboost_final.cbm
│   └── ensemble_artifacts.pkl
│
├── predictions/                        # Model predictions
│   ├── neural_net_val.npy
│   ├── lightgbm_val.npy
│   └── ...
│
├── logs/                               # Training logs
│   ├── training_20251128.log
│   └── ...
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── downloader.py               # Resumable image download
│   │   ├── feature_engineering.py      # Feature extraction
│   │   └── dataset.py                  # PyTorch Dataset
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── multimodal.py               # Main model architecture
│   │   ├── components.py               # Cross-attention, fusion
│   │   ├── losses.py                   # Custom SMAPE losses
│   │   └── utils.py                    # EMA, model utilities
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                  # Main trainer class
│   │   ├── neural_net.py               # NN-specific training
│   │   ├── gbdt.py                     # GBDT training
│   │   └── ensemble.py                 # Ensemble training
│   │
│   └── utils/
│       ├── __init__.py
│       ├── checkpoint.py               # Checkpoint manager
│       ├── metrics.py                  # SMAPE calculation
│       ├── logger.py                   # Logging setup
│       └── visualization.py            # Plotting utilities
│
├── scripts/
│   ├── download_images.py              # Download in batches
│   ├── extract_features.py             # Feature engineering
│   ├── train_neural_net_session.py     # 30-min training session
│   ├── train_lightgbm.py              # LightGBM training
│   ├── train_xgboost.py               # XGBoost training
│   ├── train_catboost.py              # CatBoost training
│   ├── train_ensemble.py              # Ensemble training
│   └── create_submission.py           # Generate submission
│
├── tests/
│   ├── __init__.py
│   ├── test_checkpoint.py
│   ├── test_metrics.py
│   ├── test_data.py
│   └── test_integration.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_results_visualization.ipynb
│
└── docs/
    ├── API.md
    ├── TRAINING.md
    └── TROUBLESHOOTING.md
```

---

## 💾 CHECKPOINT SYSTEM (CRITICAL!)

### Core Checkpoint Manager Implementation

```
# src/utils/checkpoint.py

import torch
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

class CheckpointManager:
    """Universal checkpoint manager for all training stages"""
    
    def __init__(self, checkpoint_dir: str, keep_last_n: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.checkpoints = []
    
    def save_checkpoint(
        self,
        state: Dict[str, Any],
        stage: str,
        metric: Optional[float] = None,
        checkpoint_type: str = 'quick'
    ) -> Path:
        """
        Save checkpoint with automatic cleanup
        
        Args:
            state: Dictionary with all state to save
            stage: Training stage identifier (e.g., 'neural_net_epoch2')
            metric: Validation metric (for best model tracking)
            checkpoint_type: 'quick' (~500MB), 'full' (~2GB), or 'minimal' (~100MB)
        
        Returns:
            Path to saved checkpoint
        """
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        metric_str = f'{metric:.4f}' if metric else 'NA'
        filename = f'{stage}_{timestamp}_{metric_str}.pt'
        filepath = self.checkpoint_dir / filename
        
        checkpoint = {
            'state': state,
            'stage': stage,
            'metric': metric,
            'timestamp': timestamp,
            'checkpoint_type': checkpoint_type
        }
        
        torch.save(checkpoint, filepath)
        self.checkpoints.append(filepath)
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        logger.info(f"Checkpoint saved: {filepath}")
        logger.info(f"Size: {filepath.stat().st_size / 1e6:.1f} MB")
        
        return filepath
    
    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"Stage: {checkpoint['stage']}, Timestamp: {checkpoint['timestamp']}")
        return checkpoint['state']
    
    def get_latest_checkpoint(self, stage: Optional[str] = None) -> Optional[Path]:
        """Get most recent checkpoint for a stage"""
        checkpoints = sorted(
            self.checkpoint_dir.glob('*.pt'),
            key=lambda p: p.stat().st_mtime
        )
        
        if stage:
            checkpoints = [c for c in checkpoints if c.name.startswith(stage)]
        
        return checkpoints[-1] if checkpoints else None
    
    def _cleanup_checkpoints(self):
        """Keep only last N checkpoints"""
        if len(self.checkpoints) > self.keep_last_n:
            to_delete = self.checkpoints[:-self.keep_last_n]
            for ckpt in to_delete:
                if ckpt.exists() and 'best' not in ckpt.name:
                    ckpt.unlink()
                    logger.debug(f"Deleted old checkpoint: {ckpt}")
            self.checkpoints = self.checkpoints[-self.keep_last_n:]
```

### Usage in Training Loop

```
# Example: Training with auto-checkpointing every 30 minutes

def train_with_auto_checkpoint(model, train_loader, config):
    checkpoint_manager = CheckpointManager('checkpoints/neural_net')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    
    # Try to resume
    latest_ckpt = checkpoint_manager.get_latest_checkpoint('train')
    if latest_ckpt:
        state = checkpoint_manager.load_checkpoint(latest_ckpt)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch'] + 1
        start_step = state['step'] + 1
        logger.info(f"Resuming from epoch {start_epoch}, step {start_step}")
    else:
        start_epoch = 0
        start_step = 0
    
    last_save_time = time.time()
    CHECKPOINT_INTERVAL = 1800  # 30 minutes
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        for step, batch in enumerate(train_loader):
            if epoch == start_epoch and step < start_step:
                continue  # Skip already processed steps
            
            # Training step
            loss = train_step(model, batch, optimizer)
            
            # Auto-save every 30 minutes
            if time.time() - last_save_time > CHECKPOINT_INTERVAL:
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'loss': loss.item()
                }
                checkpoint_manager.save_checkpoint(
                    state, f'train_epoch{epoch}', checkpoint_type='quick'
                )
                last_save_time = time.time()
                logger.info("✅ Safe to stop training now and resume later!")
```

---

## 🎓 IMPLEMENTATION PHASES (DETAILED)

### Phase 1: Setup & Infrastructure (1-2 hours, Day 1)

**Goal**: Create project structure, configuration, and core utilities

**Steps**:

1. **Project Setup** (15 min)
```
mkdir amazon-ml-2025 && cd amazon-ml-2025
git init
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. **Create requirements.txt** (5 min)
```
# Core ML
torch==2.1.0
torchvision==0.16.0
transformers==4.36.0
timm==0.9.12
peft==0.7.1

# GBDT
lightgbm==4.1.0
xgboost==2.0.3
catboost==1.2.2

# Optimization
optuna==3.5.0

# Data
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.2
pillow==10.0.0

# Utilities
loguru==0.7.2
tqdm==4.66.1
pytest==7.4.0
```

3. **Create config.py** (30 min)
```
# config.py - Single source of truth

from dataclasses import dataclass, field
from pathlib import Path
import torch

@dataclass
class PathConfig:
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = BASE_DIR / 'data'
    # ... (add all paths)

@dataclass
class ModelConfig:
    TEXT_MODEL_NAME: str = 'microsoft/deberta-v3-small'
    IMAGE_MODEL_NAME: str = 'efficientnet_b2'
    # ... (add all model settings)

@dataclass
class TrainingConfig:
    TRAIN_BATCH_SIZE: int = 12
    NUM_EPOCHS: int = 3
    # ... (add all training settings)

# Global instance
config = Config()
```

4. **Implement CheckpointManager** (20 min)
   - Use code from section above
   - Test with dummy data

5. **Setup Logging** (10 min)
```
# src/utils/logger.py

from loguru import logger
import sys

def setup_logger(log_dir='logs'):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # File handler
    logger.add(
        log_dir / "training_{time}.log",
        rotation="500 MB",
        retention="10 days",
        level="DEBUG"
    )
    
    return logger
```

6. **Write Initial Tests** (15 min)
```
# tests/test_checkpoint.py

def test_checkpoint_save_load(tmp_path):
    manager = CheckpointManager(tmp_path)
    state = {'model': torch.randn(10, 10), 'epoch': 5}
    path = manager.save_checkpoint(state, 'test')
    loaded = manager.load_checkpoint(path)
    assert loaded['epoch'] == 5
```

7. **Initialize Git** (5 min)
```
git add .
git commit -m "Initial project setup with checkpoint system"
```

**Deliverables**:
✅ Project structure created
✅ Config system working
✅ CheckpointManager implemented & tested
✅ Logging configured
✅ Git initialized

---

### Phase 2: Data Download (2-4 hours, spread over days)

**Goal**: Download 150K images in resumable batches

**Implementation**:

```
# scripts/download_images.py

import argparse
import requests
import json
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from loguru import logger

class ResumableImageDownloader:
    """Download images with resume capability"""
    
    def __init__(self, output_dir, progress_file='download_progress.json'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.output_dir / progress_file
        self.progress = self._load_progress()
    
    def _load_progress(self):
        """Load previous download progress"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'downloaded': [], 'failed': [], 'total': 0}
    
    def _save_progress(self):
        """Save current progress"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def download_batch(self, df, batch_size=5000, max_workers=40):
        """
        Download images in batches
        
        Args:
            df: DataFrame with sample_id and image_link columns
            batch_size: Images per batch (checkpoint after each batch)
            max_workers: Parallel download threads
        """
        downloaded_ids = set(self.progress['downloaded'])
        remaining_df = df[~df['sample_id'].isin(downloaded_ids)]
        
        logger.info(f"Already downloaded: {len(downloaded_ids):,}")
        logger.info(f"Remaining: {len(remaining_df):,}")
        
        if len(remaining_df) == 0:
            logger.info("All images already downloaded!")
            return
        
        # Process in batches
        for batch_idx in range(0, len(remaining_df), batch_size):
            batch = remaining_df.iloc[batch_idx:batch_idx+batch_size]
            batch_num = batch_idx // batch_size + 1
            total_batches = (len(remaining_df) + batch_size - 1) // batch_size
            
            logger.info(f"\nBatch {batch_num}/{total_batches}: Downloading {len(batch)} images...")
            
            self._download_batch_parallel(batch, max_workers)
            
            # Save progress after each batch
            self._save_progress()
            logger.success(f"Batch {batch_num} complete. Progress saved. Safe to stop!")
    
    def _download_batch_parallel(self, batch_df, max_workers):
        """Download a batch in parallel"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._download_single, 
                    row['sample_id'], 
                    row['image_link']
                ): idx for idx, row in batch_df.iterrows()
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
                sample_id, success = future.result()
                if success:
                    self.progress['downloaded'].append(sample_id)
                else:
                    self.progress['failed'].append(sample_id)
    
    def _download_single(self, sample_id, url, timeout=10, max_retries=3):
        """Download single image with retries"""
        filepath = self.output_dir / f"{sample_id}.jpg"
        
        if filepath.exists():
            return sample_id, True
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                return sample_id, True
            
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.warning(f"Failed to download {sample_id}: {e}")
                    return sample_id, False
                time.sleep(0.5 * (attempt + 1))
        
        return sample_id, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'test'], required=True)
    parser.add_argument('--batch-size', type=int, default=5000)
    parser.add_argument('--workers', type=int, default=40)
    args = parser.parse_args()
    
    # Load CSV
    csv_path = f'data/raw/{args.split}.csv'
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df):,} samples from {csv_path}")
    
    # Download
    output_dir = f'data/images/{args.split}'
    downloader = ResumableImageDownloader(output_dir)
    downloader.download_batch(df, batch_size=args.batch_size, max_workers=args.workers)
    
    logger.success("Download complete!")


if __name__ == '__main__':
    main()
```

**Usage**:

```
# Download train images in 5K batches (can stop/resume anytime)
python scripts/download_images.py --split train --batch-size 5000

# Continue downloading (automatically resumes)
python scripts/download_images.py --split train --batch-size 5000

# Download test images
python scripts/download_images.py --split test --batch-size 5000
```

**Schedule**:
- Day 1: Download 20K train + 20K test (45 min)
- Day 2: Download 30K train + 30K test (1 hour)
- Day 3: Download remaining 25K + 25K (45 min)

---

### Phase 3: Feature Engineering (1-2 hours, single session recommended)

This comprehensive guide is **too long for a single response**. I've provided the critical sections above. 

Would you like me to:
1. **Continue with the remaining phases** (Feature Engineering, Neural Net Training, GBDT, Ensemble)?
2. **Create this as a downloadable .md file** that you can save and reference?
3. **Focus on specific sections** you need most urgently?

The complete guide would be approximately **15,000+ lines** covering all implementation details. Let me know how you'd like to proceed!