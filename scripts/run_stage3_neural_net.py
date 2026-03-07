#!/usr/bin/env python3
"""
Stage 3: Neural Network Training

This script handles:
1. Dataset and dataloader creation
2. Model initialization with LoRA
3. Training with mixed precision
4. Checkpoint saving and resumption
5. Validation and prediction generation
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import PATHS, TRAIN_CONFIG, MODEL_CONFIG
from src.data.dataset import AmazonMLDataset, get_dataloader
from src.models.multimodal import OptimizedMultimodalModel
from src.training.train_neural_net import train_neural_network, predict_with_tta
from src.utils.metrics import calculate_smape, evaluate_predictions
from src.utils.checkpoint import CheckpointManager
from src.utils.visualization import plot_training_curves, plot_predictions

# Configure logging
PATHS['logs_dir'].mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PATHS['logs_dir'] / 'stage3_neural_net.log')
    ]
)
logger = logging.getLogger(__name__)


def load_data():
    """Load raw data and features"""
    logger.info("Loading data...")
    
    # Load raw data
    train_df = pd.read_csv(PATHS['raw_dir'] / 'train.csv')
    test_df = pd.read_csv(PATHS['raw_dir'] / 'test.csv')
    
    # Load features
    processed_dir = PATHS['processed_dir']
    train_features = pd.read_parquet(processed_dir / 'train_features.parquet')
    test_features = pd.read_parquet(processed_dir / 'test_features.parquet')
    
    logger.info(f"  Training samples: {len(train_df)}")
    logger.info(f"  Test samples: {len(test_df)}")
    logger.info(f"  Feature columns: {train_features.shape[1]}")
    
    return train_df, test_df, train_features, test_features


def create_datasets(train_df, test_df, train_features, test_features, val_size=0.1):
    """Create train/val/test datasets"""
    logger.info("\nCreating datasets...")
    
    from transformers import AutoTokenizer
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['text_config']['model_name'])
    
    # Split training data
    train_idx, val_idx = train_test_split(
        range(len(train_df)),
        test_size=val_size,
        random_state=42
    )
    
    # Create split DataFrames
    train_split_df = train_df.iloc[train_idx].reset_index(drop=True)
    val_split_df = train_df.iloc[val_idx].reset_index(drop=True)
    train_split_features = train_features.iloc[train_idx].reset_index(drop=True)
    val_split_features = train_features.iloc[val_idx].reset_index(drop=True)
    
    logger.info(f"  Train split: {len(train_split_df)}")
    logger.info(f"  Val split: {len(val_split_df)}")
    
    # Create datasets using correct constructor args
    train_dataset = AmazonMLDataset(
        raw_df=train_split_df,
        features_df=train_split_features,
        image_dir=PATHS['images_dir'],
        tokenizer=tokenizer,
        mode='train'
    )
    
    val_dataset = AmazonMLDataset(
        raw_df=val_split_df,
        features_df=val_split_features,
        image_dir=PATHS['images_dir'],
        tokenizer=tokenizer,
        mode='val'
    )
    
    test_dataset = AmazonMLDataset(
        raw_df=test_df,
        features_df=test_features,
        image_dir=PATHS['images_dir'],
        tokenizer=tokenizer,
        mode='test'
    )
    
    return train_dataset, val_dataset, test_dataset, train_split_df, val_split_df


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=16):
    """Create dataloaders"""
    train_loader = get_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = get_dataloader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = get_dataloader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(
        description='Stage 3: Neural Network Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training
  python run_stage3_neural_net.py

  # Resume from checkpoint
  python run_stage3_neural_net.py --resume checkpoints/neural_net_xxx.pt

  # Quick test with fewer epochs
  python run_stage3_neural_net.py --epochs 2 --debug

  # Predict only (using saved model)
  python run_stage3_neural_net.py --predict-only --model-path models/neural_net_best.pt
        """
    )
    
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG.get('nn_epochs', 20),
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=TRAIN_CONFIG.get('batch_size', 16),
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=TRAIN_CONFIG.get('learning_rate', 2e-5),
                        help='Learning rate')
    parser.add_argument('--predict-only', action='store_true',
                        help='Skip training, only generate predictions')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to saved model for prediction')
    parser.add_argument('--no-tta', action='store_true',
                        help='Disable test-time augmentation')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode with fewer samples')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("AMAZON ML CHALLENGE - STAGE 3: NEURAL NETWORK TRAINING")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    try:
        # Load data
        train_df, test_df, train_features, test_features = load_data()
        
        # Debug mode
        if args.debug:
            logger.info("\n*** DEBUG MODE - Using reduced data ***")
            train_df = train_df.head(500)
            test_df = test_df.head(100)
            train_features = train_features.head(500)
            test_features = test_features.head(100)
        
        # Create datasets and dataloaders
        train_dataset, val_dataset, test_dataset, train_split_df, val_split_df = create_datasets(
            train_df, test_df, train_features, test_features
        )
        
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=args.batch_size
        )
        
        # Derive actual number of numeric tabular features
        numeric_cols = train_features.select_dtypes(include=[np.number, bool]).columns
        # Exclude metadata columns
        metadata_cols = {'sample_id', 'price'}
        numeric_feature_cols = [c for c in numeric_cols if c not in metadata_cols]
        actual_tabular_dim = len(numeric_feature_cols)
        logger.info(f"  Actual numeric tabular features: {actual_tabular_dim}")
        
        if args.predict_only:
            # Load model and predict
            logger.info("\n" + "=" * 60)
            logger.info("PREDICTION MODE")
            logger.info("=" * 60)
            
            model_path = args.model_path or PATHS['models_dir'] / 'neural_net_best.pt'
            
            model = OptimizedMultimodalModel(
                num_tabular_features=actual_tabular_dim,
                hidden_dim=MODEL_CONFIG['hidden_dim'],
            )
            
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            
            # Generate predictions with TTA
            val_preds = predict_with_tta(model, val_loader, device)
            test_preds = predict_with_tta(model, test_loader, device)
            
            results = {}
            
        else:
            # Train model
            logger.info("\n" + "=" * 60)
            logger.info("TRAINING MODE")
            logger.info("=" * 60)
            
            model, results = train_neural_network(
                train_loader=train_loader,
                val_loader=val_loader,
                num_tabular_features=actual_tabular_dim,
                resume=(args.resume is not None),
                use_lora=True,
            )
            
            # Generate predictions using trained model
            model.eval()
            val_preds = predict_with_tta(model, val_loader, device)
            test_preds = predict_with_tta(model, test_loader, device)
            
            # Save training curves
            if results.get('train_losses'):
                plot_training_curves(
                    train_losses=results['train_losses'],
                    val_losses=results.get('val_losses'),
                    val_smapes=results.get('val_smapes'),
                    save_path=PATHS['logs_dir'] / 'neural_net_training_curves.png',
                    title='Neural Network Training Curves'
                )
        
        # Evaluate — model outputs are in log1p space, convert to price space
        y_val = val_split_df['price'].values
        val_preds_price = np.expm1(val_preds)
        test_preds_price = np.expm1(test_preds)
        val_smape = calculate_smape(y_val, val_preds_price)
        
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION RESULTS")
        logger.info("=" * 60)
        
        metrics = evaluate_predictions(y_val, val_preds_price, 'validation')
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Save predictions (in price space for consistency with GBDT predictions)
        predictions_dir = PATHS['predictions_dir']
        predictions_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as numpy arrays (price space)
        np.save(predictions_dir / 'nn_val_preds.npy', val_preds_price)
        np.save(predictions_dir / 'nn_test_preds.npy', test_preds_price)
        
        # Also save full predictions with sample_ids
        val_pred_df = pd.DataFrame({
            'sample_id': val_split_df['sample_id'],
            'true_price': y_val,
            'predicted_price': val_preds_price
        })
        val_pred_df.to_csv(predictions_dir / 'nn_val_predictions.csv', index=False)
        
        test_pred_df = pd.DataFrame({
            'sample_id': test_df['sample_id'],
            'predicted_price': test_preds_price
        })
        test_pred_df.to_csv(predictions_dir / 'nn_test_predictions.csv', index=False)
        
        logger.info(f"\nPredictions saved to {predictions_dir}")
        
        # Plot predictions
        plot_predictions(
            y_val, val_preds_price,
            split_name='Validation',
            save_path=PATHS['logs_dir'] / 'neural_net_predictions.png',
            smape=val_smape
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 3 COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Validation SMAPE: {val_smape:.4f}")
        logger.info("Next step: Run stage 4 for GBDT training")
        logger.info("  python scripts/run_stage4_gbdt.py")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Stage 3 failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
