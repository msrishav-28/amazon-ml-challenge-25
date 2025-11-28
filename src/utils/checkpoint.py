"""
Checkpoint management system for resumable training.

This module provides the CheckpointManager class for saving and loading
training checkpoints with automatic cleanup and resume capability.
"""

import json
import logging
import shutil
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Universal checkpoint manager for all training stages.
    
    Supports three checkpoint types:
    - quick: ~500MB, saves every 30 minutes
    - full: ~2GB, saves at epoch boundaries
    - minimal: ~100MB, saves only essential state
    
    Automatically manages checkpoint cleanup, keeping only the most recent
    checkpoints to save disk space.
    """
    
    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize the CheckpointManager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints. Defaults to config.CHECKPOINT_DIR
        """
        self.checkpoint_dir = checkpoint_dir or config.CHECKPOINT_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CheckpointManager initialized with directory: {self.checkpoint_dir}")
    
    def save_checkpoint(
        self,
        state: Dict[str, Any],
        stage: str,
        metric: Optional[float] = None,
        checkpoint_type: str = 'quick'
    ) -> Path:
        """
        Save checkpoint with automatic cleanup.
        
        Args:
            state: Dictionary containing the state to save. Should include:
                   - model: model state dict
                   - optimizer: optimizer state dict (optional)
                   - epoch: current epoch
                   - step: current step
                   - loss: current loss value
                   - best_smape: best validation SMAPE (optional)
                   - ema_state: EMA model state (optional)
            stage: Stage identifier (e.g., 'neural_net', 'gbdt', 'ensemble')
            metric: Validation metric value (optional)
            checkpoint_type: Type of checkpoint ('quick', 'full', 'minimal')
        
        Returns:
            Path to the saved checkpoint file
        """
        # Validate checkpoint type
        valid_types = ['quick', 'full', 'minimal']
        if checkpoint_type not in valid_types:
            raise ValueError(f"checkpoint_type must be one of {valid_types}, got {checkpoint_type}")
        
        # Create checkpoint metadata
        timestamp = datetime.now().isoformat()
        checkpoint_data = {
            'state': state,
            'stage': stage,
            'metric': metric,
            'timestamp': timestamp,
            'checkpoint_type': checkpoint_type
        }
        
        # Generate checkpoint filename
        metric_str = f"_metric{metric:.4f}" if metric is not None else ""
        filename = f"{stage}_{checkpoint_type}_{timestamp.replace(':', '-')}{metric_str}.pt"
        checkpoint_path = self.checkpoint_dir / filename
        
        # Save checkpoint
        try:
            torch.save(checkpoint_data, checkpoint_path)
            
            # Get file size
            size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            logger.info(f"Saved {checkpoint_type} checkpoint: {checkpoint_path.name} ({size_mb:.1f} MB)")
            
            # Perform automatic cleanup
            self._cleanup_old_checkpoints(stage, checkpoint_type, metric)
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        Load checkpoint and return state with error handling.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        
        Returns:
            Dictionary containing the checkpoint data with keys:
            - state: The saved state dictionary
            - stage: Stage identifier
            - metric: Validation metric (if available)
            - timestamp: When checkpoint was created
            - checkpoint_type: Type of checkpoint
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint is corrupted or incompatible
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            # Validate checkpoint structure
            required_keys = ['state', 'stage', 'timestamp', 'checkpoint_type']
            for key in required_keys:
                if key not in checkpoint_data:
                    raise RuntimeError(f"Checkpoint missing required key: {key}")
            
            logger.info(f"Loaded checkpoint: {checkpoint_path.name}")
            logger.info(f"  Stage: {checkpoint_data['stage']}")
            logger.info(f"  Type: {checkpoint_data['checkpoint_type']}")
            logger.info(f"  Timestamp: {checkpoint_data['timestamp']}")
            if checkpoint_data.get('metric') is not None:
                logger.info(f"  Metric: {checkpoint_data['metric']:.4f}")
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise RuntimeError(f"Checkpoint load failed: {e}")
    
    def get_latest_checkpoint(self, stage: Optional[str] = None) -> Optional[Path]:
        """
        Get most recent checkpoint for a stage (for resume logic).
        
        Args:
            stage: Stage identifier to filter by. If None, returns latest across all stages.
        
        Returns:
            Path to the most recent checkpoint, or None if no checkpoints exist
        """
        # Get all checkpoint files
        checkpoint_files = list(self.checkpoint_dir.glob("*.pt"))
        
        if not checkpoint_files:
            logger.info("No checkpoints found")
            return None
        
        # Filter by stage if specified
        if stage is not None:
            checkpoint_files = [
                f for f in checkpoint_files
                if f.name.startswith(f"{stage}_")
            ]
            
            if not checkpoint_files:
                logger.info(f"No checkpoints found for stage: {stage}")
                return None
        
        # Sort by modification time (most recent first)
        checkpoint_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        latest = checkpoint_files[0]
        logger.info(f"Latest checkpoint: {latest.name}")
        
        return latest
    
    def _cleanup_old_checkpoints(
        self,
        stage: str,
        checkpoint_type: str,
        current_metric: Optional[float] = None
    ) -> None:
        """
        Maintain only the last N checkpoints and delete older ones.
        
        Keeps the best checkpoint (by metric) plus the most recent N-1 checkpoints.
        
        Args:
            stage: Stage identifier
            checkpoint_type: Type of checkpoint
            current_metric: Current metric value (lower is better for SMAPE)
        """
        max_to_keep = config.MAX_CHECKPOINTS_TO_KEEP
        
        # Get all checkpoints for this stage and type
        pattern = f"{stage}_{checkpoint_type}_*.pt"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        
        if len(checkpoints) <= max_to_keep:
            return  # No cleanup needed
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Find best checkpoint by metric (if metrics are available)
        best_checkpoint = None
        best_metric = float('inf')
        
        for ckpt_path in checkpoints:
            try:
                # Try to extract metric from filename
                if "_metric" in ckpt_path.name:
                    metric_str = ckpt_path.name.split("_metric")[1].split(".pt")[0]
                    metric = float(metric_str)
                    if metric < best_metric:
                        best_metric = metric
                        best_checkpoint = ckpt_path
            except (ValueError, IndexError):
                continue
        
        # Keep the most recent max_to_keep checkpoints
        to_keep = set(checkpoints[:max_to_keep])
        
        # Also keep the best checkpoint if it's not in the recent ones
        if best_checkpoint is not None and best_checkpoint not in to_keep:
            to_keep.add(best_checkpoint)
        
        # Delete old checkpoints
        for ckpt_path in checkpoints:
            if ckpt_path not in to_keep:
                try:
                    ckpt_path.unlink()
                    logger.info(f"Deleted old checkpoint: {ckpt_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete checkpoint {ckpt_path.name}: {e}")
    
    def list_checkpoints(self, stage: Optional[str] = None) -> list[Tuple[Path, Dict[str, Any]]]:
        """
        List all available checkpoints with their metadata.
        
        Args:
            stage: Stage identifier to filter by. If None, lists all checkpoints.
        
        Returns:
            List of tuples (checkpoint_path, metadata_dict)
        """
        checkpoint_files = list(self.checkpoint_dir.glob("*.pt"))
        
        if stage is not None:
            checkpoint_files = [
                f for f in checkpoint_files
                if f.name.startswith(f"{stage}_")
            ]
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        result = []
        for ckpt_path in checkpoint_files:
            try:
                # Load just the metadata (not the full state)
                checkpoint_data = torch.load(ckpt_path, map_location='cpu')
                metadata = {
                    'stage': checkpoint_data.get('stage'),
                    'checkpoint_type': checkpoint_data.get('checkpoint_type'),
                    'timestamp': checkpoint_data.get('timestamp'),
                    'metric': checkpoint_data.get('metric'),
                    'size_mb': ckpt_path.stat().st_size / (1024 * 1024)
                }
                result.append((ckpt_path, metadata))
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {ckpt_path.name}: {e}")
        
        return result
    
    def delete_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        Delete a specific checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint to delete
        
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info(f"Deleted checkpoint: {checkpoint_path.name}")
                return True
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_path}: {e}")
            return False
    
    def clear_stage_checkpoints(self, stage: str) -> int:
        """
        Delete all checkpoints for a specific stage.
        
        Args:
            stage: Stage identifier
        
        Returns:
            Number of checkpoints deleted
        """
        pattern = f"{stage}_*.pt"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        
        deleted_count = 0
        for ckpt_path in checkpoints:
            if self.delete_checkpoint(ckpt_path):
                deleted_count += 1
        
        logger.info(f"Deleted {deleted_count} checkpoints for stage: {stage}")
        return deleted_count
