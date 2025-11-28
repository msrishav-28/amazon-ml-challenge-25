"""
Demonstration of the CheckpointManager functionality.

This script shows how to use the checkpoint system for saving and resuming training.
"""

import sys
import os
from pathlib import Path
from collections import OrderedDict

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from src.utils.checkpoint import CheckpointManager


def demo_basic_checkpoint():
    """Demonstrate basic checkpoint save and load."""
    print("=" * 60)
    print("Demo 1: Basic Checkpoint Save and Load")
    print("=" * 60)
    
    # Create checkpoint manager
    manager = CheckpointManager()
    
    # Create a simple model state
    state = {
        'model': OrderedDict({
            'layer1.weight': torch.randn(10, 5),
            'layer1.bias': torch.randn(10),
            'layer2.weight': torch.randn(1, 10),
            'layer2.bias': torch.randn(1),
        }),
        'optimizer': {
            'param_groups': [{'lr': 0.001, 'weight_decay': 0.01}]
        },
        'epoch': 5,
        'step': 1000,
        'loss': 0.234,
        'best_smape': 8.5,
    }
    
    # Save checkpoint
    print("\nSaving checkpoint...")
    checkpoint_path = manager.save_checkpoint(
        state=state,
        stage='demo_training',
        metric=8.5,
        checkpoint_type='quick'
    )
    print(f"Checkpoint saved to: {checkpoint_path}")
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    loaded_data = manager.load_checkpoint(checkpoint_path)
    
    print(f"Loaded state:")
    print(f"  Epoch: {loaded_data['state']['epoch']}")
    print(f"  Step: {loaded_data['state']['step']}")
    print(f"  Loss: {loaded_data['state']['loss']:.4f}")
    print(f"  Best SMAPE: {loaded_data['state']['best_smape']:.2f}")
    print(f"  Metric: {loaded_data['metric']:.2f}")
    
    # Verify data integrity
    original_weight = state['model']['layer1.weight']
    loaded_weight = loaded_data['state']['model']['layer1.weight']
    print(f"\nData integrity check:")
    print(f"  Weights match: {torch.allclose(original_weight, loaded_weight)}")


def demo_checkpoint_cleanup():
    """Demonstrate automatic checkpoint cleanup."""
    print("\n" + "=" * 60)
    print("Demo 2: Automatic Checkpoint Cleanup")
    print("=" * 60)
    
    manager = CheckpointManager()
    
    # Save multiple checkpoints
    print("\nSaving 6 checkpoints...")
    for i in range(6):
        state = {
            'model': OrderedDict({'layer.weight': torch.randn(10, 10)}),
            'optimizer': {'param_groups': [{'lr': 0.001}]},
            'epoch': i,
            'step': i * 100,
            'loss': float(i),
            'best_smape': 10.0 - i * 0.5,
        }
        
        checkpoint_path = manager.save_checkpoint(
            state=state,
            stage='cleanup_demo',
            metric=10.0 - i * 0.5,
            checkpoint_type='quick'
        )
        print(f"  Saved checkpoint {i+1}: epoch={i}, metric={10.0 - i * 0.5:.1f}")
    
    # List remaining checkpoints
    print("\nRemaining checkpoints after cleanup:")
    checkpoints = manager.list_checkpoints(stage='cleanup_demo')
    print(f"  Total: {len(checkpoints)} checkpoints")
    for ckpt_path, metadata in checkpoints:
        print(f"    - {ckpt_path.name}")
        print(f"      Epoch: {metadata.get('stage')}, Metric: {metadata.get('metric'):.2f}")


def demo_resume_training():
    """Demonstrate resuming training from checkpoint."""
    print("\n" + "=" * 60)
    print("Demo 3: Resume Training from Checkpoint")
    print("=" * 60)
    
    manager = CheckpointManager()
    
    # Simulate initial training
    print("\nSimulating initial training...")
    for epoch in range(3):
        state = {
            'model': OrderedDict({'layer.weight': torch.randn(10, 10)}),
            'optimizer': {'param_groups': [{'lr': 0.001}]},
            'epoch': epoch,
            'step': epoch * 100,
            'loss': 1.0 - epoch * 0.1,
            'best_smape': 10.0 - epoch,
        }
        
        manager.save_checkpoint(
            state=state,
            stage='resume_demo',
            checkpoint_type='full'
        )
        print(f"  Completed epoch {epoch}")
    
    # Get latest checkpoint
    print("\nResuming training...")
    latest_checkpoint = manager.get_latest_checkpoint(stage='resume_demo')
    print(f"Found latest checkpoint: {latest_checkpoint.name}")
    
    # Load and continue
    loaded_data = manager.load_checkpoint(latest_checkpoint)
    resume_epoch = loaded_data['state']['epoch'] + 1
    resume_step = loaded_data['state']['step']
    
    print(f"Resuming from:")
    print(f"  Epoch: {resume_epoch}")
    print(f"  Step: {resume_step}")
    print(f"  Previous loss: {loaded_data['state']['loss']:.4f}")
    
    # Continue training
    print("\nContinuing training for 2 more epochs...")
    for epoch in range(resume_epoch, resume_epoch + 2):
        print(f"  Training epoch {epoch}...")


def demo_checkpoint_types():
    """Demonstrate different checkpoint types."""
    print("\n" + "=" * 60)
    print("Demo 4: Different Checkpoint Types")
    print("=" * 60)
    
    manager = CheckpointManager()
    
    # Create a state
    state = {
        'model': OrderedDict({'layer.weight': torch.randn(100, 100)}),
        'optimizer': {'param_groups': [{'lr': 0.001}]},
        'epoch': 10,
        'step': 5000,
        'loss': 0.5,
        'best_smape': 7.5,
    }
    
    # Save different types
    print("\nSaving different checkpoint types...")
    for ckpt_type in ['quick', 'full', 'minimal']:
        checkpoint_path = manager.save_checkpoint(
            state=state,
            stage='types_demo',
            checkpoint_type=ckpt_type
        )
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"  {ckpt_type:8s}: {size_mb:.2f} MB")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CheckpointManager Demonstration")
    print("=" * 60)
    
    try:
        demo_basic_checkpoint()
        demo_checkpoint_cleanup()
        demo_resume_training()
        demo_checkpoint_types()
        
        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
