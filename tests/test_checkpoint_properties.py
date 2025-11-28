"""
Property-based tests for checkpoint system.

These tests verify checkpoint completeness, training resumption continuity,
and checkpoint cleanup policy as specified in the design document.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from collections import OrderedDict
from hypothesis import given, strategies as st, settings
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.checkpoint import CheckpointManager
import torch
import torch.nn as nn


# Helper strategies for generating test data
@st.composite
def model_state_dict_strategy(draw):
    """Generate a random model state dict."""
    num_layers = draw(st.integers(min_value=1, max_value=5))
    state_dict = OrderedDict()
    
    for i in range(num_layers):
        # Generate random tensor shapes
        in_features = draw(st.integers(min_value=10, max_value=100))
        out_features = draw(st.integers(min_value=10, max_value=100))
        
        # Create weight and bias tensors
        state_dict[f'layer{i}.weight'] = torch.randn(out_features, in_features)
        state_dict[f'layer{i}.bias'] = torch.randn(out_features)
    
    return state_dict


@st.composite
def optimizer_state_strategy(draw):
    """Generate a random optimizer state dict."""
    return {
        'state': {},
        'param_groups': [{
            'lr': draw(st.floats(min_value=1e-6, max_value=1e-2)),
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': draw(st.floats(min_value=0.0, max_value=0.1)),
        }]
    }


@st.composite
def checkpoint_state_strategy(draw):
    """Generate a random checkpoint state."""
    return {
        'model': draw(model_state_dict_strategy()),
        'optimizer': draw(optimizer_state_strategy()),
        'epoch': draw(st.integers(min_value=0, max_value=100)),
        'step': draw(st.integers(min_value=0, max_value=10000)),
        'loss': draw(st.floats(min_value=0.0, max_value=100.0)),
        'best_smape': draw(st.floats(min_value=0.0, max_value=100.0)),
    }


class TestCheckpointProperties:
    """Property-based tests for checkpoint system."""
    
    def setup_method(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir)
        self.manager = CheckpointManager(checkpoint_dir=self.checkpoint_dir)
    
    def teardown_method(self):
        """Clean up temporary directory after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @given(
        state=checkpoint_state_strategy(),
        stage=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        checkpoint_type=st.sampled_from(['quick', 'full', 'minimal'])
    )
    @settings(max_examples=100, deadline=None)
    def test_property_11_checkpoint_completeness(self, state, stage, checkpoint_type):
        """
        Feature: amazon-ml-price-prediction, Property 11: Checkpoint completeness
        
        For any saved checkpoint, loading it should provide all required fields:
        model state, optimizer state, epoch, step, and loss.
        
        Validates: Requirements 3.2
        """
        # Save checkpoint
        checkpoint_path = self.manager.save_checkpoint(
            state=state,
            stage=stage,
            checkpoint_type=checkpoint_type
        )
        
        # Load checkpoint
        loaded_data = self.manager.load_checkpoint(checkpoint_path)
        
        # Verify all required fields are present in the loaded data
        assert 'state' in loaded_data, "Checkpoint missing 'state' field"
        assert 'stage' in loaded_data, "Checkpoint missing 'stage' field"
        assert 'timestamp' in loaded_data, "Checkpoint missing 'timestamp' field"
        assert 'checkpoint_type' in loaded_data, "Checkpoint missing 'checkpoint_type' field"
        
        # Verify the state contains all required fields
        loaded_state = loaded_data['state']
        assert 'model' in loaded_state, "State missing 'model' field"
        assert 'optimizer' in loaded_state, "State missing 'optimizer' field"
        assert 'epoch' in loaded_state, "State missing 'epoch' field"
        assert 'step' in loaded_state, "State missing 'step' field"
        assert 'loss' in loaded_state, "State missing 'loss' field"
        
        # Verify values match
        assert loaded_data['stage'] == stage
        assert loaded_data['checkpoint_type'] == checkpoint_type
        assert loaded_state['epoch'] == state['epoch']
        assert loaded_state['step'] == state['step']
        assert abs(loaded_state['loss'] - state['loss']) < 1e-6
    
    @given(
        state=checkpoint_state_strategy(),
        stage=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        checkpoint_type=st.sampled_from(['quick', 'full', 'minimal'])
    )
    @settings(max_examples=100, deadline=None)
    def test_property_12_training_resumption_continuity(self, state, stage, checkpoint_type):
        """
        Feature: amazon-ml-price-prediction, Property 12: Training resumption continuity
        
        For any training state, saving a checkpoint at step N and resuming should
        continue training from step N+1 with the same model weights and optimizer state.
        
        Validates: Requirements 3.3
        """
        # Save checkpoint at step N
        current_step = state['step']
        checkpoint_path = self.manager.save_checkpoint(
            state=state,
            stage=stage,
            checkpoint_type=checkpoint_type
        )
        
        # Load checkpoint
        loaded_data = self.manager.load_checkpoint(checkpoint_path)
        loaded_state = loaded_data['state']
        
        # Verify we can resume from the saved step
        assert loaded_state['step'] == current_step, \
            f"Loaded step {loaded_state['step']} doesn't match saved step {current_step}"
        
        # Verify model state is preserved
        for key in state['model'].keys():
            assert key in loaded_state['model'], f"Model state missing key: {key}"
            original_tensor = state['model'][key]
            loaded_tensor = loaded_state['model'][key]
            assert torch.allclose(original_tensor, loaded_tensor, rtol=1e-5, atol=1e-8), \
                f"Model state mismatch for key: {key}"
        
        # Verify optimizer state is preserved
        assert loaded_state['optimizer']['param_groups'][0]['lr'] == \
               state['optimizer']['param_groups'][0]['lr'], \
               "Optimizer learning rate not preserved"
        
        # Verify epoch is preserved
        assert loaded_state['epoch'] == state['epoch'], \
            f"Epoch {loaded_state['epoch']} doesn't match saved epoch {state['epoch']}"
        
        # Simulate continuing training (step N+1)
        next_step = current_step + 1
        loaded_state['step'] = next_step
        
        # Save the continued state
        continued_checkpoint_path = self.manager.save_checkpoint(
            state=loaded_state,
            stage=stage,
            checkpoint_type=checkpoint_type
        )
        
        # Load and verify the continued checkpoint
        continued_data = self.manager.load_checkpoint(continued_checkpoint_path)
        assert continued_data['state']['step'] == next_step, \
            "Training continuation failed to increment step"
    
    @given(
        state=checkpoint_state_strategy(),
        stage=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        checkpoint_type=st.sampled_from(['quick', 'full', 'minimal'])
    )
    @settings(max_examples=100, deadline=None)
    def test_checkpoint_round_trip_preserves_all_data(self, state, stage, checkpoint_type):
        """
        Additional property test: Round-trip save/load preserves all data.
        
        For any checkpoint state, saving and loading should preserve all data
        with numerical precision.
        """
        # Save checkpoint
        checkpoint_path = self.manager.save_checkpoint(
            state=state,
            stage=stage,
            checkpoint_type=checkpoint_type
        )
        
        # Load checkpoint
        loaded_data = self.manager.load_checkpoint(checkpoint_path)
        loaded_state = loaded_data['state']
        
        # Verify all model tensors are preserved
        assert len(loaded_state['model']) == len(state['model']), \
            "Model state dict size mismatch"
        
        for key in state['model'].keys():
            assert torch.allclose(
                state['model'][key],
                loaded_state['model'][key],
                rtol=1e-5,
                atol=1e-8
            ), f"Tensor mismatch for key: {key}"
        
        # Verify scalar values are preserved
        assert loaded_state['epoch'] == state['epoch']
        assert loaded_state['step'] == state['step']
        assert abs(loaded_state['loss'] - state['loss']) < 1e-6
        assert abs(loaded_state['best_smape'] - state['best_smape']) < 1e-6
    
    @given(
        state=checkpoint_state_strategy(),
        stage=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        metric=st.floats(min_value=0.0, max_value=100.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_checkpoint_with_metric_preserves_metric(self, state, stage, metric):
        """
        Property test: Checkpoints with metrics preserve the metric value.
        
        For any checkpoint saved with a metric, loading should return the same metric.
        """
        # Save checkpoint with metric
        checkpoint_path = self.manager.save_checkpoint(
            state=state,
            stage=stage,
            metric=metric,
            checkpoint_type='full'
        )
        
        # Load checkpoint
        loaded_data = self.manager.load_checkpoint(checkpoint_path)
        
        # Verify metric is preserved
        assert loaded_data['metric'] is not None, "Metric not saved"
        assert abs(loaded_data['metric'] - metric) < 1e-6, \
            f"Metric mismatch: expected {metric}, got {loaded_data['metric']}"
    
    def test_get_latest_checkpoint_returns_most_recent(self):
        """
        Property test: get_latest_checkpoint returns the most recently saved checkpoint.
        
        For any sequence of checkpoint saves, get_latest_checkpoint should return
        the most recent one.
        """
        stage = "test_stage"
        
        # Save multiple checkpoints with small delays
        checkpoints = []
        for i in range(3):
            state = {
                'model': OrderedDict({'layer.weight': torch.randn(10, 10)}),
                'optimizer': {'param_groups': [{'lr': 0.001}]},
                'epoch': i,
                'step': i * 100,
                'loss': float(i),
                'best_smape': 10.0,
            }
            checkpoint_path = self.manager.save_checkpoint(
                state=state,
                stage=stage,
                checkpoint_type='quick'
            )
            checkpoints.append(checkpoint_path)
        
        # Get latest checkpoint
        latest = self.manager.get_latest_checkpoint(stage=stage)
        
        # Should be the last one saved
        assert latest is not None
        assert latest == checkpoints[-1]
    
    def test_get_latest_checkpoint_filters_by_stage(self):
        """
        Property test: get_latest_checkpoint correctly filters by stage.
        
        For any set of checkpoints from different stages, get_latest_checkpoint
        should only return checkpoints from the specified stage.
        """
        # Save checkpoints for different stages
        stages = ["stage1", "stage2", "stage3"]
        stage_checkpoints = {}
        
        for stage in stages:
            state = {
                'model': OrderedDict({'layer.weight': torch.randn(10, 10)}),
                'optimizer': {'param_groups': [{'lr': 0.001}]},
                'epoch': 0,
                'step': 0,
                'loss': 1.0,
                'best_smape': 10.0,
            }
            checkpoint_path = self.manager.save_checkpoint(
                state=state,
                stage=stage,
                checkpoint_type='quick'
            )
            stage_checkpoints[stage] = checkpoint_path
        
        # Get latest for each stage
        for stage in stages:
            latest = self.manager.get_latest_checkpoint(stage=stage)
            assert latest is not None
            assert latest.name.startswith(f"{stage}_")


class TestCheckpointCleanupProperties:
    """Property-based tests for checkpoint cleanup policy."""
    
    def setup_method(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir)
        self.manager = CheckpointManager(checkpoint_dir=self.checkpoint_dir)
    
    def teardown_method(self):
        """Clean up temporary directory after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @given(
        num_checkpoints=st.integers(min_value=5, max_value=15),
        stage=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        checkpoint_type=st.sampled_from(['quick', 'full', 'minimal'])
    )
    @settings(max_examples=100, deadline=None)
    def test_property_13_checkpoint_cleanup_policy(self, num_checkpoints, stage, checkpoint_type):
        """
        Feature: amazon-ml-price-prediction, Property 13: Checkpoint cleanup policy
        
        For any sequence of checkpoint saves, after saving N checkpoints (N > 3),
        only the most recent 3 non-best checkpoints should exist on disk.
        
        Validates: Requirements 3.4
        """
        # Save N checkpoints
        for i in range(num_checkpoints):
            state = {
                'model': OrderedDict({'layer.weight': torch.randn(10, 10)}),
                'optimizer': {'param_groups': [{'lr': 0.001}]},
                'epoch': i,
                'step': i * 100,
                'loss': float(i),
                'best_smape': 10.0,
            }
            
            # Use different metrics to test best checkpoint preservation
            metric = 10.0 - (i * 0.1)  # Decreasing metric (better over time)
            
            self.manager.save_checkpoint(
                state=state,
                stage=stage,
                metric=metric,
                checkpoint_type=checkpoint_type
            )
        
        # Count remaining checkpoints for this stage and type
        pattern = f"{stage}_{checkpoint_type}_*.pt"
        remaining_checkpoints = list(self.checkpoint_dir.glob(pattern))
        
        # Should have at most MAX_CHECKPOINTS_TO_KEEP + 1 (for best checkpoint)
        # In practice, should be exactly MAX_CHECKPOINTS_TO_KEEP since best is among recent
        max_expected = 4  # 3 recent + 1 best (if not in recent)
        
        assert len(remaining_checkpoints) <= max_expected, \
            f"Too many checkpoints remaining: {len(remaining_checkpoints)} > {max_expected}"
        
        # Verify at least MAX_CHECKPOINTS_TO_KEEP are kept (if we saved that many)
        if num_checkpoints >= 3:
            assert len(remaining_checkpoints) >= 3, \
                f"Too few checkpoints kept: {len(remaining_checkpoints)} < 3"
    
    @given(
        num_checkpoints=st.integers(min_value=5, max_value=10),
        stage=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
    )
    @settings(max_examples=100, deadline=None)
    def test_cleanup_preserves_best_checkpoint(self, num_checkpoints, stage):
        """
        Property test: Cleanup preserves the best checkpoint by metric.
        
        For any sequence of checkpoints with metrics, the checkpoint with the
        best (lowest) metric should be preserved even if it's not recent.
        """
        # Save checkpoints with varying metrics
        best_metric = float('inf')
        best_checkpoint_epoch = -1
        
        for i in range(num_checkpoints):
            state = {
                'model': OrderedDict({'layer.weight': torch.randn(10, 10)}),
                'optimizer': {'param_groups': [{'lr': 0.001}]},
                'epoch': i,
                'step': i * 100,
                'loss': float(i),
                'best_smape': 10.0,
            }
            
            # Create a metric pattern where the best is in the middle
            if i == num_checkpoints // 2:
                metric = 5.0  # Best metric
                best_metric = metric
                best_checkpoint_epoch = i
            else:
                metric = 10.0 + i
            
            self.manager.save_checkpoint(
                state=state,
                stage=stage,
                metric=metric,
                checkpoint_type='quick'
            )
        
        # Check that checkpoints exist
        pattern = f"{stage}_quick_*.pt"
        remaining_checkpoints = list(self.checkpoint_dir.glob(pattern))
        
        # Verify best checkpoint is preserved (check by loading and inspecting)
        found_best = False
        for ckpt_path in remaining_checkpoints:
            loaded = self.manager.load_checkpoint(ckpt_path)
            if loaded['state']['epoch'] == best_checkpoint_epoch:
                found_best = True
                break
        
        # If we saved enough checkpoints to trigger cleanup, best should be preserved
        if num_checkpoints > 3:
            assert found_best or best_checkpoint_epoch >= num_checkpoints - 3, \
                "Best checkpoint was not preserved during cleanup"
    
    @given(
        stage=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
    )
    @settings(max_examples=100, deadline=None)
    def test_cleanup_only_affects_same_stage_and_type(self, stage):
        """
        Property test: Cleanup only affects checkpoints of the same stage and type.
        
        For any checkpoint cleanup, only checkpoints matching the stage and type
        should be deleted, leaving others untouched.
        """
        other_stage = stage + "_other"
        
        # Save checkpoints for two different stages
        for i in range(5):
            state = {
                'model': OrderedDict({'layer.weight': torch.randn(10, 10)}),
                'optimizer': {'param_groups': [{'lr': 0.001}]},
                'epoch': i,
                'step': i * 100,
                'loss': float(i),
                'best_smape': 10.0,
            }
            
            # Save for main stage
            self.manager.save_checkpoint(
                state=state,
                stage=stage,
                checkpoint_type='quick'
            )
            
            # Save for other stage
            self.manager.save_checkpoint(
                state=state,
                stage=other_stage,
                checkpoint_type='quick'
            )
        
        # Count checkpoints for each stage
        main_checkpoints = list(self.checkpoint_dir.glob(f"{stage}_quick_*.pt"))
        other_checkpoints = list(self.checkpoint_dir.glob(f"{other_stage}_quick_*.pt"))
        
        # Both should have been cleaned up independently
        assert len(main_checkpoints) <= 4
        assert len(other_checkpoints) <= 4
        
        # Both should still have checkpoints
        assert len(main_checkpoints) >= 3
        assert len(other_checkpoints) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
