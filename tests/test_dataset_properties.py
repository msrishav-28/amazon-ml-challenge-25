"""
Property-based tests for PyTorch dataset module.

Tests verify correctness properties for:
- Data merge correctness
- Tokenization length constraint
- Image transformation shape
- Data augmentation conditional application
- NaN filling completeness
- Image loading fallback
"""

import pytest
import pandas as pd
import numpy as np
import torch
from hypothesis import given, strategies as st, settings, assume
from pathlib import Path
import tempfile
from PIL import Image
from transformers import AutoTokenizer
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import AmazonMLDataset, get_dataloader
from config import config


# ==================== Test Strategies ====================

@st.composite
def sample_id_strategy(draw):
    """Generate valid sample IDs."""
    return f"sample_{draw(st.integers(min_value=0, max_value=999999))}"


@st.composite
def raw_dataframe_strategy(draw, include_price=True):
    """Generate a raw DataFrame with sample data."""
    n_samples = draw(st.integers(min_value=5, max_value=20))
    
    sample_ids = [f"sample_{i}" for i in range(n_samples)]
    texts = [draw(st.text(min_size=10, max_size=100)) for _ in range(n_samples)]
    
    data = {
        'sample_id': sample_ids,
        'catalog_content': texts,
        'image_link': [f"http://example.com/image_{i}.jpg" for i in range(n_samples)]
    }
    
    if include_price:
        prices = [draw(st.floats(min_value=10.0, max_value=10000.0)) for _ in range(n_samples)]
        data['price'] = prices
    
    return pd.DataFrame(data)


@st.composite
def features_dataframe_strategy(draw, sample_ids):
    """Generate a features DataFrame matching the sample IDs."""
    n_features = draw(st.integers(min_value=5, max_value=20))
    
    data = {'sample_id': sample_ids}
    
    # Add random feature columns
    for i in range(n_features):
        feature_values = [
            draw(st.one_of(
                st.floats(min_value=-10.0, max_value=10.0, allow_nan=True, allow_infinity=False),
                st.just(np.nan)
            ))
            for _ in range(len(sample_ids))
        ]
        data[f'feature_{i}'] = feature_values
    
    return pd.DataFrame(data)


# ==================== Property Tests ====================

class TestDatasetProperties:
    """Property-based tests for PyTorch dataset."""
    
    @given(st.data())
    @settings(max_examples=50, deadline=None)
    def test_property_28_data_merge_correctness(self, data):
        """
        **Feature: amazon-ml-price-prediction, Property 28: Data merge correctness**
        **Validates: Requirements 8.1**
        
        For any raw DataFrame and features DataFrame, merging on sample_id should 
        preserve all rows from the raw DataFrame and add all feature columns.
        """
        # Generate raw DataFrame
        raw_df = data.draw(raw_dataframe_strategy(include_price=True))
        sample_ids = raw_df['sample_id'].tolist()
        
        # Generate features DataFrame with matching sample IDs
        features_df = data.draw(features_dataframe_strategy(sample_ids))
        
        # Create temporary directory for images
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = Path(tmpdir)
            
            # Create dummy images
            for sample_id in sample_ids:
                img = Image.new('RGB', (100, 100), color='red')
                img.save(image_dir / f"{sample_id}.jpg")
            
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
            
            # Create dataset
            dataset = AmazonMLDataset(
                raw_df=raw_df,
                features_df=features_df,
                image_dir=image_dir,
                tokenizer=tokenizer,
                mode='train',
                config_obj=config
            )
            
            # Property: Dataset should have same number of rows as raw_df
            assert len(dataset) == len(raw_df), \
                f"Dataset length {len(dataset)} != raw_df length {len(raw_df)}"
            
            # Property: All feature columns should be present
            feature_cols = [col for col in features_df.columns if col != 'sample_id']
            for feature_col in feature_cols:
                assert feature_col in dataset.feature_cols, \
                    f"Feature column {feature_col} not found in dataset"
            
            # Property: All sample IDs should be preserved
            dataset_sample_ids = dataset.data['sample_id'].tolist()
            assert set(dataset_sample_ids) == set(sample_ids), \
                "Sample IDs not preserved after merge"

    
    @given(st.text(min_size=0, max_size=1000))
    @settings(max_examples=100, deadline=None)
    def test_property_29_tokenization_length_constraint(self, text):
        """
        **Feature: amazon-ml-price-prediction, Property 29: Tokenization length constraint**
        **Validates: Requirements 8.2**
        
        For any text input, after tokenization with max_length=128, the output 
        input_ids should have exactly length 128 (padded or truncated).
        """
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        
        # Create minimal raw and features DataFrames
        raw_df = pd.DataFrame({
            'sample_id': ['test_sample'],
            'catalog_content': [text],
            'image_link': ['http://example.com/test.jpg'],
            'price': [100.0]
        })
        
        features_df = pd.DataFrame({
            'sample_id': ['test_sample'],
            'feature_0': [1.0],
            'feature_1': [2.0]
        })
        
        # Create temporary directory for images
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = Path(tmpdir)
            
            # Create dummy image
            img = Image.new('RGB', (100, 100), color='blue')
            img.save(image_dir / "test_sample.jpg")
            
            # Create dataset
            dataset = AmazonMLDataset(
                raw_df=raw_df,
                features_df=features_df,
                image_dir=image_dir,
                tokenizer=tokenizer,
                mode='train',
                config_obj=config
            )
            
            # Get sample
            sample = dataset[0]
            
            # Property: input_ids should have exactly MAX_TEXT_LENGTH tokens
            assert sample['input_ids'].shape[0] == config.MAX_TEXT_LENGTH, \
                f"input_ids length {sample['input_ids'].shape[0]} != {config.MAX_TEXT_LENGTH}"
            
            # Property: attention_mask should also have exactly MAX_TEXT_LENGTH tokens
            assert sample['attention_mask'].shape[0] == config.MAX_TEXT_LENGTH, \
                f"attention_mask length {sample['attention_mask'].shape[0]} != {config.MAX_TEXT_LENGTH}"
            
            # Property: input_ids should be a 1D tensor
            assert len(sample['input_ids'].shape) == 1, \
                f"input_ids should be 1D, got shape {sample['input_ids'].shape}"
            
            # Property: attention_mask should be a 1D tensor
            assert len(sample['attention_mask'].shape) == 1, \
                f"attention_mask should be 1D, got shape {sample['attention_mask'].shape}"
