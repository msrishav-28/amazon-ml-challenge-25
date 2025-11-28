"""
PyTorch Dataset and DataLoader for Amazon ML Price Prediction.

This module provides efficient data loading for multimodal neural network training,
handling text tokenization, image loading, and tabular feature processing.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Optional, Callable
from transformers import AutoTokenizer
import torchvision.transforms as transforms

from config import config


class AmazonMLDataset(Dataset):
    """
    PyTorch Dataset for multimodal Amazon ML data.
    
    Handles:
    - Text tokenization with padding/truncation
    - Image loading with transformations and augmentation
    - Tabular feature processing with NaN filling
    - Fallback for missing/corrupted images
    """
    
    def __init__(
        self,
        raw_df: pd.DataFrame,
        features_df: pd.DataFrame,
        image_dir: Path,
        tokenizer: AutoTokenizer,
        mode: str = 'train',
        config_obj=None
    ):
        """
        Initialize the dataset.
        
        Args:
            raw_df: DataFrame with sample_id, catalog_content, image_link, price (optional)
            features_df: DataFrame with engineered features
            image_dir: Directory containing product images
            tokenizer: Hugging Face tokenizer for text encoding
            mode: 'train', 'val', or 'test' - affects augmentation
            config_obj: Configuration object (defaults to global config)
        """
        self.config = config_obj if config_obj is not None else config
        self.mode = mode
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        
        # Merge raw data with features on sample_id
        self.data = raw_df.merge(features_df, on='sample_id', how='left')
        
        # Verify merge was successful
        if len(self.data) != len(raw_df):
            raise ValueError(
                f"Data merge failed: expected {len(raw_df)} rows, got {len(self.data)}"
            )
        
        # Get feature columns (exclude metadata columns)
        metadata_cols = {'sample_id', 'catalog_content', 'image_link', 'price', 'potential_brand'}
        self.feature_cols = [col for col in features_df.columns if col not in metadata_cols]
        
        # Setup image transformations
        self.train_transform = transforms.Compose([
            transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.IMAGE_MEAN, std=self.config.IMAGE_STD)
        ])
        
        self.eval_transform = transforms.Compose([
            transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.IMAGE_MEAN, std=self.config.IMAGE_STD)
        ])
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def _load_image(self, sample_id: str) -> Image.Image:
        """
        Load image from disk with fallback for missing/corrupted images.
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            PIL Image (RGB mode)
        """
        image_path = self.image_dir / f"{sample_id}.jpg"
        
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except (FileNotFoundError, OSError, IOError) as e:
            # Create blank gray placeholder image
            blank_image = Image.new('RGB', (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), color=(128, 128, 128))
            return blank_image
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - input_ids: Tokenized text (seq_len,)
                - attention_mask: Attention mask (seq_len,)
                - image: Transformed image (3, H, W)
                - tabular: Tabular features (num_features,)
                - target: Log-space price (scalar) - only if price available
                - sample_id: Sample identifier (string)
        """
        row = self.data.iloc[idx]
        sample_id = row['sample_id']
        
        # Tokenize text
        text = str(row['catalog_content']) if pd.notna(row['catalog_content']) else ""
        encoding = self.tokenizer(
            text,
            max_length=self.config.MAX_TEXT_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)  # (seq_len,)
        attention_mask = encoding['attention_mask'].squeeze(0)  # (seq_len,)
        
        # Load and transform image
        image = self._load_image(sample_id)
        if self.mode == 'train':
            image_tensor = self.train_transform(image)
        else:
            image_tensor = self.eval_transform(image)
        
        # Extract tabular features and fill NaN with 0
        tabular_features = row[self.feature_cols].values.astype(np.float32)
        tabular_features = np.nan_to_num(tabular_features, nan=0.0)
        tabular_tensor = torch.from_numpy(tabular_features)
        
        # Prepare output dictionary
        output = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image': image_tensor,
            'tabular': tabular_tensor,
            'sample_id': sample_id
        }
        
        # Add target if available (training/validation)
        if 'price' in row and pd.notna(row['price']):
            # Convert to log space: log(1 + price)
            target = np.log1p(float(row['price']))
            output['target'] = torch.tensor(target, dtype=torch.float32)
        
        return output


def get_dataloader(
    dataset: AmazonMLDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create DataLoader with optimized settings.
    
    Args:
        dataset: AmazonMLDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
