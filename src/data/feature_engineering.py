"""
Feature engineering module for Amazon ML Price Prediction.

This module extracts and transforms features from raw product data including:
- IPQ (Item Pack Quantity) features with unit normalization
- Text statistics (length, word count, character distributions)
- Keyword features (quality and discount indicators)
- Brand extraction
- TF-IDF vectorization
"""

import re
import pickle
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from config import config


class FeatureEngineer:
    """
    Orchestrates all feature extraction and transformation.
    
    This class provides methods to extract various features from product data
    and manages the TF-IDF vectorizer for text features.
    """
    
    def __init__(self):
        """Initialize the FeatureEngineer with a TF-IDF vectorizer."""
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._is_fitted = False
    
    def extract_ipq_features(self, text: str) -> Dict[str, any]:
        """
        Extract Item Pack Quantity features using regex patterns.
        
        Args:
            text: Product text to extract IPQ from
            
        Returns:
            Dictionary containing:
                - ipq_value: Extracted quantity value (float or None)
                - ipq_unit: Unit string (str or None)
                - ipq_normalized: Normalized value in standard units (float or None)
                - ipq_unit_type: Type of unit (weight/volume/length/count or None)
                - ipq_confidence: Extraction confidence [0-1]
                - has_ipq: Whether IPQ was found (bool)
        """
        if not isinstance(text, str) or not text:
            return {
                'ipq_value': None,
                'ipq_unit': None,
                'ipq_normalized': None,
                'ipq_unit_type': None,
                'ipq_confidence': 0.0,
                'has_ipq': False
            }
        
        text_lower = text.lower()
        
        # Try each pattern
        for pattern in config.IPQ_PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                groups = match.groups()
                value = float(groups[0])
                
                # Determine unit
                if len(groups) > 1:
                    unit = groups[1].lower()
                else:
                    # For patterns like "pack of X", unit is implicit
                    if 'pack' in pattern:
                        unit = 'pack'
                    elif 'count' in pattern:
                        unit = 'count'
                    elif 'piece' in pattern or 'pcs' in pattern:
                        unit = 'pieces'
                    else:
                        unit = 'count'
                
                # Normalize to standard units
                normalized_value = None
                unit_type = None
                if unit in config.UNIT_CONVERSIONS:
                    unit_type, conversion_factor = config.UNIT_CONVERSIONS[unit]
                    normalized_value = value * conversion_factor
                
                # Confidence based on pattern specificity
                confidence = 0.9 if unit in config.UNIT_CONVERSIONS else 0.7
                
                return {
                    'ipq_value': value,
                    'ipq_unit': unit,
                    'ipq_normalized': normalized_value,
                    'ipq_unit_type': unit_type,
                    'ipq_confidence': confidence,
                    'has_ipq': True
                }
        
        # No match found
        return {
            'ipq_value': None,
            'ipq_unit': None,
            'ipq_normalized': None,
            'ipq_unit_type': None,
            'ipq_confidence': 0.0,
            'has_ipq': False
        }
    
    def extract_text_statistics(self, text: str) -> Dict[str, float]:
        """
        Compute text statistics including length, word count, and character distributions.
        
        Args:
            text: Product text to analyze
            
        Returns:
            Dictionary containing:
                - text_length: Character count
                - word_count: Word count
                - digit_count: Number of digits
                - special_char_count: Number of special characters
                - uppercase_ratio: Ratio of uppercase letters
                - avg_word_length: Average word length
        """
        if not isinstance(text, str) or not text:
            return {
                'text_length': 0,
                'word_count': 0,
                'digit_count': 0,
                'special_char_count': 0,
                'uppercase_ratio': 0.0,
                'avg_word_length': 0.0
            }
        
        text_length = len(text)
        words = text.split()
        word_count = len(words)
        
        # Count digits
        digit_count = sum(1 for c in text if c.isdigit())
        
        # Count special characters (not alphanumeric or whitespace)
        special_char_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        
        # Calculate uppercase ratio
        alpha_chars = [c for c in text if c.isalpha()]
        uppercase_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) if alpha_chars else 0.0
        
        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0.0
        
        return {
            'text_length': text_length,
            'word_count': word_count,
            'digit_count': digit_count,
            'special_char_count': special_char_count,
            'uppercase_ratio': uppercase_ratio,
            'avg_word_length': avg_word_length
        }
    
    def extract_keyword_features(self, text: str) -> Dict[str, any]:
        """
        Identify quality and discount keywords.
        
        Args:
            text: Product text to search for keywords
            
        Returns:
            Dictionary containing:
                - has_quality_keywords: Whether quality keywords found (bool)
                - quality_keyword_count: Number of quality keywords (int)
                - has_discount_keywords: Whether discount keywords found (bool)
                - discount_keyword_count: Number of discount keywords (int)
        """
        if not isinstance(text, str) or not text:
            return {
                'has_quality_keywords': False,
                'quality_keyword_count': 0,
                'has_discount_keywords': False,
                'discount_keyword_count': 0
            }
        
        text_lower = text.lower()
        
        # Count quality keywords
        quality_count = sum(1 for keyword in config.QUALITY_KEYWORDS if keyword in text_lower)
        
        # Count discount keywords
        discount_count = sum(1 for keyword in config.DISCOUNT_KEYWORDS if keyword in text_lower)
        
        return {
            'has_quality_keywords': quality_count > 0,
            'quality_keyword_count': quality_count,
            'has_discount_keywords': discount_count > 0,
            'discount_keyword_count': discount_count
        }
    
    def extract_brand_features(self, text: str) -> Dict[str, any]:
        """
        Extract potential brand names from text.
        
        Uses heuristics to identify brand names:
        - Capitalized words at the beginning
        - Words in all caps
        - Common brand patterns
        
        Args:
            text: Product text to extract brand from
            
        Returns:
            Dictionary containing:
                - has_brand: Whether brand detected (bool)
                - brand_position: Normalized position in text [0-1] (float or None)
                - potential_brand: Extracted brand name (str or None)
        """
        if not isinstance(text, str) or not text:
            return {
                'has_brand': False,
                'brand_position': None,
                'potential_brand': None
            }
        
        words = text.split()
        if not words:
            return {
                'has_brand': False,
                'brand_position': None,
                'potential_brand': None
            }
        
        # Look for capitalized words at the beginning (common brand pattern)
        potential_brand = None
        brand_position = None
        
        # Check first few words for capitalized patterns
        for i, word in enumerate(words[:5]):
            # Remove special characters for checking
            clean_word = re.sub(r'[^a-zA-Z]', '', word)
            if clean_word and len(clean_word) > 1:
                # Check if word is capitalized or all caps
                if clean_word[0].isupper() and (clean_word.isupper() or clean_word[1:].islower()):
                    potential_brand = word
                    brand_position = i / len(words)
                    break
        
        has_brand = potential_brand is not None
        
        return {
            'has_brand': has_brand,
            'brand_position': brand_position,
            'potential_brand': potential_brand
        }
    
    def fit_tfidf(self, texts: pd.Series) -> None:
        """
        Fit TF-IDF vectorizer on training texts.
        
        Args:
            texts: Series of product texts to fit on
        """
        # Clean texts - replace NaN with empty string
        texts_clean = texts.fillna('').astype(str)
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            lowercase=True,
            strip_accents='unicode',
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.tfidf_vectorizer.fit(texts_clean)
        self._is_fitted = True
    
    def transform_tfidf(self, texts: pd.Series) -> np.ndarray:
        """
        Transform texts to TF-IDF features using fitted vectorizer.
        
        Args:
            texts: Series of product texts to transform
            
        Returns:
            TF-IDF feature matrix (n_samples, max_features)
            
        Raises:
            ValueError: If vectorizer not fitted yet
        """
        if not self._is_fitted or self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf() first.")
        
        # Clean texts - replace NaN with empty string
        texts_clean = texts.fillna('').astype(str)
        
        return self.tfidf_vectorizer.transform(texts_clean).toarray()
    
    def engineer_features(self, df: pd.DataFrame, fit_tfidf: bool = False) -> pd.DataFrame:
        """
        Engineer all features from DataFrame.
        
        This orchestrates all feature extraction methods and combines them
        into a single feature DataFrame.
        
        Args:
            df: DataFrame with 'catalog_content' column
            fit_tfidf: Whether to fit TF-IDF vectorizer (True for training data)
            
        Returns:
            DataFrame with all engineered features (~180 features)
        """
        if 'catalog_content' not in df.columns:
            raise ValueError("DataFrame must contain 'catalog_content' column")
        
        features_list = []
        
        # Extract features for each row
        for idx, row in df.iterrows():
            text = row['catalog_content']
            
            # Extract all feature types
            ipq_features = self.extract_ipq_features(text)
            text_stats = self.extract_text_statistics(text)
            keyword_features = self.extract_keyword_features(text)
            brand_features = self.extract_brand_features(text)
            
            # Combine all features
            combined_features = {
                'sample_id': row.get('sample_id', idx),
                **ipq_features,
                **text_stats,
                **keyword_features,
                **brand_features
            }
            
            features_list.append(combined_features)
        
        # Create features DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Add TF-IDF features
        if fit_tfidf:
            self.fit_tfidf(df['catalog_content'])
        
        if self._is_fitted:
            tfidf_features = self.transform_tfidf(df['catalog_content'])
            tfidf_df = pd.DataFrame(
                tfidf_features,
                columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
            )
            tfidf_df['sample_id'] = features_df['sample_id'].values
            features_df = features_df.merge(tfidf_df, on='sample_id', how='left')
        
        return features_df
    
    def save_features(self, features_df: pd.DataFrame, filepath: Path) -> None:
        """
        Save features to disk in pickle format.
        
        Args:
            features_df: DataFrame with engineered features
            filepath: Path to save pickle file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(features_df, f)
    
    def load_features(self, filepath: Path) -> pd.DataFrame:
        """
        Load features from disk.
        
        Args:
            filepath: Path to pickle file
            
        Returns:
            DataFrame with engineered features
        """
        filepath = Path(filepath)
        
        with open(filepath, 'rb') as f:
            features_df = pickle.load(f)
        
        return features_df
    
    def save_vectorizer(self, filepath: Path) -> None:
        """
        Save the fitted TF-IDF vectorizer to disk.
        
        Args:
            filepath: Path to save pickle file
        """
        if not self._is_fitted or self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Nothing to save.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
    
    def load_vectorizer(self, filepath: Path) -> None:
        """
        Load a fitted TF-IDF vectorizer from disk.
        
        Args:
            filepath: Path to pickle file
        """
        filepath = Path(filepath)
        
        with open(filepath, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        
        self._is_fitted = True
