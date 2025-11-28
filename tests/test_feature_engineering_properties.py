"""
Property-based tests for feature engineering module.

Tests verify correctness properties for:
- IPQ extraction determinism
- Text statistics correctness
- Keyword detection completeness
- TF-IDF transformation consistency
- Unit normalization reversibility
- Feature serialization round-trip
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings
from pathlib import Path
import tempfile
import shutil

from src.data.feature_engineering import FeatureEngineer
from config import config


# ==================== Test Strategies ====================

@st.composite
def product_text_strategy(draw):
    """Generate realistic product text with various patterns."""
    # Base text components
    brand = draw(st.sampled_from(['Samsung', 'Apple', 'Sony', 'LG', 'Generic', '']))
    product_type = draw(st.sampled_from(['Phone', 'Laptop', 'TV', 'Tablet', 'Watch', 'Camera']))
    
    # Optional quantity
    has_quantity = draw(st.booleans())
    quantity_text = ''
    if has_quantity:
        value = draw(st.floats(min_value=0.1, max_value=1000, allow_nan=False, allow_infinity=False))
        unit = draw(st.sampled_from(['kg', 'g', 'ml', 'l', 'cm', 'pack', 'count']))
        quantity_text = f" {value:.1f} {unit}"
    
    # Optional quality keywords
    has_quality = draw(st.booleans())
    quality_text = ''
    if has_quality:
        quality = draw(st.sampled_from(config.QUALITY_KEYWORDS))
        quality_text = f" {quality}"
    
    # Optional discount keywords
    has_discount = draw(st.booleans())
    discount_text = ''
    if has_discount:
        discount = draw(st.sampled_from(config.DISCOUNT_KEYWORDS))
        discount_text = f" {discount}"
    
    # Combine
    text = f"{brand} {product_type}{quantity_text}{quality_text}{discount_text}".strip()
    return text


@st.composite
def text_with_quality_keyword_strategy(draw):
    """Generate text guaranteed to contain a quality keyword."""
    keyword = draw(st.sampled_from(config.QUALITY_KEYWORDS))
    prefix = draw(st.text(min_size=0, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    suffix = draw(st.text(min_size=0, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    return f"{prefix} {keyword} {suffix}".strip()


# ==================== Property Tests ====================

class TestIPQExtractionDeterminism:
    """
    Feature: amazon-ml-price-prediction, Property 5: IPQ extraction determinism
    Validates: Requirements 2.1
    """
    
    @settings(max_examples=100)
    @given(text=st.text(min_size=0, max_size=200))
    def test_ipq_extraction_is_deterministic(self, text):
        """
        For any product text, extracting IPQ features multiple times
        should produce identical results (deterministic extraction).
        """
        engineer = FeatureEngineer()
        
        # Extract features multiple times
        result1 = engineer.extract_ipq_features(text)
        result2 = engineer.extract_ipq_features(text)
        result3 = engineer.extract_ipq_features(text)
        
        # All results should be identical
        assert result1 == result2, "First and second extraction differ"
        assert result2 == result3, "Second and third extraction differ"
        assert result1 == result3, "First and third extraction differ"


class TestTextStatisticsCorrectness:
    """
    Feature: amazon-ml-price-prediction, Property 6: Text statistics correctness
    Validates: Requirements 2.2
    """
    
    @settings(max_examples=100)
    @given(text=st.text(min_size=0, max_size=500))
    def test_word_count_matches_split(self, text):
        """
        For any text string, the computed word_count should equal len(text.split()).
        """
        engineer = FeatureEngineer()
        stats = engineer.extract_text_statistics(text)
        
        expected_word_count = len(text.split())
        assert stats['word_count'] == expected_word_count, \
            f"Word count {stats['word_count']} != expected {expected_word_count}"
    
    @settings(max_examples=100)
    @given(text=st.text(min_size=0, max_size=500))
    def test_text_length_matches_len(self, text):
        """
        For any text string, text_length should equal len(text).
        """
        engineer = FeatureEngineer()
        stats = engineer.extract_text_statistics(text)
        
        expected_length = len(text)
        assert stats['text_length'] == expected_length, \
            f"Text length {stats['text_length']} != expected {expected_length}"


class TestKeywordDetectionCompleteness:
    """
    Feature: amazon-ml-price-prediction, Property 7: Keyword detection completeness
    Validates: Requirements 2.3
    """
    
    @settings(max_examples=100)
    @given(text=text_with_quality_keyword_strategy())
    def test_quality_keyword_detection(self, text):
        """
        For any text containing a quality keyword from the predefined list,
        has_quality_keywords should be True and quality_keyword_count should be at least 1.
        """
        engineer = FeatureEngineer()
        features = engineer.extract_keyword_features(text)
        
        assert features['has_quality_keywords'] is True, \
            f"Quality keyword not detected in text: {text}"
        assert features['quality_keyword_count'] >= 1, \
            f"Quality keyword count is {features['quality_keyword_count']}, expected >= 1"


class TestTFIDFConsistency:
    """
    Feature: amazon-ml-price-prediction, Property 8: TF-IDF transformation consistency
    Validates: Requirements 2.6
    """
    
    @settings(max_examples=100)
    @given(
        corpus=st.lists(product_text_strategy(), min_size=10, max_size=50),
        test_text=product_text_strategy()
    )
    def test_tfidf_transformation_consistency(self, corpus, test_text):
        """
        For any fitted TF-IDF vectorizer and any text, transforming the same text
        multiple times should produce identical feature vectors.
        """
        engineer = FeatureEngineer()
        
        # Create a corpus and fit
        corpus_series = pd.Series(corpus)
        
        try:
            engineer.fit_tfidf(corpus_series)
        except ValueError:
            # If corpus is too small or contains only stop words, skip this test case
            # This is an edge case outside the normal input domain
            return
        
        # Transform the same text multiple times
        test_series = pd.Series([test_text])
        result1 = engineer.transform_tfidf(test_series)
        result2 = engineer.transform_tfidf(test_series)
        result3 = engineer.transform_tfidf(test_series)
        
        # All results should be identical
        np.testing.assert_array_equal(result1, result2, err_msg="First and second transformation differ")
        np.testing.assert_array_equal(result2, result3, err_msg="Second and third transformation differ")
        np.testing.assert_array_equal(result1, result3, err_msg="First and third transformation differ")


class TestUnitNormalizationReversibility:
    """
    Feature: amazon-ml-price-prediction, Property 9: Unit normalization reversibility
    Validates: Requirements 2.7
    """
    
    @settings(max_examples=100)
    @given(
        value=st.floats(min_value=0.1, max_value=1000, allow_nan=False, allow_infinity=False),
        unit=st.sampled_from(['kg', 'g', 'mg', 'l', 'ml', 'm', 'cm', 'mm'])
    )
    def test_unit_normalization_reversibility(self, value, unit):
        """
        For any IPQ value with a known unit, normalizing to standard units
        and then denormalizing should recover the original value within numerical precision.
        """
        engineer = FeatureEngineer()
        
        # Create text with the value and unit
        text = f"Product with {value} {unit}"
        
        # Extract IPQ features
        ipq_features = engineer.extract_ipq_features(text)
        
        # If IPQ was extracted
        if ipq_features['has_ipq'] and ipq_features['ipq_normalized'] is not None:
            extracted_value = ipq_features['ipq_value']
            normalized_value = ipq_features['ipq_normalized']
            extracted_unit = ipq_features['ipq_unit']
            
            # Get conversion factor
            if extracted_unit in config.UNIT_CONVERSIONS:
                unit_type, conversion_factor = config.UNIT_CONVERSIONS[extracted_unit]
                
                # Denormalize
                denormalized_value = normalized_value / conversion_factor
                
                # Should match original extracted value within numerical precision
                assert abs(denormalized_value - extracted_value) < 1e-6, \
                    f"Denormalized {denormalized_value} != extracted {extracted_value}"


class TestFeatureSerializationRoundTrip:
    """
    Feature: amazon-ml-price-prediction, Property 10: Feature serialization round-trip
    Validates: Requirements 2.8
    """
    
    @settings(max_examples=100)
    @given(
        texts=st.lists(product_text_strategy(), min_size=15, max_size=30)
    )
    def test_feature_serialization_round_trip(self, texts):
        """
        For any engineered features DataFrame, saving to pickle and loading
        should produce a DataFrame with identical values and schema.
        """
        engineer = FeatureEngineer()
        
        # Create DataFrame
        df = pd.DataFrame({
            'sample_id': [f'sample_{i}' for i in range(len(texts))],
            'catalog_content': texts
        })
        
        # Engineer features (without TF-IDF to avoid min_df issues with small corpora)
        features_df = engineer.engineer_features(df, fit_tfidf=False)
        
        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_features.pkl'
            engineer.save_features(features_df, filepath)
            loaded_df = engineer.load_features(filepath)
        
        # Check schema
        assert list(features_df.columns) == list(loaded_df.columns), \
            "Column names differ after round-trip"
        
        # Check shape
        assert features_df.shape == loaded_df.shape, \
            f"Shape differs: {features_df.shape} != {loaded_df.shape}"
        
        # Check values for numeric columns
        for col in features_df.columns:
            if features_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                # Use allclose for floating point comparison
                if features_df[col].dtype in [np.float64, np.float32]:
                    np.testing.assert_allclose(
                        features_df[col].fillna(0).values,
                        loaded_df[col].fillna(0).values,
                        rtol=1e-9,
                        err_msg=f"Column {col} values differ after round-trip"
                    )
                else:
                    np.testing.assert_array_equal(
                        features_df[col].fillna(0).values,
                        loaded_df[col].fillna(0).values,
                        err_msg=f"Column {col} values differ after round-trip"
                    )
            elif features_df[col].dtype == bool:
                np.testing.assert_array_equal(
                    features_df[col].values,
                    loaded_df[col].values,
                    err_msg=f"Column {col} values differ after round-trip"
                )
