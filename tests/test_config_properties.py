"""
Property-based tests for configuration completeness.

These tests verify that the configuration module provides all required
paths, hyperparameters, and feature engineering parameters as specified
in the design document.
"""

import pytest
from pathlib import Path
from hypothesis import given, strategies as st
import sys
import os

# Add parent directory to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config, config


class TestConfigurationProperties:
    """Property-based tests for configuration completeness."""
    
    def test_property_38_configuration_path_completeness(self):
        """
        Feature: amazon-ml-price-prediction, Property 38: Configuration path completeness
        
        For any initialized config object, it should provide valid Path objects for:
        DATA_DIR, IMAGE_DIR, MODEL_DIR, CHECKPOINT_DIR, PRED_DIR.
        
        Validates: Requirements 10.2
        """
        # Test that all required path attributes exist
        required_paths = [
            'DATA_DIR',
            'IMAGE_DIR',
            'MODEL_DIR',
            'CHECKPOINT_DIR',
            'PRED_DIR',
        ]
        
        for path_name in required_paths:
            # Check attribute exists
            assert hasattr(config, path_name), f"Config missing required path: {path_name}"
            
            # Get the path value
            path_value = getattr(config, path_name)
            
            # Check it's a Path object
            assert isinstance(path_value, Path), \
                f"{path_name} should be a Path object, got {type(path_value)}"
            
            # Check it's not None or empty
            assert path_value is not None, f"{path_name} should not be None"
            assert str(path_value), f"{path_name} should not be empty"
    
    @given(st.integers(min_value=0, max_value=100))
    def test_property_38_paths_are_valid_across_instances(self, seed):
        """
        Property test: Configuration paths should be consistent across multiple accesses.
        
        For any number of times we access the config, the paths should remain valid
        Path objects with the same values.
        """
        # Access paths multiple times
        data_dir_1 = config.DATA_DIR
        data_dir_2 = config.DATA_DIR
        
        # Should be the same
        assert data_dir_1 == data_dir_2
        assert isinstance(data_dir_1, Path)
        assert isinstance(data_dir_2, Path)
    
    def test_property_39_configuration_hyperparameter_completeness(self):
        """
        Feature: amazon-ml-price-prediction, Property 39: Configuration hyperparameter completeness
        
        For any initialized config object, it should provide all required hyperparameters:
        LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, LORA_R, LORA_ALPHA.
        
        Validates: Requirements 10.3, 10.4
        """
        # Test neural network hyperparameters
        required_hyperparams = {
            'LEARNING_RATE': (float, int),
            'BATCH_SIZE': (int,),
            'NUM_EPOCHS': (int,),
            'LORA_R': (int,),
            'LORA_ALPHA': (int, float),
        }
        
        for param_name, expected_types in required_hyperparams.items():
            # Check attribute exists
            assert hasattr(config, param_name), \
                f"Config missing required hyperparameter: {param_name}"
            
            # Get the value
            param_value = getattr(config, param_name)
            
            # Check it's not None
            assert param_value is not None, \
                f"{param_name} should not be None"
            
            # Check it's the right type
            assert isinstance(param_value, expected_types), \
                f"{param_name} should be one of {expected_types}, got {type(param_value)}"
            
            # Check it's a reasonable value (positive)
            assert param_value > 0, \
                f"{param_name} should be positive, got {param_value}"
    
    @given(st.integers(min_value=1, max_value=10))
    def test_property_39_hyperparameters_are_numeric_and_positive(self, multiplier):
        """
        Property test: All hyperparameters should be positive numbers.
        
        For any hyperparameter, it should be a positive numeric value.
        """
        # Test that learning rate is positive
        assert config.LEARNING_RATE > 0
        
        # Test that batch size is positive integer
        assert config.BATCH_SIZE > 0
        assert isinstance(config.BATCH_SIZE, int)
        
        # Test that epochs is positive integer
        assert config.NUM_EPOCHS > 0
        assert isinstance(config.NUM_EPOCHS, int)
        
        # Test LoRA parameters are positive
        assert config.LORA_R > 0
        assert config.LORA_ALPHA > 0
    
    def test_property_40_feature_engineering_config_completeness(self):
        """
        Feature: amazon-ml-price-prediction, Property 40: Feature engineering config completeness
        
        For any initialized config object, it should provide IPQ_PATTERNS (list of regex),
        UNIT_CONVERSIONS (dict), and TFIDF_MAX_FEATURES (int).
        
        Validates: Requirements 10.5
        """
        # Test IPQ_PATTERNS
        assert hasattr(config, 'IPQ_PATTERNS'), \
            "Config missing IPQ_PATTERNS"
        assert isinstance(config.IPQ_PATTERNS, list), \
            f"IPQ_PATTERNS should be a list, got {type(config.IPQ_PATTERNS)}"
        assert len(config.IPQ_PATTERNS) > 0, \
            "IPQ_PATTERNS should not be empty"
        
        # Check all patterns are strings (regex patterns)
        for i, pattern in enumerate(config.IPQ_PATTERNS):
            assert isinstance(pattern, str), \
                f"IPQ_PATTERNS[{i}] should be a string, got {type(pattern)}"
            assert len(pattern) > 0, \
                f"IPQ_PATTERNS[{i}] should not be empty"
        
        # Test UNIT_CONVERSIONS
        assert hasattr(config, 'UNIT_CONVERSIONS'), \
            "Config missing UNIT_CONVERSIONS"
        assert isinstance(config.UNIT_CONVERSIONS, dict), \
            f"UNIT_CONVERSIONS should be a dict, got {type(config.UNIT_CONVERSIONS)}"
        assert len(config.UNIT_CONVERSIONS) > 0, \
            "UNIT_CONVERSIONS should not be empty"
        
        # Check structure of UNIT_CONVERSIONS
        for unit, conversion in config.UNIT_CONVERSIONS.items():
            assert isinstance(unit, str), \
                f"Unit key should be string, got {type(unit)}"
            assert isinstance(conversion, tuple), \
                f"Conversion value should be tuple, got {type(conversion)}"
            assert len(conversion) == 2, \
                f"Conversion tuple should have 2 elements, got {len(conversion)}"
            
            unit_type, conversion_factor = conversion
            assert isinstance(unit_type, str), \
                f"Unit type should be string, got {type(unit_type)}"
            assert isinstance(conversion_factor, (int, float)), \
                f"Conversion factor should be numeric, got {type(conversion_factor)}"
            assert conversion_factor > 0, \
                f"Conversion factor should be positive, got {conversion_factor}"
        
        # Test TFIDF_MAX_FEATURES
        assert hasattr(config, 'TFIDF_MAX_FEATURES'), \
            "Config missing TFIDF_MAX_FEATURES"
        assert isinstance(config.TFIDF_MAX_FEATURES, int), \
            f"TFIDF_MAX_FEATURES should be int, got {type(config.TFIDF_MAX_FEATURES)}"
        assert config.TFIDF_MAX_FEATURES > 0, \
            f"TFIDF_MAX_FEATURES should be positive, got {config.TFIDF_MAX_FEATURES}"
    
    @given(st.integers(min_value=0, max_value=100))
    def test_property_40_unit_conversions_are_consistent(self, seed):
        """
        Property test: Unit conversions should have consistent structure.
        
        For any unit in UNIT_CONVERSIONS, it should map to a tuple of
        (unit_type: str, conversion_factor: positive number).
        """
        # Pick a random unit from the conversions
        if len(config.UNIT_CONVERSIONS) > 0:
            units = list(config.UNIT_CONVERSIONS.keys())
            # Use seed to pick a unit deterministically
            unit_idx = seed % len(units)
            unit = units[unit_idx]
            
            conversion = config.UNIT_CONVERSIONS[unit]
            
            # Verify structure
            assert isinstance(conversion, tuple)
            assert len(conversion) == 2
            
            unit_type, factor = conversion
            assert isinstance(unit_type, str)
            assert isinstance(factor, (int, float))
            assert factor > 0
    
    @given(st.integers(min_value=0, max_value=100))
    def test_property_40_ipq_patterns_are_valid_regex(self, pattern_idx):
        """
        Property test: All IPQ patterns should be valid regex patterns.
        
        For any pattern in IPQ_PATTERNS, it should be a valid regex that can be compiled.
        """
        import re
        
        if len(config.IPQ_PATTERNS) > 0:
            # Pick a pattern using the index
            idx = pattern_idx % len(config.IPQ_PATTERNS)
            pattern = config.IPQ_PATTERNS[idx]
            
            # Should be able to compile without error
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                assert compiled is not None
            except re.error as e:
                pytest.fail(f"Pattern '{pattern}' is not valid regex: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
