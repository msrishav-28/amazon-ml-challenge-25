"""
Demo script to showcase the feature engineering module.

This demonstrates:
- Extracting IPQ features
- Computing text statistics
- Detecting keywords
- Extracting brand features
- TF-IDF vectorization
- Full feature engineering pipeline
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data.feature_engineering import FeatureEngineer

# Sample product data
sample_products = [
    "Samsung Galaxy S23 Ultra 256GB Premium Smartphone with 5000mAh Battery",
    "Apple iPhone 14 Pro 128GB - Discount Sale - High Quality Display",
    "Sony WH-1000XM5 Wireless Headphones - Premium Noise Cancelling",
    "LG 55 inch 4K Smart TV - Best Deal - 120Hz Refresh Rate",
    "Generic USB Cable 2m - Affordable and Durable",
    "Philips Air Fryer 4.1l - Healthy Cooking - Top Rated",
    "Nike Running Shoes Size 10 - Professional Athletic Footwear",
    "Logitech MX Master 3 Mouse - Ergonomic Design - Excellent Performance",
    "Canon EOS R6 Camera Body - Professional Photography - Certified Original",
    "Bose QuietComfort Earbuds - Superior Sound Quality - Luxury Audio"
]

def main():
    print("=" * 80)
    print("Feature Engineering Demo")
    print("=" * 80)
    
    # Create engineer
    engineer = FeatureEngineer()
    
    # Demo 1: IPQ Extraction
    print("\n1. IPQ (Item Pack Quantity) Extraction:")
    print("-" * 80)
    test_texts = [
        "Product with 500ml volume",
        "Pack of 12 items",
        "2.5 kg weight",
        "No quantity mentioned"
    ]
    for text in test_texts:
        ipq = engineer.extract_ipq_features(text)
        print(f"Text: {text}")
        print(f"  → Value: {ipq['ipq_value']}, Unit: {ipq['ipq_unit']}, "
              f"Normalized: {ipq['ipq_normalized']}, Has IPQ: {ipq['has_ipq']}")
    
    # Demo 2: Text Statistics
    print("\n2. Text Statistics:")
    print("-" * 80)
    sample_text = sample_products[0]
    stats = engineer.extract_text_statistics(sample_text)
    print(f"Text: {sample_text}")
    print(f"  → Length: {stats['text_length']}, Words: {stats['word_count']}")
    print(f"  → Digits: {stats['digit_count']}, Special chars: {stats['special_char_count']}")
    print(f"  → Uppercase ratio: {stats['uppercase_ratio']:.2f}, Avg word length: {stats['avg_word_length']:.2f}")
    
    # Demo 3: Keyword Detection
    print("\n3. Keyword Detection:")
    print("-" * 80)
    for text in sample_products[:3]:
        keywords = engineer.extract_keyword_features(text)
        print(f"Text: {text[:60]}...")
        print(f"  → Quality keywords: {keywords['has_quality_keywords']} "
              f"(count: {keywords['quality_keyword_count']})")
        print(f"  → Discount keywords: {keywords['has_discount_keywords']} "
              f"(count: {keywords['discount_keyword_count']})")
    
    # Demo 4: Brand Extraction
    print("\n4. Brand Extraction:")
    print("-" * 80)
    for text in sample_products[:5]:
        brand = engineer.extract_brand_features(text)
        print(f"Text: {text[:60]}...")
        print(f"  → Has brand: {brand['has_brand']}, Brand: {brand['potential_brand']}, "
              f"Position: {brand['brand_position']}")
    
    # Demo 5: Full Feature Engineering Pipeline
    print("\n5. Full Feature Engineering Pipeline:")
    print("-" * 80)
    
    # Create DataFrame
    df = pd.DataFrame({
        'sample_id': [f'prod_{i:03d}' for i in range(len(sample_products))],
        'catalog_content': sample_products
    })
    
    print(f"Input DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Engineer features (without TF-IDF for small demo dataset)
    # Note: TF-IDF requires min_df=5 documents, so we skip it for this small demo
    features_df = engineer.engineer_features(df, fit_tfidf=False)
    
    print(f"\nOutput Features DataFrame shape: {features_df.shape}")
    print(f"Number of features: {features_df.shape[1] - 1}")  # -1 for sample_id
    print(f"\nFeature columns (first 20):")
    for col in list(features_df.columns)[:20]:
        print(f"  - {col}")
    
    # Show sample of engineered features
    print(f"\nSample features for first product:")
    first_row = features_df.iloc[0]
    print(f"  Sample ID: {first_row['sample_id']}")
    print(f"  IPQ Value: {first_row['ipq_value']}")
    print(f"  Text Length: {first_row['text_length']}")
    print(f"  Word Count: {first_row['word_count']}")
    print(f"  Has Quality Keywords: {first_row['has_quality_keywords']}")
    print(f"  Has Brand: {first_row['has_brand']}")
    print(f"  Potential Brand: {first_row['potential_brand']}")
    
    # Demo 6: Feature Serialization
    print("\n6. Feature Serialization:")
    print("-" * 80)
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / 'demo_features.pkl'
        
        # Save
        engineer.save_features(features_df, filepath)
        print(f"Features saved to: {filepath}")
        print(f"File size: {filepath.stat().st_size / 1024:.2f} KB")
        
        # Load
        loaded_df = engineer.load_features(filepath)
        print(f"Features loaded successfully")
        print(f"Loaded shape: {loaded_df.shape}")
        print(f"Shapes match: {features_df.shape == loaded_df.shape}")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
