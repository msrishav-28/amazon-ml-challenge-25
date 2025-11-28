"""
Setup script for Amazon ML Price Prediction package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

setup(
    name="amazon-ml-price-prediction",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multimodal ML system for Amazon product price prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/amazon-ml-price-prediction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.36.0",
        "timm>=0.9.12",
        "lightgbm>=4.1.0",
        "xgboost>=2.0.3",
        "catboost>=1.2.2",
        "optuna>=3.5.0",
        "pandas>=2.1.4",
        "numpy>=1.26.2",
        "scikit-learn>=1.3.2",
        "scipy>=1.11.4",
        "Pillow>=10.1.0",
        "regex>=2023.10.3",
        "matplotlib>=3.8.2",
        "seaborn>=0.13.0",
        "tqdm>=4.66.1",
        "pyyaml>=6.0.1",
        "requests>=2.31.0",
        "peft>=0.7.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "hypothesis>=6.92.1",
            "black>=23.12.0",
            "flake8>=7.0.0",
            "mypy>=1.7.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "amazon-ml-train=scripts.run_stage3_neural_net:main",
            "amazon-ml-predict=scripts.create_submission:main",
        ],
    },
)
