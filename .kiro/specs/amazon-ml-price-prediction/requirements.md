# Requirements Document

## Introduction

This document specifies the requirements for a portfolio-quality multimodal machine learning system for the Amazon ML Challenge 2025. The system predicts product prices using text (product descriptions), images, and tabular features. The solution is designed for incremental training on consumer hardware (AMD Ryzen 7 5800H + RTX 3050 6GB + 16GB RAM) with the ability to stop and resume training at any point. The system must achieve less than 9% SMAPE (Symmetric Mean Absolute Percentage Error) on the validation set.

## Glossary

- **SMAPE**: Symmetric Mean Absolute Percentage Error - the primary evaluation metric for price prediction accuracy
- **System**: The complete multimodal ML pipeline including data processing, model training, and prediction generation
- **Checkpoint**: A saved state of training progress that allows resumption from any point
- **LoRA**: Low-Rank Adaptation - a parameter-efficient fine-tuning technique
- **GBDT**: Gradient Boosted Decision Trees - ensemble learning algorithms (LightGBM, XGBoost, CatBoost)
- **Multimodal Model**: A neural network that processes text, images, and tabular features simultaneously
- **Feature Engineering**: The process of extracting and transforming raw data into meaningful features
- **Ensemble**: A combination of multiple models to improve prediction accuracy
- **IPQ**: Item Pack Quantity - extracted features about product quantities and units
- **TF-IDF**: Term Frequency-Inverse Document Frequency - text vectorization technique
- **Cross-Modal Attention**: A mechanism allowing text and image features to interact
- **Meta-Learner**: A model that combines predictions from multiple base models
- **Isotonic Calibration**: A technique to improve prediction calibration

## Requirements

### Requirement 1: Data Download and Management

**User Story:** As a developer, I want to download and manage 150K product images automatically with resume capability, so that I can handle interrupted downloads and work with limited internet connectivity.

#### Acceptance Criteria

1. WHEN the system downloads images THEN the System SHALL download images in configurable batches with parallel workers
2. WHEN a download is interrupted THEN the System SHALL save progress state and resume from the last completed batch
3. WHEN an image download fails THEN the System SHALL retry up to 3 times with exponential backoff
4. WHEN all images are downloaded THEN the System SHALL verify image integrity and identify corrupted files
5. WHERE automatic download is enabled THEN the System SHALL download the dataset from the competition source without manual intervention

### Requirement 2: Feature Engineering

**User Story:** As a data scientist, I want to extract comprehensive features from product data, so that I can provide rich inputs to the machine learning models.

#### Acceptance Criteria

1. WHEN processing product text THEN the System SHALL extract IPQ (Item Pack Quantity) features using regex patterns
2. WHEN processing product text THEN the System SHALL compute text statistics including length, word count, and character distributions
3. WHEN processing product text THEN the System SHALL identify quality and discount keywords
4. WHEN processing product text THEN the System SHALL extract potential brand names from text
5. WHEN processing training data THEN the System SHALL fit a TF-IDF vectorizer with configurable max features
6. WHEN processing validation or test data THEN the System SHALL transform text using the fitted TF-IDF vectorizer
7. WHEN IPQ features are extracted THEN the System SHALL normalize quantities to standard units
8. WHEN all features are engineered THEN the System SHALL save features to disk in pickle format

### Requirement 3: Checkpoint System

**User Story:** As a developer with limited hardware availability, I want to save and resume training at any point, so that I can train models in multiple short sessions without losing progress.

#### Acceptance Criteria

1. WHEN training progresses for 30 minutes THEN the System SHALL automatically save a checkpoint
2. WHEN a checkpoint is saved THEN the System SHALL include model state, optimizer state, epoch number, and step number
3. WHEN resuming training THEN the System SHALL load the most recent checkpoint and continue from the saved state
4. WHEN multiple checkpoints exist THEN the System SHALL maintain the last 3 checkpoints and delete older ones
5. WHEN a checkpoint is created THEN the System SHALL log the checkpoint size and location
6. WHEN training completes a stage THEN the System SHALL save a stage checkpoint with metadata

### Requirement 4: Multimodal Neural Network Architecture

**User Story:** As a machine learning engineer, I want a memory-efficient multimodal architecture, so that I can train on 6GB VRAM while achieving high accuracy.

#### Acceptance Criteria

1. WHEN encoding text THEN the System SHALL use DeBERTa-small with LoRA fine-tuning
2. WHEN encoding images THEN the System SHALL use EfficientNet-B2 with frozen early layers
3. WHEN fusing modalities THEN the System SHALL apply bidirectional cross-modal attention between text and image features
4. WHEN processing tabular features THEN the System SHALL project them through a learned embedding layer
5. WHEN making predictions THEN the System SHALL combine fused multimodal features with tabular embeddings through a regression head
6. WHERE gradient checkpointing is enabled THEN the System SHALL use gradient checkpointing to reduce memory usage
7. WHEN training THEN the System SHALL use mixed precision (FP16) training
8. WHEN applying LoRA THEN the System SHALL make only 1.6% of parameters trainable

### Requirement 5: Neural Network Training

**User Story:** As a machine learning practitioner, I want to train the neural network with proper optimization and regularization, so that I can achieve the best possible validation performance.

#### Acceptance Criteria

1. WHEN training THEN the System SHALL use Huber-smoothed SMAPE loss function
2. WHEN optimizing THEN the System SHALL use AdamW optimizer with different learning rates for different components
3. WHEN scheduling learning rate THEN the System SHALL apply linear warmup followed by linear decay
4. WHEN accumulating gradients THEN the System SHALL accumulate gradients over 4 steps to simulate larger batch sizes
5. WHEN updating weights THEN the System SHALL clip gradients to maximum norm of 1.0
6. WHEN training THEN the System SHALL maintain an Exponential Moving Average (EMA) of model parameters
7. WHEN validating THEN the System SHALL evaluate on validation set after each epoch
8. WHEN validation SMAPE improves THEN the System SHALL save the model as the best checkpoint
9. WHEN training completes THEN the System SHALL generate predictions using the best model

### Requirement 6: GBDT Model Training

**User Story:** As a machine learning engineer, I want to train multiple GBDT models with custom SMAPE objectives, so that I can create diverse base models for ensembling.

#### Acceptance Criteria

1. WHEN training LightGBM THEN the System SHALL use custom SMAPE objective and evaluation functions
2. WHEN training XGBoost THEN the System SHALL use custom SMAPE objective with proper gradient and hessian
3. WHEN training CatBoost THEN the System SHALL use built-in regression with early stopping
4. WHERE hyperparameter optimization is enabled THEN the System SHALL use Optuna to optimize each GBDT model
5. WHEN GBDT training completes THEN the System SHALL save each trained model to disk
6. WHEN GBDT training completes THEN the System SHALL generate and save predictions for train, validation, and test sets

### Requirement 7: Ensemble and Stacking

**User Story:** As a data scientist, I want to combine multiple models through 2-level stacking, so that I can achieve better performance than any single model.

#### Acceptance Criteria

1. WHEN creating level-1 meta-features THEN the System SHALL stack predictions from all base models
2. WHEN training level-1 meta-learners THEN the System SHALL train Ridge, ElasticNet, and shallow LightGBM models
3. WHEN creating level-2 ensemble THEN the System SHALL optimize weights for combining level-1 predictions
4. WHEN optimizing level-2 weights THEN the System SHALL minimize SMAPE on validation set
5. WHEN final predictions are generated THEN the System SHALL apply isotonic regression calibration
6. WHEN ensemble training completes THEN the System SHALL save all meta-learners, weights, and calibration models
7. WHEN ensemble is evaluated THEN the System SHALL report individual model performance and ensemble improvement

### Requirement 8: PyTorch Dataset and DataLoader

**User Story:** As a developer, I want efficient data loading with proper preprocessing, so that I can maximize GPU utilization during training.

#### Acceptance Criteria

1. WHEN loading data THEN the System SHALL merge raw data with engineered features on sample_id
2. WHEN tokenizing text THEN the System SHALL truncate or pad to maximum sequence length
3. WHEN loading images THEN the System SHALL resize to configured dimensions and apply normalization
4. WHERE training mode is active THEN the System SHALL apply data augmentation to images
5. WHEN loading tabular features THEN the System SHALL fill NaN values with zeros
6. WHEN creating batches THEN the System SHALL use pin_memory and prefetching for efficiency
7. WHEN an image fails to load THEN the System SHALL create a blank placeholder image

### Requirement 9: Metrics and Evaluation

**User Story:** As a data scientist, I want comprehensive evaluation metrics, so that I can understand model performance across different price ranges.

#### Acceptance Criteria

1. WHEN calculating SMAPE THEN the System SHALL use the formula: mean(|pred - true| / ((|true| + |pred|) / 2)) * 100
2. WHEN predictions are in log space THEN the System SHALL convert to original space before calculating SMAPE
3. WHEN evaluating THEN the System SHALL compute MAE, RMSE, MAPE, and R-squared in addition to SMAPE
4. WHEN analyzing errors THEN the System SHALL calculate SMAPE by price quantiles
5. WHEN evaluation completes THEN the System SHALL print formatted metrics summary

### Requirement 10: Configuration Management

**User Story:** As a developer, I want centralized configuration, so that I can easily adjust hyperparameters and paths without modifying code.

#### Acceptance Criteria

1. WHEN the system initializes THEN the System SHALL load all configuration from a single config module
2. WHEN configuration is accessed THEN the System SHALL provide paths for data, models, checkpoints, and predictions
3. WHEN configuration is accessed THEN the System SHALL provide model hyperparameters for neural network and GBDT models
4. WHEN configuration is accessed THEN the System SHALL provide training settings including batch size, learning rate, and epochs
5. WHEN configuration is accessed THEN the System SHALL provide feature engineering parameters including IPQ patterns and unit conversions

### Requirement 11: Visualization and Analysis

**User Story:** As a data scientist, I want to visualize training progress and prediction quality, so that I can diagnose issues and communicate results.

#### Acceptance Criteria

1. WHEN training completes THEN the System SHALL plot training and validation loss curves
2. WHEN training completes THEN the System SHALL plot validation SMAPE over epochs
3. WHEN predictions are generated THEN the System SHALL create scatter plots of predicted vs actual values
4. WHEN predictions are generated THEN the System SHALL create residual plots
5. WHEN predictions are generated THEN the System SHALL plot error distribution histograms
6. WHEN plots are created THEN the System SHALL save them as high-resolution PNG files

### Requirement 12: Submission Generation

**User Story:** As a competition participant, I want to generate properly formatted submission files, so that I can submit predictions to the competition platform.

#### Acceptance Criteria

1. WHEN generating submission THEN the System SHALL load the best ensemble predictions for test set
2. WHEN generating submission THEN the System SHALL convert predictions from log space to original price space
3. WHEN generating submission THEN the System SHALL create a CSV with sample_id and predicted_price columns
4. WHEN generating submission THEN the System SHALL validate that all test samples have predictions
5. WHEN submission is saved THEN the System SHALL log the output file path and basic statistics
