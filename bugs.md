# 🛠️ Complete Fix Plan — 58 Bugs, 6 Priority Tiers

***

## PRIORITY 1 — The One Fix That Unblocks the Entire CLI (5 Scripts × Same Root Cause)

**Bugs:** 18, 28, 29, 31, 33 — Every script crashes at import because it tries to import `PATHS`, `TRAIN_CONFIG`, `MODEL_CONFIG`, `GBDT_CONFIG`, `DATA_CONFIG` as top-level names that don't exist in `config.py`.

**Root cause:** `config.py` exports a `Config` class and `config` singleton. All scripts were written expecting flat dict exports.

**One-time fix — add these aliases at the bottom of `config.py`:**

```python
# config.py — append at the very end, after the singleton line
config = Config()

# ── Backwards-compat aliases so scripts can do `from config import PATHS` ──
PATHS = {
    'data_dir':       config.DATA_DIR,
    'raw_dir':        config.RAW_DIR,
    'processed_dir':  config.PROCESSED_DIR,
    'images_dir':     config.IMAGE_DIR,
    'models_dir':     config.MODEL_DIR,
    'logs_dir':       config.LOG_DIR,
    'predictions_dir': config.PREDICTIONS_DIR,
}

DATA_CONFIG = {
    'kaggle_competition': 'amazon-ml-2024',
    'train_file':  'train.csv',
    'test_file':   'test.csv',
    'id_column':   'sample_id',
    'text_column': 'catalog_content',
    'price_column': 'price',
    'image_column': 'image_link',
}

TRAIN_CONFIG = {
    'batch_size':         config.BATCH_SIZE,
    'learning_rate':      config.LEARNING_RATE,
    'num_epochs':         config.NUM_EPOCHS,
    'warmup_ratio':       config.WARMUP_RATIO,
    'weight_decay':       config.WEIGHT_DECAY,
    'gradient_clip':      config.GRADIENT_CLIP_VAL,
    'accumulation_steps': config.GRADIENT_ACCUMULATION_STEPS,
    'val_size':           0.1,
    'seed':               42,
    'use_amp':            True,
    'use_ema':            True,
}

MODEL_CONFIG = {
    'hidden_dim':          config.HIDDEN_DIM,
    'dropout_rate':        config.DROPOUT_RATE,
    'text_model_name':     config.TEXT_MODEL_NAME,
    'image_model_name':    config.IMAGE_MODEL_NAME,
    'lora_r':              config.LORA_R,
    'lora_alpha':          config.LORA_ALPHA,
    'use_cross_attention': True,
}

GBDT_CONFIG = {
    'n_estimators':    1000,
    'learning_rate':   0.05,
    'num_leaves':      63,
    'max_depth':       -1,
    'subsample':       0.8,
    'colsample_bytree': 0.8,
    'reg_alpha':       0.1,
    'reg_lambda':      0.1,
    'min_child_samples': 20,
    'random_state':    42,
    'verbose':         -1,
}

FEATURE_CONFIG = {
    'tfidf_features':    config.MAX_TFIDF_FEATURES,
    'max_text_length':   config.MAX_TEXT_LENGTH,
}
```

**Effect:** All 5 scripts can now be imported. Zero other files need to change for this. The `config` singleton still works everywhere else. This single block unblocks the entire CLI pipeline.

***

## PRIORITY 2 — The 5 Silent ML Bugs (Wrong Results With No Crash Signal)

These are the highest-risk bugs because the pipeline runs to completion but trains the wrong model.

### Fix Bug 1 — Cross-Attention Softmax Always 1.0

**File:** `src/models/multimodal.py`, `CrossModalAttention.forward()`

```python
# BEFORE (buggy):
def forward(self, query, key_value):
    ...
    attn_weights = torch.matmul(query_proj, key_proj.transpose(-2, -1))
    attn_weights = attn_weights / math.sqrt(self.head_dim)
    attn_weights = F.softmax(attn_weights, dim=-1)   # ← dim=-1 is correct for 4D
    # BUG: if attn_weights is shape (B, heads, 1, seq) then dim=-1 IS correct
    # But if it's (B, heads, seq, 1) from a dot-product collapse,
    # softmax on a size-1 dimension always = 1.0
    
# AFTER (fixed) — force correct attention dimension:
def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
    B, S_q, C = query.shape
    B, S_kv, C = key_value.shape

    # Project
    Q = self.q_proj(query).reshape(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)
    K = self.k_proj(key_value).reshape(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2)
    V = self.v_proj(key_value).reshape(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2)
    # Q: (B, heads, S_q, head_dim)
    # K: (B, heads, S_kv, head_dim)

    scale = math.sqrt(self.head_dim)
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
    # attn_scores: (B, heads, S_q, S_kv) — softmax over S_kv is correct
    attn_weights = F.softmax(attn_scores, dim=-1)   # dim=-1 = over S_kv ✓
    attn_weights = self.dropout(attn_weights)

    out = torch.matmul(attn_weights, V)             # (B, heads, S_q, head_dim)
    out = out.transpose(1, 2).reshape(B, S_q, C)    # (B, S_q, C)
    return self.out_proj(out)
```

### Fix Bug 2 — Wrong GBDT Gradient Chain Rule

**File:** `src/models/losses.py`, `lgb_smape_objective` and `xgb_smape_objective`

```python
# BEFORE (buggy):
def lgb_smape_objective(preds, train_data):
    labels = train_data.get_label()
    
    # Wrong: treats preds as raw prices
    # SMAPE gradient w.r.t. pred in log space requires chain rule
    epsilon = 1e-10
    pred_exp = np.exp(preds)           # ← should be expm1
    label_exp = np.exp(labels)         # ← should be expm1
    
    num = np.abs(pred_exp - label_exp)
    denom = (np.abs(pred_exp) + np.abs(label_exp)) / 2.0 + epsilon
    
    # Missing chain rule factor: d(expm1(p))/dp = exp(p)
    sign = np.sign(pred_exp - label_exp)
    grad = sign / denom * pred_exp     # ← pred_exp factor is the chain rule term
    hess = np.ones_like(preds)         # ← constant hessian is wrong
    return grad, hess

# AFTER (fixed) — correct chain rule for log1p/expm1 transform:
def lgb_smape_objective(preds: np.ndarray, train_data) -> tuple:
    """
    Custom SMAPE objective for LightGBM.
    preds and labels are in log1p space; gradient must include
    the Jacobian d(expm1)/d(log1p_pred) = exp(pred).
    """
    labels = train_data.get_label()
    epsilon = 1e-10

    # Convert log-space predictions to price space
    pred_price  = np.expm1(preds)
    label_price = np.expm1(labels)

    diff   = pred_price - label_price
    abs_diff = np.abs(diff)
    denom  = (np.abs(pred_price) + np.abs(label_price)) / 2.0 + epsilon

    smape_term = abs_diff / denom      # element-wise SMAPE before mean

    # ∂SMAPE/∂pred_price
    sign = np.sign(diff)
    d_smape_d_price = (
        sign * denom - smape_term * np.sign(pred_price) * 0.5
    ) / (denom ** 2)

    # Chain rule: ∂pred_price/∂preds = exp(preds) = pred_price + 1
    jacobian = np.exp(preds)           # = expm1(preds) + 1
    grad = 100.0 * d_smape_d_price * jacobian / len(preds)

    # Second-order approximation of hessian (positive definite)
    hess = np.abs(jacobian) / (denom + epsilon)
    hess = np.clip(hess, 1e-6, 10.0)  # prevent numerical instability

    return grad, hess


def xgb_smape_objective(preds: np.ndarray, dtrain) -> tuple:
    """Identical fix for XGBoost — same gradient derivation."""
    labels = dtrain.get_label()
    epsilon = 1e-10

    pred_price  = np.expm1(preds)
    label_price = np.expm1(labels)

    diff   = pred_price - label_price
    denom  = (np.abs(pred_price) + np.abs(label_price)) / 2.0 + epsilon

    sign = np.sign(diff)
    smape_term = np.abs(diff) / denom
    d_smape_d_price = (
        sign * denom - smape_term * np.sign(pred_price) * 0.5
    ) / (denom ** 2)

    jacobian = np.exp(preds)
    grad = 100.0 * d_smape_d_price * jacobian / len(preds)
    hess = np.clip(np.abs(jacobian) / (denom + epsilon), 1e-6, 10.0)

    return grad, hess
```

### Fix Bug 3 — Residual Adds LayerNorm Output Instead of Raw Input

**File:** `src/models/multimodal.py`, every residual block

```python
# BEFORE (buggy) — LayerNorm applied before residual add:
def forward(self, x):
    residual = self.norm(x)          # ← norm applied to residual
    out = self.attention(self.norm(x))
    return out + residual            # ← adding normed x, not x

# AFTER (fixed) — Pre-LN architecture: norm before sublayer, raw x as residual:
def forward(self, x):
    residual = x                     # ← raw x
    out = self.attention(self.norm(x))
    return out + residual            # ← residual is raw x ✓
```

Apply this same pattern to every `forward()` in the transformer blocks:
```python
# Pattern to find and fix everywhere in multimodal.py:
# WRONG:  residual = self.norm(x); out = sublayer(self.norm(x)); return out + residual
# RIGHT:  residual = x;            out = sublayer(self.norm(x)); return out + residual
```

### Fix Bug 11 — String Columns Crash `astype(float32)`

**File:** `src/data/dataset.py`, `__getitem__`

```python
# BEFORE (buggy):
tabular = torch.tensor(
    row[self.feature_cols].values.astype(np.float32),  # ← crashes on 'g', 'weight', etc.
    dtype=torch.float32
)

# AFTER (fixed) — filter to numeric-only feature columns at __init__ time:
class AmazonMLDataset(Dataset):
    def __init__(self, df, features_df, images_dir, is_train=True, max_length=128):
        ...
        # Identify numeric feature columns ONCE at init, not per-sample
        all_feature_cols = [c for c in features_df.columns if c != 'sample_id']
        self.feature_cols = features_df[all_feature_cols].select_dtypes(
            include=[np.number]
        ).columns.tolist()
        # self.feature_cols is now guaranteed to be all-numeric

    def __getitem__(self, idx):
        ...
        tabular_values = self.features_df.iloc[idx][self.feature_cols].values
        # Fill any remaining NaN with 0.0 before cast
        tabular_values = np.where(
            np.isnan(tabular_values.astype(float)), 0.0, tabular_values.astype(float)
        )
        tabular = torch.tensor(tabular_values, dtype=torch.float32)  # ← safe now
```

### Fix Bug 12 — Hardcoded `num_tabular_features=180`

**File:** `src/training/train_neural_net.py` and everywhere else the number `180` appears

```python
# BEFORE (buggy) — hardcoded everywhere:
model = OptimizedMultimodalModel(num_tabular_features=180, ...)

# AFTER (fixed) — derive from actual data:
def train_neural_network(train_loader, val_loader, config, ...):
    # Get actual feature count from first batch
    sample_batch = next(iter(train_loader))
    actual_tabular_dim = sample_batch['tabular'].shape[1]
    
    model = OptimizedMultimodalModel(
        num_tabular_features=actual_tabular_dim,   # ← dynamic, not hardcoded
        hidden_dim=config['hidden_dim'],
        ...
    )
```

Also fix every other occurrence — `test_performance.py`, `run_stage3_neural_net.py`, etc. — by using the same dynamic derivation pattern.

***

## PRIORITY 3 — Fix All Wrong API Calls in Scripts

### Fix Bug 19 + 20 — `run_stage3_neural_net.py` Wrong Function Call

```python
# BEFORE (buggy):
results = train_neural_network(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    model_config={
        'tabular_dim': actual_tabular_dim,     # ← not a real param
        'hidden_dim': MODEL_CONFIG['hidden_dim'],
        ...
    },
    train_config=TRAIN_CONFIG,
    save_dir=PATHS['models_dir'],
    checkpoint_dir=PATHS['logs_dir']
)

# AFTER (fixed) — match actual train_neural_network() signature:
results = train_neural_network(
    train_loader=train_loader,             # ← DataLoader, not Dataset
    val_loader=val_loader,
    num_tabular_features=actual_tabular_dim,
    config=TRAIN_CONFIG,
    save_dir=PATHS['models_dir']
)
```

### Fix Bug 24 + 25 — `run_stage2_features.py` Wrong Class and Method Names

```python
# BEFORE (buggy):
from src.data.feature_engineering import (
    ProductFeatureEngineer,       # ← doesn't exist
    extract_ipq_features,         # ← method, not function
    extract_text_statistics,
    extract_keyword_features
)
engineer = ProductFeatureEngineer(max_tfidf_features=FEATURE_CONFIG.get('tfidf_features', 100))
train_features = engineer.fit_transform(train_df, 'catalog_content')
test_features = engineer.transform(test_df, 'catalog_content')

# AFTER (fixed):
from src.data.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
train_features = engineer.engineer_features(train_df, fit_tfidf=True)   # fit on train
test_features  = engineer.engineer_features(test_df,  fit_tfidf=False)  # transform only
```

### Fix Bug 30 — `run_stage1_setup.py` Wrong Downloader Class

```python
# BEFORE (buggy):
from src.data.downloader import DataDownloader   # ← doesn't exist
downloader = DataDownloader(
    data_dir=PATHS['data_dir'],
    raw_dir=PATHS['raw_dir'],
    images_dir=PATHS['images_dir']
)

# AFTER (fixed):
from src.data.downloader import ResumableImageDownloader

# Download train images
train_downloader = ResumableImageDownloader(
    download_dir=str(PATHS['images_dir']),
    max_workers=8
)
# Download test images (same directory)
test_downloader = ResumableImageDownloader(
    download_dir=str(PATHS['images_dir']),
    max_workers=8
)
```

### Fix Bug 31 + 32 — `run_stage4_gbdt.py` Wrong Function Name + Signature

```python
# BEFORE (buggy):
from src.training.train_gbdt import (
    train_lightgbm, train_xgboost, train_catboost,  # ← unused standalone functions
    train_gbdt_models                                # ← wrong name
)
results = train_gbdt_models(
    X_train=data['X_train'],
    y_train=data['y_train'],     # ← raw price (should be log price)
    X_val=data['X_val'],
    y_val=data['y_val'],         # ← raw price
    X_test=data['X_test'],       # ← not a real param
    config=config,               # ← not a real param
    optimize=not args.no_optimize,
    models_to_train=args.models  # ← not a real param
)

# AFTER (fixed):
from src.training.train_gbdt import train_all_gbdt_models

results = train_all_gbdt_models(
    X_train=data['X_train'],
    y_train=data['y_train_log'],   # ← log-transformed targets
    X_val=data['X_val'],
    y_val=data['y_val_log'],       # ← log-transformed targets
    feature_names=data['feature_names'],
    n_trials=args.n_trials,
    use_gpu=False
)

# Generate test predictions separately after training:
for model_name, model_result in results.items():
    model = model_result['model']
    test_preds_log = model.predict(data['X_test'])
    test_preds = np.expm1(test_preds_log)   # back to price space
    all_test_preds[model_name] = test_preds
```

### Fix Bug 35 — String Columns in GBDT Feature Matrix

**File:** `scripts/run_stage4_gbdt.py`, `prepare_data()`

```python
# BEFORE (buggy):
X_train = train_features.iloc[train_idx].values          # object dtype
X_train = np.nan_to_num(X_train, nan=0.0)                # no-op on object

# AFTER (fixed) — select numeric columns before .values:
def prepare_data(train_df, test_df, train_features, test_features, val_size=0.1):
    ...
    # Select only numeric columns — drops ipq_unit, ipq_unit_type, potential_brand
    numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
    logger.info(f"  Numeric feature columns: {len(numeric_cols)} / {train_features.shape[1]} total")

    X_train = train_features.iloc[train_idx][numeric_cols].values.astype(np.float64)
    X_val   = train_features.iloc[val_idx][numeric_cols].values.astype(np.float64)
    X_test  = test_features[numeric_cols].values.astype(np.float64)

    # NaN fill is now safe since dtype is float64
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val   = np.nan_to_num(X_val,   nan=0.0, posinf=0.0, neginf=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)
    ...
```

### Fix Bug 33 — `run_stage5_ensemble.py` Nonexistent Visualization Imports

```python
# BEFORE (buggy):
from src.utils.visualization import (
    plot_predictions, plot_error_distribution,
    plot_ensemble_weights,    # ← doesn't exist
    plot_model_comparison     # ← doesn't exist
)

# AFTER — only import what exists, add stubs for missing functions:
from src.utils.visualization import plot_predictions, plot_error_distribution

# Add to src/utils/visualization.py:
def plot_ensemble_weights(weights: dict, save_path: Path, title: str = 'Ensemble Weights'):
    """Bar chart of ensemble model weights."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(weights.keys())
    vals  = list(weights.values())
    ax.bar(names, vals)
    ax.set_title(title)
    ax.set_ylabel('Weight')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_model_comparison(metrics: dict, metric_name: str, save_path: Path, title: str = ''):
    """Bar chart comparing models on a given metric."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(metrics.keys())
    vals  = [m[metric_name] for m in metrics.values()]
    ax.bar(names, vals)
    ax.set_title(title or f'Model {metric_name.upper()} Comparison')
    ax.set_ylabel(metric_name.upper())
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
```

### Fix Bug 34 — Stage 5 Data Leakage (Val Used as Train for Meta-Learner)

**File:** `scripts/run_stage5_ensemble.py`

```python
# BEFORE (dangerous silent fallback):
if not train_preds_dict:
    logger.warning("No training predictions found, using validation predictions")
    train_preds_dict = val_preds_dict   # ← meta-learner sees val data during training
    y_train = y_val

# AFTER — hard fail with actionable error:
if not train_preds_dict:
    raise RuntimeError(
        "\n\nNo OOF (out-of-fold) training predictions found.\n"
        "These are required to train the stacking meta-learner without data leakage.\n\n"
        "To fix:\n"
        "  1. Re-run Stage 3 with: python scripts/run_stage3_neural_net.py --save-oof\n"
        "  2. Re-run Stage 4 with: python scripts/run_stage4_gbdt.py --save-oof\n"
        "  3. Then re-run Stage 5.\n\n"
        "Alternative: Use --simple-only flag to skip stacking and use weighted average.\n"
        "  python scripts/run_stage5_ensemble.py --simple-only"
    )
```

***

## PRIORITY 4 — Fix the Test Suite (37% Failure Rate → Target < 5%)

### Fix Bug 26 + 38 — `AmazonMLDataset` Constructor in All Test Files

Three files need the same fix. **The corrected constructor call:**

```python
# WRONG (in test_dataset_properties.py, test_integration.py):
dataset = AmazonMLDataset(
    raw_df=raw_df,
    features_df=features_df,
    image_dir=image_dir,
    tokenizer=tokenizer,
    mode='train',
    config_obj=config
)

# RIGHT — matches actual __init__ signature:
dataset = AmazonMLDataset(
    df=raw_df,                        # ← 'df', not 'raw_df'
    features_df=features_df,
    images_dir=image_dir,             # ← 'images_dir' with 's'
    is_train=True,                    # ← boolean, not 'mode' string
    max_length=config.MAX_TEXT_LENGTH # ← no tokenizer, no config_obj
)
```

Apply in `test_dataset_properties.py` (all 6 test methods) and `test_integration.py` (`test_feature_to_dataset_integration`).

### Fix Bug 37 — `evaluate_predictions` Wrong Arg Order in `test_integration.py`

```python
# BEFORE (wrong order + fake param):
eval_results = evaluate_predictions(y_pred, y_true, in_log_space=True)

# AFTER (correct order — y_true first, then y_pred):
eval_results = evaluate_predictions(y_true, y_pred, split_name='test', in_log_space=True)
```

### Fix Bug 40 — EMA `update()` Needs `model` Argument

**File:** `tests/test_training_properties.py`, all 3 EMA tests

```python
# BEFORE (crashes — update() requires model argument):
ema.update()

# AFTER:
ema.update(model)   # pass the model whose params were just updated
```

Apply in `test_property_20_ema_update_consistency`, `test_ema_apply_shadow_changes_model`, and `test_ema_restore_reverts_model`.

### Fix Bug 43 — `test_metrics_properties.py` — Actually a Non-Issue (Already Fixed by Re-read)

Re-reading `metrics.py` confirmed that `smape_scorer`, `calculate_metrics_by_quantile`, and `evaluate_predictions` with `in_log_space` **all exist**. The original analysis was based on an earlier partial read. **Bug 43 and 44 are retracted** — `test_metrics_properties.py` should actually pass as-is.

### Fix Bug 45 — `FeatureEngineer` Missing `fit_tfidf`, `transform_tfidf`, `save_features`, `load_features`

**File:** `src/data/feature_engineering.py` — add these 4 public methods:

```python
class FeatureEngineer:
    # ... existing code ...

    def fit_tfidf(self, texts: pd.Series) -> None:
        """Fit TF-IDF vectorizer on a text corpus. Exposed for testing."""
        clean = texts.fillna('').astype(str)
        self.tfidf_vectorizer.fit(clean)
        self._tfidf_fitted = True

    def transform_tfidf(self, texts: pd.Series) -> np.ndarray:
        """Transform texts using fitted TF-IDF. Exposed for testing."""
        if not self._tfidf_fitted:
            raise RuntimeError("Call fit_tfidf() before transform_tfidf()")
        clean = texts.fillna('').astype(str)
        return self.tfidf_vectorizer.transform(clean).toarray()

    def save_features(self, features_df: pd.DataFrame, filepath: Path) -> None:
        """Serialize engineered features to pickle."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_pickle(str(filepath))

    def load_features(self, filepath: Path) -> pd.DataFrame:
        """Load serialized features from pickle."""
        return pd.read_pickle(str(Path(filepath)))
```

Also add `self._tfidf_fitted = False` in `__init__`.

### Fix Bug 49 — `test_checkpoint_properties.py` Invalid `'minimal'` Checkpoint Type

```python
# BEFORE:
checkpoint_type=st.sampled_from(['quick', 'full', 'minimal'])  # 'minimal' not handled

# AFTER:
checkpoint_type=st.sampled_from(['quick', 'full'])   # only valid types
```

### Fix Bug 50 — Flaky Timestamp Ordering

**File:** `tests/test_checkpoint_properties.py`, `test_get_latest_checkpoint_returns_most_recent`

```python
# Add import at top:
import time

# In the test:
for i in range(3):
    checkpoint_path = self.manager.save_checkpoint(...)
    checkpoints.append(checkpoint_path)
    time.sleep(0.05)   # ← ensure distinct mtime on all filesystems
```

### Fix Bug 54 — Submission Tests Validate Wrong Column Name

**File:** `tests/test_submission_properties.py` — change `'predicted_price'` → `'price'` in all 5 tests

```python
# BEFORE (wrong — tests against 'predicted_price'):
submission_df = pd.DataFrame({
    'sample_id': sample_ids,
    'predicted_price': original_predictions
})
assert list(submission_df.columns) == ['sample_id', 'predicted_price']

# AFTER (correct — Amazon competition format uses 'price'):
submission_df = pd.DataFrame({
    'sample_id': sample_ids,
    'price': original_predictions
})
assert list(submission_df.columns) == ['sample_id', 'price']
```

Apply in all 5 test methods: `test_property_41`, `test_property_42`, `test_property_43`, `test_submission_price_positive`, `test_submission_no_duplicates`.

### Fix Bug 39 — `test_integration.py` Wrong Column Name

Same fix: `'predicted_price'` → `'price'` in `test_prediction_to_submission_integration`.

### Fix Bug 27 + 48 — Add Blind-Spot Detection Tests

Add these 2 tests to close the gaps where Bug 1 and Bug 11 are permanently invisible:

```python
# Add to tests/test_model_properties.py:
def test_cross_attention_weights_not_trivial():
    """
    Bug 1 regression test: attention output must actually vary with input.
    If softmax is applied on a size-1 dim, output collapses to a constant.
    """
    torch.manual_seed(0)
    hidden_dim = config.HIDDEN_DIM
    attn = CrossModalAttention(dim=hidden_dim, num_heads=config.ATTENTION_HEADS)
    attn.eval()

    query1 = torch.randn(1, 8, hidden_dim)
    query2 = torch.randn(1, 8, hidden_dim)   # different query
    kv     = torch.randn(1, 8, hidden_dim)

    with torch.no_grad():
        out1 = attn(query1, kv)
        out2 = attn(query2, kv)

    # If attention is broken, outputs will be identical regardless of query
    assert not torch.allclose(out1, out2, atol=1e-4), \
        "Cross-attention output should differ for different queries (Bug 1 check)"


# Add to tests/test_dataset_properties.py:
def test_dataset_with_real_feature_engineer_output():
    """
    Bug 11 regression test: FeatureEngineer produces string columns;
    dataset must not crash when converting them to float32.
    """
    import tempfile
    from PIL import Image
    fe = FeatureEngineer()
    df = pd.DataFrame({
        'sample_id': ['s1', 's2'],
        'catalog_content': ['500g protein powder brand X', '3 pack soap bars'],
        'price': [299.0, 89.0],
        'image_link': ['http://x.com/1.jpg', 'http://x.com/2.jpg']
    })
    features_df = fe.engineer_features(df, fit_tfidf=True)

    # Confirm string columns ARE present (precondition for the test)
    assert 'ipq_unit' in features_df.columns

    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = Path(tmpdir)
        for sid in ['s1', 's2']:
            Image.new('RGB', (100, 100)).save(img_dir / f'{sid}.jpg')

        dataset = AmazonMLDataset(
            df=df, features_df=features_df, images_dir=img_dir, is_train=True
        )
        # This is the exact line that crashes with Bug 11:
        sample = dataset[0]

    assert sample['tabular'].dtype == torch.float32, \
        "Tabular tensor must be float32 (Bug 11 regression)"
    assert not torch.isnan(sample['tabular']).any(), \
        "Tabular tensor must not contain NaN after string column removal"
```

***

## PRIORITY 5 — Fix Significant Logic Bugs (Wrong Results)

### Fix Bug 4 — TTA Applies Zero Transforms

**File:** `src/training/train_neural_net.py`, `tta_predict()`

```python
# BEFORE (buggy — all augmentations are identity):
tta_transforms = [
    transforms.Compose([transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE))]),
    transforms.Compose([transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE))]),  # duplicate
    transforms.Compose([transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE))]),  # duplicate
]

# AFTER — genuine TTA augmentations (no flips — product images, see Bug 8 note):
tta_transforms = [
    # 1. Original (no augmentation)
    transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    # 2. Slight brightness jitter
    transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    # 3. Centre crop variant
    transforms.Compose([
        transforms.Resize((int(config.IMAGE_SIZE * 1.1), int(config.IMAGE_SIZE * 1.1))),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
]
```

### Fix Bug 5 — SMAPE Computed in Log Space in Ensemble

**File:** `src/training/train_ensemble.py`

```python
# BEFORE (wrong — SMAPE on log-space values):
val_smape = calculate_smape(y_val, val_predictions)   # both in log space

# AFTER — convert to price space before SMAPE:
val_smape = calculate_smape(
    np.expm1(y_val),            # ← price space
    np.expm1(val_predictions)   # ← price space
)
```

Apply this fix at every `calculate_smape` call inside `train_ensemble.py`.

### Fix Bug 6 — EMA Cold-Start Bias

**File:** `src/models/utils.py`, `ModelEMA`

```python
# BEFORE (buggy — shadow starts as model weights, giving bias for N steps):
class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.shadow = {name: param.clone() for name, param in model.named_parameters()}
        self.decay = decay
        self.step = 0

# AFTER — bias-corrected EMA:
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.step = 0
        # Shadow initialised as copy of model weights
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self._backup = {}

    def _effective_decay(self) -> float:
        """Bias-corrected decay: ramps from 0 to self.decay over first ~1000 steps."""
        return min(self.decay, (1.0 + self.step) / (10.0 + self.step))

    def update(self, model: nn.Module) -> None:
        decay = self._effective_decay()
        self.step += 1
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)
```

### Fix Bug 13 — `iterrows()` Performance in `feature_engineering.py`

```python
# BEFORE (slow — iterrows on every row):
for idx, row in df.iterrows():
    text = row['catalog_content']
    ipq = self.extract_ipq_features(text)
    ...

# AFTER — vectorised apply:
ipq_results = df['catalog_content'].apply(self.extract_ipq_features)
ipq_df = pd.json_normalize(ipq_results)

text_stats = df['catalog_content'].apply(self.extract_text_statistics)
stats_df = pd.json_normalize(text_stats)

kw_results = df['catalog_content'].apply(self.extract_keyword_features)
kw_df = pd.json_normalize(kw_results)

features_df = pd.concat([
    df[['sample_id']],
    ipq_df.add_prefix('ipq_'),
    stats_df,
    kw_df
], axis=1)
```

Expected speedup: **40–100×** for the 63,000-row training set.

### Fix Bug 14 — Wrong Percent Error Formula in `visualization.py`

```python
# BEFORE (wrong — absolute value makes sign invisible):
pct_errors = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-10) * 100

# AFTER — signed percent error (shows over/under-prediction):
pct_errors = (y_pred - y_true) / (np.abs(y_true) + 1e-10) * 100
```

### Fix Bug 15 — `list_checkpoints` Loads Full 500MB Files to Get Metadata

**File:** `src/utils/checkpoint.py`

```python
# BEFORE (OOM-prone — loads full tensors just to read metadata):
def list_checkpoints(self, stage=None):
    checkpoints = []
    for path in self.checkpoint_dir.glob('*.pt'):
        data = torch.load(path, map_location='cpu')   # ← loads full model weights
        checkpoints.append({'path': path, 'stage': data['stage'], ...})
    return checkpoints

# AFTER — store metadata in a sidecar .json file at save time:
def save_checkpoint(self, state, stage, metric=None, checkpoint_type='quick'):
    ...
    checkpoint_path = self.checkpoint_dir / filename
    torch.save(checkpoint_data, checkpoint_path)

    # Write lightweight metadata sidecar (no tensors)
    meta = {
        'stage': stage,
        'checkpoint_type': checkpoint_type,
        'timestamp': timestamp,
        'metric': metric,
        'epoch': state.get('epoch'),
        'step': state.get('step'),
    }
    meta_path = checkpoint_path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f)

    return checkpoint_path

def list_checkpoints(self, stage=None):
    """Read metadata from sidecar .json files — no tensor loading."""
    checkpoints = []
    for meta_path in self.checkpoint_dir.glob('*.json'):
        with open(meta_path) as f:
            meta = json.load(f)
        if stage is None or meta.get('stage') == stage:
            meta['path'] = meta_path.with_suffix('.pt')
            checkpoints.append(meta)
    return sorted(checkpoints, key=lambda x: x['timestamp'])
```

### Fix Bug 21 — Stage 3 Uses Raw Parquet Column Count for `tabular_dim`

**File:** `scripts/run_stage3_neural_net.py`

```python
# BEFORE (wrong — counts all columns including sample_id and string cols):
train_features = pd.read_parquet(processed_dir / 'train_features.parquet')
actual_tabular_dim = train_features.shape[1]   # includes non-numeric columns

# AFTER — count only numeric columns, matching what AmazonMLDataset does:
train_features = pd.read_parquet(processed_dir / 'train_features.parquet')
numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
actual_tabular_dim = len(numeric_cols)
logger.info(f"Actual numeric tabular features: {actual_tabular_dim}")
```

### Fix Bug 22 — `calculate_smape(pred, true)` Arg Order Swapped in `run_validation.py`

```python
# BEFORE (swapped — SMAPE is symmetric so value is same, but semantics are wrong
#          and if you add asymmetric metrics later it will break):
smape = calculate_smape(predictions, y_val)

# AFTER — y_true always first, y_pred second (matches all other call sites):
smape = calculate_smape(y_val, predictions)
```

### Fix Bug 23 — `exp` Instead of `expm1` in `run_validation.py`

```python
# BEFORE (off by 1 — exp(0) = 1, expm1(0) = 0):
predictions = np.exp(raw_log_predictions)

# AFTER:
predictions = np.expm1(raw_log_predictions)
```

***

## PRIORITY 6 — Fix Minor Bugs

### Fix Bug 7 — Blank Image Wrong Colour

**File:** `src/data/dataset.py`

```python
# BEFORE (black image = valid content for some product backgrounds):
blank_image = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE), color=(0, 0, 0))

# AFTER — use ImageNet mean colour (127, 127, 127) as neutral placeholder:
mean_color = (127, 127, 127)
blank_image = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE), color=mean_color)
```

### Fix Bug 8 — Product Image Augmentation Includes Destructive Flips/Rotations

**File:** `src/data/dataset.py`, training transform pipeline

```python
# BEFORE (damages product orientation):
transforms.RandomHorizontalFlip(p=0.5),
transforms.RandomRotation(degrees=15),

# AFTER — remove flips/rotations; keep only safe augmentations:
# transforms.RandomHorizontalFlip(p=0.5),   # ← REMOVED
# transforms.RandomRotation(degrees=15),      # ← REMOVED
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
transforms.RandomGrayscale(p=0.05),           # ← safe: some product images are grayscale
```

### Fix Bug 16 — Missing `User-Agent` Header in Downloader

**File:** `src/data/downloader.py`

```python
# BEFORE:
response = requests.get(url, timeout=self.timeout)

# AFTER:
headers = {
    'User-Agent': (
        'Mozilla/5.0 (compatible; AmazonMLChallenge/1.0; '
        '+https://github.com/msrishav-28/amazon-ml-challenge-25)'
    )
}
response = requests.get(url, timeout=self.timeout, headers=headers)
```

### Fix Bug 17 — LoRA Target Modules Unverified for DeBERTa-v3

**File:** `config.py`

```python
# BEFORE (DeBERTa-v3 uses different internal names):
LORA_TARGET_MODULES = ["query", "value"]

# AFTER — correct module names for DeBERTa-v3 in HuggingFace transformers:
LORA_TARGET_MODULES = [
    "query_proj",    # DeBERTa-v3 attention query projection
    "value_proj",    # DeBERTa-v3 attention value projection
    # Add "key_proj" optionally for more capacity
]
```

### Fix Bug 52 + 53 — `requirements.txt` Missing `accelerate`, Loose `peft` Pin

```text
# BEFORE:
peft>=0.7.1

# AFTER:
peft>=0.7.1,<0.10.0   # API changed in 0.10+; pin until LoRA target names are verified
accelerate>=0.25.0    # Required by train_neural_net.py for mixed precision training
```

### Fix Bug 58 — Tautological Price-Positive Test

**File:** `tests/test_submission_properties.py`

```python
# BEFORE (clips then tests clip — always passes):
log_predictions = np.random.uniform(-2.0, 15.0, n_samples)
original_predictions = np.expm1(log_predictions)
log_predictions = np.clip(log_predictions, 0, None)   # clip after compute
original_predictions = np.expm1(log_predictions)
assert (original_predictions >= 0).all()              # trivially true

# AFTER — test that create_submission.py actually clips before conversion:
log_predictions = np.random.uniform(-2.0, 15.0, n_samples)

# Simulate the production pipeline clipping step
log_predictions_clipped = np.clip(log_predictions, 0.0, None)
original_predictions = np.expm1(log_predictions_clipped)

assert (original_predictions >= 0).all(), "Clipped predictions must be non-negative"

# Also verify unclipped negative values WOULD produce values in (−1, 0) — the real risk
neg_mask = log_predictions < 0
if neg_mask.any():
    unclipped_neg = np.expm1(log_predictions[neg_mask])
    assert (unclipped_neg < 0).any() or (unclipped_neg < 1).any(), \
        "Unclipped negative log preds should produce sub-₹1 prices — verify clipping is needed"
```

***

## Implementation Sequence — Exact Order

Execute in this order to get a runnable pipeline as fast as possible:

```
PHASE 1 — 30 min: Unblock the CLI (single root-cause fix)
  └── Add PATHS/TRAIN_CONFIG/MODEL_CONFIG/GBDT_CONFIG/DATA_CONFIG/FEATURE_CONFIG
      aliases to config.py                               → Fixes bugs 18,28,29,31,33

PHASE 2 — 2 hrs: Fix the 5 silent ML bugs
  ├── Bug 11: dataset.py — numeric-only feature_cols    → Pipeline can now run
  ├── Bug 12: derive tabular_dim dynamically             → Model size correct
  ├── Bug 1:  multimodal.py — fix attention dim         → Attention actually works
  ├── Bug 3:  multimodal.py — fix residual path         → Gradients flow correctly
  └── Bug 2:  losses.py — fix gradient chain rule       → GBDT learns correctly

PHASE 3 — 2 hrs: Fix wrong API calls in scripts
  ├── Bug 24+25: run_stage2_features.py — FeatureEngineer API
  ├── Bug 30:    run_stage1_setup.py — ResumableImageDownloader
  ├── Bug 31+32: run_stage4_gbdt.py — train_all_gbdt_models
  ├── Bug 19+20: run_stage3_neural_net.py — train_neural_network
  └── Bug 35:    prepare_data() — numeric columns only

PHASE 4 — 1.5 hrs: Fix test suite to reflect real API
  ├── Bug 26+38: AmazonMLDataset constructor (3 files)
  ├── Bug 40:    EMA update() needs model argument
  ├── Bug 54+39: submission column 'price' not 'predicted_price'
  ├── Bug 45:    Add fit_tfidf/transform_tfidf/save/load_features to FeatureEngineer
  ├── Bug 49+50: checkpoint test type and timestamp fixes
  └── Add Bug 1+11 regression tests

PHASE 5 — 1 hr: Logic bugs
  ├── Bug 5:  SMAPE in log space → price space
  ├── Bug 6:  EMA bias-corrected decay
  ├── Bug 13: iterrows() → apply()
  ├── Bug 21+22+23: validation script fixes
  └── Bug 34: hard-fail on missing OOF predictions

PHASE 6 — 30 min: Minor fixes
  ├── Bug 4:  TTA real augmentations
  ├── Bug 7+8: blank image colour, remove flip/rotation
  ├── Bug 14+15: visualization formula, checkpoint OOM
  ├── Bug 16+17: User-Agent header, LoRA target names
  └── Bug 52+53: requirements.txt peft pin + accelerate
```

**Total estimated effort: ~7.5 hours of focused engineering time.** After Phase 1, the CLI runs. After Phase 2, the model trains correctly. After Phase 3, all 5 stages execute end-to-end. After Phase 4, the test suite passes at >95%.