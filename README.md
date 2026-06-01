# ERDERA ERN Predictor

Predicts which [European Reference Network (ERN)](https://health.ec.europa.eu/european-reference-networks_en) a patient most likely belongs to, given their HPO-annotated phenopacket. All hyperparameters are controlled via `config.toml`.

## Setup

```bash
conda env create -f src/environment.yml
conda activate find-ern-env
```

## Usage

```bash
# train
python src/main.py --input data/ --train

# predict all samples
python src/main.py --input data/ --predict

# train and predict in one run
python src/main.py --input data/ --train --predict

# predict using a specific model bundle
python src/main.py --input data/ --predict --model /path/to/model.joblib
```

> **Parallel training** — by default, training runs 4 concurrent Optuna trials each using 4 LightGBM threads (16 cores total). Adjust `optuna_jobs` and `lgbm_threads` in `config.toml` to match your hardware. Running more concurrent trials than available cores will slow training down rather than speed it up. Parallel Optuna trials also require a thread-safe environment. If you encounter crashes or inconsistent results, set `optuna_jobs = 1` to run sequentially.

## Methodology

### Input data and prediction target

Input is a directory of [Phenopacket](https://phenopacket-schema.readthedocs.io/) JSON files. Each file must contain at least one observed (non-excluded) HPO term and an ERN identifier in `metaData.externalReferences`. Samples without an ERN label are excluded from training but can be included in prediction.

Each HPO term is propagated to all its ancestors using the HPO ontology, producing a binary feature vector per patient (1 = term or ancestor present, 0 = absent). This captures semantic similarity between patients even when different but related terms were recorded.

ERN classes with fewer than `min_samples` (default: 20) training samples are merged into a single **Other** class rather than dropped. This means the model can predict `Other` for patients who likely belong to a rare or underrepresented ERN, which are then flagged for human review. The `confidence` column in the prediction output gives the predicted probability for the top class, regardless of which class was predicted.

### Training process

1. **Stratified train/test split** (default 80/20) is applied to preserve class proportions in both sets.
2. **Optuna TPE search** runs `n_iter` trials, each evaluating a candidate hyperparameter set via stratified `cv_folds`-fold cross-validation on the training set. The objective is mean macro F1 across folds. Trials run in parallel (`optuna_jobs` concurrent trials, each using `lgbm_threads` LightGBM threads).
3. **Sample weights** are computed per class using inverse frequency weighting, capped at a multiple of the median weight. The cap multiplier is also tuned by Optuna to prevent rare classes from dominating.
4. The best parameter set is refitted on the full training set using the mean optimal tree count from early stopping across CV folds.

### Results

Each run writes to a timestamped directory under `output/`:

| File | Description |
|---|---|
| `data/ern_per_class_metrics.csv` | Precision, recall, F1, and support per ERN class on the test set |
| `data/ern_confusion_matrix.csv` | Full confusion matrix |
| `data/ern_feature_importances.csv` | Top HPO feature importances by gain |
| `data/ern_predictions.csv` | Predicted ERN and confidence per sample (predict mode) |
| `plots/ern_confusion_matrix.png` | Confusion matrix heatmap |
| `plots/ern_feature_importances.png` | Top feature importances bar chart |
| `plots/ern_cv_scores.png` | Cross-validation score distribution |
| `plots/ern_roc_curves.png` | ROC curves per ERN with Youden J threshold markers |

