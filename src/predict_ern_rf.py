#!/usr/bin/env python3
"""
Train a Random Forest classifier to predict ERN from HPO terms and submitter.

Features:
  - Ontology-propagated binary HPO term vectors (same representation as
    cluster_phenopackets.py)
  - One-hot encoded submitter label (omit with --no-submitter)

Target:
  - ERN (European Reference Network) from phenopacket metaData.externalReferences

ERN classes with fewer than --min-class-size patients are excluded from training
and evaluation.

Evaluation:
  - Stratified train/test split
  - Stratified k-fold cross-validation (on training set)
  - Per-class precision / recall / F1
  - Confusion matrix (CSV + heatmap PNG)
  - Top feature importances (HPO terms and submitter categories)

Usage:
    python predict_ern.py --input /path/to/phenopackets/ [options]

Options:
    --input,           -i  DIR   Directory with phenopacket JSON files (required)
    --output,          -o  FILE  Output directory (default: output/<timestamp>/)
    --n-estimators         INT   Number of trees (default: 300)
    --test-size          FLOAT   Fraction held out for testing (default: 0.20)
    --cv-folds             INT   Stratified CV folds on training set (default: 5)
    --min-class-size       INT   Drop ERN classes smaller than this (default: 10)
    --top-features         INT   Feature importances to report (default: 30)
    --no-submitter               Exclude submitter from the feature set
    --save-model                 Persist the fitted model with joblib

Dependencies:
    pyhpo scikit-learn scipy numpy pandas tqdm matplotlib
"""

import os
import sys
import warnings
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_DISPLAY_ENV", "FALSE")
os.environ.setdefault("KMP_WARNINGS", "FALSE")
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack as sp_hstack
from tqdm import tqdm

from pyhpo import Ontology
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from phenopackets import Sample, build_ancestry_matrix


# Feature construction


def build_feature_matrix(
    samples: list[Sample],
    vocabulary: list[str],
    X_hpo: csr_matrix,
    include_submitter: bool,
) -> tuple[csr_matrix, list[str]]:
    """Stack HPO binary matrix with optional one-hot submitter. Returns (X, feature_names)."""
    feature_names = [_term_label(t) for t in vocabulary]

    if not include_submitter:
        return X_hpo, feature_names

    submitters = np.array([s.submitter for s in samples]).reshape(-1, 1)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X_sub = ohe.fit_transform(submitters)
    sub_names = [f"submitter={v}" for v in ohe.categories_[0]]

    print(f"  Submitter one-hot: {X_sub.shape[1]} categories")
    return sp_hstack([X_hpo, X_sub], format="csr"), feature_names + sub_names


def _term_label(tid: str) -> str:
    try:
        return f"{tid} {Ontology.get_hpo_object(tid).name}"
    except RuntimeError:
        return tid


# Class filtering


def filter_by_class_size(
    samples: list[Sample], min_size: int
) -> tuple[list[Sample], list[str]]:
    counts = Counter(s.ern for s in samples)
    kept = {ern for ern, n in counts.items() if n >= min_size}
    dropped = {ern: n for ern, n in counts.items() if n < min_size}

    if dropped:
        print(f"  Dropping {len(dropped)} ERN class(es) with < {min_size} patients:")
        for ern, n in sorted(dropped.items(), key=lambda x: -x[1]):
            print(f"    {ern}: {n}")

    filtered = [s for s in samples if s.ern in kept]
    classes = sorted(kept)
    print(f"  Keeping {len(filtered)} patients across {len(classes)} ERN classes")
    return filtered, classes


# Training & evaluation


def train_and_evaluate(
    X: csr_matrix,
    y: np.ndarray,
    class_names: list[str],
    test_size: float,
    cv_folds: int,
    n_estimators: int,
    seed: int = 42,
) -> tuple[RandomForestClassifier, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
    )

    # Cross-validation on training set
    print(f"  Running {cv_folds}-fold stratified CV on training set...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_results = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring=["accuracy", "f1_macro", "f1_weighted"],
        n_jobs=-1,
    )
    print(
        f"  CV accuracy:    {cv_results['test_accuracy'].mean():.3f}"
        f" ± {cv_results['test_accuracy'].std():.3f}"
    )
    print(
        f"  CV F1 (macro):  {cv_results['test_f1_macro'].mean():.3f}"
        f" ± {cv_results['test_f1_macro'].std():.3f}"
    )

    # Final fit on full training set, evaluate on held-out test set
    print(f"  Fitting final model on {X_train.shape[0]} training samples...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred)

    return model, {
        "cv": cv_results,
        "report": report,
        "confusion_matrix": cm,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred": y_pred,
    }


# Feature importances


def top_feature_importances(
    model: RandomForestClassifier,
    feature_names: list[str],
    top_n: int,
) -> pd.DataFrame:
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    return pd.DataFrame(
        {"feature": [feature_names[i] for i in idx], "importance": imp[idx]}
    )


# Output


def save_results(
    results: dict,
    model: RandomForestClassifier,
    feature_names: list[str],
    class_names: list[str],
    data_dir: Path,
    plots_dir: Path,
    top_n: int,
    save_model: bool,
) -> None:
    from plot_model_performance import (
        plot_confusion_matrix,
        plot_feature_importances,
        plot_cv_scores,
    )

    cv = results["cv"]
    report = results["report"]

    # Classification report
    report_path = data_dir / "ern_prediction_report.txt"
    with open(report_path, "w") as fh:
        fh.write("ERN PREDICTION — RANDOM FOREST\n")
        fh.write("=" * 50 + "\n\n")

        fh.write("Cross-validation (training set):\n")
        fh.write(f"  Accuracy:   {cv['test_accuracy'].mean():.4f} ± {cv['test_accuracy'].std():.4f}\n")
        fh.write(f"  F1 macro:   {cv['test_f1_macro'].mean():.4f} ± {cv['test_f1_macro'].std():.4f}\n")
        fh.write(f"  F1 weighted:{cv['test_f1_weighted'].mean():.4f} ± {cv['test_f1_weighted'].std():.4f}\n\n")

        fh.write("Hold-out test set:\n")
        fh.write(f"  Accuracy:   {report['accuracy']:.4f}\n")
        fh.write(f"  F1 macro:   {report['macro avg']['f1-score']:.4f}\n")
        fh.write(f"  F1 weighted:{report['weighted avg']['f1-score']:.4f}\n\n")

        fh.write("Per-class results:\n")
        fh.write(f"  {'ERN':<25}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}  {'Support':>7}\n")
        fh.write(f"  {'-' * 60}\n")
        for cls in class_names:
            r = report.get(cls, {})
            fh.write(
                f"  {cls:<25}  {r.get('precision', 0):>9.3f}"
                f"  {r.get('recall', 0):>6.3f}"
                f"  {r.get('f1-score', 0):>6.3f}"
                f"  {int(r.get('support', 0)):>7}\n"
            )
    print(f"Classification report:        {report_path}")

    # Confusion matrix
    cm = results["confusion_matrix"]
    cm_path = data_dir / "ern_confusion_matrix.csv"
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_path)
    print(f"Confusion matrix CSV:         {cm_path}")
    plot_confusion_matrix(cm, class_names, plots_dir / "ern_confusion_matrix.png")

    # Feature importances
    imp_df = top_feature_importances(model, feature_names, top_n)
    imp_path = data_dir / "ern_feature_importances.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"Feature importances CSV:      {imp_path}")
    plot_feature_importances(imp_df, top_n, plots_dir / "ern_feature_importances.png")

    # CV score plot
    plot_cv_scores(cv, plots_dir / "ern_cv_scores.png")

    # Model
    if save_model:
        try:
            import joblib
            model_path = data_dir / "ern_rf_model.joblib"
            joblib.dump(model, model_path)
            print(f"Model saved:                  {model_path}")
        except ImportError:
            print("[warn] joblib not installed; model not saved.", file=sys.stderr)


