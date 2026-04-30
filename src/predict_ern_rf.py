#!/usr/bin/env python3
"""
Random Forest model for predicting ERN from HPO terms (and optionally submitter).

This module is imported by main.py and is not intended to be run standalone.
Entry points:
  - filter_by_class_size   Drop under-represented ERN classes before training
  - build_feature_matrix   HPO binary matrix + optional one-hot submitter (training)
  - build_predict_feature_matrix  Aligned feature matrix for inference
  - train_and_evaluate     Train RF with CV + held-out test evaluation
  - save_results           Write classification report, confusion matrix, feature importances
  - save_model_bundle      Persist model + encoders to DEFAULT_MODEL_PATH (or custom path)
  - predict_and_save       Load a saved bundle and write ERN predictions to CSV

Dependencies:
    pyhpo scikit-learn scipy numpy pandas matplotlib joblib
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

from pyhpo import Ontology
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix

from phenopackets import Sample, build_ancestry_matrix


# Feature construction


def build_feature_matrix(
    samples: list[Sample],
    vocabulary: list[str],
    X_hpo: csr_matrix,
    include_submitter: bool,
) -> tuple[csr_matrix, list[str], "OneHotEncoder | None"]:
    """Stack HPO binary matrix with optional one-hot submitter.

    Returns (X, feature_names, ohe) where ohe is the fitted OneHotEncoder
    (or None when include_submitter is False). Save ohe in the model bundle
    so it can be reused during prediction.
    """
    feature_names = [_term_label(t) for t in vocabulary]

    if not include_submitter:
        return X_hpo, feature_names, None

    submitters = np.array([s.submitter for s in samples]).reshape(-1, 1)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X_sub = ohe.fit_transform(submitters)
    sub_names = [f"submitter={v}" for v in ohe.categories_[0]]

    print(f"  Submitter one-hot: {X_sub.shape[1]} categories")
    return sp_hstack([X_hpo, X_sub], format="csr"), feature_names + sub_names, ohe


def build_predict_feature_matrix(
    samples: list[Sample],
    vocabulary: list[str],
    ohe: "OneHotEncoder | None",
    include_submitter: bool,
) -> csr_matrix:
    """Build a feature matrix for *prediction*, aligned to the training vocabulary.

    HPO columns outside the training vocabulary are silently ignored (zero).
    Unknown submitter categories are handled by the saved OHE (sparse zeros).
    """
    vocab_idx = {t: i for i, t in enumerate(vocabulary)}
    rows: list[int] = []
    cols: list[int] = []

    for i, sample in enumerate(samples):
        term_set: set[str] = set()
        for tid in sample.terms:
            try:
                hpo_term = Ontology.get_hpo_object(tid)
                term_set.add(hpo_term.id)
                for ancestor in hpo_term.all_parents:
                    term_set.add(ancestor.id)
            except RuntimeError:
                pass
        for t in term_set:
            if t in vocab_idx:
                rows.append(i)
                cols.append(vocab_idx[t])

    n, v = len(samples), len(vocabulary)
    if rows:
        X_hpo = csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(n, v),
        )
    else:
        X_hpo = csr_matrix((n, v), dtype=np.float32)

    if not include_submitter or ohe is None:
        return X_hpo

    submitters = np.array([s.submitter for s in samples]).reshape(-1, 1)
    X_sub = ohe.transform(submitters)
    return sp_hstack([X_hpo, X_sub], format="csr")


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


# Model bundle


NOT_APPLICABLE = "Not applicable"
DEFAULT_MODEL_PATH: Path = Path.home() / ".cache" / "find-ern" / "ern_rf_model.joblib"


def save_model_bundle(
    model: RandomForestClassifier,
    vocabulary: list[str],
    le: LabelEncoder,
    ohe: "OneHotEncoder | None",
    include_submitter: bool,
    path: Path = DEFAULT_MODEL_PATH,
) -> "Path | None":
    """Persist a self-contained model bundle (model + vocabulary + encoders).

    Saved to *path*, which defaults to DEFAULT_MODEL_PATH so that --predict
    can find the model without needing --model to be specified.
    """
    try:
        import joblib
    except ImportError:
        print("[warn] joblib not installed; model bundle not saved.", file=sys.stderr)
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "vocabulary": vocabulary,
        "label_encoder": le,
        "ohe": ohe,
        "include_submitter": include_submitter,
    }
    joblib.dump(bundle, path)
    return path


def predict_and_save(
    bundle_path: Path,
    samples: list[Sample],
    missing_only: bool,
    output_path: Path,
) -> None:
    """Load a saved model bundle and write ERN predictions to *output_path*.

    If *missing_only* is True, only samples whose current ERN equals
    'Not applicable' are predicted; otherwise all samples are predicted.
    """
    try:
        import joblib
    except ImportError:
        print("[error] joblib is required for prediction; install it first.", file=sys.stderr)
        return

    bundle = joblib.load(bundle_path)
    model: RandomForestClassifier = bundle["model"]
    vocabulary: list[str] = bundle["vocabulary"]
    le: LabelEncoder = bundle["label_encoder"]
    ohe = bundle["ohe"]
    include_submitter: bool = bundle["include_submitter"]

    if missing_only:
        predict_samples = [s for s in samples if s.ern == NOT_APPLICABLE]
        print(f"  Found {len(predict_samples)} samples with ERN = '{NOT_APPLICABLE}'")
    else:
        predict_samples = list(samples)
        print(f"  Predicting ERN for {len(predict_samples)} samples")

    if not predict_samples:
        print("[skip] No samples to predict.", file=sys.stderr)
        return

    print(f"  Building feature matrix aligned to training vocabulary ({len(vocabulary)} terms)...")
    X = build_predict_feature_matrix(predict_samples, vocabulary, ohe, include_submitter)

    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    ern_pred = le.inverse_transform(y_pred)
    confidence = proba.max(axis=1)

    df = pd.DataFrame({
        "sample": [s.name for s in predict_samples],
        "current_ern": [s.ern for s in predict_samples],
        "predicted_ern": ern_pred,
        "confidence": confidence.round(4),
    })
    df.to_csv(output_path, index=False)
    print(f"  Predictions saved:  {output_path}")
    print(f"  {len(df)} samples predicted across {df['predicted_ern'].nunique()} ERN classes")


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




