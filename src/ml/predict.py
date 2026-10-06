"""Load a saved model bundle and write ERN predictions to CSV."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyhpo import Ontology
from scipy.sparse import csr_matrix

from utils.phenopackets import Sample, is_unlabeled


def calibrate_proba(proba: np.ndarray, log_w: np.ndarray, temperature: float) -> np.ndarray:
    # divide out the class weights used in training, then temperature-scale the logits
    logits = (np.log(np.clip(proba, 1e-12, 1.0)) - log_w) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


# this is here because it is a mandatory processing step before inferencing
def build_predict_feature_matrix(
    samples: list[Sample],
    vocabulary: list[str],
) -> csr_matrix:
    """Build a feature matrix for prediction, aligned to the training vocabulary."""
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
    return (
        csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, v))
        if rows
        else csr_matrix((n, v), dtype=np.float32)
    )


def predict_and_save(
    bundle_path: Path,
    samples: list[Sample],
    missing_only: bool,
    output_path: Path,
) -> None:
    try:
        import joblib
    except ImportError:
        print("joblib is required for prediction.", file=sys.stderr)
        return

    # load model, vocabulary and label encoder
    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    vocabulary: list[str] = bundle["vocabulary"]
    le = bundle["label_encoder"]

    if missing_only:
        predict_samples = [s for s in samples if is_unlabeled(s.ern)]
        print(f"Found {len(predict_samples)} samples with unlabeled ERN")
    else:
        predict_samples = list(samples)
        print(f"Predicting ERN for {len(predict_samples)} samples")

    if not predict_samples:
        print("No samples to predict.", file=sys.stderr)
        return

    print(f"Building feature matrix aligned to training vocabulary ({len(vocabulary)} terms).")
    X = build_predict_feature_matrix(predict_samples, vocabulary)

    # predict class and probability distribution
    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    ern_pred = le.inverse_transform(y_pred)
    # probability of the predicted ERN, raw and calibrated
    pred_idx = (np.arange(len(y_pred)), np.searchsorted(model.classes_, y_pred))
    confidence_raw = proba[pred_idx]
    if "calibration" in bundle:
        proba_cal = calibrate_proba(proba, bundle["calibration"]["log_w"], bundle["calibration"]["temperature"])
        confidence_cal = proba_cal[pred_idx]
    else:
        print("[warn] model bundle has no calibration, retrain to get calibrated confidence.", file=sys.stderr)
        confidence_cal = np.full(len(y_pred), np.nan)

    df = pd.DataFrame({
        "sample": [s.name for s in predict_samples],
        "submitter": [s.submitter for s in predict_samples],
        "current_ern": [s.ern for s in predict_samples],
        "ern_was_missing": [is_unlabeled(s.ern) for s in predict_samples],
        "predicted_ern": ern_pred,
        "confidence_raw": confidence_raw.round(4),
        "confidence_calibrated": confidence_cal.round(4),
    })
    df.to_csv(output_path, index=False)
    print(f"Predictions saved:  {output_path}")
    print(f"{len(df)} samples predicted across {df['predicted_ern'].nunique()} ERN classes")
