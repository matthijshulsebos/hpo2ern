"""Load a saved model bundle and write ERN predictions to CSV."""

import sys
from pathlib import Path

import pandas as pd

from utils.phenopackets import Sample
from ml.features import build_predict_feature_matrix
from ml.train import NOT_APPLICABLE


def predict_and_save(
    bundle_path: Path,
    samples: list[Sample],
    missing_only: bool,
    output_path: Path,
) -> None:
    try:
        import joblib
    except ImportError:
        print("[error] joblib is required for prediction; install it first.", file=sys.stderr)
        return

    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    vocabulary: list[str] = bundle["vocabulary"]
    le = bundle["label_encoder"]

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
    X = build_predict_feature_matrix(predict_samples, vocabulary)

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
