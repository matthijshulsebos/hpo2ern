#!/usr/bin/env python3
"""
ERDERA ERN prediction pipeline.

Hyperparameters are read from config.toml (repo root).

Usage:
    python src/main.py --input data/ --train
    python src/main.py --input data/ --predict
    python src/main.py --input data/ --predict --predict-missing-only
    python src/main.py --input data/ --predict --model /path/to/model.joblib
    python src/main.py --input data/ --train --predict
"""

import argparse
import datetime
import os
import sys
import tomllib
import warnings
from pathlib import Path

import numpy as np
from pyhpo import Ontology
from sklearn.preprocessing import LabelEncoder

from utils.phenopackets import load_all_phenopackets, validate_terms, build_ancestry_matrix, is_unlabeled
from ml.train import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PARAM_SPACE,
    train_and_evaluate,
    save_results as save_train_results,
    save_model_bundle,
)
from ml.predict import predict_and_save


# Suppress LightGBM warnings
os.environ.setdefault("OMP_DISPLAY_ENV", "FALSE")
os.environ.setdefault("KMP_WARNINGS", "FALSE")
warnings.filterwarnings("ignore", category=UserWarning)

CONFIG_PATH = Path("config.toml")


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ERDERA ERN prediction pipeline: train · predict",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  "-i", required=True, type=Path, metavar="DIR",
                   help="Directory with phenopacket JSON files")
    p.add_argument("--output", "-o", type=Path, default=None, metavar="DIR",
                   help="Output directory (default: output/<timestamp>/)")
    p.add_argument("--config", type=Path, default=CONFIG_PATH, metavar="FILE",
                   help="TOML config file")

    steps = p.add_argument_group("Steps")
    steps.add_argument("--train",   action="store_true", help="Train the ERN prediction model")
    steps.add_argument("--predict", action="store_true", help="Predict ERNs using a saved model")

    pr_g = p.add_argument_group("Prediction")
    pr_g.add_argument("--model", type=Path, default=None, metavar="FILE",
                      help="Saved model bundle (.joblib); defaults to config or ~/.cache path")
    pr_g.add_argument("--predict-missing-only", action="store_true",
                      help="Only score samples whose ERN is 'Not_Applicable'")

    return p.parse_args()


def _section(title: str) -> None:
    print(f"\n {title}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    tr = cfg.get("training", {})
    n_iter         = cfg.get("tuning",   {}).get("n_iter",        50)
    cv_folds       = cfg.get("tuning",   {}).get("cv_folds",       5)
    param_space    = cfg.get("tuning",   {}).get("param_space",    DEFAULT_PARAM_SPACE)
    optuna_jobs    = cfg.get("tuning",   {}).get("optuna_jobs",     4)
    lgbm_threads   = cfg.get("tuning",   {}).get("lgbm_threads",   4)
    test_size      = tr.get("test_size",      0.20)
    top_features   = tr.get("top_features",   30)
    model_path     = args.model or Path(
        cfg.get("paths", {}).get("model", str(DEFAULT_MODEL_PATH))
    ).expanduser()

    if not args.train and not args.predict:
        sys.exit("ERROR: specify at least one of --train or --predict.")

    if args.output is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = Path("output") / timestamp
    else:
        run_dir = args.output
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = run_dir / "data"
    data_dir.mkdir(exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    active = [s for s, on in [("train", args.train), ("predict", args.predict)] if on]
    print(f"Output : {run_dir}/")
    print(f"Steps  : {' · '.join(active)}")

    _section("Loading")
    # Loads the HPO ontology into memory (initialisation)
    Ontology()
    samples = load_all_phenopackets(args.input)
    samples = validate_terms(samples)

    if args.train:
        _section("Training")

        labeled = [s for s in samples if not is_unlabeled(s.ern)]
        n_unlabeled = len(samples) - len(labeled)
        if n_unlabeled:
            print(f"Excluded {n_unlabeled} samples with unlabeled ERN (no label)")

        class_names = sorted(set(s.ern for s in labeled))
        print(f"Training on {len(labeled)} patients across {len(class_names)} ERN classes")
        if len(class_names) < 2:
            sys.exit("[error] Fewer than 2 ERN classes, cannot train.")

        X_hpo, vocabulary = build_ancestry_matrix(labeled)
        feature_names = []
        for t in vocabulary:
            try:
                feature_names.append(f"{t} {Ontology.get_hpo_object(t).name}")
            except RuntimeError:
                feature_names.append(t)
        X = X_hpo
        print(f"  Features : {X.shape[1]}")

        le = LabelEncoder()
        le.fit(class_names)
        y = le.transform([s.ern for s in labeled])

        model, results = train_and_evaluate(
            X, y, class_names,
            test_size=test_size,
            cv_folds=cv_folds,
            n_iter=n_iter,
            param_space=param_space,
            optuna_jobs=optuna_jobs,
            lgbm_threads=lgbm_threads,
        )

        print(f"  Test accuracy : {results['accuracy']:.4f}")
        print(f"  Test F1 macro : {results['report']['macro avg']['f1-score']:.4f}")

        # patch le to include "Other" if small classes were relabeled during training
        if "Other" in results["class_names_remapped"]:
            le.classes_ = np.append(le.classes_, "Other")

        save_train_results(results, model, feature_names, data_dir, plots_dir, top_features)

        bundle_path = save_model_bundle(model, vocabulary, le, path=model_path)
        if bundle_path:
            print(f"  Model bundle  : {bundle_path}")

    if args.predict:
        _section("Predicting")
        print(f"  Model bundle  : {model_path}")
        if not model_path.exists():
            sys.exit(
                f"[error] Model bundle not found: {model_path}\n"
                "        Run --train first, or pass --model."
            )
        predict_and_save(
            bundle_path=model_path,
            samples=samples,
            missing_only=args.predict_missing_only,
            output_path=data_dir / "ern_predictions.csv",
        )


if __name__ == "__main__":
    main()
