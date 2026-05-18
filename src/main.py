#!/usr/bin/env python3
"""
ERDERA ERN prediction pipeline.

Loads hyperparameters from config.toml (repo root). Any CLI flag overrides
the corresponding config value.

Usage:
    python src/main.py --input data/ --train
    python src/main.py --input data/ --predict
    python src/main.py --input data/ --predict --predict-missing-only
    python src/main.py --input data/ --predict --model /path/to/model.joblib
    python src/main.py --input data/ --train --predict   # train then immediately predict
"""

import argparse
import datetime
import os
import sys
import tomllib
import warnings
from pathlib import Path

os.environ.setdefault("OMP_DISPLAY_ENV", "FALSE")
os.environ.setdefault("KMP_WARNINGS", "FALSE")
warnings.filterwarnings("ignore", category=UserWarning)

from pyhpo import Ontology
from sklearn.preprocessing import LabelEncoder

from utils.phenopackets import load_all_phenopackets, validate_terms, build_ancestry_matrix
from ml.features import build_feature_matrix
from ml.train import (
    NOT_APPLICABLE,
    is_unlabeled,
    DEFAULT_MODEL_PATH,
    DEFAULT_PARAM_SPACE,
    filter_by_class_size,
    train_and_evaluate,
    save_results as save_train_results,
    save_model_bundle,
)
from ml.predict import predict_and_save


CONFIG_PATH = Path("config.toml")


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def parse_args(cfg: dict) -> argparse.Namespace:
    m  = cfg.get("model",    {})
    tr = cfg.get("training", {})
    pa = cfg.get("paths",    {})

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

    tr_g = p.add_argument_group("Training  (override config.toml [training] / [tuning])")
    tr_g.add_argument("--n-iter",       type=int,  default=cfg.get("tuning", {}).get("n_iter",  50),
                      help="Randomised search iterations")
    tr_g.add_argument("--class-weight", type=str,  default=m.get("class_weight", "balanced"))
    tr_g.add_argument("--test-size",    type=float, default=tr.get("test_size",  0.20))
    tr_g.add_argument("--cv-folds",       type=int,   default=tr.get("cv_folds",      5))
    tr_g.add_argument("--min-class-size", type=int,   default=tr.get("min_class_size", 10))
    tr_g.add_argument("--top-features",   type=int,   default=tr.get("top_features",  30))
    tr_g.add_argument("--no-save-model",  action="store_true",
                      help="Do not persist the fitted model bundle")

    pr_g = p.add_argument_group("Prediction")
    default_model = Path(pa.get("model", str(DEFAULT_MODEL_PATH))).expanduser()
    pr_g.add_argument("--model", type=Path, default=default_model, metavar="FILE",
                      help="Saved model bundle (.joblib)")
    pr_g.add_argument("--predict-missing-only", action="store_true",
                      help="Only score samples whose ERN is 'Not_Applicable'")

    return p.parse_args()


def _section(title: str) -> None:
    print(f"\n {title}")


def main() -> None:
    # Load config first so argparse defaults reflect it
    cfg_path = CONFIG_PATH
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            cfg_path = Path(sys.argv[i + 1])
    cfg = load_config(cfg_path)

    args = parse_args(cfg)

    if not args.train and not args.predict:
        sys.exit("ERROR: specify at least one of --train or --predict.")

    if args.output is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = Path("output") / timestamp
    else:
        run_dir = args.output
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir  = run_dir / "data";  data_dir.mkdir(exist_ok=True)
    plots_dir = run_dir / "plots"; plots_dir.mkdir(exist_ok=True)

    active = [s for s, on in [("train", args.train), ("predict", args.predict)] if on]
    print(f"Output : {run_dir}/")
    print(f"Steps  : {' · '.join(active)}")

    _section("Loading")
    Ontology()
    samples = load_all_phenopackets(args.input)
    samples = validate_terms(samples)

    if args.train:
        _section("Training")

        labeled = [s for s in samples if not is_unlabeled(s.ern)]
        n_unlabeled = len(samples) - len(labeled)
        if n_unlabeled:
            print(f"  Excluded {n_unlabeled} samples with unlabeled ERN (no label)")

        train_samples, class_names = filter_by_class_size(labeled, args.min_class_size)
        if len(class_names) < 2:
            sys.exit("[error] Fewer than 2 ERN classes remain after filtering; cannot train.")

        X_hpo, vocabulary = build_ancestry_matrix(train_samples)
        X, feature_names = build_feature_matrix(vocabulary, X_hpo)
        print(f"  Features : {X.shape[1]}")

        le = LabelEncoder()
        le.fit(class_names)
        y = le.transform([s.ern for s in train_samples])

        param_space = cfg.get("tuning", {}).get("param_space", DEFAULT_PARAM_SPACE)
        cv_folds    = cfg.get("tuning", {}).get("cv_folds", 5)

        model, results = train_and_evaluate(
            X, y, class_names,
            test_size=args.test_size,
            cv_folds=cv_folds,
            n_iter=args.n_iter,
            param_space=param_space,
            class_weight=args.class_weight,
        )

        report = results["report"]
        print(f"  Test accuracy : {report['accuracy']:.4f}")
        print(f"  Test F1 macro : {report['macro avg']['f1-score']:.4f}")

        save_train_results(results, model, feature_names, class_names,
                           data_dir, plots_dir, args.top_features)

        if not args.no_save_model:
            bundle_path = save_model_bundle(model, vocabulary, le, path=args.model)
            if bundle_path:
                print(f"  Model bundle  : {bundle_path}")

    if args.predict:
        _section("Predicting")
        print(f"  Model bundle  : {args.model}")
        if not args.model.exists():
            sys.exit(
                f"[error] Model bundle not found: {args.model}\n"
                "        Run --train first, or pass --model."
            )
        predict_and_save(
            bundle_path=args.model,
            samples=samples,
            missing_only=args.predict_missing_only,
            output_path=data_dir / "ern_predictions.csv",
        )


if __name__ == "__main__":
    main()
