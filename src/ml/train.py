"""LightGBM training with randomised hyperparameter search for ERN prediction."""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from utils.phenopackets import Sample


NOT_APPLICABLE = "Not_Applicable"
DEFAULT_MODEL_PATH: Path = Path.home() / ".cache" / "find-ern" / "ern_lgbm_model.joblib"

DEFAULT_PARAM_SPACE: dict = {
    "n_estimators":      [100, 200, 300, 500, 800],
    "learning_rate":     [0.01, 0.05, 0.1, 0.2, 0.3],
    "num_leaves":        [20, 31, 50, 80, 100],
    "max_depth":         [-1, 5, 7, 10],
    "min_child_samples": [5, 10, 20, 50],
    "subsample":         [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":  [0.7, 0.8, 0.9, 1.0],
    "reg_alpha":         [0.0, 0.1, 0.5, 1.0],
    "reg_lambda":        [0.0, 0.1, 0.5, 1.0],
}


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


def train_and_evaluate(
    X: csr_matrix,
    y: np.ndarray,
    class_names: list[str],
    test_size: float,
    cv_folds: int,
    n_iter: int,
    param_space: dict,
    class_weight: str | None = "balanced",
    seed: int = 42,
) -> tuple[LGBMClassifier, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    base_model = LGBMClassifier(
        class_weight=class_weight,
        importance_type="gain",
        random_state=seed,
        verbose=-1,
        n_jobs=1,
    )

    scoring = {
        "accuracy":   "accuracy",
        "f1_macro":   "f1_macro",
        "f1_weighted": "f1_weighted",
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    print(f"  Randomised search: {n_iter} iterations × {cv_folds}-fold CV...")
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_space,
        n_iter=n_iter,
        scoring=scoring,
        refit="f1_macro",
        cv=cv,
        n_jobs=-1,
        random_state=seed,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_idx = search.best_index_
    cv_fold_scores = {
        f"test_{m}": np.array([
            search.cv_results_[f"split{i}_test_{m}"][best_idx]
            for i in range(cv_folds)
        ])
        for m in ["accuracy", "f1_macro", "f1_weighted"]
    }

    print(f"  Best params : {search.best_params_}")
    print(
        f"  Best CV F1 macro  : {cv_fold_scores['test_f1_macro'].mean():.3f}"
        f" ± {cv_fold_scores['test_f1_macro'].std():.3f}"
    )
    print(
        f"  Best CV accuracy  : {cv_fold_scores['test_accuracy'].mean():.3f}"
        f" ± {cv_fold_scores['test_accuracy'].std():.3f}"
    )

    model: LGBMClassifier = search.best_estimator_
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred)

    return model, {
        "best_params": search.best_params_,
        "cv": cv_fold_scores,
        "report": report,
        "confusion_matrix": cm,
    }


def top_feature_importances(
    model: LGBMClassifier, feature_names: list[str], top_n: int
) -> pd.DataFrame:
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    return pd.DataFrame(
        {"feature": [feature_names[i] for i in idx], "importance": imp[idx]}
    )


def save_model_bundle(
    model: LGBMClassifier,
    vocabulary: list[str],
    le: LabelEncoder,
    path: Path = DEFAULT_MODEL_PATH,
) -> "Path | None":
    try:
        import joblib
    except ImportError:
        print("[warn] joblib not installed; model bundle not saved.", file=sys.stderr)
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "vocabulary": vocabulary, "label_encoder": le}, path)
    return path


def save_results(
    results: dict,
    model: LGBMClassifier,
    feature_names: list[str],
    class_names: list[str],
    data_dir: Path,
    plots_dir: Path,
    top_n: int,
) -> None:
    from explore.model_plots import plot_confusion_matrix, plot_cv_scores, plot_feature_importances

    cv = results["cv"]
    report = results["report"]
    best_params = results["best_params"]

    report_path = data_dir / "ern_prediction_report.txt"
    with open(report_path, "w") as fh:
        fh.write("ERN PREDICTION — LIGHTGBM\n")
        fh.write("=" * 50 + "\n\n")
        fh.write("Best hyperparameters:\n")
        for k, v in sorted(best_params.items()):
            fh.write(f"  {k:<25} {v}\n")
        fh.write("\nCross-validation (best parameter combination):\n")
        fh.write(f"  Accuracy:    {cv['test_accuracy'].mean():.4f} ± {cv['test_accuracy'].std():.4f}\n")
        fh.write(f"  F1 macro:    {cv['test_f1_macro'].mean():.4f} ± {cv['test_f1_macro'].std():.4f}\n")
        fh.write(f"  F1 weighted: {cv['test_f1_weighted'].mean():.4f} ± {cv['test_f1_weighted'].std():.4f}\n\n")
        fh.write("Hold-out test set:\n")
        fh.write(f"  Accuracy:    {report['accuracy']:.4f}\n")
        fh.write(f"  F1 macro:    {report['macro avg']['f1-score']:.4f}\n")
        fh.write(f"  F1 weighted: {report['weighted avg']['f1-score']:.4f}\n\n")
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

    cm = results["confusion_matrix"]
    cm_path = data_dir / "ern_confusion_matrix.csv"
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_path)
    print(f"Confusion matrix CSV:         {cm_path}")
    plot_confusion_matrix(cm, class_names, plots_dir / "ern_confusion_matrix.png")

    imp_df = top_feature_importances(model, feature_names, top_n)
    imp_df.to_csv(data_dir / "ern_feature_importances.csv", index=False)
    print(f"Feature importances CSV:      {data_dir / 'ern_feature_importances.csv'}")
    plot_feature_importances(imp_df, top_n, plots_dir / "ern_feature_importances.png")

    plot_cv_scores(cv, plots_dir / "ern_cv_scores.png")
