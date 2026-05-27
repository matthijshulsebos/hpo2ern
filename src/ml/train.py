"""LightGBM training with randomised hyperparameter search for ERN prediction."""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy.sparse import csr_matrix
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)


DEFAULT_MODEL_PATH: Path = Path.home() / ".cache" / "find-ern" / "ern_lgbm_model.joblib"

DEFAULT_PARAM_SPACE: dict = {
    # n_estimators is no longer a search parameter: early stopping determines
    # the optimal tree count automatically during each CV fold
    "learning_rate":     [0.03, 0.05, 0.1, 0.15],
    "num_leaves":        [31, 63, 127, 255],
    "max_depth":         [-1, 7, 10, 15],
    # lower floor: tail classes may have only ~10-30 samples; 200 prevents any splits
    "min_child_samples": [5, 10, 20, 50],
    "min_child_weight":  [1e-3, 1e-2, 1e-1, 1.0],
    # small gains are legitimate on rare classes; cap at 0.1 not 1.0
    "min_split_gain":    [0.0, 0.01, 0.05, 0.1],
    # subsample_freq=1 activates row subsampling (without it, subsample has no effect)
    "subsample":         [0.7, 0.8, 0.9, 1.0],
    "subsample_freq":    [1],
    # raised floor: at ~1% feature density, 0.3 column sampling leaves most nodes
    # with near-zero signal; 0.5 is the practical minimum
    "colsample_bytree":  [0.5, 0.7, 0.9, 1.0],
    "reg_alpha":         [0.0, 0.1, 0.5, 1.0],
    "reg_lambda":        [0.0, 0.1, 1.0, 5.0],
}


def train_and_evaluate(
    X: csr_matrix,
    y: np.ndarray,
    class_names: list[str],
    test_size: float,
    cv_folds: int,
    n_iter: int,
    param_space: dict,
    min_samples: int = 10,
    weight_cap_factor: float = 5.0,
    seed: int = 42,
) -> tuple[LGBMClassifier, dict]:
    # drop classes with fewer than min_samples; they can't support stratified CV
    # and would destabilise sample weights if kept
    class_counts = np.bincount(y, minlength=len(class_names))
    large_mask = np.array([class_counts[yi] >= min_samples for yi in y])

    X_large, y_large = X[large_mask], y[large_mask]

    dropped = [class_names[i] for i in range(len(class_names)) if 0 < class_counts[i] < min_samples]
    if dropped:
        print(f"  {len(dropped)} ERN class(es) with < {min_samples} samples excluded: {', '.join(dropped)}")

    if X_large.shape[0] == 0:
        sys.exit("[error] No ERN class has enough samples for stratified CV.")

    # stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_large, y_large, test_size=test_size, stratify=y_large, random_state=seed
    )
    X_cv, y_cv = X_train, y_train

    # capped balanced sample weights — prevents tiny classes from dominating gradients
    large_counts = np.bincount(y_large, minlength=len(class_names))
    raw_w = np.where(
        large_counts > 0,
        len(y_large) / (len(class_names) * large_counts.astype(float)),
        0.0,
    )
    cap = np.median(raw_w[raw_w > 0]) * weight_cap_factor
    capped = np.minimum(raw_w, cap)
    sw_train = capped[y_train]

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = {k: trial.suggest_categorical(k, v) for k, v in param_space.items()}
        fold_acc, fold_f1m, fold_f1w, best_iters = [], [], [], []
        for train_idx, val_idx in cv.split(X_cv, y_cv):
            X_tr, X_val = X_cv[train_idx], X_cv[val_idx]
            y_tr, y_val = y_cv[train_idx], y_cv[val_idx]
            # fresh instance per fold so early-stopping state doesn't carry over
            m = LGBMClassifier(
                importance_type="gain",
                random_state=seed,
                verbose=-1,
                n_jobs=-1,
                n_estimators=1000,
                **params,
            )
            m.fit(
                X_tr, y_tr,
                sample_weight=sw_train[train_idx],
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
            best_iters.append(m.best_iteration_)
            y_pred = m.predict(X_val)
            fold_acc.append(accuracy_score(y_val, y_pred))
            fold_f1m.append(f1_score(y_val, y_pred, average="macro", zero_division=0))
            fold_f1w.append(f1_score(y_val, y_pred, average="weighted", zero_division=0))
        trial.set_user_attr("fold_accuracy", fold_acc)
        trial.set_user_attr("fold_f1_macro", fold_f1m)
        trial.set_user_attr("fold_f1_weighted", fold_f1w)
        trial.set_user_attr("mean_best_iter", int(np.mean(best_iters)))
        return float(np.mean(fold_f1m))

    width = len(str(n_iter))

    def _log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        improved = trial.value == study.best_value
        marker = " *" if improved else ""
        trees = trial.user_attrs.get("mean_best_iter", "?")
        print(
            f"[{trial.number + 1:{width}d}/{n_iter}]"
            f"  F1={trial.value:.4f}"
            f"  best={study.best_value:.4f}{marker}"
            f"  trees={trees}"
        )

    print(f"Optuna TPE search: {n_iter} trials x {cv_folds}-fold CV...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    # n_jobs=1: trials run sequentially
    # LightGBM's internal thread pool is initialised concurrently across workers
    study.optimize(objective, n_trials=n_iter, n_jobs=1, show_progress_bar=False, callbacks=[_log_trial])

    best_trial = study.best_trial
    best_params = best_trial.params
    cv_fold_scores = {
        "test_accuracy": np.array(best_trial.user_attrs["fold_accuracy"]),
        "test_f1_macro": np.array(best_trial.user_attrs["fold_f1_macro"]),
        "test_f1_weighted": np.array(best_trial.user_attrs["fold_f1_weighted"]),
    }

    print(f"Best params : {best_params}")
    print(
        f"Best CV F1 macro  : {cv_fold_scores['test_f1_macro'].mean():.3f}"
        f" ± {cv_fold_scores['test_f1_macro'].std():.3f}"
    )
    print(
        f"Best CV accuracy  : {cv_fold_scores['test_accuracy'].mean():.3f}"
        f" ± {cv_fold_scores['test_accuracy'].std():.3f}"
    )

    # refit best model on full training set using mean CV tree count
    best_iter = best_trial.user_attrs.get("mean_best_iter", 300)
    model = LGBMClassifier(
        importance_type="gain",
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
        n_estimators=best_iter,
        **best_params,
    )
    model.fit(X_train, y_train, sample_weight=sw_train)

    # evaluate on large-class test set
    large_class_indices = sorted(set(y_large.tolist()))
    eval_class_names = [class_names[i] for i in large_class_indices]
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test, y_pred,
        labels=large_class_indices,
        target_names=eval_class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=large_class_indices)
    acc = accuracy_score(y_test, y_pred)

    return model, {
        "best_params": best_params,
        "cv": cv_fold_scores,
        "report": report,
        "confusion_matrix": cm,
        "eval_class_names": eval_class_names,
        "accuracy": acc,
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
        print("[warn] joblib not installed: model bundle not saved.", file=sys.stderr)
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "vocabulary": vocabulary, "label_encoder": le}, path)
    return path


def save_results(
    results: dict,
    model: LGBMClassifier,
    feature_names: list[str],
    data_dir: Path,
    plots_dir: Path,
    top_n: int,
) -> None:
    from ml.plots import plot_confusion_matrix, plot_cv_scores, plot_feature_importances

    cv = results["cv"]
    cm = results["confusion_matrix"]
    report = results["report"]
    eval_class_names = results["eval_class_names"]

    # per-class metrics CSV
    report_rows = [
        {
            "ern": cls,
            "precision": report[cls]["precision"],
            "recall": report[cls]["recall"],
            "f1_score": report[cls]["f1-score"],
            "support": int(report[cls]["support"]),
        }
        for cls in eval_class_names
        if cls in report
    ]
    metrics_path = data_dir / "ern_per_class_metrics.csv"
    pd.DataFrame(report_rows).to_csv(metrics_path, index=False)
    print(f"Per-class metrics:            {metrics_path}")
    cm_path = data_dir / "ern_confusion_matrix.csv"
    pd.DataFrame(cm, index=eval_class_names, columns=eval_class_names).to_csv(cm_path)
    plot_confusion_matrix(cm, eval_class_names, plots_dir / "ern_confusion_matrix.png")

    # Feature importance plot generation
    imp_df = top_feature_importances(model, feature_names, top_n)
    imp_df.to_csv(data_dir / "ern_feature_importances.csv", index=False)
    plot_feature_importances(imp_df, top_n, plots_dir / "ern_feature_importances.png")

    # Cross-validation scores plot generation
    plot_cv_scores(cv, plots_dir / "ern_cv_scores.png")
