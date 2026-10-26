"""LightGBM training with Optuna TPE hyperparameter search for ERN prediction."""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from ml.predict import calibrate_proba

optuna.logging.set_verbosity(optuna.logging.WARNING)


# I prefer .cache but you can store it elsewhere
DEFAULT_MODEL_PATH: Path = Path.home() / ".cache" / "find-ern" / "ern_lgbm_model.joblib"

DEFAULT_PARAM_SPACE: dict = {
    "learning_rate":     [0.03, 0.05, 0.1, 0.15],
    "num_leaves":        [31, 63, 127, 255],
    "max_depth":         [-1, 7, 10, 15],
    "min_child_samples": [5, 10, 20, 50],
    "min_child_weight":  [1e-3, 1e-2, 1e-1, 1.0],
    "min_split_gain":    [0.0, 0.01, 0.05, 0.1],
    "subsample":         [0.7, 0.8, 0.9, 1.0],
    "subsample_freq":    [1],
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
    min_samples: int = 20,
    weight_cap_factors: list[float] | None = None,
    optuna_jobs: int = 4,
    lgbm_threads: int = 4,
    seed: int = 42,
) -> tuple[LGBMClassifier, dict]:
    # remap classes with fewer than min_samples to "Not available" so no samples are discarded
    class_counts = np.bincount(y, minlength=len(class_names))
    small_set = {i for i in range(len(class_names)) if 0 < class_counts[i] < min_samples}

    y_remapped = y.copy()
    class_names_remapped = list(class_names)

    if small_set:
        if "Not available" not in class_names_remapped:
            class_names_remapped.append("Not available")
        other_idx = class_names_remapped.index("Not available")
        y_remapped = np.where(np.isin(y_remapped, list(small_set)), other_idx, y_remapped)

    if X.shape[0] == 0:
        sys.exit("No samples available for training.")

    # stratified train/test split
    X_train, X_test, y_train, y_test, sample_train_idx, sample_test_idx = train_test_split(
        X, y_remapped, np.arange(len(y_remapped)), test_size=test_size, stratify=y_remapped, random_state=seed
    )
    X_cv, y_cv = X_train, y_train

    # balanced base weights due to class imbalance
    remapped_counts = np.bincount(y_remapped, minlength=len(class_names_remapped))
    safe_counts = np.where(remapped_counts > 0, remapped_counts, 1).astype(float)
    raw_w = np.where(
        remapped_counts > 0,
        len(y_remapped) / (len(class_names_remapped) * safe_counts),
        0.0,
    )
    median_raw_w = float(np.median(raw_w[raw_w > 0]))
    _wcf_choices = weight_cap_factors if weight_cap_factors is not None else [2.0, 3.0, 5.0, 7.0, 10.0]

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # Optuna min/maxes the objective function for CV F1 macro
    def objective(trial: optuna.Trial) -> float:
        wcf = trial.suggest_categorical("weight_cap_factor", _wcf_choices)
        sw_trial = np.minimum(raw_w, median_raw_w * wcf)[y_train]
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
                n_jobs=lgbm_threads,
                n_estimators=1000,
                **params,
            )
            m.fit(
                X_tr, y_tr,
                sample_weight=sw_trial[train_idx],
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

        # store scores in Optuna trial
        trial.set_user_attr("fold_accuracy", fold_acc)
        trial.set_user_attr("fold_f1_macro", fold_f1m)
        trial.set_user_attr("fold_f1_weighted", fold_f1w)
        trial.set_user_attr("mean_best_iter", int(np.mean(best_iters)))

        return float(np.mean(fold_f1m))

    width = len(str(n_iter))

    # callback to show training progress
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

    print(f"Optuna TPE search: {n_iter} trials x {cv_folds}-fold CV ({optuna_jobs} parallel trials, {lgbm_threads} threads each).")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_iter, n_jobs=optuna_jobs, show_progress_bar=False, callbacks=[_log_trial])

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_wcf = best_params.pop("weight_cap_factor")
    sw_train = np.minimum(raw_w, median_raw_w * best_wcf)[y_train]
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

    # calibration: undo the class-weight prior shift, then fit one temperature on
    # out-of-fold probabilities so the test set stays untouched
    log_w = np.log(np.minimum(raw_w, median_raw_w * best_wcf)[model.classes_])
    oof = np.zeros((len(y_train), len(model.classes_)))
    for train_idx, val_idx in cv.split(X_train, y_train):
        m = LGBMClassifier(
            importance_type="gain",
            random_state=seed,
            verbose=-1,
            n_jobs=-1,
            n_estimators=best_iter,
            **best_params,
        )
        m.fit(X_train[train_idx], y_train[train_idx], sample_weight=sw_train[train_idx])
        oof[val_idx] = m.predict_proba(X_train[val_idx])
    train_cols = np.searchsorted(model.classes_, y_train)
    temperature = minimize_scalar(
        lambda t: -np.mean(np.log(calibrate_proba(oof, log_w, t)[np.arange(len(y_train)), train_cols])),
        bounds=(0.05, 20.0),
        method="bounded",
    ).x
    calibration = {"log_w": log_w, "temperature": float(temperature)}
    print(f"Calibration temperature: {temperature:.3f}")

    # evaluate on test set
    large_class_indices = sorted(set(y_remapped.tolist()))
    eval_class_names = [class_names_remapped[i] for i in large_class_indices]
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    report = classification_report(
        y_test, y_pred,
        labels=large_class_indices,
        target_names=eval_class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=large_class_indices)
    acc = accuracy_score(y_test, y_pred)

    # top-class confidence before/after calibration on the test set
    rows = np.arange(len(y_test))
    pred_cols = np.searchsorted(model.classes_, y_pred)
    test_cols = np.searchsorted(model.classes_, y_test)
    y_proba_cal = calibrate_proba(y_proba, log_w, temperature)
    print(
        f"Test log loss raw / calibrated: {-np.mean(np.log(np.clip(y_proba[rows, test_cols], 1e-12, 1))):.3f}"
        f" / {-np.mean(np.log(y_proba_cal[rows, test_cols])):.3f}"
    )

    return model, {
        "best_params": best_params,
        "cv": cv_fold_scores,
        "report": report,
        "confusion_matrix": cm,
        "eval_class_names": eval_class_names,
        "class_names_remapped": class_names_remapped,
        "large_class_indices": large_class_indices,
        "y_test": y_test,
        "y_proba": y_proba,
        "accuracy": acc,
        "train_idx": sample_train_idx,
        "test_idx": sample_test_idx,
        "calibration": calibration,
        "test_correct": y_pred == y_test,
        "test_conf_raw": y_proba[rows, pred_cols],
        "test_conf_cal": y_proba_cal[rows, pred_cols],
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
    calibration: dict,
    cohorts: dict[str, str],
    path: Path = DEFAULT_MODEL_PATH,
) -> "Path | None":
    try:
        import joblib
    except ImportError:
        print("[warn] joblib not installed: model not saved.", file=sys.stderr)
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "vocabulary": vocabulary, "label_encoder": le, "calibration": calibration, "cohorts": cohorts},
        path,
    )
    return path


def save_results(
    results: dict,
    model: LGBMClassifier,
    feature_names: list[str],
    data_dir: Path,
    plots_dir: Path,
    top_n: int,
) -> None:
    from ml.plots import plot_calibration, plot_confusion_matrix, plot_cv_scores, plot_feature_importances, plot_roc_curves

    cv = results["cv"]
    cm = results["confusion_matrix"]
    report = results["report"]
    eval_class_names = results["eval_class_names"]

    # class level metrics output csv
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
    cm_path = data_dir / "ern_confusion_matrix.csv"
    pd.DataFrame(cm, index=eval_class_names, columns=eval_class_names).to_csv(cm_path)
    plot_confusion_matrix(cm, eval_class_names, plots_dir / "ern_confusion_matrix.png")

    # Feature importance plot generation
    imp_df = top_feature_importances(model, feature_names, top_n)
    imp_df.to_csv(data_dir / "ern_feature_importances.csv", index=False)
    plot_feature_importances(imp_df, top_n, plots_dir / "ern_feature_importances.png")

    # Cross-validation scores plot generation
    plot_cv_scores(cv, plots_dir / "ern_cv_scores.png")

    # calibration curve of top-class confidence
    plot_calibration(
        results["test_correct"],
        results["test_conf_raw"],
        results["test_conf_cal"],
        plots_dir / "ern_calibration.png",
    )

    # ROC curves
    plot_roc_curves(
        results["y_test"],
        results["y_proba"],
        results["large_class_indices"],
        eval_class_names,
        plots_dir / "ern_roc_curves.png",
    )
