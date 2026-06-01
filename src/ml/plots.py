"""Plotting functions for model performance."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc


def plot_confusion_matrix(
    cm: np.ndarray, class_names: list[str], output_path: Path
) -> None:
    fig_size = max(8, len(class_names) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="YlOrRd", xticks_rotation=45)
    ax.set_title("ERN prediction: confusion matrix (test set)", fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix plot:        {output_path}")


def plot_feature_importances(
    imp_df: pd.DataFrame, top_n: int, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.28)))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="steelblue")
    ax.set_xlabel("Feature importance (gain)", fontsize=9)
    ax.set_title(f"Top {top_n} features for ERN prediction", fontsize=11)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Feature importances plot:     {output_path}")


def plot_cv_scores(cv_results: dict, output_path: Path) -> None:
    metrics = {
        "Accuracy": cv_results["test_accuracy"],
        "F1 macro": cv_results["test_f1_macro"],
        "F1 weighted": cv_results["test_f1_weighted"],
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(
        list(metrics.values()),
        labels=list(metrics.keys()),
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.6),
        medianprops=dict(color="black", linewidth=1.5),
    )
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title("Cross-validation scores (training set)", fontsize=11)
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"CV score plot:                {output_path}")


def plot_roc_curves(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    class_indices: list[int],
    class_names: list[str],
    output_path: Path,
) -> None:
    # OVR ROC curves
    fpr: dict[int, np.ndarray] = {}
    tpr: dict[int, np.ndarray] = {}
    thresholds: dict[int, np.ndarray] = {}
    roc_auc: dict[int, float] = {}
    for col, (ci, cn) in enumerate(zip(class_indices, class_names)):
        y_bin = (y_test == ci).astype(int)
        if y_bin.sum() == 0:
            continue
        fpr[col], tpr[col], thresholds[col] = roc_curve(y_bin, y_proba[:, col])
        roc_auc[col] = auc(fpr[col], tpr[col])

    ithaca_col = next(
        (col for col, cn in enumerate(class_names) if "ITHACA" in cn.upper()),
        None,
    )
    other_col = next(
        (col for col, cn in enumerate(class_names) if cn == "Other"),
        None,
    )
    named_cols = [col for col in roc_auc if col != other_col and col != ithaca_col]

    base_fpr = np.linspace(0, 1, 300)

    def _youden_point(col: int) -> tuple[float, float, float]:
        j = tpr[col] - fpr[col]
        idx = int(np.argmax(j))
        return float(fpr[col][idx]), float(tpr[col][idx]), float(thresholds[col][idx])

    def _mark_youden(col: int, color: str) -> None:
        fx, ty, thr = _youden_point(col)
        ax.plot(fx, ty, "o", color=color, ms=7, zorder=5)
        ax.annotate(f"  p={thr:.2f}", (fx, ty), fontsize=7.5, color=color, va="center")

    fig, ax = plt.subplots(figsize=(8, 6))

    # faint line per named ERN
    for col in named_cols:
        if col == ithaca_col:
            continue
        ax.plot(fpr[col], tpr[col], lw=1, alpha=0.35, color="steelblue")

    # macro avg of named ERNs
    if named_cols:
        mean_tpr = np.mean([np.interp(base_fpr, fpr[c], tpr[c]) for c in named_cols], axis=0)
        macro_auc = auc(base_fpr, mean_tpr)
        ax.plot(base_fpr, mean_tpr, lw=2, color="steelblue", linestyle="--",
                label=f"Named ERNs macro avg (AUC={macro_auc:.3f})")

    # ITHACA highlighted with Youdens J marker
    if ithaca_col is not None and ithaca_col in roc_auc:
        ax.plot(fpr[ithaca_col], tpr[ithaca_col], lw=2.5, color="tomato",
                label=f"{class_names[ithaca_col]} (AUC={roc_auc[ithaca_col]:.3f})")
        _mark_youden(ithaca_col, "tomato")

    # other with optimal threshold marker
    if other_col is not None and other_col in roc_auc:
        ax.plot(fpr[other_col], tpr[other_col], lw=2.5, color="goldenrod",
                label=f"Other / rare ERNs (AUC={roc_auc[other_col]:.3f})")
        _mark_youden(other_col, "goldenrod")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.set_title("ROC curves (test set)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ROC curve plot:               {output_path}")
