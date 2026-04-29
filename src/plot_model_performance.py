"""Visualisation utilities for ERN Random Forest model performance."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Save a heatmap of the confusion matrix to *output_path*."""
    fig_size = max(8, len(class_names) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="YlOrRd", xticks_rotation=45)
    ax.set_title("ERN RF: confusion matrix (test set)", fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix plot:        {output_path}")


def plot_feature_importances(
    imp_df: pd.DataFrame,
    top_n: int,
    output_path: Path,
) -> None:
    """Save a horizontal bar chart of the top-N feature importances to *output_path*."""
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.28)))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="steelblue")
    ax.set_xlabel("Mean decrease in impurity", fontsize=9)
    ax.set_title(f"Top {top_n} features for ERN prediction", fontsize=11)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Feature importances plot:     {output_path}")


def plot_cv_scores(
    cv_results: dict,
    output_path: Path,
) -> None:
    """Box plot of per-fold CV scores (accuracy, F1 macro, F1 weighted)."""
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
