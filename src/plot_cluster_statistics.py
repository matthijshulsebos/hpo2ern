"""Visualisation utilities for phenopacket cluster analysis."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_clusters(
    X_2d: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    X_disease_2d: np.ndarray | None = None,
    dist_matrix: np.ndarray | None = None,
    disease_names: list[str] | None = None,
    top_omim_per_cluster: int = 1,
) -> None:
    """
    Scatter plot of patients in 2-D UMAP space with cluster-associated OMIM
    disease labels as reference.

    All OMIM diseases are shown as faint gray background dots to give spatial
    context. The top N closest diseases per cluster are highlighted as coloured
    star markers (matching their cluster colour) with text labels.

    Args:
        X_2d:                N×2 patient UMAP coordinates.
        labels:              Integer cluster assignment per patient (0-indexed).
        output_path:         Base output path; plot saved as *_umap.png alongside it.
        X_disease_2d:        D×2 OMIM disease UMAP coordinates (optional).
        dist_matrix:         K×D average Jaccard distances, cluster × disease (optional).
        disease_names:       List of OMIM disease name strings (optional).
        top_omim_per_cluster: Number of top OMIM diseases to label per cluster.
    """
    cluster_ids = sorted(set(labels))
    n_clusters = len(cluster_ids)
    cmap = plt.get_cmap("tab20" if n_clusters > 10 else "tab10")

    fig, ax = plt.subplots(figsize=(13, 9))

    # All OMIM diseases as faint background context
    if X_disease_2d is not None:
        ax.scatter(
            X_disease_2d[:, 0], X_disease_2d[:, 1],
            c="lightgray", s=6, alpha=0.20, linewidths=0, zorder=1,
        )

    # Patient scatter, one colour per cluster
    for k in cluster_ids:
        mask = labels == k
        color = cmap(k / max(n_clusters - 1, 1))
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            color=color,
            label=f"{k + 1}  (n={int(mask.sum())})",
            s=28, alpha=0.80,
            edgecolors="white", linewidths=0.3,
            zorder=2,
        )

    # Cluster centroid labels
    for k in cluster_ids:
        mask = labels == k
        cx = float(X_2d[mask, 0].mean())
        cy = float(X_2d[mask, 1].mean())
        color = cmap(k / max(n_clusters - 1, 1))
        ax.annotate(
            str(k + 1),
            xy=(cx, cy),
            fontsize=7, fontweight="bold",
            ha="center", va="center", color="white",
            bbox=dict(boxstyle="round,pad=0.18", facecolor=color, alpha=0.90, linewidth=0),
            zorder=3,
        )

    # Top OMIM disease labels per cluster
    if X_disease_2d is not None and dist_matrix is not None and disease_names is not None:
        # Collect top-N diseases per cluster; deduplicate by assigning each
        # disease to the single cluster it is closest to.
        assigned: dict[int, int] = {}   # disease_idx → cluster k
        for k in cluster_ids:
            top_idx = np.argsort(dist_matrix[k])[:top_omim_per_cluster]
            for j in top_idx:
                if j not in assigned:
                    assigned[j] = k

        for disease_idx, cluster_k in assigned.items():
            dx, dy = float(X_disease_2d[disease_idx, 0]), float(X_disease_2d[disease_idx, 1])
            color = cmap(cluster_k / max(n_clusters - 1, 1))

            name = disease_names[disease_idx]
            label = name if len(name) <= 35 else name[:32] + "…"

            ax.annotate(
                label,
                xy=(dx, dy),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=5.5,
                color=color,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    alpha=0.80,
                    linewidth=0,
                ),
                zorder=5,
            )

    # Axes, legend, save
    ax.set_title(
        f"Patient phenotype clusters  (K={n_clusters})\n"
        "Representation: ontology-propagated binary HPO  "
        "·  Distance: Jaccard  ·  Reduction: UMAP",
        fontsize=11, pad=10,
    )
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.legend(
        *ax.get_legend_handles_labels(),
        title="Cluster",
        title_fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7,
        markerscale=1.6,
        framealpha=0.8,
        ncol=1 if n_clusters <= 20 else 2,
    )

    plot_path = output_path.with_name(output_path.stem + "_umap.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Cluster plot saved to:        {plot_path}")
