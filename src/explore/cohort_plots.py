"""Visualisation utilities for cohort-level phenopacket statistics."""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.phenopackets import Sample


def plot_ern_distribution(samples: list[Sample], output_path: Path) -> None:
    counts = Counter(s.ern for s in samples)
    erns, vals = zip(*sorted(counts.items(), key=lambda x: -x[1]))

    fig, ax = plt.subplots(figsize=(max(7, len(erns) * 0.55), 5))
    ax.bar(range(len(erns)), vals, color="steelblue", alpha=0.85)
    ax.set_xticks(range(len(erns)))
    ax.set_xticklabels(erns, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of patients", fontsize=9)
    ax.set_title(f"Patient distribution across ERNs  (N={len(samples)})", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ERN distribution plot:        {output_path}")


def plot_submitter_distribution(samples: list[Sample], output_path: Path) -> None:
    sub_hpo: dict[str, list[int]] = defaultdict(list)
    for s in samples:
        sub_hpo[s.submitter].append(len(s.terms))

    sorted_subs = sorted(sub_hpo.items(), key=lambda x: -len(x[1]))
    subs = [k for k, _ in sorted_subs]
    n_samples = [len(v) for _, v in sorted_subs]
    means = [np.mean(v) for _, v in sorted_subs]
    stds = [np.std(v) for _, v in sorted_subs]

    x = range(len(subs))
    fig, ax1 = plt.subplots(figsize=(max(7, len(subs) * 0.65), 5))
    ax1.bar(x, n_samples, color="darkorange", alpha=0.75, label="Samples")
    ax1.set_ylabel("Number of samples", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(subs, rotation=45, ha="right", fontsize=7)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.errorbar(
        x, means, yerr=stds,
        fmt="o-", color="steelblue", linewidth=1.5, markersize=4,
        capsize=3, label="Mean HPO terms / sample",
    )
    ax2.set_ylabel("HPO terms per sample  (mean ± SD)", fontsize=9)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax1.set_title(
        f"Samples and HPO annotation depth per submitter  (N={len(samples)})", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Submitter distribution plot:  {output_path}")


def plot_hpo_term_counts(samples: list[Sample], output_path: Path) -> None:
    counts = [len(s.terms) for s in samples]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(counts, bins=30, color="mediumseagreen", alpha=0.85, edgecolor="white")
    ax.axvline(np.mean(counts), color="black", linestyle="--", linewidth=1,
               label=f"Mean {np.mean(counts):.1f}")
    ax.axvline(np.median(counts), color="gray", linestyle=":", linewidth=1,
               label=f"Median {int(np.median(counts))}")
    ax.set_xlabel("HPO terms per patient (direct annotations)", fontsize=9)
    ax.set_ylabel("Patients", fontsize=9)
    ax.set_title("Distribution of HPO annotation depth", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"HPO term-count histogram:     {output_path}")


def plot_top_hpo_terms(
    samples: list[Sample],
    output_path: Path,
    top_n: int = 25,
    term_names: dict[str, str] | None = None,
) -> None:
    freq: Counter = Counter()
    for s in samples:
        freq.update(s.terms)

    items = freq.most_common(top_n)
    terms, counts = zip(*items)
    labels = (
        [f"{term_names.get(t, t)}  ({t})" for t in terms]
        if term_names
        else list(terms)
    )

    fig, ax = plt.subplots(figsize=(12, max(5, top_n * 0.38)))
    ax.barh(labels[::-1], counts[::-1], color="steelblue", alpha=0.85)
    ax.set_xlabel("Patients (direct annotation)", fontsize=9)
    ax.set_title(f"Top {top_n} HPO terms in cohort  (N={len(samples)})", fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Top HPO terms plot:           {output_path}")
