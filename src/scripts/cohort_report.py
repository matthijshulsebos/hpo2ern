#!/usr/bin/env python3
"""
Generate a cohort summary report and distribution plots from phenopackets.

Usage:
    python src/scripts/cohort_report.py --input <phenopacket_dir> [--output <dir>] [--top-terms N]
"""

import argparse
import datetime
import os
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("OMP_DISPLAY_ENV", "FALSE")
os.environ.setdefault("KMP_WARNINGS", "FALSE")
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyhpo import Ontology

from utils.phenopackets import Sample, load_all_phenopackets, validate_terms


# plots

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


# summary

def _term_name(tid: str) -> str:
    try:
        return Ontology.get_hpo_object(tid).name
    except RuntimeError:
        return tid


def _fmt_counter(
    counter: Counter, total: int, indent: str = "    ", max_rows: int | None = 10
) -> str:
    items = counter.most_common(max_rows)
    lines = [f"{indent}{val:<40}  {cnt:>5}  ({cnt / total:.1%})" for val, cnt in items]
    if max_rows is not None and len(counter) > max_rows:
        lines.append(f"{indent}  … and {len(counter) - max_rows} more")
    return "\n".join(lines)


def _fmt_top_terms(
    hpo_freq: Counter,
    n_group: int,
    n_total: int,
    top_n: int,
    cohort_freq: Counter | None = None,
) -> list[str]:
    lines = [
        f"Top {min(top_n, len(hpo_freq))} HPO terms:",
        f"    {'Term ID':<14}  {'Name':<50}  {'Count':>6}  {'In group':>8}  {'In cohort':>9}",
        f"    {'-' * 95}",
    ]
    for tid, cnt in hpo_freq.most_common(top_n):
        cohort_cnt = cohort_freq[tid] if cohort_freq is not None else cnt
        lines.append(
            f"    {tid:<14}  {_term_name(tid):<50}  {cnt:>6}"
            f"  {cnt / n_group:>7.1%}  {cohort_cnt / n_total:>8.1%}"
        )
    return lines


def _format_cohort_section(samples: list[Sample], top_n_terms: int) -> str:
    n = len(samples)
    tc = [len(s.terms) for s in samples]
    hpo_freq: Counter = Counter()
    for s in samples:
        hpo_freq.update(s.terms)

    lines = [
        "=" * 62,
        f"COHORT SUMMARY  ({n} patients)",
        "=" * 62,
        "",
        "ERNs:",
        _fmt_counter(Counter(s.ern for s in samples), n, max_rows=None),
        "",
        "Submitters:",
        _fmt_counter(Counter(s.submitter for s in samples), n),
        "",
        "HPO terms per patient (direct annotations):",
        f"    Mean {np.mean(tc):.1f}  ·  Median {int(np.median(tc))}"
        f"  ·  Min {min(tc)}  ·  Max {max(tc)}",
        "",
        *_fmt_top_terms(hpo_freq, n, n, top_n_terms, cohort_freq=hpo_freq),
    ]
    return "\n".join(lines)


def _format_ern_section(samples: list[Sample], top_n_terms: int) -> str:
    n_total = len(samples)
    global_freq: Counter = Counter()
    for s in samples:
        global_freq.update(s.terms)

    sections = []
    for ern in sorted(set(s.ern for s in samples)):
        members = [s for s in samples if s.ern == ern]
        n = len(members)
        tc = [len(s.terms) for s in members]
        hpo_freq: Counter = Counter()
        for s in members:
            hpo_freq.update(s.terms)

        lines = [
            "=" * 62,
            f"ERN: {ern}  ({n} patients)",
            "=" * 62,
            "",
            "Submitters:",
            _fmt_counter(Counter(s.submitter for s in members), n),
            "",
            "HPO terms per patient (direct annotations):",
            f"    Mean {np.mean(tc):.1f}  ·  Median {int(np.median(tc))}"
            f"  ·  Min {min(tc)}  ·  Max {max(tc)}",
            "",
            *_fmt_top_terms(hpo_freq, n, n_total, top_n_terms, cohort_freq=global_freq),
        ]
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def write_summary(samples: list[Sample], output_path: Path, top_terms: int = 15) -> None:
    log_path = output_path.with_suffix(".log")
    with open(log_path, "w") as fh:
        fh.write(_format_cohort_section(samples, top_terms))
        fh.write("\n\n── PER-ERN PROFILES ─────────────────────────────────────────────────────────\n\n")
        fh.write(_format_ern_section(samples, top_terms))
    print(f"Summary log saved to:         {log_path}")


# cli

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input",     "-i", required=True, type=Path, metavar="DIR",
                   help="Directory with phenopacket JSON files")
    p.add_argument("--output",    "-o", type=Path, default=None, metavar="DIR",
                   help="Output directory (default: output/<timestamp>_cohort/)")
    p.add_argument("--top-terms", type=int, default=25, metavar="N",
                   help="Top HPO terms to show in summary and bar chart")
    args = p.parse_args()

    if args.output is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = Path("output") / f"{timestamp}_cohort"
    else:
        run_dir = args.output
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    Ontology()
    samples = load_all_phenopackets(args.input)
    samples = validate_terms(samples)

    write_summary(samples, run_dir / "cohort_summary", top_terms=args.top_terms)

    term_names: dict[str, str] = {}
    for tid in {t for s in samples for t in s.terms}:
        try:
            term_names[tid] = Ontology.get_hpo_object(tid).name
        except RuntimeError:
            pass

    plot_ern_distribution(samples, plots_dir / "ern_distribution.png")
    plot_submitter_distribution(samples, plots_dir / "submitter_distribution.png")
    plot_hpo_term_counts(samples, plots_dir / "hpo_term_counts.png")
    plot_top_hpo_terms(samples, plots_dir / "top_hpo_terms.png",
                       top_n=args.top_terms, term_names=term_names)

    print(f"\nDone. Output in {run_dir}/")


if __name__ == "__main__":
    main()
