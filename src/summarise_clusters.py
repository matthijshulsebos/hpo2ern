#!/usr/bin/env python3
"""
Generate cohort, cluster, and ERN summary reports from phenopacket clusters.

Can be run standalone to (re)generate a summary from an existing clusters CSV
without re-running the full clustering pipeline:

    python src/summarise.py \\
        --input data/ \\
        --clusters output/2026-04-29_143201/clusters.csv

Or imported by cluster_phenopackets.py and called via write_summary().

Output (written next to the clusters CSV):
    clusters_summary.log    Cohort → per-cluster → per-ERN profiles

Options:
    --input,    -i  DIR   Phenopacket directory (required)
    --clusters, -c  FILE  Cluster assignment CSV from cluster_phenopackets.py (required)
    --top-terms     INT   Top HPO terms to show per section (default: 15)
    --top-omim      INT   Top OMIM disease matches per section (default: 10)

Dependencies:
    pyhpo scikit-learn scipy numpy pandas tqdm
"""

import os
import sys
import warnings
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_DISPLAY_ENV", "FALSE")
os.environ.setdefault("KMP_WARNINGS", "FALSE")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm

from pyhpo import Ontology

from phenopackets import Sample, build_ancestry_matrix


def load_omim_diseases(vocabulary: list[str]) -> tuple[list[str], list[str], csr_matrix]:
    print("Loading OMIM disease annotations...")
    raw = {}
    for term in Ontology:
        for d in term.omim_diseases:
            raw[d.id] = d

    vocab_set = set(vocabulary)
    idx = {t: i for i, t in enumerate(vocabulary)}
    V = len(vocabulary)
    disease_ids, disease_names, rows, cols, d_row = [], [], [], [], 0

    for did, disease in raw.items():
        term_set = set()
        for hpo_int in disease.hpo:
            try:
                t = Ontology.get_hpo_object(hpo_int)
                if t.id in vocab_set:
                    term_set.add(t.id)
                for anc in t.all_parents:
                    if anc.id in vocab_set:
                        term_set.add(anc.id)
            except RuntimeError:
                pass
        if len(term_set) < 2:
            continue
        for t in term_set:
            rows.append(d_row)
            cols.append(idx[t])
        disease_ids.append(f"OMIM:{did}")
        disease_names.append(disease.name)
        d_row += 1

    X_disease = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(d_row, V)
    )
    print(f"  {d_row} OMIM diseases with ≥2 terms in patient vocabulary")
    return disease_ids, disease_names, X_disease


# Distance helpers


def _group_disease_distances(
    X_bin: csr_matrix,
    mask: np.ndarray,
    X_disease: csr_matrix,
) -> np.ndarray:
    """Average Jaccard distance from a group of patients (mask) to each OMIM disease."""
    X_g = X_bin[mask]
    patient_sums = np.asarray(X_g.sum(axis=1)).flatten()
    disease_sums = np.asarray(X_disease.sum(axis=1)).flatten()
    intersections = (X_g @ X_disease.T).toarray().astype(np.float64)
    unions = patient_sums[:, None] + disease_sums[None, :] - intersections
    jac = np.where(unions > 0, 1.0 - intersections / unions, 1.0)
    return jac.mean(axis=0)


def compute_cluster_disease_distances(
    X_bin: csr_matrix, labels: np.ndarray, X_disease: csr_matrix
) -> np.ndarray:
    cluster_ids = sorted(set(labels))
    D = X_disease.shape[0]
    dist = np.zeros((len(cluster_ids), D))
    for k in tqdm(cluster_ids, desc="Cluster–OMIM distances"):
        dist[k] = _group_disease_distances(X_bin, labels == k, X_disease)
    return dist


def compute_ern_disease_distances(
    samples: list[Sample], X_bin: csr_matrix, X_disease: csr_matrix
) -> tuple[list[str], np.ndarray]:
    ern_ids = sorted(set(s.ern for s in samples))
    D = X_disease.shape[0]
    dist = np.zeros((len(ern_ids), D))
    sample_arr = np.array([s.ern for s in samples])
    for i, ern in enumerate(tqdm(ern_ids, desc="ERN–OMIM distances")):
        dist[i] = _group_disease_distances(X_bin, sample_arr == ern, X_disease)
    return ern_ids, dist


# Formatting helpers


def _term_name(tid: str) -> str:
    try:
        return Ontology.get_hpo_object(tid).name
    except RuntimeError:
        return tid


def _fmt_counter(
    counter: Counter, total: int, indent: str = "    ", max_rows: int = 10
) -> str:
    lines = []
    for val, cnt in counter.most_common(max_rows):
        lines.append(f"{indent}{val:<40}  {cnt:>5}  ({cnt / total:.1%})")
    if len(counter) > max_rows:
        lines.append(f"{indent}  … and {len(counter) - max_rows} more")
    return "\n".join(lines)


def _fmt_top_terms(
    hpo_freq: Counter, n_group: int, n_total: int, top_n: int,
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


def _fmt_top_omim(top_omim: list[tuple]) -> list[str]:
    lines = [
        f"Top {len(top_omim)} OMIM disease matches (avg. Jaccard distance — lower = more similar):",
        f"    {'OMIM ID':<15}  {'Avg dist':>8}  Name",
        f"    {'-' * 70}",
    ]
    for dist, omim_id, name in top_omim:
        lines.append(f"    {omim_id:<15}  {dist:>8.4f}  {name}")
    return lines


def _top_omim_for(dist_row: np.ndarray, disease_ids: list[str], disease_names: list[str], n: int):
    top_idx = np.argsort(dist_row)[:n]
    return [(dist_row[j], disease_ids[j], disease_names[j]) for j in top_idx]


# Section formatters


def format_cohort_section(
    samples: list[Sample],
    labels: np.ndarray,
    cluster_dist: np.ndarray,
    disease_ids: list[str],
    disease_names: list[str],
    top_n_terms: int,
    top_n_omim: int,
) -> str:
    n = len(samples)
    tc = [len(s.terms) for s in samples]
    hpo_freq: Counter = Counter()
    for s in samples:
        hpo_freq.update(s.terms)

    cluster_ids = sorted(set(labels))
    cluster_sizes = np.array([int((labels == k).sum()) for k in cluster_ids])
    cohort_dist = np.average(cluster_dist, axis=0, weights=cluster_sizes)
    top_omim = _top_omim_for(cohort_dist, disease_ids, disease_names, top_n_omim)

    lines = [
        "=" * 62,
        f"COHORT SUMMARY  ({n} patients)",
        "=" * 62,
        "",
        "Submitters:",
        _fmt_counter(Counter(s.submitter for s in samples), n),
        "",
        "ERNs:",
        _fmt_counter(Counter(s.ern for s in samples), n),
        "",
        "HPO terms per patient (direct annotations):",
        f"    Mean {np.mean(tc):.1f}  ·  Median {int(np.median(tc))}"
        f"  ·  Min {min(tc)}  ·  Max {max(tc)}",
        "",
        *_fmt_top_terms(hpo_freq, n, n_total, top_n_terms, cohort_freq=hpo_freq),
    samples: list[Sample],
    labels: np.ndarray,
    cluster_dist: np.ndarray,
    disease_ids: list[str],
    disease_names: list[str],
    top_n_terms: int,
    top_n_omim: int,
) -> str:
    n_total = len(samples)
    global_freq: Counter = Counter()
    for s in samples:
        global_freq.update(s.terms)

    sections = []
    for k in sorted(set(labels)):
        members = [s for s, lbl in zip(samples, labels) if lbl == k]
        n = len(members)
        tc = [len(s.terms) for s in members]
        hpo_freq: Counter = Counter()
        for s in members:
            hpo_freq.update(s.terms)

        top_omim = _top_omim_for(cluster_dist[k], disease_ids, disease_names, top_n_omim)

        lines = [
            "=" * 62,
            f"Cluster {k + 1}  ({n} patients)",
            "=" * 62,
            "",
            "Submitters:",
            _fmt_counter(Counter(s.submitter for s in members), n),
            "",
            "ERNs:",
            _fmt_counter(Counter(s.ern for s in members), n),
            "",
            "HPO terms per patient (direct annotations):",
            f"    Mean {np.mean(tc):.1f}  ·  Median {int(np.median(tc))}"
            f"  ·  Min {min(tc)}  ·  Max {max(tc)}",
            "",
        *_fmt_top_terms(hpo_freq, n, n_total, top_n_terms, cohort_freq=global_freq),
    samples: list[Sample],
    labels: np.ndarray,
    ern_ids: list[str],
    ern_dist: np.ndarray,
    disease_ids: list[str],
    disease_names: list[str],
    top_n_terms: int,
    top_n_omim: int,
) -> str:
    n_total = len(samples)
    global_freq: Counter = Counter()
    for s in samples:
        global_freq.update(s.terms)

    label_map = {s.name: lbl for s, lbl in zip(samples, labels)}

    sections = []
    for i, ern in enumerate(ern_ids):
        members = [s for s in samples if s.ern == ern]
        n = len(members)
        tc = [len(s.terms) for s in members]
        hpo_freq: Counter = Counter()
        for s in members:
            hpo_freq.update(s.terms)

        cluster_dist_in_ern = Counter(
            label_map[s.name] + 1 for s in members if s.name in label_map
        )
        top_omim = _top_omim_for(ern_dist[i], disease_ids, disease_names, top_n_omim)

        lines = [
            "=" * 62,
            f"ERN: {ern}  ({n} patients)",
            "=" * 62,
            "",
            "Submitters:",
            _fmt_counter(Counter(s.submitter for s in members), n),
            "",
            "Cluster distribution:",
            _fmt_counter(cluster_dist_in_ern, n),
            "",
            "HPO terms per patient (direct annotations):",
            f"    Mean {np.mean(tc):.1f}  ·  Median {int(np.median(tc))}"
            f"  ·  Min {min(tc)}  ·  Max {max(tc)}",
            "",
        *_fmt_top_terms(hpo_freq, n, n_total, top_n_terms, cohort_freq=global_freq),


def write_summary(
    samples: list,          # list[Sample] — typed loosely so cross-module import works
    labels: np.ndarray,
    X_bin: csr_matrix,
    X_disease: csr_matrix,
    cluster_dist: np.ndarray,
    disease_ids: list[str],
    disease_names: list[str],
    output_path: Path,
    n_input: int,
    silhouette_scores: list[tuple[int, float]] | None,
    top_terms: int,
    top_omim: int,
) -> None:
    counts = Counter(int(lbl) + 1 for lbl in labels)

    print("Computing ERN–OMIM disease distances...")
    ern_ids, ern_dist = compute_ern_disease_distances(samples, X_bin, X_disease)

    cohort_text = format_cohort_section(
        samples, labels, cluster_dist, disease_ids, disease_names, top_terms, top_omim
    )
    cluster_text = format_cluster_section(
        samples, labels, cluster_dist, disease_ids, disease_names, top_terms, top_omim
    )
    ern_text = format_ern_section(
        samples, labels, ern_ids, ern_dist, disease_ids, disease_names, top_terms, top_omim
    )

    log_path = output_path.with_name(output_path.stem + "_summary.log")
    with open(log_path, "w") as log:
        log.write(f"Input phenopackets:  {n_input}\n")
        log.write(f"Used in clustering:  {sum(counts.values())}\n")
        log.write(f"Clusters:            {len(counts)}\n")
        sizes = list(counts.values())
        log.write(
            f"Min / max / mean:    {min(sizes)} / {max(sizes)} / {np.mean(sizes):.1f}\n"
        )
        if silhouette_scores:
            log.write("\nSilhouette sweep (UMAP + Euclidean):\n")
            log.write(f"  {'K':>4}  {'Silhouette':>10}\n")
            log.write(f"  {'-' * 18}\n")
            best_k = len(counts)
            for k, score in silhouette_scores:
                marker = " *" if k == best_k else ""
                log.write(f"  {k:>4}   {score:>10.4f}{marker}\n")

        log.write("\n\n")
        log.write(cohort_text)
        log.write("\n\n── PER-CLUSTER PROFILES ─────────────────────────────────────────────────────\n\n")
        log.write(cluster_text)
        log.write("\n\n── PER-ERN PROFILES ─────────────────────────────────────────────────────────\n\n")
        log.write(ern_text)

    print(f"Summary log saved to:         {log_path}")

    # Brief stdout table
    print()
    label_arr = np.asarray(labels)
    for k in sorted(set(label_arr)):
        members = [s for s, lbl in zip(samples, label_arr) if lbl == k]
        tc = [len(s.terms) for s in members]
        hpo_freq: Counter = Counter()
        for s in members:
            hpo_freq.update(s.terms)
        top3 = ", ".join(_term_name(t) for t, _ in hpo_freq.most_common(3))
        best_omim = disease_names[int(np.argmin(cluster_dist[k]))]
        print(
            f"  Cluster {k + 1:>2}: {len(members):>5} patients"
            f"  |  median {int(np.median(tc))} HPO terms"
            f"  |  top HPO: {top3}"
            f"  |  closest OMIM: {best_omim}"
        )


