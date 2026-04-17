#!/usr/bin/env python3
"""
Cluster phenopackets into N groups using BMA Resnik HPO semantic similarity.

Each phenopacket is represented by its set of HPO terms. Pairwise semantic
similarity is computed using Best Match Average (BMA) Resnik similarity via
the pyhpo library. Agglomerative clustering with complete linkage partitions
the cohort into the requested number of clusters.

If --n-clusters is omitted, silhouette analysis is run over --k-min..--k-max
to select the best K automatically.

Usage:
    python cluster_phenopackets.py --input /path/to/phenopackets/ [options]

Options:
    --input,      -i  DIR   Directory with phenopacket JSON files (required)
    --n-clusters, -n  INT   Number of clusters (omit to auto-select via silhouette)
    --k-min           INT   Minimum K for silhouette sweep (default: 2)
    --k-max           INT   Maximum K for silhouette sweep (default: 30)
    --output,     -o  FILE  Output CSV path (default: clusters.csv)
    --ic-source       STR   IC source: omim | orpha | decipher | gene
                            (default: omim)
    --save-matrix       Flag: also write the full similarity matrix to CSV.
    --linkage         STR   Agglomerative linkage: complete | average | single
                            (default: complete)

Dependencies (pip install):
    pyhpo scikit-learn numpy pandas tqdm
"""

import json
import argparse
import sys
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from tqdm import tqdm

from pyhpo import Ontology, HPOSet
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


@dataclass
class Sample:
    name: str
    terms: list[str]
    hpo_set: HPOSet | None = field(default=None, repr=False)


# Phenopacket parsing


def load_hpo_terms(filepath: Path) -> list[str]:
    """Return observed HPO term IDs from a phenopacket JSON file.

    Handles both Phenopackets v1 and v2 schemas:
      v1: phenotypicFeatures[].type.id  (negated via .negated)
      v2: phenotypicFeatures[].type.id  (negated via .excluded)
    """
    with open(filepath) as fh:
        data = json.load(fh)

    terms: list[str] = []
    for feature in data.get("phenotypicFeatures", []):
        if feature.get("excluded", False) or feature.get("negated", False):
            continue
        term_id = feature.get("type", {}).get("id", "")
        if term_id.startswith("HP:"):
            terms.append(term_id)

    return terms


def load_all_phenopackets(phenopacket_dir: Path) -> list[Sample]:
    """Discover and load all phenopacket JSON files under phenopacket_dir."""
    files = sorted(phenopacket_dir.rglob("*.json"))
    if not files:
        sys.exit(f"ERROR: no .json files found under {phenopacket_dir}")

    samples: list[Sample] = []
    n_skipped = 0

    for f in files:
        terms = load_hpo_terms(f)
        if not terms:
            print(f"[skip] {f.name}: no observed HPO terms", file=sys.stderr)
            n_skipped += 1
            continue
        samples.append(Sample(name=f.stem, terms=terms))

    print(
        f"Loaded {len(samples)} phenopackets"
        + (f" ({n_skipped} skipped: no HPO terms)" if n_skipped else "")
    )
    return samples


# HPO set construction


def build_hpo_sets(samples: list[Sample]) -> list[Sample]:
    """Populate each sample's hpo_set after validating terms against the ontology.

    Samples whose terms are all unknown to the ontology are dropped.
    """
    valid: list[Sample] = []

    for sample in samples:
        valid_terms: list[str] = []
        for t in sample.terms:
            try:
                Ontology.get_hpo_object(t)
                valid_terms.append(t)
            except RuntimeError:
                print(
                    f"[warn] {sample.name}: unknown HPO term {t!r} skipped",
                    file=sys.stderr,
                )

        if not valid_terms:
            print(
                f"[skip] {sample.name}: no valid HPO terms after ontology check",
                file=sys.stderr,
            )
            continue

        sample.terms = valid_terms
        sample.hpo_set = HPOSet.from_queries(valid_terms)
        valid.append(sample)

    return valid


# Similarity matrix


def compute_similarity_matrix(
    samples: list[Sample],
    ic_source: str,
) -> np.ndarray:
    """Compute the full pairwise BMA Resnik similarity matrix (N×N)."""
    n = len(samples)
    sim = np.zeros((n, n), dtype=np.float64)

    total_pairs = n * (n - 1) // 2
    with tqdm(total=total_pairs, desc="Computing pairwise BMA Resnik similarity") as bar:
        for i, j in combinations(range(n), 2):
            val = samples[i].hpo_set.similarity(
                samples[j].hpo_set,
                kind=ic_source,
                method="resnik",
                combine="BMA",
            )
            sim[i, j] = val
            sim[j, i] = val
            bar.update(1)

    np.fill_diagonal(sim, 1.0)

    return sim


# Distance conversion and clustering


def sim_to_dist(sim_matrix: np.ndarray) -> np.ndarray:
    """Convert Resnik similarity to distance: dist = 1 - sim_norm."""
    max_val = sim_matrix.max()
    if max_val == 0:
        sys.exit(
            "ERROR: all pairwise similarities are zero. "
            "Check that your HPO terms are present in the ontology."
        )
    dist = 1.0 - (sim_matrix / max_val)
    np.fill_diagonal(dist, 0.0)
    return dist


def cluster(dist_matrix: np.ndarray, n_clusters: int, linkage: str) -> np.ndarray:
    """Agglomerative clustering on a precomputed distance matrix."""
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage=linkage,
    )
    return model.fit_predict(dist_matrix)


def select_k_by_silhouette(
    dist_matrix: np.ndarray,
    linkage: str,
    k_min: int,
    k_max: int,
) -> tuple[int, list[tuple[int, float]]]:
    """Sweep K from k_min to k_max and return the K with the highest silhouette score."""
    scores: list[tuple[int, float]] = []

    for k in tqdm(range(k_min, k_max + 1), desc="Silhouette sweep"):
        labels = cluster(dist_matrix, k, linkage)
        score = silhouette_score(dist_matrix, labels, metric="precomputed")
        scores.append((k, score))

    best_k, best_score = max(scores, key=lambda x: x[1])
    print(f"  Best K={best_k} (silhouette={best_score:.4f})")
    return best_k, scores


# Output


def save_results(
    samples: list[Sample],
    labels: np.ndarray,
    sim_matrix: np.ndarray,
    output_path: Path,
    save_matrix: bool,
    n_input: int,
    silhouette_scores: list[tuple[int, float]] | None = None,
) -> None:
    df = (
        pd.DataFrame(
            {
                "phenopacket": [s.name for s in samples],
                "cluster": labels + 1,  # 1-indexed for human readability
            }
        )
        .sort_values(["cluster", "phenopacket"])
        .reset_index(drop=True)
    )

    df.to_csv(output_path, index=False)

    if save_matrix:
        mat_path = output_path.with_name(output_path.stem + "_similarity_matrix.csv")
        names = [s.name for s in samples]
        sim_df = pd.DataFrame(sim_matrix, index=names, columns=names)
        sim_df.to_csv(mat_path)

    counts = df["cluster"].value_counts().sort_index()

    log_path = output_path.with_name(output_path.stem + "_summary.log")
    with open(log_path, "w") as log:
        log.write(f"Input phenopackets:  {n_input}\n")
        log.write(f"Used in clustering:  {len(df)}\n")
        log.write(f"Clusters:            {counts.nunique()}\n")
        log.write(
            f"Min / max / mean:    {counts.min()} / {counts.max()} / {counts.mean():.1f}\n"
        )
        log.write("\n")
        log.write(f"{'Cluster':>9}  {'Count':>6}\n")
        log.write("-" * 20 + "\n")
        for cid, cnt in counts.items():
            log.write(f"  {cid:>7}   {cnt:>5}\n")

        if silhouette_scores:
            log.write("\n")
            log.write("Silhouette sweep:\n")
            log.write(f"{'K':>5}  {'Silhouette':>10}\n")
            log.write("-" * 20 + "\n")
            for k, score in silhouette_scores:
                marker = " *" if k == counts.nunique() else ""
                log.write(f"  {k:>3}   {score:>10.4f}{marker}\n")

    print(f"Cluster assignments saved to: {output_path}")
    print(f"Summary log saved to:         {log_path}")
    if save_matrix:
        print(f"Similarity matrix saved to:   {mat_path}")


# CLI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cluster phenopackets using BMA Resnik HPO similarity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", "-i", required=True, type=Path, metavar="DIR",
        help="Directory containing phenopacket JSON files (searched recursively)",
    )
    p.add_argument(
        "--n-clusters", "-n", type=int, default=None, metavar="N",
        help="Number of clusters (omit to auto-select via silhouette analysis)",
    )
    p.add_argument(
        "--k-min", type=int, default=2, metavar="N",
        help="Minimum K for silhouette sweep (ignored when --n-clusters is set)",
    )
    p.add_argument(
        "--k-max", type=int, default=30, metavar="N",
        help="Maximum K for silhouette sweep (ignored when --n-clusters is set)",
    )
    p.add_argument(
        "--output", "-o", type=Path, default=Path("clusters.csv"), metavar="FILE",
        help="Output CSV file",
    )
    p.add_argument(
        "--ic-source", choices=["omim", "orpha", "decipher", "gene"], default="omim",
        help="Information content source for Resnik similarity",
    )
    p.add_argument(
        "--save-matrix", action="store_true",
        help="Also write the N×N similarity matrix to CSV",
    )
    p.add_argument(
        "--linkage", choices=["complete", "average", "single"], default="complete",
        help="Agglomerative linkage strategy",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load HPO ontology (downloaded automatically on first run by pyhpo)
    print("Loading HPO ontology...")
    Ontology()
    print("  Ontology ready.")

    # 2. Parse phenopackets
    print(f"\nScanning phenopackets in: {args.input}")
    samples = load_all_phenopackets(args.input)
    n_input = len(samples)

    if args.n_clusters is not None and n_input < args.n_clusters:
        sys.exit(
            f"ERROR: only {n_input} usable phenopackets but "
            f"{args.n_clusters} clusters were requested. "
            f"Lower --n-clusters or add more phenopackets."
        )

    # 3. Build HPO sets (ontology-validated); may drop further phenopackets
    print("\nBuilding HPO sets...")
    samples = build_hpo_sets(samples)

    # 4. Compute similarity and distance matrices
    print(f"\nComputing {len(samples)}×{len(samples)} pairwise BMA Resnik similarity matrix...")
    print(f"  IC source: {args.ic_source}")
    sim_matrix = compute_similarity_matrix(samples, args.ic_source)
    dist_matrix = sim_to_dist(sim_matrix)

    # 5. Select K via silhouette sweep if not specified
    silhouette_scores: list[tuple[int, float]] | None = None
    if args.n_clusters is None:
        k_max = min(args.k_max, len(samples) - 1)
        print(f"\nNo --n-clusters given; running silhouette sweep (K={args.k_min}..{k_max})...")
        n_clusters, silhouette_scores = select_k_by_silhouette(
            dist_matrix, args.linkage, args.k_min, k_max
        )
    else:
        n_clusters = args.n_clusters

    # 6. Cluster with chosen K
    print(f"\nClustering into {n_clusters} groups (linkage={args.linkage})...")
    labels = cluster(dist_matrix, n_clusters, args.linkage)

    # 7. Save
    save_results(samples, labels, sim_matrix, args.output, args.save_matrix, n_input, silhouette_scores)


if __name__ == "__main__":
    main()
