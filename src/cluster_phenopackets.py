#!/usr/bin/env python3
"""
Cluster phenopackets into N groups using BMA Resnik HPO semantic similarity.

Each phenopacket is represented by its set of HPO terms. Pairwise semantic
similarity is computed using Best Match Average (BMA) Resnik similarity via
the pyhpo library. Agglomerative clustering (UPGMA) partitions the cohort
into the requested number of clusters (default: 24, one per ERN).

Usage:
    python cluster_phenopackets.py --input /path/to/phenopackets/ [options]

Options:
    --input,      -i  DIR   Directory with phenopacket JSON files (required)
    --n-clusters, -n  INT   Number of clusters / ERNs (default: 24)
    --output,     -o  FILE  Output CSV path (default: clusters.csv)
    --ic-source       STR   IC source: omim | orpha | decipher | gene
                            (default: omim). Use --corpus-ic to override with
                            corpus-derived IC instead.
    --corpus-ic         Flag: compute IC from phenopacket corpus frequencies
                            rather than HPO annotation databases.
    --save-matrix       Flag: also write the full similarity matrix to CSV.
    --linkage         STR   Agglomerative linkage: average | complete | single
                            (default: average = UPGMA)

Dependencies (pip install):
    pyhpo scikit-learn numpy pandas tqdm
"""

import json
import argparse
import math
import sys
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from tqdm import tqdm

from pyhpo import Ontology, HPOSet, HPOTerm
from sklearn.cluster import AgglomerativeClustering


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
                    f"[warn] {sample.name}: unknown HPO term {t!r} — skipped",
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


# Corpus-based information content


def compute_corpus_ic(samples: list[Sample]) -> dict[str, float]:
    """Compute information content from the phenopacket corpus.

    For each HPO term (and all its ancestors), count how many phenopackets
    annotate that term directly or via a more specific descendant.

        IC(t) = -log( freq(t) / N )

    where freq(t) = number of phenopackets in which t or a descendant of t
    appears, and N = total number of phenopackets.

    This is the standard corpus-based IC used when external annotation
    databases are not available or not desired.
    """
    n_total = len(samples)
    term_freq: dict[str, int] = defaultdict(int)

    for sample in samples:
        ancestors_union: set[str] = set()
        for t in set(sample.terms):
            try:
                hpo_obj = Ontology.get_hpo_object(t)
            except RuntimeError:
                continue
            ancestors_union.update(a.id for a in hpo_obj.path_to_top())
        for anc in ancestors_union:
            term_freq[anc] += 1

    ic: dict[str, float] = {}
    for term_id, freq in term_freq.items():
        if freq > 0:
            ic[term_id] = -math.log(freq / n_total)
        else:
            ic[term_id] = 0.0

    return ic


# BMA Resnik similarity


def resnik_sim_pyhpo(
    set_a: HPOSet,
    set_b: HPOSet,
    ic_source: str,
) -> float:
    """BMA Resnik similarity using pyhpo's built-in annotation-based IC."""
    return set_a.similarity(
        set_b,
        kind=ic_source,  # "omim" | "orpha" | "decipher" | "gene"
        method="resnik",
        combine="BMA",
    )


def _mica_ic_corpus(
    term_a: HPOTerm,
    term_b: HPOTerm,
    corpus_ic: dict[str, float],
) -> float:
    """IC of the Most Informative Common Ancestor for corpus-based IC."""
    ancestors_a = {a.id for a in term_a.path_to_top()}
    ancestors_b = {b.id for b in term_b.path_to_top()}
    common = ancestors_a & ancestors_b
    if not common:
        return 0.0
    return max(corpus_ic.get(t, 0.0) for t in common)


def resnik_bma_corpus(
    sample_a: Sample,
    sample_b: Sample,
    corpus_ic: dict[str, float],
) -> float:
    """BMA Resnik similarity using corpus-derived IC values.

    BMA(A, B) = [ sum_a max_b sim(a,b) + sum_b max_a sim(a,b) ] / (|A| + |B|)
    """
    hpo_a = [Ontology.get_hpo_object(t) for t in sample_a.terms]
    hpo_b = [Ontology.get_hpo_object(t) for t in sample_b.terms]

    def one_way(source: list, target: list) -> float:
        total = 0.0
        for s in source:
            best = max(_mica_ic_corpus(s, t, corpus_ic) for t in target)
            total += best
        return total

    return (one_way(hpo_a, hpo_b) + one_way(hpo_b, hpo_a)) / (
        len(hpo_a) + len(hpo_b)
    )


# Similarity matrix


def compute_similarity_matrix(
    samples: list[Sample],
    ic_source: str,
    use_corpus_ic: bool,
) -> np.ndarray:
    """Compute the full pairwise BMA Resnik similarity matrix (N×N)."""
    n = len(samples)
    sim = np.zeros((n, n), dtype=np.float64)

    corpus_ic: dict[str, float] = {}
    if use_corpus_ic:
        print("Computing corpus-based information content...")
        corpus_ic = compute_corpus_ic(samples)
        print(f"  IC computed for {len(corpus_ic)} HPO terms")

    total_pairs = n * (n - 1) // 2
    with tqdm(
        total=total_pairs, desc="Computing pairwise BMA Resnik similarity"
    ) as bar:
        for i, j in combinations(range(n), 2):
            if use_corpus_ic:
                val = resnik_bma_corpus(samples[i], samples[j], corpus_ic)
            else:
                val = resnik_sim_pyhpo(samples[i].hpo_set, samples[j].hpo_set, ic_source)
            sim[i, j] = val
            sim[j, i] = val
            bar.update(1)

    # Self-similarity on the diagonal: use the max off-diagonal value
    # so that clustering treats self as the most similar possible point.
    max_off = sim.max()
    diag_val = max_off if max_off > 0 else 1.0
    np.fill_diagonal(sim, diag_val)

    return sim


# Distance conversion and clustering


def sim_to_dist(sim_matrix: np.ndarray) -> np.ndarray:
    """Normalise similarity to [0, 1] then convert to distance (1 - sim_norm)."""
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


# Output


def save_results(
    samples: list[Sample],
    labels: np.ndarray,
    sim_matrix: np.ndarray,
    output_path: Path,
    save_matrix: bool,
    n_input: int,
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

    # Write summary log
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

    print(f"Cluster assignments saved to: {output_path}")
    print(f"Summary log saved to:         {log_path}")
    if save_matrix:
        print(f"Similarity matrix saved to:   {mat_path}")


# CLI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cluster phenopackets into N groups using BMA Resnik HPO similarity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory containing phenopacket JSON files (searched recursively)",
    )
    p.add_argument(
        "--n-clusters",
        "-n",
        type=int,
        default=24,
        metavar="N",
        help="Number of clusters",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("clusters.csv"),
        metavar="FILE",
        help="Output CSV file",
    )
    p.add_argument(
        "--ic-source",
        choices=["omim", "orpha", "decipher", "gene"],
        default="omim",
        help=(
            "Information content source for Resnik similarity "
            "(ignored when --corpus-ic is set)"
        ),
    )
    p.add_argument(
        "--corpus-ic",
        action="store_true",
        help=(
            "Derive IC from term frequencies in the phenopacket corpus "
            "instead of HPO annotation databases"
        ),
    )
    p.add_argument(
        "--save-matrix",
        action="store_true",
        help="Also write the N×N similarity matrix to CSV",
    )
    p.add_argument(
        "--linkage",
        choices=["average", "complete", "single"],
        default="average",
        help="Agglomerative clustering linkage strategy (average = UPGMA)",
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

    if n_input < args.n_clusters:
        sys.exit(
            f"ERROR: only {n_input} usable phenopackets but "
            f"{args.n_clusters} clusters were requested. "
            f"Lower --n-clusters or add more phenopackets."
        )

    # 3. Build HPO sets (ontology-validated); may drop further phenopackets
    print("\nBuilding HPO sets...")
    samples = build_hpo_sets(samples)

    # 4. Compute similarity matrix
    print(f"\nComputing {len(samples)}×{len(samples)} pairwise BMA Resnik similarity matrix...")
    ic_label = "corpus-based IC" if args.corpus_ic else f"IC source: {args.ic_source}"
    print(f"  Method: BMA Resnik ({ic_label})")
    sim_matrix = compute_similarity_matrix(samples, args.ic_source, args.corpus_ic)

    # 5. Convert to distance
    dist_matrix = sim_to_dist(sim_matrix)

    # 6. Cluster
    print(f"\nClustering into {args.n_clusters} groups (linkage={args.linkage})...")
    labels = cluster(dist_matrix, args.n_clusters, args.linkage)

    # 7. Save
    save_results(samples, labels, sim_matrix, args.output, args.save_matrix, n_input)


if __name__ == "__main__":
    main()
