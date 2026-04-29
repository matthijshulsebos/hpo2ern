#!/usr/bin/env python3
"""
Cluster phenopackets using ontology-propagated binary HPO vectors + Jaccard K-Means.

Each patient is represented as a sparse binary vector over every HPO term
present in the cohort (direct annotations + all ontology ancestors). Jaccard
distance naturally up-weights rare, specific terms without an explicit IC
calculation. UMAP with Jaccard metric projects the sparse binary matrix into a
dense low-dimensional space for K-Means clustering and 2-D visualisation.

OMIM diseases are projected into the same UMAP space as a reference background,
and the top closest OMIM diseases per cluster are reported in the summary.

NOTE on cluster count: silhouette analysis maximises compactness + separation
and typically selects a small K (broad groups). Use --n-clusters to specify
a domain-informed K, or lower --n-neighbors to expose finer local structure.

Usage:
    python embedding_cluster_phenopackets.py --input /path/to/phenopackets/ [options]

Options:
    --input,          -i  DIR   Directory with phenopacket JSON files (required)
    --n-clusters,     -n  INT   Number of clusters (omit to auto-select via silhouette)
    --k-min               INT   Minimum K for silhouette sweep (default: 2)
    --k-max               INT   Maximum K for silhouette sweep (default: 30)
    --output,         -o  FILE  Output CSV path (default: clusters_embed.csv)
    --umap-components     INT   UMAP dimensions used for clustering (default: 25)
    --n-neighbors         INT   UMAP neighbourhood size; lower = more local structure
                                and more potential clusters (default: 15)
    --min-dist          FLOAT   UMAP minimum distance; lower = tighter packing (default: 0.1)
    --top-terms           INT   Top HPO terms to show per cluster in summary (default: 15)
    --top-omim            INT   Top OMIM disease matches to show per cluster (default: 10)
    --save-matrix               Write the N×N Jaccard-distance matrix to CSV
    --save-embeddings           Write the N×D UMAP cluster-embedding matrix to CSV
    --save-plot                 Save a 2-D UMAP scatter plot with OMIM disease background

Dependencies (pip install):
    pyhpo umap-learn scikit-learn scipy numpy pandas tqdm matplotlib
"""

import os
import sys
import warnings
from pathlib import Path

# Suppress cosmetic runtime messages before any heavy imports
os.environ.setdefault("OMP_DISPLAY_ENV", "FALSE")
os.environ.setdefault("KMP_WARNINGS", "FALSE")
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm

from pyhpo import Ontology
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from phenopackets import Sample, build_ancestry_matrix


# OMIM diseases


def load_omim_diseases(
    vocabulary: list[str],
) -> tuple[list[str], list[str], csr_matrix]:
    """Build binary HPO vectors for all OMIM diseases restricted to the patient vocabulary.

    Returns (disease_ids, disease_names, X_disease) where X_disease is (D × V).
    Only diseases with ≥2 terms in the patient vocabulary are included.
    """
    print("Loading OMIM disease HPO annotations from ontology...")
    raw_diseases: dict[int, object] = {}
    for term in Ontology:
        for d in term.omim_diseases:
            raw_diseases[d.id] = d

    vocab_set = set(vocabulary)
    idx = {t: i for i, t in enumerate(vocabulary)}
    V = len(vocabulary)

    disease_ids: list[str] = []
    disease_names: list[str] = []
    rows, cols = [], []
    d_row = 0

    for did, disease in raw_diseases.items():
        term_set: set[str] = set()
        for hpo_int in disease.hpo:
            try:
                term = Ontology.get_hpo_object(hpo_int)
                if term.id in vocab_set:
                    term_set.add(term.id)
                for ancestor in term.all_parents:
                    if ancestor.id in vocab_set:
                        term_set.add(ancestor.id)
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
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(d_row, V),
    )
    print(f"  {d_row} OMIM diseases with ≥2 terms overlapping patient vocabulary")
    return disease_ids, disease_names, X_disease


def cluster_disease_distances(
    X_bin: csr_matrix,
    labels: np.ndarray,
    X_disease: csr_matrix,
) -> np.ndarray:
    """Compute average Jaccard distance from each cluster to each OMIM disease.

    Returns a K × D array (K clusters, D diseases), values in [0, 1].
    Lower = more similar.
    """
    cluster_ids = sorted(set(labels))
    D = X_disease.shape[0]
    dist_matrix = np.zeros((len(cluster_ids), D), dtype=np.float64)
    disease_sums = np.asarray(X_disease.sum(axis=1)).flatten()

    for k in tqdm(cluster_ids, desc="Computing cluster–OMIM distances"):
        X_k = X_bin[labels == k]
        patient_sums = np.asarray(X_k.sum(axis=1)).flatten()
        intersections = (X_k @ X_disease.T).toarray().astype(np.float64)
        unions = patient_sums[:, None] + disease_sums[None, :] - intersections
        jac_dist = np.where(unions > 0, 1.0 - intersections / unions, 1.0)
        dist_matrix[k] = jac_dist.mean(axis=0)

    return dist_matrix


# UMAP projection


def umap_fit(
    X: csr_matrix,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    seed: int = 42,
):
    """Fit UMAP and return (embedding, fitted_reducer)."""
    try:
        import umap  # noqa: PLC0415
    except ImportError:
        sys.exit("ERROR: umap-learn is not installed.\n  pip install umap-learn")

    n_components = min(n_components, X.shape[0] - 1)
    reducer = umap.UMAP(
        n_components=n_components,
        metric="jaccard",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
        low_memory=True,
    )
    return reducer.fit_transform(X), reducer


# Clustering


def cluster_kmeans(X: np.ndarray, n_clusters: int, seed: int = 42) -> np.ndarray:
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(X)


def select_k_by_silhouette(
    X: np.ndarray, k_min: int, k_max: int
) -> tuple[int, list[tuple[int, float]]]:
    scores: list[tuple[int, float]] = []
    for k in tqdm(range(k_min, k_max + 1), desc="Silhouette sweep"):
        labels = cluster_kmeans(X, k)
        score = silhouette_score(X, labels, metric="euclidean")
        scores.append((k, score))
    best_k, best_score = max(scores, key=lambda x: x[1])
    print(f"  Best K={best_k} (silhouette={best_score:.4f})")
    if best_k <= 5:
        print(
            "  [hint] Silhouette favours compact groups and may undercount subtypes.\n"
            "         Use --n-clusters to set a domain-informed K, or lower --n-neighbors\n"
            "         to expose finer local structure."
        )
    return best_k, scores


# Output


def save_results(
    samples: list[Sample],
    labels: np.ndarray,
    X_bin: csr_matrix,
    X_umap: np.ndarray,
    output_path: Path,
    save_matrix: bool,
    save_embeddings: bool,
) -> None:
    """Save cluster assignment CSV and optional Jaccard matrix / UMAP embeddings."""
    df = (
        pd.DataFrame({"phenopacket": [s.name for s in samples], "cluster": labels + 1})
        .sort_values(["cluster", "phenopacket"])
        .reset_index(drop=True)
    )
    df.to_csv(output_path, index=False)
    print(f"Cluster assignments saved to: {output_path}")

    names = [s.name for s in samples]

    if save_matrix:
        intersection = (X_bin @ X_bin.T).toarray().astype(np.float64)
        row_sums = np.asarray(X_bin.sum(axis=1)).flatten()
        union = row_sums[:, None] + row_sums[None, :] - intersection
        jaccard_dist = np.where(union > 0, 1.0 - intersection / union, 0.0)
        mat_path = output_path.with_name(output_path.stem + "_jaccard_dist.csv")
        pd.DataFrame(jaccard_dist, index=names, columns=names).to_csv(mat_path)
        print(f"Jaccard-distance matrix saved to: {mat_path}")

    if save_embeddings:
        emb_path = output_path.with_name(output_path.stem + "_umap_embedding.csv")
        pd.DataFrame(X_umap, index=names).to_csv(emb_path)
        print(f"UMAP cluster embedding saved to:  {emb_path}")
