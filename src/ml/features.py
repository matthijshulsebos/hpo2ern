"""Feature construction for ERN prediction.

Builds the ontology-propagated HPO binary matrix for training and provides
a separate path for constructing a prediction-time matrix aligned to the
training vocabulary.
"""

import numpy as np

from scipy.sparse import csr_matrix
from pyhpo import Ontology
from utils.phenopackets import Sample


def _term_label(tid: str) -> str:
    try:
        return f"{tid} {Ontology.get_hpo_object(tid).name}"
    except RuntimeError:
        return tid


def build_feature_matrix(
    vocabulary: list[str],
    X_hpo: csr_matrix,
) -> tuple[csr_matrix, list[str]]:
    """Return (X_hpo, feature_names) with human-readable HPO term labels."""
    return X_hpo, [_term_label(t) for t in vocabulary]


def build_predict_feature_matrix(
    samples: list[Sample],
    vocabulary: list[str],
) -> csr_matrix:
    """Build a feature matrix for prediction, aligned to the training vocabulary.

    HPO terms outside the training vocabulary are silently ignored.
    """
    vocab_idx = {t: i for i, t in enumerate(vocabulary)}
    rows: list[int] = []
    cols: list[int] = []

    for i, sample in enumerate(samples):
        term_set: set[str] = set()
        for tid in sample.terms:
            try:
                hpo_term = Ontology.get_hpo_object(tid)
                term_set.add(hpo_term.id)
                for ancestor in hpo_term.all_parents:
                    term_set.add(ancestor.id)
            except RuntimeError:
                pass
        for t in term_set:
            if t in vocab_idx:
                rows.append(i)
                cols.append(vocab_idx[t])

    n, v = len(samples), len(vocabulary)
    return (
        csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, v))
        if rows
        else csr_matrix((n, v), dtype=np.float32)
    )
