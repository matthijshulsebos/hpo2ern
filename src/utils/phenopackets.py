"""Phenopacket data model, JSON parsing, and ontology-propagated HPO matrix."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from pyhpo import Ontology


@dataclass
class Sample:
    name: str
    terms: list[str]        # direct (non-excluded) HPO term IDs
    submitter: str = ""
    ern: str = ""


def load_phenopacket(filepath: Path) -> tuple[list[str], str, str]:
    """Return (hpo_terms, submitter, ern) from a phenopacket JSON file."""
    with open(filepath) as fh:
        data = json.load(fh)

    terms: list[str] = []
    for feature in data.get("phenotypicFeatures", []):
        if feature.get("excluded", False) or feature.get("negated", False):
            continue
        tid = feature.get("type", {}).get("id", "")
        if tid.startswith("HP:"):
            terms.append(tid)

    meta = data.get("metaData", {})
    submitter = meta.get("submittedBy", "unknown")
    refs = meta.get("externalReferences", [])
    ern = refs[0].get("id", "unknown") if refs else "unknown"

    return terms, submitter, ern


def load_all_phenopackets(phenopacket_dir: Path) -> list[Sample]:
    files = sorted(phenopacket_dir.rglob("*.json"))
    if not files:
        sys.exit(f"ERROR: no .json files found under {phenopacket_dir}")

    samples: list[Sample] = []
    n_skipped = 0
    for f in files:
        terms, submitter, ern = load_phenopacket(f)
        if not terms:
            print(f"[skip] {f.name}: no observed HPO terms", file=sys.stderr)
            n_skipped += 1
            continue
        samples.append(Sample(name=f.stem, terms=terms, submitter=submitter, ern=ern))

    print(
        f"Loaded {len(samples)} phenopackets"
        + (f" ({n_skipped} skipped: no HPO terms)" if n_skipped else "")
    )
    return samples


def validate_terms(samples: list[Sample]) -> list[Sample]:
    """Drop HPO IDs unknown to the loaded ontology; drop samples with none left."""
    valid: list[Sample] = []
    for sample in samples:
        ok = []
        for t in sample.terms:
            try:
                Ontology.get_hpo_object(t)
                ok.append(t)
            except RuntimeError:
                print(f"[warn] {sample.name}: unknown term {t!r} skipped", file=sys.stderr)
        if not ok:
            print(f"[skip] {sample.name}: no valid HPO terms after ontology check", file=sys.stderr)
            continue
        sample.terms = ok
        valid.append(sample)
    return valid


def build_ancestry_matrix(samples: list[Sample]) -> tuple[csr_matrix, list[str]]:
    """Build a sparse N×V binary matrix of ontology-propagated HPO presence.

    Every directly annotated term AND all its ontology ancestors are marked 1.
    V is the union of all such terms across the cohort.
    """
    patient_sets: list[set[str]] = []
    vocab: set[str] = set()

    for sample in tqdm(samples, desc="Propagating ontology ancestors"):
        term_set: set[str] = set()
        for tid in sample.terms:
            try:
                hpo_term = Ontology.get_hpo_object(tid)
                term_set.add(hpo_term.id)
                for ancestor in hpo_term.all_parents:
                    term_set.add(ancestor.id)
            except RuntimeError:
                pass
        patient_sets.append(term_set)
        vocab.update(term_set)

    vocabulary = sorted(vocab)
    idx = {t: i for i, t in enumerate(vocabulary)}

    rows, cols = [], []
    for i, term_set in enumerate(patient_sets):
        for t in term_set:
            rows.append(i)
            cols.append(idx[t])

    n, v = len(samples), len(vocabulary)
    X = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n, v),
    )
    print(
        f"  Binary matrix: {n} patients × {v} HPO terms (incl. ancestors)"
        f"  ·  density {X.nnz / (n * v):.1%}"
    )
    return X, vocabulary
