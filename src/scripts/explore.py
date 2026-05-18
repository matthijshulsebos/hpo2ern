#!/usr/bin/env python3
"""
Generate cohort summary and distribution plots from phenopackets.

Usage:
    python src/scripts/explore.py --input <phenopacket_dir> [--output <dir>] [--top-terms N]
"""

import argparse
import datetime
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_DISPLAY_ENV", "FALSE")
os.environ.setdefault("KMP_WARNINGS", "FALSE")
warnings.filterwarnings("ignore", category=UserWarning)

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyhpo import Ontology

from utils.phenopackets import load_all_phenopackets, validate_terms
from explore.cohort_plots import (
    plot_ern_distribution,
    plot_hpo_term_counts,
    plot_submitter_distribution,
    plot_top_hpo_terms,
)
from explore.summary import write_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input",  "-i", required=True, type=Path, metavar="DIR",
                   help="Directory with phenopacket JSON files")
    p.add_argument("--output", "-o", type=Path, default=None, metavar="DIR",
                   help="Output directory (default: output/<timestamp>/)")
    p.add_argument("--top-terms", type=int, default=25, metavar="N",
                   help="Top HPO terms to show in summary and bar chart")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.output is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = Path("output") / f"{timestamp}_explore"
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
