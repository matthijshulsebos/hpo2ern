"""Summarise ERN cohort composition of phenopackets in a directory.

Usage (from src/):
    python utils/statistics.py --input ../data/some-folder/
    python utils/statistics.py --input ../data/  # recurses into subdirectories
"""

import argparse
import sys

from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.phenopackets import is_unlabeled, load_all_phenopackets


def main() -> None:
    p = argparse.ArgumentParser(description="ERN statistics for a phenopacket directory")
    p.add_argument("--input", "-i", required=True, type=Path, metavar="DIR")
    args = p.parse_args()

    samples = load_all_phenopackets(args.input)

    missing = [s for s in samples if is_unlabeled(s.ern)]
    labeled = [s for s in samples if not is_unlabeled(s.ern)]

    print(f"\nTotal samples : {len(samples)}")
    print(f"  Labeled     : {len(labeled)}")
    print(f"  Missing ERN : {len(missing)}")

    if labeled:
        counts = Counter(s.ern for s in labeled)
        print(f"\nERN breakdown ({len(counts)} classes):")
        width = max(len(ern) for ern in counts)
        for ern, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {ern:<{width}}  {n:>5}  ({100 * n / len(samples):>5.1f}%)")


if __name__ == "__main__":
    main()
