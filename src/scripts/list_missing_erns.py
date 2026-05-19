#!/usr/bin/env python3
"""
List phenopacket IDs that are missing an assigned ERN and write to CSV.

Usage:
    python src/scripts/list_missing_erns.py --input <phenopacket_dir> [--output <out.csv>]
"""

import argparse
import csv
import sys
from pathlib import Path

from utils.phenopackets import load_phenopacket, is_unlabeled


def list_missing_erns(phenopacket_dir: Path, output_path: Path) -> None:
    files = sorted(phenopacket_dir.rglob("*.json"))
    if not files:
        sys.exit(f"ERROR: no .json files found under {phenopacket_dir}")

    rows: list[tuple[str, str]] = []
    for f in files:
        phenopacket_id, _, _, ern = load_phenopacket(f)
        if is_unlabeled(ern):
            rows.append((phenopacket_id, ern))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "ern_value"])
        writer.writerows(rows)

    print(f"Scanned {len(files)} phenopackets — {len(rows)} missing ERN")
    print(f"Output written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", "-i", type=Path, required=True, metavar="DIR",
                        help="Directory containing phenopacket .json files (searched recursively)")
    parser.add_argument("--output", "-o", type=Path, default=Path("output/missing_erns.csv"),
                        metavar="FILE", help="Path for the output CSV (default: output/missing_erns.csv)")
    args = parser.parse_args()

    if not args.input.is_dir():
        sys.exit(f"ERROR: {args.input} is not a directory")

    list_missing_erns(args.input, args.output)


if __name__ == "__main__":
    main()
