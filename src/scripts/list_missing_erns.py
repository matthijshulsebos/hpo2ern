#!/usr/bin/env python3
"""
List phenopacket IDs that are missing an assigned ERN.

"Missing" means either:
  - metaData.externalReferences is absent or empty  →  ern_value left blank
  - the ERN value is a known "not applicable" sentinel  →  ern_value shown as-is

Usage:
    python src/utils/list_missing_erns.py --input <phenopacket_dir> [--output <out.csv>]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

_NOT_APPLICABLE = {"Not_Applicable", "Not applicable"}


def _ern_for_file(filepath: Path) -> tuple[str, str] | None:
    """Return (phenopacket_id, ern_value) if the ERN is missing or not applicable.

    Returns None when the ERN is properly assigned.
    ern_value is an empty string when the property is entirely absent.
    """
    with open(filepath) as fh:
        data = json.load(fh)

    phenopacket_id = data.get("id", filepath.stem)
    refs = data.get("metaData", {}).get("externalReferences", [])

    if not refs or "id" not in refs[0]:
        return phenopacket_id, ""

    ern = refs[0]["id"]
    if ern in _NOT_APPLICABLE:
        return phenopacket_id, ern

    return None


def list_missing_erns(phenopacket_dir: Path, output_path: Path) -> None:
    files = sorted(phenopacket_dir.rglob("*.json"))
    if not files:
        sys.exit(f"ERROR: no .json files found under {phenopacket_dir}")

    rows: list[tuple[str, str]] = []
    for f in files:
        result = _ern_for_file(f)
        if result is not None:
            rows.append(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "ern_value"])
        writer.writerows(rows)

    print(f"Scanned {len(files)} phenopackets — {len(rows)} missing ERN")
    print(f"Output written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory containing phenopacket .json files (searched recursively)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output/missing_erns.csv"),
        metavar="FILE",
        help="Path for the output CSV (default: output/missing_erns.csv)",
    )
    args = parser.parse_args()

    if not args.input.is_dir():
        sys.exit(f"ERROR: {args.input} is not a directory")

    list_missing_erns(args.input, args.output)


if __name__ == "__main__":
    main()
