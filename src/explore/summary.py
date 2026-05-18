"""Cohort and per-ERN summary reports from phenopacket samples."""

from collections import Counter
from pathlib import Path

import numpy as np
from pyhpo import Ontology

from utils.phenopackets import Sample


def _term_name(tid: str) -> str:
    try:
        return Ontology.get_hpo_object(tid).name
    except RuntimeError:
        return tid


def _fmt_counter(
    counter: Counter, total: int, indent: str = "    ", max_rows: int | None = 10
) -> str:
    items = counter.most_common(max_rows)
    lines = [f"{indent}{val:<40}  {cnt:>5}  ({cnt / total:.1%})" for val, cnt in items]
    if max_rows is not None and len(counter) > max_rows:
        lines.append(f"{indent}  … and {len(counter) - max_rows} more")
    return "\n".join(lines)


def _fmt_top_terms(
    hpo_freq: Counter,
    n_group: int,
    n_total: int,
    top_n: int,
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


def format_cohort_section(samples: list[Sample], top_n_terms: int) -> str:
    n = len(samples)
    tc = [len(s.terms) for s in samples]
    hpo_freq: Counter = Counter()
    for s in samples:
        hpo_freq.update(s.terms)

    lines = [
        "=" * 62,
        f"COHORT SUMMARY  ({n} patients)",
        "=" * 62,
        "",
        "ERNs:",
        _fmt_counter(Counter(s.ern for s in samples), n, max_rows=None),
        "",
        "Submitters:",
        _fmt_counter(Counter(s.submitter for s in samples), n),
        "",
        "HPO terms per patient (direct annotations):",
        f"    Mean {np.mean(tc):.1f}  ·  Median {int(np.median(tc))}"
        f"  ·  Min {min(tc)}  ·  Max {max(tc)}",
        "",
        *_fmt_top_terms(hpo_freq, n, n, top_n_terms, cohort_freq=hpo_freq),
    ]
    return "\n".join(lines)


def format_ern_section(samples: list[Sample], top_n_terms: int) -> str:
    n_total = len(samples)
    global_freq: Counter = Counter()
    for s in samples:
        global_freq.update(s.terms)

    ern_ids = sorted(set(s.ern for s in samples))
    sections = []
    for ern in ern_ids:
        members = [s for s in samples if s.ern == ern]
        n = len(members)
        tc = [len(s.terms) for s in members]
        hpo_freq: Counter = Counter()
        for s in members:
            hpo_freq.update(s.terms)

        lines = [
            "=" * 62,
            f"ERN: {ern}  ({n} patients)",
            "=" * 62,
            "",
            "Submitters:",
            _fmt_counter(Counter(s.submitter for s in members), n),
            "",
            "HPO terms per patient (direct annotations):",
            f"    Mean {np.mean(tc):.1f}  ·  Median {int(np.median(tc))}"
            f"  ·  Min {min(tc)}  ·  Max {max(tc)}",
            "",
            *_fmt_top_terms(hpo_freq, n, n_total, top_n_terms, cohort_freq=global_freq),
        ]
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def write_summary(
    samples: list[Sample],
    output_path: Path,
    top_terms: int = 15,
) -> None:
    cohort_text = format_cohort_section(samples, top_terms)
    ern_text = format_ern_section(samples, top_terms)

    log_path = output_path.with_suffix(".log")
    with open(log_path, "w") as fh:
        fh.write(cohort_text)
        fh.write("\n\n── PER-ERN PROFILES ─────────────────────────────────────────────────────────\n\n")
        fh.write(ern_text)

    print(f"Summary log saved to:         {log_path}")
