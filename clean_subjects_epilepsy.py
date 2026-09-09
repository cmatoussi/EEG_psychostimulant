"""Utility to normalize the Epilepsy column to binary values."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable


DEFAULT_INPUT = Path("data/results/dl/subjects.csv")
DEFAULT_OUTPUT = Path("data/results/dl/subjects_cleaned.csv")
DEFAULT_LIMIT = 978


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize Epilepsy column to 0/1 and write cleaned CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Maximum number of subjects to keep (default: {DEFAULT_LIMIT})",
    )
    return parser.parse_args()


def normalize_epilepsy(value: str) -> int:
    """Return 0 or 1 from a raw Epilepsy field."""
    text = (value or "").strip()
    if text.startswith("0"):
        return 0
    if text.startswith("1"):
        return 1
    raise ValueError(f"Unexpected Epilepsy value: {value!r}")


def clean_rows(rows: Iterable[Dict[str, str]], limit: int | None) -> Iterable[Dict[str, str]]:
    for idx, row in enumerate(rows):
        if limit is not None and idx >= limit:
            break
        row = dict(row)
        row["Epilepsy"] = str(normalize_epilepsy(row.get("Epilepsy", "")))
        yield row


def main() -> None:
    args = parse_args()
    with args.input.open(newline="", encoding="utf-8") as rf:
        reader = csv.DictReader(rf, delimiter=";")
        fieldnames = reader.fieldnames
        if not fieldnames or "Epilepsy" not in fieldnames:
            raise ValueError("Epilepsy column is missing from the input file.")
        cleaned_rows = list(clean_rows(reader, limit=args.limit))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as wf:
        writer = csv.DictWriter(wf, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(cleaned_rows)


if __name__ == "__main__":
    main()
