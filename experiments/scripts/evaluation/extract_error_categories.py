#!/usr/bin/env python3
"""
Create a mapping between paper IDs and their associated flaw/error categories.

The script reads the master flaw catalogue (`categorized_flaw_cleaned.csv`)
and optionally filters it to the set of papers present in `filtered_pairs.csv`.
The output is a tidy CSV where each row corresponds to one paper/category pair,
optionally including the originating flaw identifier.  This file can be consumed
by downstream analyses (for example, aggregating treatment effects by error
type).

Example:
    python extract_error_categories.py \
        --categorized_csv ../../data/categorized_flaw_cleaned.csv \
        --filtered_pairs_csv ../../sampled_data/ICLR2024/filtered_pairs.csv \
        --output_csv ../../data/paper_error_categories.csv
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


# Human-readable labels used in downstream summaries.
CATEGORY_GROUP_LABELS = {
    "1": "Empirical Evaluation Flaws",
    "2": "Methodological & Theoretical Flaws",
    "3": "Positioning & Contribution Flaws",
    "4": "Presentation & Reproducibility Flaws",
    "5": "Failure to Address Limitations or Ethical Concerns",
}

CATEGORY_LABELS = {
    "1a": "Insufficient Baselines/Comparisons",
    "1b": "Weak or Limited Scope of Experiments",
    "1c": "Lack of Necessary Ablation or Analysis",
    "1d": "Flawed Evaluation Metrics or Setup",
    "2a": "Fundamental Technical Limitation",
    "2b": "Missing or Incomplete Theoretical Foundation",
    "2c": "Technical or Mathematical Error",
    "3a": "Insufficient Novelty / Unacknowledged Prior Work",
    "3b": "Overstated Claims or Mismatch Between Claim and Evidence",
    "4a": "Lack of Clarity / Ambiguity",
    "4b": "Missing Implementation or Methodological Details",
    "5a": "Unacknowledged Technical Limitations",
    "5b": "Unaddressed Ethical or Societal Impact",
}


def parse_categories(raw_value: str) -> List[str]:
    """
    Normalise the category list stored in the CSV.

    The source file contains a mixture of quoting styles such as:
        1a
        "1a,1b"
        """1a,1b"""
    This helper returns a list of unique category identifiers (e.g., ["1a", "1b"]).
    """
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return []

    text = str(raw_value).strip()
    # Remove triple/double quotes and surrounding brackets.
    text = text.replace('"""', "").replace('"', "").replace("'", "")
    text = text.strip()

    # Split on commas or whitespace.
    parts = re.split(r"[,\s]+", text)
    categories = []
    for part in parts:
        token = part.strip()
        if not token:
            continue
        if token not in categories:
            categories.append(token)
    return categories


def build_rows(df: pd.DataFrame, paper_ids: Optional[Iterable[str]]) -> List[dict]:
    """
    Convert the wide flaw catalogue into a long-form table.
    """
    if paper_ids is not None:
        paper_ids = set(paper_ids)
        df = df[df["openreview_id"].isin(paper_ids)]

    rows: List[dict] = []
    for _, record in df.iterrows():
        paper_id = record["openreview_id"]
        flaw_id = record.get("flaw_id")
        categories = parse_categories(record.get("category_ids"))

        for category in categories:
            category_group = category[0]
            rows.append(
                {
                    "paper_id": paper_id,
                    "flaw_id": flaw_id,
                    "category_id": category,
                    "category_label": CATEGORY_LABELS.get(category),
                    "category_group": category_group,
                    "category_group_label": CATEGORY_GROUP_LABELS.get(category_group),
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract paper-to-error-category mappings from the flaw catalogue. "
            "If --filtered_pairs_csv is provided, the output is limited to those papers."
        )
    )
    parser.add_argument(
        "--categorized_csv",
        type=Path,
        required=True,
        help="Path to categorized_flaw_cleaned.csv",
    )
    parser.add_argument(
        "--filtered_pairs_csv",
        type=Path,
        required=False,
        help="Optional path to filtered_pairs.csv to limit the paper set",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Destination CSV path for the flattened mapping",
    )

    args = parser.parse_args()

    categorized_path: Path = args.categorized_csv
    filtered_pairs_path: Optional[Path] = args.filtered_pairs_csv
    output_path: Path = args.output_csv

    if not categorized_path.exists():
        raise FileNotFoundError(f"Categorized flaw file not found: {categorized_path}")

    categorized_df = pd.read_csv(categorized_path)

    if filtered_pairs_path:
        if not filtered_pairs_path.exists():
            raise FileNotFoundError(
                f"Filtered pairs file not found: {filtered_pairs_path}"
            )
        filtered_df = pd.read_csv(filtered_pairs_path)
        paper_ids = filtered_df["paperid"].unique()
    else:
        paper_ids = None

    rows = build_rows(categorized_df, paper_ids)
    if not rows:
        raise ValueError(
            "No category rows were generated. Check that the input files share paper IDs."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "paper_id",
        "flaw_id",
        "category_id",
        "category_label",
        "category_group",
        "category_group_label",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} category rows to {output_path}")


if __name__ == "__main__":
    main()

