#!/usr/bin/env python3
"""
Generate Cohen's d summaries comparing AI reviewer scores against human scores.

This script consumes the `ai_vs_human_detailed.csv` file produced by the v1 vs
human evaluation pipeline and creates:

- `ai_vs_human_effects.csv`
- `ai_vs_human_effects.json`
- `cohens_d_summary.png` (with expectation-coloured bars)

Example usage:

    python analyze_ai_vs_human_effects.py \
        --input_csv /path/to/ai_vs_human_detailed.csv \
        --output_dir /path/to/v1_human \
        --treatment_label "AI Reviewer" \
        --baseline_label "Human Reviewer"
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ERROR_BAR_SCALE = 0.2
COLOR_MATCH = "#2ecc71"
COLOR_MISMATCH = "#e74c3c"
COLOR_NEUTRAL = "#95a5a6"


def evaluate_expectation(
    cohen_d: Optional[float], expectation: str, tolerance: float
) -> Optional[bool]:
    """Determine whether the observed effect matches the expected direction."""
    if cohen_d is None or math.isnan(cohen_d):
        return None

    if expectation == "positive":
        return cohen_d > tolerance
    if expectation == "negative":
        return cohen_d < -tolerance
    if expectation == "zero":
        return abs(cohen_d) <= tolerance
    return None


def cohen_d_se_one_sample(cohen_d: float, n: int) -> Optional[float]:
    """
    Approximate standard error for Cohen's d in a one-sample / paired design.

    Based on the large-sample approximation:
        Var(d) ≈ (1 / n) + (d^2 / (2n))
    """
    if n <= 1:
        return None
    variance = (1.0 / n) + ((cohen_d ** 2) / (2.0 * n))
    return math.sqrt(variance)


def analyse_metric(
    df_metric: pd.DataFrame,
    treatment_label: str,
    baseline_label: str,
    expected_direction: str,
    tolerance: float,
) -> Dict:
    """Compute summary statistics for a single metric."""
    diffs = df_metric["ai_score"] - df_metric["human_score"]
    diffs = diffs.dropna()

    n = int(len(diffs))
    if n == 0:
        raise ValueError("No overlapping AI/Human scores found for this metric.")

    mean_diff = float(diffs.mean())
    std_diff = float(diffs.std(ddof=1)) if n > 1 else 0.0

    if std_diff > 0:
        cohen_d = mean_diff / std_diff
    else:
        cohen_d = 0.0

    se = cohen_d_se_one_sample(cohen_d, n)
    if se is not None:
        ci_low = cohen_d - 1.96 * se
        ci_high = cohen_d + 1.96 * se
    else:
        ci_low = ci_high = None

    matches = evaluate_expectation(cohen_d, expected_direction, tolerance)

    return {
        "comparison": f"{treatment_label} vs {baseline_label}",
        "treatment_label": treatment_label,
        "baseline_label": baseline_label,
        "expected_direction": expected_direction,
        "matches_expectation": matches,
        "n_pairs": n,
        "mean_difference": mean_diff,
        "std_difference": std_diff if std_diff > 0 else None,
        "cohen_d": cohen_d,
        "standard_error": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def create_effect_plot(
    summaries: Dict[str, Dict],
    output_path: Path,
    treatment_label: str,
    baseline_label: str,
):
    """Create colour-coded Cohen's d bar plot."""
    metrics = list(summaries.keys())
    if not metrics:
        raise ValueError("No metrics were processed; cannot create plot.")

    num_metrics = len(metrics)
    ncols = min(2, num_metrics)
    nrows = math.ceil(num_metrics / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 4.5 * nrows),
    )
    if num_metrics == 1:
        axes = np.array([[axes]]) if not isinstance(axes, np.ndarray) else axes.reshape(1, 1)
    axes = axes.flatten()

    legend_handles = []

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        result = summaries[metric]

        cohen_d = result["cohen_d"]
        se = result.get("standard_error")
        if se is not None:
            lower = cohen_d - result["ci_low"] if result["ci_low"] is not None else 0.0
            upper = result["ci_high"] - cohen_d if result["ci_high"] is not None else 0.0
        else:
            lower = upper = 0.0

        lower *= ERROR_BAR_SCALE
        upper *= ERROR_BAR_SCALE

        matches = result["matches_expectation"]
        if matches is None:
            color = COLOR_NEUTRAL
        elif matches:
            color = COLOR_MATCH
        else:
            color = COLOR_MISMATCH

        bars = ax.bar(
            [0],
            [cohen_d],
            yerr=[[lower], [upper]],
            capsize=8,
            color=color,
            edgecolor="black",
            alpha=0.85,
        )

        bar = bars[0]
        text_y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            text_y,
            f"d={cohen_d:.2f}",
            ha="center",
            va="bottom" if text_y >= 0 else "top",
            fontsize=11,
            fontweight="bold",
        )

        expectation = result["expected_direction"]
        if expectation:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                -0.1,
                f"Expected: {expectation}",
                ha="center",
                va="top",
                fontsize=10,
                rotation=45,
                color="dimgray",
                transform=ax.get_xaxis_transform(),
            )

        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks([0])
        ax.set_xticklabels([f"{treatment_label} − {baseline_label}"])
        ax.set_ylabel("Cohen's d")
        ax.set_title(metric.title())
        ax.grid(True, axis="y", alpha=0.3)

    # Hide unused subplots if any
    for j in range(len(metrics), len(axes)):
        axes[j].set_visible(False)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_MATCH, ec="black", label="Matches expectation"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_MISMATCH, ec="black", label="Contradicts expectation"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_NEUTRAL, ec="black", label="No expectation"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=12)

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    fig.suptitle(
        f"Cohen's d: {treatment_label} vs {baseline_label}",
        fontsize=18,
        fontweight="bold",
    )
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Cohen's d summaries for AI vs human reviewer scores."
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="Path to ai_vs_human_detailed.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to store outputs (CSV, JSON, PNG)",
    )
    parser.add_argument(
        "--treatment_label",
        type=str,
        default="AI Reviewer",
        help="Label for the AI reviewer scores",
    )
    parser.add_argument(
        "--baseline_label",
        type=str,
        default="Human Reviewer",
        help="Label for the human scores",
    )
    parser.add_argument(
        "--expected_direction",
        choices=["positive", "negative", "zero"],
        default="zero",
        help="Expected direction of the effect (used for colour coding).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="Tolerance around zero when expectation is 'zero' (default: 0.1).",
    )

    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    required_columns = {"metric", "human_score", "ai_score"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    df["metric"] = df["metric"].astype(str)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: Dict[str, Dict] = {}
    rows: List[Dict] = []

    for metric, df_metric in df.groupby("metric"):
        summary = analyse_metric(
            df_metric,
            treatment_label=args.treatment_label,
            baseline_label=args.baseline_label,
            expected_direction=args.expected_direction,
            tolerance=args.tolerance,
        )
        summaries[metric] = summary
        rows.append(
            {
                "metric": metric,
                **summary,
            }
        )

    summary_df = pd.DataFrame(rows)
    csv_path = output_dir / "ai_vs_human_effects.csv"
    json_path = output_dir / "ai_vs_human_effects.json"
    plot_path = output_dir / "cohens_d_summary.png"

    summary_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    create_effect_plot(
        summaries,
        plot_path,
        treatment_label=args.treatment_label,
        baseline_label=args.baseline_label,
    )

    print(f"Saved summary CSV to: {csv_path}")
    print(f"Saved summary JSON to: {json_path}")
    print(f"Saved Cohen's d plot to: {plot_path}")


if __name__ == "__main__":
    main()

