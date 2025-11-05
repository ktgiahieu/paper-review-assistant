#!/usr/bin/env python3
"""
Aggregate and visualize model performance across all reviews_* folders in sampled_data.

This script scans experiments/sampled_data for directories starting with "reviews_" and
collects evaluation results for three analyses:

1) Author/Affiliation factor (confounding): anonymized vs prestige vs unknown author
   - Source: evaluation_results/author_affiliation/summary_statistics.csv
   
2) Quality factor (causal): v1 vs latest differentiation
   - Source: evaluation_results/v1_latest/evaluation_detailed_results.json

3) v1 Human comparison: How well LLM scores match human scores for v1 papers
   - Source: evaluation_results/v1_human/ai_vs_human_results.json

It then produces a single comprehensive heatmap showing all models' performance
across all metrics and analyses.

Usage:
  python3 aggregate_model_performance.py \
      --sampled_root ../../sampled_data \
      --dataset ICLR2024 \
      --output_dir ../../sampled_data/combined_visualizations
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def find_review_roots(sampled_root: Path) -> List[Path]:
    review_roots = []
    for child in sampled_root.iterdir():
        if child.is_dir() and child.name.startswith("reviews_"):
            review_roots.append(child)
    return sorted(review_roots)


def safe_read_json(path: Path) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_v1_latest_results(review_root: Path, dataset: str) -> Optional[pd.DataFrame]:
    """
    Read evaluation_detailed_results.json and extract mean_difference and p-value per metric.
    
    Tries multiple possible paths:
    1. {review_root}/evaluation_results/v1_latest/evaluation_detailed_results.json
    2. {review_root}/{dataset}/evaluation_results/evaluation_detailed_results.json
    3. {review_root}/evaluation_results/evaluation_detailed_results.json
    """
    # Try multiple possible paths
    possible_paths = [
        review_root / "evaluation_results" / "v1_latest" / "evaluation_detailed_results.json",
        review_root / dataset / "evaluation_results" / "evaluation_detailed_results.json",
        review_root / "evaluation_results" / "evaluation_detailed_results.json",
    ]
    
    results = None
    for path in possible_paths:
        results = safe_read_json(path)
        if results:
            break
    
    if not results:
        return None

    # results_by_model dict. Our pipeline typically uses model_type "default".
    records: List[Dict] = []
    for model_type, metrics_map in results.items():
        for metric, stats in metrics_map.items():
            if "error" in stats or not isinstance(stats, dict):
                continue
            records.append({
                "metric": metric,
                "mean_difference": stats.get("mean_difference"),
                "p_value": stats.get("p_value"),
                "cohens_d": stats.get("cohens_d"),
                "n_pairs": stats.get("n_pairs"),
                "model": review_root.name,
            })

    if not records:
        return None
    return pd.DataFrame(records)


def load_author_affiliation_results(review_root: Path, dataset: str) -> Optional[pd.DataFrame]:
    """
    Read author affiliation summary CSV or JSON.
    
    Tries multiple possible paths:
    1. {review_root}/evaluation_results/author_affiliation/summary_statistics.csv
    2. {review_root}/evaluation_results/author_affiliation/statistical_results.json
    3. {review_root}/{dataset}/evaluation_results/author_affiliation/summary_statistics.csv
    """
    # Try CSV first
    csv_paths = [
        review_root / "evaluation_results" / "author_affiliation" / "summary_statistics.csv",
        review_root / dataset / "evaluation_results" / "author_affiliation" / "summary_statistics.csv",
    ]
    
    for csv_path in csv_paths:
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                return df
            except Exception:
                continue
    
    # Try JSON as fallback
    json_paths = [
        review_root / "evaluation_results" / "author_affiliation" / "statistical_results.json",
        review_root / dataset / "evaluation_results" / "author_affiliation" / "statistical_results.json",
    ]
    
    for json_path in json_paths:
        results = safe_read_json(json_path)
        if results:
            # Convert JSON structure to DataFrame format
            records = []
            for metric, metric_data in results.items():
                if not isinstance(metric_data, dict):
                    continue
                good = metric_data.get("good", {})
                bad = metric_data.get("bad", {})
                
                records.append({
                    "metric": metric,
                    "good_mean_diff": good.get("mean") if isinstance(good, dict) else None,
                    "good_p_value": good.get("p_value") if isinstance(good, dict) else None,
                    "good_significant": good.get("significant_at_0.05") if isinstance(good, dict) else None,
                    "bad_mean_diff": bad.get("mean") if isinstance(bad, dict) else None,
                    "bad_p_value": bad.get("p_value") if isinstance(bad, dict) else None,
                    "bad_significant": bad.get("significant_at_0.05") if isinstance(bad, dict) else None,
                })
            
            if records:
                return pd.DataFrame(records)
    
    return None


def load_v1_human_results(review_root: Path, dataset: str) -> Optional[pd.DataFrame]:
    """
    Read v1 human comparison results (MAE, correlation, etc.).
    
    Tries multiple possible paths:
    1. {review_root}/evaluation_results/v1_human/ai_vs_human_results.json
    2. {review_root}/evaluation_results/v1_human/ai_vs_human_results.csv
    3. {review_root}/{dataset}/evaluation_results/v1_human/ai_vs_human_results.json
    """
    # Try JSON first
    json_paths = [
        review_root / "evaluation_results" / "v1_human" / "ai_vs_human_results.json",
        review_root / dataset / "evaluation_results" / "v1_human" / "ai_vs_human_results.json",
    ]
    
    for json_path in json_paths:
        results = safe_read_json(json_path)
        if results:
            records = []
            for metric, stats in results.items():
                if not isinstance(stats, dict):
                    continue
                records.append({
                    "metric": metric,
                    "mae": stats.get("mae"),
                    "rmse": stats.get("rmse"),
                    "correlation": stats.get("correlation"),
                    "n_pairs": stats.get("n_pairs"),
                })
            if records:
                return pd.DataFrame(records)
    
    # Try CSV as fallback
    csv_paths = [
        review_root / "evaluation_results" / "v1_human" / "ai_vs_human_results.csv",
        review_root / dataset / "evaluation_results" / "v1_human" / "ai_vs_human_results.csv",
    ]
    
    for csv_path in csv_paths:
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                return df
            except Exception:
                continue
    
    return None


def pretty_model_name(review_root: Path) -> str:
    # reviews_gemini_2-5_pro__generic_prompt -> Gemini 2.5 Pro (generic_prompt)
    name = review_root.name.replace("reviews_", "")
    parts = name.split("__")
    model = parts[0].replace("_", " ")
    suffix = f" ({parts[1]})" if len(parts) > 1 else ""
    return (model + suffix).title()


def create_comprehensive_heatmap(
    all_aff_df: pd.DataFrame,
    all_v1_latest_df: pd.DataFrame,
    all_v1_human_df: pd.DataFrame,
    output_dir: Path
):
    """
    Create a single comprehensive heatmap showing all analyses.
    
    Structure:
    - Rows: Metrics × Models (e.g., Soundness_Model1, Soundness_Model2, ...)
    - Columns: Analysis types
    - Analyses: Prestige Author Effect, Unknown Author Effect, v1 vs Latest Effect, v1 Human MAE
    """
    metrics = ["soundness", "presentation", "contribution", "rating"]
    
    # Collect all models
    all_models = set()
    if not all_aff_df.empty:
        all_models.update(all_aff_df["model"].unique())
    if not all_v1_latest_df.empty:
        all_models.update([pretty_model_name(Path(m)) for m in all_v1_latest_df["model"].unique()])
    if not all_v1_human_df.empty:
        all_models.update(all_v1_human_df["model"].unique())
    
    all_models = sorted(list(all_models))
    
    if not all_models:
        print("Warning: No models found to plot")
        return
    
    # Analysis types in order (columns)
    analysis_types = [
        ("Prestige Author\nEffect", "prestige"),  # Prestige - anonymized camera-ready
        ("Unknown Author\nEffect", "unknown"),    # Unknown - anonymized camera-ready
        ("v1 vs Latest\nEffect", "v1_latest"),    # Latest - v1 (quality factor)
        ("v1 Human MAE", "v1_human_mae"),         # MAE between AI and human for v1
    ]
    
    col_labels = [label for label, _ in analysis_types]
    
    # Build the heatmap matrix: rows = Metrics × Models
    heatmap_data = []
    row_labels = []
    
    for model in all_models:
        for metric in metrics:
            row_labels.append(f"{metric.capitalize()}\n{model}")
            row_data = []
            
            for analysis_label, analysis_key in analysis_types:
                value = None
                
                if analysis_key == "prestige":
                    # Prestige author effect (prestige - anonymized camera-ready)
                    aff_row = all_aff_df[
                        (all_aff_df["model"] == model) & 
                        (all_aff_df["metric"] == metric)
                    ]
                    if not aff_row.empty:
                        value = aff_row.iloc[0].get("good_mean_diff")
                
                elif analysis_key == "unknown":
                    # Unknown author effect (unknown - anonymized camera-ready)
                    aff_row = all_aff_df[
                        (all_aff_df["model"] == model) & 
                        (all_aff_df["metric"] == metric)
                    ]
                    if not aff_row.empty:
                        value = aff_row.iloc[0].get("bad_mean_diff")
                
                elif analysis_key == "v1_latest":
                    # v1 vs latest effect (latest - v1)
                    model_raw = None
                    # Find the raw model name from v1_latest_df
                    for raw_name in all_v1_latest_df["model"].unique():
                        if pretty_model_name(Path(raw_name)) == model:
                            model_raw = raw_name
                            break
                    if model_raw:
                        v1_row = all_v1_latest_df[
                            (all_v1_latest_df["model"] == model_raw) & 
                            (all_v1_latest_df["metric"] == metric)
                        ]
                        if not v1_row.empty:
                            value = v1_row.iloc[0].get("mean_difference")
                
                elif analysis_key == "v1_human_mae":
                    # v1 human MAE
                    human_row = all_v1_human_df[
                        (all_v1_human_df["model"] == model) & 
                        (all_v1_human_df["metric"] == metric)
                    ]
                    if not human_row.empty:
                        value = human_row.iloc[0].get("mae")
                
                row_data.append(float(value) if pd.notna(value) else np.nan)
            
            heatmap_data.append(row_data)
    
    # Create DataFrame
    heatmap_df = pd.DataFrame(heatmap_data, index=row_labels, columns=col_labels)
    
    # Create the heatmap
    plt.figure(figsize=(max(10, len(col_labels) * 2.5), max(8, len(row_labels) * 0.4)))
    
    # Create a diverging colormap for effects, and sequential for MAE
    # We'll use a combined approach: center at 0 for effects, but MAE should be positive
    # For simplicity, use a diverging colormap centered at 0
    vmin = heatmap_df.min().min()
    vmax = heatmap_df.max().max()
    center = 0 if vmin < 0 else None
    
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r" if center is not None else "YlOrRd",
        center=center,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Effect Size / MAE"},
        xticklabels=True,
        yticklabels=True,
    )
    
    plt.title("Comprehensive Model Performance Heatmap\nAll Metrics and Analyses Combined", 
              fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Analysis Types", fontsize=12, fontweight="bold")
    plt.ylabel("Metrics × Models", fontsize=12, fontweight="bold")
    plt.xticks(rotation=0, ha="center")
    plt.yticks(rotation=0)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "comprehensive_performance_heatmap.png"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and visualize model performance across reviews_* folders"
    )
    parser.add_argument(
        "--sampled_root",
        type=str,
        default="../../sampled_data",
        help="Root directory containing reviews_* folders (default: ../../sampled_data)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ICLR2024",
        help="Dataset subfolder to read (default: ICLR2024)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../sampled_data/combined_visualizations",
        help="Directory to save combined plots"
    )

    args = parser.parse_args()
    sampled_root = Path(args.sampled_root)
    output_dir = Path(args.output_dir)

    print("Scanning:", sampled_root)
    review_roots = find_review_roots(sampled_root)
    print(f"Found {len(review_roots)} review folders")

    all_aff_rows: List[pd.DataFrame] = []
    all_v1_latest_rows: List[pd.DataFrame] = []
    all_v1_human_rows: List[pd.DataFrame] = []

    for rr in review_roots:
        model_label = pretty_model_name(rr)
        
        # Author/Affiliation
        df_aff = load_author_affiliation_results(rr, args.dataset)
        if df_aff is not None and not df_aff.empty:
            df_aff = df_aff.copy()
            df_aff["model"] = model_label
            all_aff_rows.append(df_aff)

        # v1 vs latest
        df_v1 = load_v1_latest_results(rr, args.dataset)
        if df_v1 is not None and not df_v1.empty:
            df_v1 = df_v1.copy()
            df_v1["model"] = rr.name  # preserve raw name; pretty later
            all_v1_latest_rows.append(df_v1)

        # v1 human comparison
        df_v1_human = load_v1_human_results(rr, args.dataset)
        if df_v1_human is not None and not df_v1_human.empty:
            df_v1_human = df_v1_human.copy()
            df_v1_human["model"] = model_label
            all_v1_human_rows.append(df_v1_human)

    # Combine all data
    print("\n" + "="*80)
    print("Collecting data...")
    print("="*80)
    
    aff_df = pd.DataFrame()
    if all_aff_rows:
        aff_df = pd.concat(all_aff_rows, ignore_index=True)
        print(f"✅ Author/Affiliation data: {len(aff_df)} rows from {len(all_aff_rows)} models")
    else:
        print("⚠️  No author/affiliation summaries found.")

    v1_latest_df = pd.DataFrame()
    if all_v1_latest_rows:
        v1_latest_df = pd.concat(all_v1_latest_rows, ignore_index=True)
        print(f"✅ v1/Latest data: {len(v1_latest_df)} rows from {len(all_v1_latest_rows)} models")
    else:
        print("⚠️  No v1/latest evaluation results found.")

    v1_human_df = pd.DataFrame()
    if all_v1_human_rows:
        v1_human_df = pd.concat(all_v1_human_rows, ignore_index=True)
        print(f"✅ v1/Human comparison data: {len(v1_human_df)} rows from {len(all_v1_human_rows)} models")
    else:
        print("⚠️  No v1/human comparison results found.")

    # Create comprehensive heatmap
    print("\n" + "="*80)
    print("Generating comprehensive heatmap...")
    print("="*80)
    
    create_comprehensive_heatmap(aff_df, v1_latest_df, v1_human_df, output_dir)
    
    print("\n" + "="*80)
    print("✅ Visualization complete!")
    print("="*80)


if __name__ == "__main__":
    main()


