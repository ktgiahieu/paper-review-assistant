#!/usr/bin/env python3
"""
Evaluate the effect of planted errors on review scores for category-specific reviews.

This script analyzes reviews from category-specific review formats (e.g., 2a, 2b) that may have
non-standard score schemas (e.g., nested practicality_assessment scores).

Compares:
- Latest (baseline) vs Planted Error

Usage:
    python evaluate_category_reviews.py \
        --reviews_dir reviews_by_category/NeurIPS2024 \
        --category 2a \
        --output_dir evaluation_results/category_2a
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set style
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    plt.style.use("ggplot")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
COLOR_MATCH = "#2ecc71"
COLOR_MISMATCH = "#e74c3c"
COLOR_NEUTRAL = "#95a5a6"
ERROR_BAR_SCALE = 1

def extract_numerical_value(score_str) -> Optional[float]:
    """Extract numerical value from score string or number."""
    if score_str is None:
        return None
    
    if isinstance(score_str, (int, float)):
        return float(score_str)
    
    if not isinstance(score_str, str):
        try:
            return float(score_str)
        except:
            return None
    
    if not score_str.strip():
        return None
    
    match = re.match(r'^(\d+(?:\.\d+)?)', str(score_str).strip())
    if match:
        return float(match.group(1))
    
    return None

def extract_all_scores_from_review(review_file: Path) -> Optional[Dict[str, Any]]:
    """
    Extract all numerical scores from a category-specific review JSON file.
    
    Handles multiple formats:
    1. Standard format: soundness, presentation, contribution, rating
    2. Category 2a: practicality_assessment (nested), presentation, contribution, overall_score, confidence
    3. Category 2b: theoretical_rigor fields, soundness, presentation, contribution, overall_score, confidence
    
    Returns dict with all found scores, using dot notation for nested fields.
    """
    try:
        with open(review_file, 'r', encoding='utf-8') as f:
            review_data = json.load(f)
        
        if not review_data.get('success', False):
            return None
        
        scores = {}
        
        # Standard top-level scores
        standard_fields = ['soundness', 'presentation', 'contribution', 'rating', 'overall_score', 'confidence']
        for field in standard_fields:
            value = review_data.get(field)
            parsed = extract_numerical_value(value)
            if parsed is not None:
                scores[field] = parsed
        
        # Handle nested practicality_assessment (category 2a)
        if 'practicality_assessment' in review_data:
            pa = review_data['practicality_assessment']
            if isinstance(pa, dict):
                # Handle both alias names (1_input_realism) and regular names (input_realism)
                for key, value in pa.items():
                    if isinstance(value, dict):
                        # Extract score from nested object
                        score = value.get('score') if isinstance(value.get('score'), (int, float)) else None
                        if score is not None:
                            # Normalize key name (remove leading number and underscore if present)
                            normalized_key = re.sub(r'^\d+_', '', key)
                            scores[f'practicality_assessment.{normalized_key}'] = float(score)
                    elif isinstance(value, (int, float)):
                        # Direct score value
                        normalized_key = re.sub(r'^\d+_', '', key)
                        scores[f'practicality_assessment.{normalized_key}'] = float(value)
        
        # Handle theoretical rigor fields (category 2b)
        theoretical_fields = [
            'theoretical_rigor_summary',  # Skip this, it's text
            'assumption_justification_score',
            'proof_completeness_score',
            'heuristic_linkage_score',
            'definition_rigor_score'
        ]
        for field in theoretical_fields:
            if field == 'theoretical_rigor_summary':
                continue  # Skip text fields
            value = review_data.get(field)
            parsed = extract_numerical_value(value)
            if parsed is not None:
                scores[field] = parsed
        
        # Fallback: try to extract from any nested structure
        # Look for common patterns like "score" fields in nested objects
        def extract_nested_scores(obj, prefix=""):
            """Recursively extract score fields from nested structures."""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == 'score' and isinstance(value, (int, float)):
                        # Found a score field
                        if prefix:
                            scores[f"{prefix}.score"] = float(value)
                        else:
                            scores["score"] = float(value)
                    elif isinstance(value, dict):
                        # Recurse into nested dict
                        new_prefix = f"{prefix}.{key}" if prefix else key
                        extract_nested_scores(value, new_prefix)
                    elif isinstance(value, list):
                        # Skip lists for now
                        pass
        
        # Only do recursive extraction if we haven't found many scores yet
        if len(scores) < 3:
            extract_nested_scores(review_data)
        
        return scores if scores else None
    
    except Exception as e:
        return None

def collect_scores_from_category_folder(reviews_dir: Path, category_id: str, folder_name: str) -> pd.DataFrame:
    """
    Collect all scores from a specific category folder.
    
    Returns DataFrame with columns: paper_id, run_id, version_id, and all score fields found.
    """
    folder_path = reviews_dir / category_id / folder_name
    if not folder_path.exists():
        print(f"Warning: Folder {folder_path} does not exist")
        return pd.DataFrame()
    
    records = []
    
    # Find all paper directories
    paper_dirs = [d for d in folder_path.iterdir() if d.is_dir()]
    
    for paper_dir in paper_dirs:
        paper_id = paper_dir.name
        
        # Find all review files
        review_files = list(paper_dir.glob("review_run*.json"))
        # Also check for version-specific files (e.g., flaw_X_review_runY.json)
        version_files = list(paper_dir.glob("*_review_run*.json"))
        review_files.extend([f for f in version_files if f not in review_files])
        
        for review_file in review_files:
            # Extract run_id and version_id from filename
            run_id = 0
            version_id = None
            
            # Try version-specific format: {version_id}_review_run{run_id}.json
            match = re.search(r'(.+?)_review_run(\d+)\.json', review_file.name)
            if match:
                potential_version_id = match.group(1)
                run_id = int(match.group(2))
                # Check if it's not just "review" (old format)
                if potential_version_id != "review":
                    version_id = potential_version_id
            else:
                # Try old format: review_run{run_id}.json
                match = re.search(r'review_run(\d+)\.json', review_file.name)
                if match:
                    run_id = int(match.group(1))
            
            scores = extract_all_scores_from_review(review_file)
            if scores:
                record = {
                    'paper_id': paper_id,
                    'run_id': run_id,
                    'folder': folder_name,
                    **scores
                }
                if version_id:
                    record['version_id'] = version_id
                records.append(record)
    
    return pd.DataFrame(records)

def calculate_differences(df: pd.DataFrame, baseline_folder: str, treatment_folder: str) -> pd.DataFrame:
    """
    Calculate differences between baseline and treatment for each paper and metric.
    
    For papers with version_id, matches by version_id. Otherwise matches by paper_id and run_id.
    """
    results = []
    
    # Get all score columns (exclude metadata columns)
    metadata_cols = ['paper_id', 'run_id', 'folder', 'version_id']
    score_cols = [col for col in df.columns if col not in metadata_cols]
    
    if not score_cols:
        print("Warning: No score columns found in data")
        return pd.DataFrame()
    
    # Check if we have version_id column
    has_version_id = 'version_id' in df.columns
    
    if has_version_id:
        group_keys = ['paper_id', 'run_id', 'version_id']
    else:
        group_keys = ['paper_id', 'run_id']
    
    for group_key, group in df.groupby(group_keys):
        if has_version_id:
            paper_id, run_id, version_id = group_key
        else:
            paper_id, run_id = group_key
            version_id = None
        
        # Get scores for each version
        baseline_scores = group[group['folder'] == baseline_folder]
        treatment_scores = group[group['folder'] == treatment_folder]
        
        # Fallback: if baseline doesn't have version_id but treatment does, use first baseline
        if baseline_scores.empty and has_version_id:
            fallback_mask = (
                (df['folder'] == baseline_folder) &
                (df['paper_id'] == paper_id) &
                (df['run_id'] == run_id)
            )
            fallback_scores = df[fallback_mask]
            if not fallback_scores.empty:
                baseline_scores = fallback_scores.iloc[[0]]
        
        if baseline_scores.empty or treatment_scores.empty:
            continue
        
        baseline_row = baseline_scores.iloc[0]
        treatment_row = treatment_scores.iloc[0]
        
        # Calculate differences for each score metric
        for metric in score_cols:
            baseline_val = baseline_row.get(metric)
            treatment_val = treatment_row.get(metric)
            
            if baseline_val is None or pd.isna(baseline_val):
                continue
            if treatment_val is None or pd.isna(treatment_val):
                continue
            
            result = {
                'paper_id': paper_id,
                'run_id': run_id,
                'metric': metric,
                'baseline_score': float(baseline_val),
                'treatment_score': float(treatment_val),
                'difference': float(treatment_val) - float(baseline_val)
            }
            if version_id:
                result['version_id'] = version_id
            
            results.append(result)
    
    return pd.DataFrame(results)

def pooled_standard_deviation(treatment: np.ndarray, control: np.ndarray) -> Optional[float]:
    """Compute pooled standard deviation for two independent samples."""
    n_t = len(treatment)
    n_c = len(control)
    if n_t <= 1 or n_c <= 1:
        return None
    
    var_t = np.var(treatment, ddof=1)
    var_c = np.var(control, ddof=1)
    pooled_var = ((n_t - 1) * var_t + (n_c - 1) * var_c) / (n_t + n_c - 2)
    return np.sqrt(pooled_var) if pooled_var > 0 else 0.0

def cohen_d_standard_error(cohen_d: float, n_t: int, n_c: int) -> Optional[float]:
    """Approximate standard error for Cohen's d for independent samples."""
    if n_t <= 1 or n_c <= 1:
        return None
    return np.sqrt((n_t + n_c) / (n_t * n_c) + (cohen_d ** 2) / (2 * (n_t + n_c - 2)))

def compute_statistics(df_diffs: pd.DataFrame) -> Dict[str, Dict]:
    """
    Compute Cohen's d effect sizes for each metric.
    
    Returns dict mapping metric name to statistics dict.
    """
    results = {}
    
    metrics = df_diffs['metric'].unique()
    
    for metric in metrics:
        metric_rows = df_diffs[df_diffs['metric'] == metric]
        
        if metric_rows.empty:
            continue
        
        valid_rows = metric_rows[
            metric_rows['baseline_score'].notna() &
            metric_rows['treatment_score'].notna()
        ]
        
        if valid_rows.empty:
            continue
        
        baseline_scores = valid_rows['baseline_score'].astype(float).values
        treatment_scores = valid_rows['treatment_score'].astype(float).values
        
        if len(baseline_scores) == 0 or len(treatment_scores) == 0:
            continue
        
        diffs = treatment_scores - baseline_scores
        mean_diff = float(np.mean(diffs))
        pooled_sd = pooled_standard_deviation(treatment_scores, baseline_scores)
        
        if pooled_sd is None:
            cohen_d = np.nan
        elif pooled_sd == 0:
            cohen_d = float(np.sign(mean_diff)) * np.inf if mean_diff != 0 else 0.0
        else:
            cohen_d = float(mean_diff / pooled_sd)
        
        se = cohen_d_standard_error(cohen_d, len(treatment_scores), len(baseline_scores)) if np.isfinite(cohen_d) else None
        ci_low = None
        ci_high = None
        if se is not None:
            ci_low = float(cohen_d - 1.96 * se)
            ci_high = float(cohen_d + 1.96 * se)
        
        results[metric] = {
            'n_pairs': int(len(valid_rows)),
            'baseline_mean': float(np.mean(baseline_scores)),
            'treatment_mean': float(np.mean(treatment_scores)),
            'mean_difference': mean_diff,
            'cohen_d': float(cohen_d) if np.isfinite(cohen_d) else None,
            'standard_error': float(se) if se is not None else None,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'diffs': diffs.tolist()
        }
    
    return results

def create_effect_size_plot(stats_results: Dict[str, Dict], output_dir: Path, category_id: str):
    """Create bar plot showing Cohen's d with error bars for each metric."""
    if not stats_results:
        print("No statistics to plot")
        return
    
    metrics = sorted(stats_results.keys())
    if not metrics:
        return
    
    # Group metrics by category for better organization
    standard_metrics = []
    practicality_metrics = []
    theoretical_metrics = []
    other_metrics = []
    
    for metric in metrics:
        if metric in ['soundness', 'presentation', 'contribution', 'rating', 'overall_score', 'confidence']:
            standard_metrics.append(metric)
        elif metric.startswith('practicality_assessment.'):
            practicality_metrics.append(metric)
        elif 'theoretical' in metric.lower() or 'rigor' in metric.lower() or metric.endswith('_score'):
            theoretical_metrics.append(metric)
        else:
            other_metrics.append(metric)
    
    # Create subplots
    n_groups = sum([
        bool(standard_metrics),
        bool(practicality_metrics),
        bool(theoretical_metrics),
        bool(other_metrics)
    ])
    
    if n_groups == 0:
        return
    
    fig, axes = plt.subplots(n_groups, 1, figsize=(14, 4 * n_groups))
    if n_groups == 1:
        axes = [axes]
    
    fig.suptitle(f"Cohen's d Effect Sizes: Category {category_id} (Planted Error vs Latest)", 
                 fontsize=16, fontweight='bold')
    
    plot_idx = 0
    
    # Plot standard metrics
    if standard_metrics:
        ax = axes[plot_idx]
        plot_metrics_on_axis(ax, standard_metrics, stats_results, "Standard Metrics")
        plot_idx += 1
    
    # Plot practicality assessment metrics
    if practicality_metrics:
        ax = axes[plot_idx]
        plot_metrics_on_axis(ax, practicality_metrics, stats_results, "Practicality Assessment")
        plot_idx += 1
    
    # Plot theoretical rigor metrics
    if theoretical_metrics:
        ax = axes[plot_idx]
        plot_metrics_on_axis(ax, theoretical_metrics, stats_results, "Theoretical Rigor")
        plot_idx += 1
    
    # Plot other metrics
    if other_metrics:
        ax = axes[plot_idx]
        plot_metrics_on_axis(ax, other_metrics, stats_results, "Other Metrics")
        plot_idx += 1
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plot_path = output_dir / "cohens_d_summary.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")

def plot_metrics_on_axis(ax, metrics, stats_results, title):
    """Plot metrics on a single axis."""
    cohen_ds = []
    yerr = [[], []]
    labels = []
    
    for metric in metrics:
        stats = stats_results.get(metric)
        if not stats or stats['cohen_d'] is None:
            continue
        
        cohen_d = stats['cohen_d']
        labels.append(metric.replace('_', ' ').title())
        cohen_ds.append(cohen_d)
        
        if stats['ci_low'] is not None and stats['ci_high'] is not None:
            lower = cohen_d - stats['ci_low']
            upper = stats['ci_high'] - cohen_d
        elif stats['standard_error'] is not None:
            lower = upper = 1.96 * stats['standard_error']
        else:
            lower = upper = 0.0
        
        yerr[0].append(lower * ERROR_BAR_SCALE)
        yerr[1].append(upper * ERROR_BAR_SCALE)
    
    if not cohen_ds:
        ax.set_axis_off()
        return
    
    x_positions = np.arange(len(cohen_ds))
    colors = [COLOR_MISMATCH if d < 0 else COLOR_NEUTRAL for d in cohen_ds]
    bars = ax.bar(x_positions, cohen_ds, yerr=yerr, capsize=8, color=colors, 
                  edgecolor='black', alpha=0.85)
    
    for bar, cohen_d in zip(bars, cohen_ds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"d={cohen_d:.2f}",
            ha='center',
            va='bottom' if bar.get_height() >= 0 else 'top',
            fontsize=10,
            fontweight='bold',
            color='black'
        )
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel("Cohen's d", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the effect of planted errors on category-specific review scores"
    )
    parser.add_argument(
        "--reviews_dir",
        type=str,
        required=True,
        help="Directory containing category review folders (e.g., reviews_by_category/NeurIPS2024)"
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="Category ID to evaluate (e.g., 2a, 2b)"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="latest",
        help="Baseline folder name (default: latest)"
    )
    parser.add_argument(
        "--treatment",
        type=str,
        default="planted_error",
        help="Treatment folder name (default: planted_error)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: reviews_dir/../evaluation_results/category_{category})"
    )
    
    args = parser.parse_args()
    
    reviews_dir = Path(args.reviews_dir)
    
    if args.output_dir is None:
        output_dir = reviews_dir.parent / "evaluation_results" / f"category_{args.category}"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"Category {args.category} Review Evaluation")
    print("="*80)
    print(f"Reviews directory: {reviews_dir}")
    print(f"Category: {args.category}")
    print(f"Baseline folder: {args.baseline}")
    print(f"Treatment folder: {args.treatment}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Collect scores from both folders
    print("Collecting scores from review files...")
    baseline_df = collect_scores_from_category_folder(reviews_dir, args.category, args.baseline)
    treatment_df = collect_scores_from_category_folder(reviews_dir, args.category, args.treatment)
    
    if len(baseline_df) == 0:
        print(f"Error: No scores found in {args.baseline} folder")
        return
    
    if len(treatment_df) == 0:
        print(f"Error: No scores found in {args.treatment} folder")
        return
    
    print(f"  {args.baseline}: {len(baseline_df)} score records from {baseline_df['paper_id'].nunique()} papers")
    if 'version_id' in baseline_df.columns:
        print(f"    Unique versions: {baseline_df['version_id'].nunique()}")
    
    print(f"  {args.treatment}: {len(treatment_df)} score records from {treatment_df['paper_id'].nunique()} papers")
    if 'version_id' in treatment_df.columns:
        print(f"    Unique versions: {treatment_df['version_id'].nunique()}")
    
    # Combine dataframes
    df_all = pd.concat([baseline_df, treatment_df], ignore_index=True)
    print(f"\nTotal: {len(df_all)} score records")
    
    # Show available metrics
    metadata_cols = ['paper_id', 'run_id', 'folder', 'version_id']
    score_cols = [col for col in df_all.columns if col not in metadata_cols]
    print(f"Available metrics: {', '.join(score_cols)}")
    
    # Calculate differences
    print("\nCalculating differences...")
    df_diffs = calculate_differences(df_all, args.baseline, args.treatment)
    
    if len(df_diffs) == 0:
        print("Error: Could not calculate differences. Check that papers exist in both folders.")
        return
    
    print(f"Calculated differences for {df_diffs['paper_id'].nunique()} papers")
    if 'version_id' in df_diffs.columns:
        print(f"  Unique versions: {df_diffs['version_id'].nunique()}")
    print(f"  Metrics: {df_diffs['metric'].nunique()}")
    
    # Compute statistics
    print("\nComputing effect sizes (Cohen's d)...")
    stats_results = compute_statistics(df_diffs)
    
    # Print results
    print("\n" + "="*80)
    print("EFFECT SIZE SUMMARY")
    print("="*80)
    
    for metric in sorted(stats_results.keys()):
        stats = stats_results[metric]
        print(f"\n{metric}:")
        print(f"  Pairs: {stats['n_pairs']}")
        print(f"  Baseline mean: {stats['baseline_mean']:.3f}")
        print(f"  Treatment mean: {stats['treatment_mean']:.3f}")
        print(f"  Mean difference (treatment - baseline): {stats['mean_difference']:.3f}")
        cohen_d = stats['cohen_d']
        if cohen_d is not None:
            ci_text = ""
            if stats['ci_low'] is not None and stats['ci_high'] is not None:
                ci_text = f" (95% CI: [{stats['ci_low']:.3f}, {stats['ci_high']:.3f}])"
            print(f"  Cohen's d: {cohen_d:.3f}{ci_text}")
        else:
            print("  Cohen's d: undefined (insufficient variance)")
    
    # Save results
    print("\n" + "="*80)
    print("Saving results...")
    print("="*80)
    
    # Save detailed differences
    diff_csv_path = output_dir / "score_differences.csv"
    df_diffs.to_csv(diff_csv_path, index=False)
    print(f"✅ Saved detailed differences to: {diff_csv_path}")
    
    # Save statistics
    stats_json_path = output_dir / "statistical_results.json"
    with open(stats_json_path, 'w') as f:
        json.dump(stats_results, f, indent=2)
    print(f"✅ Saved statistics to: {stats_json_path}")
    
    # Create summary table
    summary_rows = []
    for metric, stats in stats_results.items():
        summary_rows.append({
            'metric': metric,
            'n_pairs': stats['n_pairs'],
            'baseline_mean': stats['baseline_mean'],
            'treatment_mean': stats['treatment_mean'],
            'mean_difference': stats['mean_difference'],
            'cohen_d': stats['cohen_d'],
            'standard_error': stats['standard_error'],
            'ci_low': stats['ci_low'],
            'ci_high': stats['ci_high']
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / "summary_statistics.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"✅ Saved summary to: {summary_csv_path}")
    
    # Generate plots
    print("\n" + "="*80)
    print("Generating plots...")
    print("="*80)
    
    try:
        create_effect_size_plot(stats_results, output_dir, args.category)
        print("\n✅ Cohen's d plot generated successfully!")
    except Exception as e:
        print(f"\n⚠️  Warning: Failed to generate plots: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)
    print(f"\nOutput files in: {output_dir}/")
    print(f"  - score_differences.csv (detailed differences per paper)")
    print(f"  - summary_statistics.csv (summary by metric)")
    print(f"  - statistical_results.json (full statistics)")
    print(f"  - cohens_d_summary.png (visualization)")

if __name__ == "__main__":
    main()

