#!/usr/bin/env python3
"""
Evaluate the effect of paper manipulations on review scores.

This script analyzes reviews from three versions:
1. Original (baseline, e.g., latest/)
2. Good manipulation (e.g., authors_affiliation_good/, abstract_good/)
3. Bad manipulation (e.g., authors_affiliation_bad/, abstract_bad/)

Expected effects:
- Good manipulation: Should improve scores (positive difference) OR decrease scores if it's a negative manipulation
- Bad manipulation: Should decrease scores (negative difference) OR increase scores if it's a negative manipulation

Usage:
    # Author/affiliation effect (default)
    python evaluate_good_bad_plant_effect.py --reviews_dir ../sampled_data/reviews/ICLR2024
    
    # Abstract manipulation effect
    python evaluate_good_bad_plant_effect.py --reviews_dir ../sampled_data/reviews/ICLR2024 \
                                              --pattern abstract \
                                              --baseline latest
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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
ERROR_BAR_SCALE = 0.2

def extract_numerical_value(score_str) -> Optional[float]:
    """
    Extract numerical value from score string or number.
    
    Examples:
        "3 good" -> 3.0
        "2 fair" -> 2.0
        "6: marginally above the acceptance threshold" -> 6.0
        "3.5" -> 3.5
        3 -> 3.0
    """
    if score_str is None:
        return None
    
    # If already a number, return it
    if isinstance(score_str, (int, float)):
        return float(score_str)
    
    # If not a string, try to convert
    if not isinstance(score_str, str):
        try:
            return float(score_str)
        except:
            return None
    
    if not score_str.strip():
        return None
    
    # Try to extract number from various formats
    match = re.match(r'^(\d+(?:\.\d+)?)', str(score_str).strip())
    if match:
        return float(match.group(1))
    
    return None

def extract_scores_from_review(review_file: Path) -> Optional[Dict]:
    """
    Extract numerical scores from a review JSON file.
    
    Handles both single-review formats and multi-review formats (e.g., CycleReviewer with multiple reviewers).
    For multi-review formats, scores are averaged across all reviewers.
    
    Returns dict with scores or None if extraction fails.
    """
    try:
        with open(review_file, 'r', encoding='utf-8') as f:
            review_data = json.load(f)
        
        # Check if review was successful
        if not review_data.get('success', False):
            return None
        
        # Check if this is a multi-review format (e.g., CycleReviewer)
        if 'reviewers' in review_data and isinstance(review_data['reviewers'], list) and len(review_data['reviewers']) > 0:
            # Aggregate scores from multiple reviewers
            all_scores = {
                'soundness': [],
                'presentation': [],
                'contribution': [],
                'rating': []
            }
            
            for reviewer in review_data['reviewers']:
                # Extract scores from each reviewer
                for metric in ['soundness', 'presentation', 'contribution', 'rating']:
                    value = reviewer.get(metric)
                    parsed = extract_numerical_value(value)
                    if parsed is not None:
                        all_scores[metric].append(parsed)
            
            # Average the scores across reviewers
            scores = {}
            for metric in ['soundness', 'presentation', 'contribution', 'rating']:
                if len(all_scores[metric]) > 0:
                    scores[metric] = np.mean(all_scores[metric])
                else:
                    scores[metric] = None
            
            # If we still don't have all scores, try top-level fields as fallback
            for metric in ['soundness', 'presentation', 'contribution', 'rating']:
                if scores[metric] is None:
                    value = review_data.get(metric)
                    parsed = extract_numerical_value(value)
                    if parsed is not None:
                        scores[metric] = parsed
            
            return scores
        
        # Single-review format (original logic)
        # Extract scores - handle both direct and mapped fields
        scores = {
            'soundness': None,
            'presentation': None,
            'contribution': None,
            'rating': None
        }
        
        # Try direct extraction first (already mapped fields)
        for metric in ['soundness', 'presentation', 'contribution', 'rating']:
            value = review_data.get(metric)
            parsed = extract_numerical_value(value)
            if parsed is not None:
                scores[metric] = parsed
        
        # Fallback to original field names if needed
        if scores['soundness'] is None:
            scores['soundness'] = extract_numerical_value(review_data.get('technical_quality_score'))
        if scores['presentation'] is None:
            scores['presentation'] = extract_numerical_value(review_data.get('clarity_score'))
        if scores['contribution'] is None:
            scores['contribution'] = extract_numerical_value(review_data.get('novelty_score'))
        if scores['rating'] is None:
            scores['rating'] = extract_numerical_value(review_data.get('overall_score'))
        
        return scores
    
    except Exception as e:
        return None

def collect_scores_from_folder(reviews_dir: Path, folder_name: str) -> pd.DataFrame:
    """
    Collect all scores from a specific folder.
    Supports both old format (review_run*.json) and new format (flaw_*_review_run*.json).
    
    Returns DataFrame with columns: paper_id, run_id, version_id, soundness, presentation, contribution, rating
    """
    folder_path = reviews_dir / folder_name
    if not folder_path.exists():
        print(f"Warning: Folder {folder_path} does not exist")
        return pd.DataFrame()
    
    records = []
    
    # Find all paper directories
    paper_dirs = [d for d in folder_path.iterdir() if d.is_dir()]
    
    for paper_dir in paper_dirs:
        paper_id = paper_dir.name
        
        # Find all review files - support both formats
        # Old format: review_run*.json
        old_format_files = list(paper_dir.glob("review_run*.json"))
        # New format: flaw_*_review_run*.json
        new_format_files = list(paper_dir.glob("flaw_*_review_run*.json"))
        
        review_files = old_format_files + new_format_files
        
        for review_file in review_files:
            # Extract run_id from filename
            run_id = 0
            version_id = None
            
            # Try new format first: flaw_X_review_runY.json
            match = re.search(r'flaw_(\d+)_review_run(\d+)\.json', review_file.name)
            if match:
                version_id = f"flaw_{match.group(1)}"
                run_id = int(match.group(2))
            else:
                # Try old format: review_runY.json
                match = re.search(r'review_run(\d+)\.json', review_file.name)
                if match:
                    run_id = int(match.group(1))
            
            scores = extract_scores_from_review(review_file)
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

def calculate_differences(df: pd.DataFrame, baseline_folder: str, good_folder: str = None, bad_folder: str = None, 
                          compare_folder: str = None, compare_label: str = None) -> pd.DataFrame:
    """
    Calculate differences between versions for each paper.
    Supports both old format (single version per paper) and new format (multiple flaw versions).
    
    For new format (with version_id), matches versions by version_id (e.g., flaw_1 with flaw_1).
    For old format, matches by paper_id and run_id only.
    
    Creates:
    - good_diff = good_folder - baseline (if good_folder provided)
    - bad_diff = bad_folder - baseline (if bad_folder provided)
    - compare_diff = compare_folder - baseline (if compare_folder provided)
    """
    results = []
    
    # Check if we have version_id column (new format)
    has_version_id = 'version_id' in df.columns
    
    if has_version_id:
        # New format: group by paper_id, run_id, and version_id
        group_keys = ['paper_id', 'run_id', 'version_id']
    else:
        # Old format: group by paper_id and run_id only
        group_keys = ['paper_id', 'run_id']
    
    for group_key, group in df.groupby(group_keys):
        if has_version_id:
            paper_id, run_id, version_id = group_key
        else:
            paper_id, run_id = group_key
            version_id = None
        
        # Get scores for each version
        baseline_scores = group[group['folder'] == baseline_folder]
        
        # Fallback: when baseline folder does not provide per-version scores (e.g., latest without flaw IDs)
        if baseline_scores.empty:
            fallback_mask = (
                (df['folder'] == baseline_folder) &
                (df['paper_id'] == paper_id) &
                (df['run_id'] == run_id)
            )
            fallback_scores = df[fallback_mask]
            if not fallback_scores.empty:
                # Use the first available baseline record and treat it as the baseline for all versions
                baseline_scores = fallback_scores.iloc[[0]]
        good_scores = group[group['folder'] == good_folder] if good_folder else pd.DataFrame()
        bad_scores = group[group['folder'] == bad_folder] if bad_folder else pd.DataFrame()
        compare_scores = group[group['folder'] == compare_folder] if compare_folder else pd.DataFrame()
        
        if baseline_scores.empty:
            continue
        
        baseline_row = baseline_scores.iloc[0]
        
        for metric in ['soundness', 'presentation', 'contribution', 'rating']:
            baseline_val = baseline_row.get(metric)
            
            if baseline_val is None or pd.isna(baseline_val):
                continue
            
            result = {
                'paper_id': paper_id,
                'run_id': run_id,
                'metric': metric,
                'baseline_score': float(baseline_val)
            }
            if version_id:
                result['version_id'] = version_id
            
            # Calculate good diff (always create columns, even if None)
            if good_folder and not good_scores.empty:
                good_row = good_scores.iloc[0]
                good_val = good_row.get(metric)
                if good_val is not None and not pd.isna(good_val):
                    result['good_score'] = float(good_val)
                    result['good_diff'] = float(good_val) - float(baseline_val)
                else:
                    result['good_score'] = None
                    result['good_diff'] = None
            elif good_folder:
                result['good_score'] = None
                result['good_diff'] = None
            
            # Calculate bad diff (always create columns, even if None)
            if bad_folder and not bad_scores.empty:
                bad_row = bad_scores.iloc[0]
                bad_val = bad_row.get(metric)
                if bad_val is not None and not pd.isna(bad_val):
                    result['bad_score'] = float(bad_val)
                    result['bad_diff'] = float(bad_val) - float(baseline_val)
                else:
                    result['bad_score'] = None
                    result['bad_diff'] = None
            elif bad_folder:
                result['bad_score'] = None
                result['bad_diff'] = None
            
            # Calculate compare diff (for sham_surgery vs planted_error, etc.)
            if compare_folder and compare_label and not compare_scores.empty:
                compare_row = compare_scores.iloc[0]
                compare_val = compare_row.get(metric)
                if compare_val is not None and not pd.isna(compare_val):
                    result[f'{compare_label}_score'] = float(compare_val)
                    result[f'{compare_label}_diff'] = float(compare_val) - float(baseline_val)
                else:
                    result[f'{compare_label}_score'] = None
                    result[f'{compare_label}_diff'] = None
            elif compare_folder and compare_label:
                result[f'{compare_label}_score'] = None
                result[f'{compare_label}_diff'] = None
            
            results.append(result)
    
    return pd.DataFrame(results)

def infer_expected_direction(treatment_label: str, baseline_label: str) -> Optional[str]:
    """
    Infer the expected direction of the effect size based on treatment/baseline labels.
    Returns one of {'positive', 'negative', 'zero', None}.
    """
    label = treatment_label.lower()
    baseline = baseline_label.lower()
    
    # Specific rules first
    if "planted" in label:
        return "negative"
    if "sham" in label or "placebo" in label:
        return "zero"
    if "camera_ready" in label or "camera-ready" in label or "camera" in label:
        return "positive"
    if "human" in label and "llm" in baseline:
        return "positive"
    
    # Generic keyword rules
    if "good" in label or "improved" in label or "privileged" in label:
        return "positive"
    if "bad" in label or "degraded" in label or "under-privileged" in label or "underserved" in label:
        return "negative"
    
    # Fallbacks
    if "baseline" in label:
        return "zero"
    
    return None

def evaluate_expectation(cohen_d: float, expected: Optional[str], tolerance: float = 0.1) -> Optional[bool]:
    """
    Evaluate whether the observed Cohen's d matches the expected direction.
    Returns True/False if expectation exists, otherwise None.
    """
    if expected is None or cohen_d is None or np.isnan(cohen_d):
        return None
    
    if expected == "positive":
        return cohen_d > tolerance
    if expected == "negative":
        return cohen_d < -tolerance
    if expected == "zero":
        return abs(cohen_d) <= tolerance
    
    return None

def pooled_standard_deviation(treatment: np.ndarray, control: np.ndarray) -> Optional[float]:
    """
    Compute pooled standard deviation for two independent samples.
    """
    n_t = len(treatment)
    n_c = len(control)
    if n_t <= 1 or n_c <= 1:
        return None
    
    var_t = np.var(treatment, ddof=1)
    var_c = np.var(control, ddof=1)
    pooled_var = ((n_t - 1) * var_t + (n_c - 1) * var_c) / (n_t + n_c - 2)
    return np.sqrt(pooled_var) if pooled_var > 0 else 0.0

def cohen_d_standard_error(cohen_d: float, n_t: int, n_c: int) -> Optional[float]:
    """
    Approximate standard error for Cohen's d for independent samples.
    """
    if n_t <= 1 or n_c <= 1:
        return None
    return np.sqrt((n_t + n_c) / (n_t * n_c) + (cohen_d ** 2) / (2 * (n_t + n_c - 2)))

def compute_statistics(df_diffs: pd.DataFrame, comparisons: List[Dict], metrics: List[str]) -> Dict[str, List[Dict]]:
    """
    Compute Cohen's d effect sizes for the specified comparisons and metrics.
    """
    results: Dict[str, List[Dict]] = {metric: [] for metric in metrics}
    
    for comp in comparisons:
        treatment_col = comp['treatment_score_col']
        control_col = comp['control_score_col']
        
        if treatment_col not in df_diffs.columns or control_col not in df_diffs.columns:
            continue
        
        for metric in metrics:
            metric_rows = df_diffs[df_diffs['metric'] == metric]
            if metric_rows.empty:
                continue
            
            valid_rows = metric_rows[
                metric_rows[treatment_col].notna() &
                metric_rows[control_col].notna()
            ]
            
            if valid_rows.empty:
                continue
            
            treatment_scores = valid_rows[treatment_col].astype(float).values
            control_scores = valid_rows[control_col].astype(float).values
            
            if len(treatment_scores) == 0 or len(control_scores) == 0:
                continue
            
            diffs = treatment_scores - control_scores
            mean_diff = float(np.mean(diffs))
            pooled_sd = pooled_standard_deviation(treatment_scores, control_scores)
            
            if pooled_sd is None:
                cohen_d = np.nan
            elif pooled_sd == 0:
                cohen_d = float(np.sign(mean_diff)) * np.inf if mean_diff != 0 else 0.0
            else:
                cohen_d = float(mean_diff / pooled_sd)
            
            se = cohen_d_standard_error(cohen_d, len(treatment_scores), len(control_scores)) if np.isfinite(cohen_d) else None
            ci_low = None
            ci_high = None
            if se is not None:
                ci_low = float(cohen_d - 1.96 * se)
                ci_high = float(cohen_d + 1.96 * se)
            
            expected_direction = comp.get('expected_direction')
            matches = evaluate_expectation(cohen_d, expected_direction) if np.isfinite(cohen_d) else None
            
            result_entry = {
                'comparison': comp['label'],
                'treatment_label': comp['treatment_label'],
                'control_label': comp['control_label'],
                'expected_direction': expected_direction,
                'matches_expectation': matches,
                'n_pairs': int(len(valid_rows)),
                'treatment_mean': float(np.mean(treatment_scores)),
                'control_mean': float(np.mean(control_scores)),
                'mean_difference': mean_diff,
                'cohen_d': float(cohen_d) if np.isfinite(cohen_d) else None,
                'standard_error': float(se) if se is not None else None,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'diffs': diffs.tolist()
            }
            
            results[metric].append(result_entry)
    
    return results

def create_effect_size_plots(effect_results: Dict[str, List[Dict]], output_dir: Path,
                             pattern_name: str = "Treatment Effect Summary"):
    """
    Create bar plots showing Cohen's d with error bars for each metric and comparison.
    """
    metrics = ['soundness', 'presentation', 'contribution', 'rating']
    metric_labels = {
        'soundness': 'Soundness',
        'presentation': 'Presentation',
        'contribution': 'Contribution',
        'rating': 'Overall Rating'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Cohen\'s d by Metric: {pattern_name}', fontsize=18, fontweight='bold')
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        effects = effect_results.get(metric, [])
        
        if not effects:
            ax.set_axis_off()
            continue
        
        cohen_ds = []
        yerr = [[], []]
        colors = []
        labels = []
        
        for effect in effects:
            cohen_d = effect['cohen_d']
            if cohen_d is None:
                continue
            
            labels.append(effect['comparison'])
            cohen_ds.append(cohen_d)
            
            if effect['ci_low'] is not None and effect['ci_high'] is not None:
                lower = cohen_d - effect['ci_low']
                upper = effect['ci_high'] - cohen_d
            elif effect['standard_error'] is not None:
                lower = upper = 1.96 * effect['standard_error']
            else:
                lower = upper = 0.0
            
            lower *= ERROR_BAR_SCALE
            upper *= ERROR_BAR_SCALE
            
            yerr[0].append(lower)
            yerr[1].append(upper)
            
            matches = effect['matches_expectation']
            if matches is None:
                colors.append(COLOR_NEUTRAL)
            elif matches:
                colors.append(COLOR_MATCH)
            else:
                colors.append(COLOR_MISMATCH)
        
        if not cohen_ds:
            ax.set_axis_off()
            continue
        
        x_positions = np.arange(len(cohen_ds))
        bars = ax.bar(x_positions, cohen_ds, yerr=yerr, capsize=8, color=colors, edgecolor='black', alpha=0.85)
        
        for bar, effect in zip(bars, [e for e in effects if e['cohen_d'] is not None]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"d={effect['cohen_d']:.2f}",
                ha='center',
                va='bottom' if bar.get_height() >= 0 else 'top',
                fontsize=11,
                fontweight='bold',
                color='black'
            )
            expectation = effect['expected_direction']
            if expectation:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    -0.1,
                    f"Expected: {expectation}",
                    ha='center',
                    va='top',
                    fontsize=10,
                    rotation=45,
                    color='dimgray',
                    transform=ax.get_xaxis_transform()
                )
        
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=11)
        ax.set_ylabel("Cohen's d", fontsize=12)
        ax.set_title(metric_labels[metric], fontsize=15, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
    
    # Custom legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_MATCH, ec='black', label='Matches expectation'),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_MISMATCH, ec='black', label='Contradicts expectation'),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_NEUTRAL, ec='black', label='No expectation')
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=12)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plot_path = output_dir / "cohens_d_summary.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the effect of paper manipulations on review scores. "
                    "Supports both old format (single version per paper) and new format (multiple flaw versions)."
    )
    parser.add_argument(
        "--reviews_dir",
        type=str,
        required=True,
        help="Directory containing review folders"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: reviews_dir/../evaluation_results/{pattern})"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Pattern prefix for manipulation folders (e.g., 'authors_affiliation' or 'abstract'). "
             "Script will look for {pattern}_good and {pattern}_bad folders. "
             "For new format, use 'planted_error' or 'sham_surgery'."
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="latest",
        help="Baseline folder name for comparison (default: latest)"
    )
    parser.add_argument(
        "--folders",
        type=str,
        nargs="+",
        default=None,
        help="DEPRECATED: Use --pattern and --baseline instead. Folders to analyze. "
             "If provided, will override --pattern and --baseline."
    )
    parser.add_argument(
        "--new_format",
        action="store_true",
        help="Use new format evaluation: compare latest vs planted_error, latest vs sham_surgery, "
             "and sham_surgery vs planted_error"
    )
    
    args = parser.parse_args()
    
    reviews_dir = Path(args.reviews_dir)
    
    comparisons: List[Dict] = []
    
    # Handle new format evaluation
    if args.new_format or (args.pattern and args.pattern in ['planted_error', 'sham_surgery']):
        # New format: compare latest vs planted_error, latest vs sham_surgery, sham_surgery vs planted_error
        baseline_folder = args.baseline
        planted_error_folder = "planted_error"
        sham_surgery_folder = "sham_surgery"
        
        # Set default output directory
        if args.output_dir is None:
            output_dir = reviews_dir.parent / "evaluation_results" / "planted_errors"
        else:
            output_dir = Path(args.output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print("Planted Error Effect Evaluation (New Format)")
        print("="*80)
        print(f"Reviews directory: {reviews_dir}")
        print(f"Baseline folder: {baseline_folder}")
        print(f"Planted error folder: {planted_error_folder}")
        print(f"Sham surgery folder: {sham_surgery_folder}")
        print(f"Output directory: {output_dir}")
        print()
        
        # Collect scores from all folders
        print("Collecting scores from review files...")
        all_scores = []
        
        folders_to_check = [baseline_folder, planted_error_folder, sham_surgery_folder]
        for folder in folders_to_check:
            df = collect_scores_from_folder(reviews_dir, folder)
            if len(df) > 0:
                version_info = ""
                if 'version_id' in df.columns:
                    unique_versions = df['version_id'].nunique()
                    version_info = f" ({unique_versions} unique versions)"
                print(f"  {folder}: {len(df)} score records from {df['paper_id'].nunique()} papers{version_info}")
                all_scores.append(df)
            else:
                print(f"  {folder}: No scores found")
        
        if not all_scores:
            print("Error: No scores collected from any folder!")
            return
        
        df_all = pd.concat(all_scores, ignore_index=True)
        print(f"\nTotal: {len(df_all)} score records")
        
        # Calculate differences for each comparison
        print("\nCalculating differences...")
        
        # 1. Latest vs Planted Error
        print("  1. Latest vs Planted Error...")
        df_diffs_planted = calculate_differences(
            df_all, baseline_folder, bad_folder=planted_error_folder
        )
        
        # 2. Latest vs Sham Surgery
        print("  2. Latest vs Sham Surgery...")
        df_diffs_sham = calculate_differences(
            df_all, baseline_folder, good_folder=sham_surgery_folder
        )
        
        # 3. Sham Surgery vs Planted Error (using sham_surgery as baseline)
        print("  3. Sham Surgery vs Planted Error...")
        df_diffs_sham_vs_planted = calculate_differences(
            df_all, sham_surgery_folder, compare_folder=planted_error_folder, compare_label="planted_error_vs_sham"
        )
        
        # Combine all differences for analysis
        # Merge on paper_id, run_id, metric, and version_id (if exists)
        merge_keys = ['paper_id', 'run_id', 'metric']
        if len(df_diffs_planted) > 0 and 'version_id' in df_diffs_planted.columns:
            merge_keys.append('version_id')
        elif len(df_diffs_sham) > 0 and 'version_id' in df_diffs_sham.columns:
            merge_keys.append('version_id')
        elif len(df_diffs_sham_vs_planted) > 0 and 'version_id' in df_diffs_sham_vs_planted.columns:
            merge_keys.append('version_id')
        
        # Start with planted error differences (or create empty dataframe with merge keys if empty)
        if len(df_diffs_planted) > 0:
            df_diffs = df_diffs_planted.copy()
        else:
            # Create empty dataframe with merge keys
            df_diffs = pd.DataFrame(columns=merge_keys + ['baseline_score', 'bad_score', 'bad_diff'])
        
        # Merge sham surgery differences if available
        if len(df_diffs_sham) > 0:
            # Ensure all required columns exist
            sham_merge_cols = merge_keys.copy()
            if 'good_score' in df_diffs_sham.columns:
                sham_merge_cols.append('good_score')
            if 'good_diff' in df_diffs_sham.columns:
                sham_merge_cols.append('good_diff')
            
            # Only merge if we have columns beyond merge keys
            if len(sham_merge_cols) > len(merge_keys):
                df_diffs = df_diffs.merge(
                    df_diffs_sham[sham_merge_cols],
                    on=merge_keys, how='outer', suffixes=('', '_sham')
                )
            else:
                # If no sham columns, just add empty columns
                if len(df_diffs) > 0:
                    df_diffs['good_score'] = None
                    df_diffs['good_diff'] = None
        else:
            # If no sham data, just add empty columns
            if len(df_diffs) > 0:
                df_diffs['good_score'] = None
                df_diffs['good_diff'] = None
        
        # Merge sham vs planted differences if available
        if len(df_diffs_sham_vs_planted) > 0:
            # Ensure all required columns exist
            sham_vs_planted_merge_cols = merge_keys.copy()
            if 'planted_error_vs_sham_score' in df_diffs_sham_vs_planted.columns:
                sham_vs_planted_merge_cols.append('planted_error_vs_sham_score')
            if 'planted_error_vs_sham_diff' in df_diffs_sham_vs_planted.columns:
                sham_vs_planted_merge_cols.append('planted_error_vs_sham_diff')
            
            # Only merge if we have columns beyond merge keys
            if len(sham_vs_planted_merge_cols) > len(merge_keys):
                df_diffs = df_diffs.merge(
                    df_diffs_sham_vs_planted[sham_vs_planted_merge_cols],
                    on=merge_keys, how='outer'
                )
            else:
                # If no sham vs planted columns, just add empty columns
                if len(df_diffs) > 0:
                    df_diffs['planted_error_vs_sham_score'] = None
                    df_diffs['planted_error_vs_sham_diff'] = None
        else:
            # If no sham vs planted data, just add empty columns
            if len(df_diffs) > 0:
                df_diffs['planted_error_vs_sham_score'] = None
                df_diffs['planted_error_vs_sham_diff'] = None
        
        # Rename columns for clarity
        df_diffs = df_diffs.rename(columns={
            'bad_score': 'planted_error_score',
            'bad_diff': 'planted_error_diff',  # This is planted_error - latest
            'good_score': 'sham_surgery_score',
            'good_diff': 'sham_surgery_diff'  # This is sham_surgery - latest
            # planted_error_vs_sham_diff is planted_error - sham_surgery (already correctly named)
        })
        
        if len(df_diffs) == 0:
            print("Error: Could not calculate differences. Check that versions exist for papers.")
            return
        
        print(f"Calculated differences for {df_diffs['paper_id'].nunique()} papers")
        if 'version_id' in df_diffs.columns:
            print(f"  Unique versions: {df_diffs['version_id'].nunique()}")
        print(f"  Metrics: {df_diffs['metric'].unique()}")
        
        # Update compute_statistics to handle new column names
        # We'll need to adapt the statistics computation
        pattern_name = "Planted Errors"
        baseline_label = baseline_folder.replace('_', ' ').title()
        good_label = "Sham Surgery"
        bad_label = "Planted Error"
        
        comparisons = []
        if 'planted_error_score' in df_diffs.columns:
            comparisons.append({
                'label': f'{bad_label} vs {baseline_label}',
                'treatment_score_col': 'planted_error_score',
                'control_score_col': 'baseline_score',
                'treatment_label': bad_label,
                'control_label': baseline_label,
                'expected_direction': infer_expected_direction(bad_label, baseline_label)
            })
        if 'sham_surgery_score' in df_diffs.columns:
            comparisons.append({
                'label': f'{good_label} vs {baseline_label}',
                'treatment_score_col': 'sham_surgery_score',
                'control_score_col': 'baseline_score',
                'treatment_label': good_label,
                'control_label': baseline_label,
                'expected_direction': infer_expected_direction(good_label, baseline_label)
            })
        if 'planted_error_vs_sham_score' in df_diffs.columns and 'sham_surgery_score' in df_diffs.columns:
            comparisons.append({
                'label': f'{bad_label} vs {good_label}',
                'treatment_score_col': 'planted_error_vs_sham_score',
                'control_score_col': 'sham_surgery_score',
                'treatment_label': bad_label,
                'control_label': good_label,
                'expected_direction': infer_expected_direction(bad_label, good_label)
            })
        
    else:
        # Old format evaluation
        # Handle deprecated --folders argument for backward compatibility
        if args.folders and len(args.folders) >= 3:
            # Extract pattern from folders
            baseline_folder = args.folders[0]
            good_folder = args.folders[1]
            bad_folder = args.folders[2]
            # Try to extract pattern from good/bad folder names
            if good_folder.endswith('_good') and bad_folder.endswith('_bad'):
                pattern = good_folder[:-5]  # Remove '_good'
                if pattern == bad_folder[:-4]:  # Remove '_bad'
                    args.pattern = pattern
                    args.baseline = baseline_folder
        else:
            # Use pattern-based approach
            if not args.pattern:
                args.pattern = "authors_affiliation"
            baseline_folder = args.baseline
            good_folder = f"{args.pattern}_good"
            bad_folder = f"{args.pattern}_bad"
        
        # Set default output directory
        if args.output_dir is None:
            output_dir = reviews_dir.parent / "evaluation_results" / args.pattern
        else:
            output_dir = Path(args.output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create readable labels
        pattern_name = args.pattern.replace('_', ' ').title()
        baseline_label = baseline_folder.replace('_', ' ').title()
        # Replace "Latest" with "Anonymized" for better clarity
        if baseline_label.lower() == "latest" and args.pattern.lower() == "authors_affiliation":
            baseline_label = "Anonymized"
        good_label = good_folder.replace('_', ' ').title()
        bad_label = bad_folder.replace('_', ' ').title()
        
        comparisons = []
        if 'good_score' in df_diffs.columns:
            comparisons.append({
                'label': f'{good_label} vs {baseline_label}',
                'treatment_score_col': 'good_score',
                'control_score_col': 'baseline_score',
                'treatment_label': good_label,
                'control_label': baseline_label,
                'expected_direction': infer_expected_direction(good_label, baseline_label)
            })
        if 'bad_score' in df_diffs.columns:
            comparisons.append({
                'label': f'{bad_label} vs {baseline_label}',
                'treatment_score_col': 'bad_score',
                'control_score_col': 'baseline_score',
                'treatment_label': bad_label,
                'control_label': baseline_label,
                'expected_direction': infer_expected_direction(bad_label, baseline_label)
            })
        
        print("="*80)
        print(f"{pattern_name} Effect Evaluation")
        print("="*80)
        print(f"Reviews directory: {reviews_dir}")
        print(f"Baseline folder: {baseline_folder}")
        print(f"Good folder: {good_folder}")
        print(f"Bad folder: {bad_folder}")
        print(f"Output directory: {output_dir}")
        print()
        
        # Collect scores from all folders
        print("Collecting scores from review files...")
        all_scores = []
        
        folders_to_check = [baseline_folder, good_folder, bad_folder]
        for folder in folders_to_check:
            df = collect_scores_from_folder(reviews_dir, folder)
            if len(df) > 0:
                print(f"  {folder}: {len(df)} score records from {df['paper_id'].nunique()} papers")
                all_scores.append(df)
            else:
                print(f"  {folder}: No scores found")
        
        if not all_scores:
            print("Error: No scores collected from any folder!")
            return
        
        df_all = pd.concat(all_scores, ignore_index=True)
        print(f"\nTotal: {len(df_all)} score records")
        
        # Calculate differences
        print("\nCalculating differences...")
        df_diffs = calculate_differences(df_all, baseline_folder, good_folder, bad_folder)
        
        if len(df_diffs) == 0:
            print("Error: Could not calculate differences. Check that all three versions exist for papers.")
            return
        
        print(f"Calculated differences for {df_diffs['paper_id'].nunique()} papers")
        print(f"  Metrics: {df_diffs['metric'].unique()}")
    
    metrics = ['soundness', 'presentation', 'contribution', 'rating']
    
    # Compute statistics
    print("\nComputing effect sizes (Cohen's d)...")
    stats_results = compute_statistics(df_diffs, comparisons, metrics)
    
    # Print results
    print("\n" + "="*80)
    print("EFFECT SIZE SUMMARY")
    print("="*80)
    
    if not comparisons:
        print("No treatment/control comparisons were configured. Nothing to report.")
    
    for metric in metrics:
        effects = stats_results.get(metric, [])
        print(f"\n{metric.upper()}:")
        if not effects:
            print("  No paired data available for this metric.")
            continue
        
        for effect in effects:
            expectation = effect['expected_direction']
            matches = effect['matches_expectation']
            match_icon = "✅" if matches else ("❌" if matches is False else "⚪️")
            expectation_text = f"expected {expectation}" if expectation else "no expectation"
            print(f"  {match_icon} {effect['comparison']} ({expectation_text})")
            print(f"    Pairs: {effect['n_pairs']}")
            print(f"    Treatment mean: {effect['treatment_mean']:.3f}")
            print(f"    Control mean: {effect['control_mean']:.3f}")
            print(f"    Mean diff (treatment-control): {effect['mean_difference']:.3f}")
            cohen_d = effect['cohen_d']
            if cohen_d is not None:
                ci_text = ""
                if effect['ci_low'] is not None and effect['ci_high'] is not None:
                    ci_text = f" (95% CI: [{effect['ci_low']:.3f}, {effect['ci_high']:.3f}])"
                print(f"    Cohen's d: {cohen_d:.3f}{ci_text}")
            else:
                print("    Cohen's d: undefined (insufficient variance)")
    
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
    for metric, effects in stats_results.items():
        for effect in effects:
            summary_rows.append({
                'metric': metric,
                'comparison': effect['comparison'],
                'treatment_label': effect['treatment_label'],
                'control_label': effect['control_label'],
                'expected_direction': effect['expected_direction'],
                'matches_expectation': effect['matches_expectation'],
                'n_pairs': effect['n_pairs'],
                'treatment_mean': effect['treatment_mean'],
                'control_mean': effect['control_mean'],
                'mean_difference': effect['mean_difference'],
                'cohen_d': effect['cohen_d'],
                'standard_error': effect['standard_error'],
                'ci_low': effect['ci_low'],
                'ci_high': effect['ci_high']
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
        create_effect_size_plots(stats_results, output_dir, pattern_name)
        print("\n✅ Cohen's d plot generated successfully!")
    except Exception as e:
        print(f"\n⚠️  Warning: Failed to generate some plots: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)
    print(f"\nOutput files in: {output_dir}/")
    print(f"  - score_differences.csv (detailed differences per paper)")
    print(f"  - summary_statistics.csv (summary by metric)")
    print(f"  - statistical_results.json (full statistics)")
    print(f"\nVisualization plots:")
    print(f"  - cohens_d_summary.png")

if __name__ == "__main__":
    main()

