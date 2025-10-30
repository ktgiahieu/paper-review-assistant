#!/usr/bin/env python3
"""
Numerical Score Evaluator for Paper Reviews

Extracts numerical scores (soundness, presentation, contribution, rating) from 
review JSON files across different formats and performs statistical analysis 
to test whether AI reviewers can differentiate between v1 and latest versions.

Supports:
- SEA-E format
- CycleReviewer format (4 reviewers per paper)
- GenericStructured format
- Default format

Usage:
    python evaluate_numerical_scores.py --reviews_dir ./reviews_output --output_dir ./evaluation_results
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict

def extract_numerical_value(score_str: str) -> Optional[float]:
    """
    Extract numerical value from score string.
    
    Examples:
        "3 good" -> 3.0
        "2 fair" -> 2.0
        "6: marginally above the acceptance threshold" -> 6.0
        "3.5" -> 3.5
    
    Args:
        score_str: String containing numerical score
        
    Returns:
        Extracted numerical value or None if not found
    """
    if not score_str:
        return None
    
    # Try to extract number from various formats
    # Pattern 1: "3 good", "2 fair"
    match = re.match(r'^(\d+(?:\.\d+)?)', str(score_str).strip())
    if match:
        return float(match.group(1))
    
    return None

def extract_scores_seae(review_data: dict) -> Dict[str, Optional[float]]:
    """Extract numerical scores from SEA-E format review."""
    scores = {
        'soundness': extract_numerical_value(review_data.get('soundness')),
        'presentation': extract_numerical_value(review_data.get('presentation')),
        'contribution': extract_numerical_value(review_data.get('contribution')),
        'rating': extract_numerical_value(review_data.get('rating'))
    }
    return scores

def extract_scores_cyclereviewer(review_data: dict) -> List[Dict[str, Optional[float]]]:
    """
    Extract numerical scores from CycleReviewer format review.
    
    Returns list of score dicts, one per reviewer (typically 4 reviewers).
    """
    reviewers_scores = []
    
    reviewers = review_data.get('reviewers', [])
    for reviewer_idx, reviewer in enumerate(reviewers):
        scores = {
            'reviewer_id': reviewer_idx,
            'soundness': extract_numerical_value(reviewer.get('soundness')),
            'presentation': extract_numerical_value(reviewer.get('presentation')),
            'contribution': extract_numerical_value(reviewer.get('contribution')),
            'rating': extract_numerical_value(reviewer.get('rating'))
        }
        reviewers_scores.append(scores)
    
    return reviewers_scores

def extract_scores_generic_structured(review_data: dict) -> Dict[str, Optional[float]]:
    """Extract numerical scores from GenericStructured format review."""
    scores = {
        'soundness': extract_numerical_value(review_data.get('soundness')),
        'presentation': extract_numerical_value(review_data.get('presentation')),
        'contribution': extract_numerical_value(review_data.get('contribution')),
        'rating': extract_numerical_value(review_data.get('rating'))
    }
    return scores

def extract_scores_default(review_data: dict) -> Dict[str, Optional[float]]:
    """Extract numerical scores from default format review."""
    # Default format uses different field names
    scores = {
        'soundness': None,  # Not in default format
        'presentation': None,  # Not in default format
        'contribution': None,  # Not in default format
        'rating': review_data.get('overall_score')  # Use overall_score as rating
    }
    return scores

def extract_scores_from_review(review_file: Path) -> List[Dict]:
    """
    Extract numerical scores from a review JSON file.
    
    Returns list of score records. For most formats, this is a single record.
    For CycleReviewer, this is 4 records (one per reviewer).
    """
    try:
        with open(review_file, 'r', encoding='utf-8') as f:
            review_data = json.load(f)
        
        # Check if review was successful
        if not review_data.get('success', False):
            print(f"Warning: Review failed for {review_file}")
            return []
        
        # Extract basic metadata
        paper_id = review_data.get('paper_id')
        version = review_data.get('version')
        run_id = review_data.get('run_id', 0)
        model_type = review_data.get('model_type', 'unknown')
        
        # Extract scores based on format
        records = []
        
        if model_type == "SEA-E":
            scores = extract_scores_seae(review_data)
            records.append({
                'paper_id': paper_id,
                'version': version,
                'run_id': run_id,
                'model_type': model_type,
                'reviewer_id': 0,  # Single reviewer
                **scores
            })
        
        elif model_type == "CycleReviewer":
            reviewers_scores = extract_scores_cyclereviewer(review_data)
            for reviewer_scores in reviewers_scores:
                reviewer_id = reviewer_scores.pop('reviewer_id')
                records.append({
                    'paper_id': paper_id,
                    'version': version,
                    'run_id': run_id,
                    'model_type': model_type,
                    'reviewer_id': reviewer_id,
                    **reviewer_scores
                })
        
        elif model_type == "GenericStructured":
            scores = extract_scores_generic_structured(review_data)
            records.append({
                'paper_id': paper_id,
                'version': version,
                'run_id': run_id,
                'model_type': model_type,
                'reviewer_id': 0,
                **scores
            })
        
        else:  # default or unknown
            scores = extract_scores_default(review_data)
            records.append({
                'paper_id': paper_id,
                'version': version,
                'run_id': run_id,
                'model_type': model_type,
                'reviewer_id': 0,
                **scores
            })
        
        return records
    
    except Exception as e:
        print(f"Error processing {review_file}: {e}")
        return []

def collect_all_scores(reviews_dir: Path) -> pd.DataFrame:
    """
    Collect all numerical scores from review files in directory.
    
    Returns DataFrame with columns:
    - paper_id
    - version (v1 or latest)
    - run_id
    - model_type
    - reviewer_id (0 for single-reviewer formats, 0-3 for CycleReviewer)
    - soundness (1-4)
    - presentation (1-4)
    - contribution (1-4)
    - rating (1-10)
    """
    all_records = []
    
    # Find all paper directories
    paper_dirs = [d for d in reviews_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(paper_dirs)} paper directories")
    
    for paper_dir in paper_dirs:
        # Find all review JSON files
        review_files = list(paper_dir.glob("*_review_run*.json"))
        
        for review_file in review_files:
            records = extract_scores_from_review(review_file)
            all_records.extend(records)
    
    df = pd.DataFrame(all_records)
    
    if len(df) > 0:
        print(f"\nCollected {len(df)} score records")
        print(f"Papers: {df['paper_id'].nunique()}")
        print(f"Model types: {df['model_type'].unique()}")
        print(f"Versions: {df['version'].unique()}")
    else:
        print("Warning: No scores collected!")
    
    return df

def compute_paired_statistics(df: pd.DataFrame, metric: str) -> Dict:
    """
    Compute paired t-test statistics for a metric comparing v1 vs latest.
    
    For CycleReviewer (multiple reviewers), we aggregate scores first.
    For formats with multiple runs, we handle each run separately then aggregate.
    
    Args:
        df: DataFrame with scores
        metric: One of 'soundness', 'presentation', 'contribution', 'rating'
    
    Returns:
        Dictionary with statistical results
    """
    # Filter out None values
    df_valid = df[df[metric].notna()].copy()
    
    if len(df_valid) == 0:
        return {
            'metric': metric,
            'n_pairs': 0,
            'error': f'No valid {metric} scores found'
        }
    
    # Aggregate scores by paper_id, version, run_id
    # For CycleReviewer, average across reviewers
    # For other formats, reviewer_id is always 0
    aggregated = df_valid.groupby(['paper_id', 'version', 'run_id'])[metric].mean().reset_index()
    
    # Pivot to get v1 and latest side-by-side
    pivot = aggregated.pivot_table(
        index=['paper_id', 'run_id'],
        columns='version',
        values=metric
    ).reset_index()
    
    # Check if we have both versions
    if 'v1' not in pivot.columns or 'latest' not in pivot.columns:
        return {
            'metric': metric,
            'n_pairs': 0,
            'error': 'Missing v1 or latest version'
        }
    
    # Drop rows with missing values
    pivot_complete = pivot.dropna(subset=['v1', 'latest'])
    
    if len(pivot_complete) < 2:
        return {
            'metric': metric,
            'n_pairs': len(pivot_complete),
            'error': 'Insufficient paired data for t-test (need at least 2 pairs)'
        }
    
    v1_scores = pivot_complete['v1'].values
    latest_scores = pivot_complete['latest'].values
    
    # Compute differences
    differences = latest_scores - v1_scores
    
    # Paired t-test
    t_statistic, p_value = stats.ttest_rel(latest_scores, v1_scores)
    
    # Effect size (Cohen's d for paired samples)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0
    
    # Confidence interval for mean difference
    ci_95 = stats.t.interval(
        0.95,
        len(differences) - 1,
        loc=mean_diff,
        scale=stats.sem(differences)
    )
    
    results = {
        'metric': metric,
        'n_pairs': len(pivot_complete),
        'n_papers': pivot_complete['paper_id'].nunique(),
        'v1_mean': float(np.mean(v1_scores)),
        'v1_std': float(np.std(v1_scores, ddof=1)),
        'latest_mean': float(np.mean(latest_scores)),
        'latest_std': float(np.std(latest_scores, ddof=1)),
        'mean_difference': float(mean_diff),
        'std_difference': float(std_diff),
        't_statistic': float(t_statistic),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'ci_95_lower': float(ci_95[0]),
        'ci_95_upper': float(ci_95[1]),
        'significant_at_0.05': p_value < 0.05,
        'significant_at_0.01': p_value < 0.01,
        'interpretation': interpret_results(mean_diff, p_value, cohens_d)
    }
    
    return results

def interpret_results(mean_diff: float, p_value: float, cohens_d: float) -> str:
    """Generate human-readable interpretation of results."""
    direction = "higher" if mean_diff > 0 else "lower"
    
    if p_value < 0.01:
        significance = "highly significant"
    elif p_value < 0.05:
        significance = "significant"
    else:
        significance = "not significant"
    
    if abs(cohens_d) < 0.2:
        effect = "negligible"
    elif abs(cohens_d) < 0.5:
        effect = "small"
    elif abs(cohens_d) < 0.8:
        effect = "medium"
    else:
        effect = "large"
    
    return (f"Latest version scored {abs(mean_diff):.3f} points {direction} than v1 "
            f"({significance}, p={p_value:.4f}). Effect size: {effect} (Cohen's d={cohens_d:.3f}).")

def analyze_by_model_type(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Perform analysis separately for each model type.
    
    Returns dict mapping model_type -> metrics -> results
    """
    results_by_model = {}
    
    for model_type in df['model_type'].unique():
        df_model = df[df['model_type'] == model_type]
        
        print(f"\n{'='*80}")
        print(f"Analyzing model type: {model_type}")
        print(f"{'='*80}")
        
        model_results = {}
        
        for metric in ['soundness', 'presentation', 'contribution', 'rating']:
            print(f"\n{metric.upper()}:")
            results = compute_paired_statistics(df_model, metric)
            
            if 'error' in results:
                print(f"  {results['error']}")
            else:
                print(f"  Pairs: {results['n_pairs']} (from {results['n_papers']} papers)")
                print(f"  v1:     {results['v1_mean']:.3f} ± {results['v1_std']:.3f}")
                print(f"  Latest: {results['latest_mean']:.3f} ± {results['latest_std']:.3f}")
                print(f"  Diff:   {results['mean_difference']:.3f} ± {results['std_difference']:.3f}")
                print(f"  t({results['n_pairs']-1}) = {results['t_statistic']:.3f}, p = {results['p_value']:.4f}")
                print(f"  Cohen's d = {results['cohens_d']:.3f}")
                print(f"  95% CI: [{results['ci_95_lower']:.3f}, {results['ci_95_upper']:.3f}]")
                print(f"  {results['interpretation']}")
            
            model_results[metric] = results
        
        results_by_model[model_type] = model_results
    
    return results_by_model

def generate_summary_table(results_by_model: Dict[str, Dict]) -> pd.DataFrame:
    """Generate summary table of all results."""
    rows = []
    
    for model_type, model_results in results_by_model.items():
        for metric, results in model_results.items():
            if 'error' not in results:
                rows.append({
                    'Model': model_type,
                    'Metric': metric,
                    'N': results['n_pairs'],
                    'v1_mean': f"{results['v1_mean']:.3f}",
                    'latest_mean': f"{results['latest_mean']:.3f}",
                    'Difference': f"{results['mean_difference']:.3f}",
                    't_statistic': f"{results['t_statistic']:.3f}",
                    'p_value': f"{results['p_value']:.4f}",
                    "Cohen's_d": f"{results['cohens_d']:.3f}",
                    'Significant': '***' if results['p_value'] < 0.01 else ('**' if results['p_value'] < 0.05 else 'ns')
                })
    
    return pd.DataFrame(rows)

def analyze_cyclereviewer_agreement(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Special analysis for CycleReviewer: compute inter-reviewer agreement.
    
    Returns DataFrame with agreement statistics per paper.
    """
    df_cycle = df[df['model_type'] == 'CycleReviewer']
    
    if len(df_cycle) == 0:
        return None
    
    print(f"\n{'='*80}")
    print("CycleReviewer Inter-Reviewer Agreement Analysis")
    print(f"{'='*80}")
    
    agreement_records = []
    
    # Group by paper, version, run
    for (paper_id, version, run_id), group in df_cycle.groupby(['paper_id', 'version', 'run_id']):
        if len(group) < 2:  # Need at least 2 reviewers
            continue
        
        for metric in ['soundness', 'presentation', 'contribution', 'rating']:
            scores = group[metric].dropna()
            
            if len(scores) < 2:
                continue
            
            agreement_records.append({
                'paper_id': paper_id,
                'version': version,
                'run_id': run_id,
                'metric': metric,
                'n_reviewers': len(scores),
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max(),
                'range': scores.max() - scores.min()
            })
    
    if not agreement_records:
        return None
    
    agreement_df = pd.DataFrame(agreement_records)
    
    # Summary statistics
    print("\nAverage inter-reviewer statistics by metric:")
    summary = agreement_df.groupby('metric').agg({
        'std': 'mean',
        'range': 'mean'
    }).round(3)
    print(summary)
    
    return agreement_df

def main():
    parser = argparse.ArgumentParser(
        description="Extract and analyze numerical scores from paper reviews"
    )
    parser.add_argument(
        "--reviews_dir",
        type=str,
        required=True,
        help="Directory containing review output (with paper subdirectories)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="evaluation",
        help="Prefix for output files"
    )
    
    args = parser.parse_args()
    
    reviews_dir = Path(args.reviews_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Numerical Score Evaluator for Paper Reviews")
    print("="*80)
    print(f"Reviews directory: {reviews_dir}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Collect all scores
    print("\n" + "="*80)
    print("Step 1: Collecting scores from review files...")
    print("="*80)
    
    df_scores = collect_all_scores(reviews_dir)
    
    if len(df_scores) == 0:
        print("\nError: No scores collected. Check reviews_dir path and file structure.")
        return
    
    # Save raw scores
    scores_file = output_dir / f"{args.output_prefix}_scores.csv"
    df_scores.to_csv(scores_file, index=False)
    print(f"\nSaved raw scores to: {scores_file}")
    
    # Step 2: Perform paired t-tests by model type
    print("\n" + "="*80)
    print("Step 2: Performing paired t-tests (v1 vs latest)...")
    print("="*80)
    
    results_by_model = analyze_by_model_type(df_scores)
    
    # Save detailed results as JSON
    results_file = output_dir / f"{args.output_prefix}_detailed_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_by_model, f, indent=2)
    print(f"\nSaved detailed results to: {results_file}")
    
    # Step 3: Generate summary table
    print("\n" + "="*80)
    print("Step 3: Generating summary table...")
    print("="*80)
    
    summary_df = generate_summary_table(results_by_model)
    
    if len(summary_df) > 0:
        print("\n" + summary_df.to_string(index=False))
        
        summary_file = output_dir / f"{args.output_prefix}_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSaved summary table to: {summary_file}")
    
    # Step 4: CycleReviewer-specific analysis
    if 'CycleReviewer' in df_scores['model_type'].values:
        agreement_df = analyze_cyclereviewer_agreement(df_scores)
        
        if agreement_df is not None:
            agreement_file = output_dir / f"{args.output_prefix}_cyclereviewer_agreement.csv"
            agreement_df.to_csv(agreement_file, index=False)
            print(f"\nSaved CycleReviewer agreement analysis to: {agreement_file}")
    
    print("\n" + "="*80)
    print("✅ Evaluation complete!")
    print("="*80)
    print(f"\nOutput files in: {output_dir}/")
    print(f"  - {args.output_prefix}_scores.csv (raw scores)")
    print(f"  - {args.output_prefix}_summary.csv (summary table)")
    print(f"  - {args.output_prefix}_detailed_results.json (full statistics)")
    if 'CycleReviewer' in df_scores['model_type'].values:
        print(f"  - {args.output_prefix}_cyclereviewer_agreement.csv (inter-reviewer agreement)")

if __name__ == "__main__":
    main()

