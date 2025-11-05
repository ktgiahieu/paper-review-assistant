#!/usr/bin/env python3
"""
Evaluate the effect of author/affiliation modifications on paper review scores.

This script analyzes reviews from three versions:
1. Original (latest/)
2. Good author/affiliation (authors_affiliation_good/)
3. Bad author/affiliation (authors_affiliation_bad/)

Expected effects:
- Good author: Should improve scores (positive difference)
- Bad author: Should decrease scores (negative difference)

Usage:
    python evaluate_author_affiliation_effect.py --reviews_dir ../sampled_data/reviews/ICLR2024
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

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
    
    Returns dict with scores or None if extraction fails.
    """
    try:
        with open(review_file, 'r', encoding='utf-8') as f:
            review_data = json.load(f)
        
        # Check if review was successful
        if not review_data.get('success', False):
            return None
        
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
    
    Returns DataFrame with columns: paper_id, run_id, soundness, presentation, contribution, rating
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
        
        # Find all review files
        review_files = list(paper_dir.glob("review_run*.json"))
        
        for review_file in review_files:
            # Extract run_id from filename
            match = re.search(r'review_run(\d+)\.json', review_file.name)
            run_id = int(match.group(1)) if match else 0
            
            scores = extract_scores_from_review(review_file)
            if scores:
                record = {
                    'paper_id': paper_id,
                    'run_id': run_id,
                    'folder': folder_name,
                    **scores
                }
                records.append(record)
    
    return pd.DataFrame(records)

def calculate_differences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate differences between versions for each paper.
    
    Creates:
    - good_diff = authors_affiliation_good - latest (expected positive)
    - bad_diff = authors_affiliation_bad - latest (expected negative)
    """
    results = []
    
    # Group by paper_id and run_id
    for (paper_id, run_id), group in df.groupby(['paper_id', 'run_id']):
        # Get scores for each version
        latest_scores = group[group['folder'] == 'latest']
        good_scores = group[group['folder'] == 'authors_affiliation_good']
        bad_scores = group[group['folder'] == 'authors_affiliation_bad']
        
        if latest_scores.empty:
            continue
        
        latest_row = latest_scores.iloc[0]
        
        for metric in ['soundness', 'presentation', 'contribution', 'rating']:
            latest_val = latest_row.get(metric)
            
            if latest_val is None or pd.isna(latest_val):
                continue
            
            result = {
                'paper_id': paper_id,
                'run_id': run_id,
                'metric': metric,
                'latest_score': float(latest_val)
            }
            
            # Calculate good diff
            if not good_scores.empty:
                good_row = good_scores.iloc[0]
                good_val = good_row.get(metric)
                if good_val is not None and not pd.isna(good_val):
                    result['good_score'] = float(good_val)
                    result['good_diff'] = float(good_val) - float(latest_val)
                else:
                    result['good_score'] = None
                    result['good_diff'] = None
            else:
                result['good_score'] = None
                result['good_diff'] = None
            
            # Calculate bad diff
            if not bad_scores.empty:
                bad_row = bad_scores.iloc[0]
                bad_val = bad_row.get(metric)
                if bad_val is not None and not pd.isna(bad_val):
                    result['bad_score'] = float(bad_val)
                    result['bad_diff'] = float(bad_val) - float(latest_val)
                else:
                    result['bad_score'] = None
                    result['bad_diff'] = None
            else:
                result['bad_score'] = None
                result['bad_diff'] = None
            
            results.append(result)
    
    return pd.DataFrame(results)

def compute_statistics(df_diffs: pd.DataFrame) -> Dict:
    """
    Compute statistical tests for differences.
    
    For good_diff: Test if mean > 0 (one-tailed t-test)
    For bad_diff: Test if mean < 0 (one-tailed t-test)
    """
    results = {}
    
    for metric in ['soundness', 'presentation', 'contribution', 'rating']:
        metric_data = df_diffs[df_diffs['metric'] == metric].copy()
        
        # Good diff statistics
        good_diffs = metric_data['good_diff'].dropna()
        bad_diffs = metric_data['bad_diff'].dropna()
        
        metric_results = {
            'metric': metric,
            'n_papers': metric_data['paper_id'].nunique(),
            'n_complete_triplets': len(metric_data[
                metric_data['good_diff'].notna() & 
                metric_data['bad_diff'].notna()
            ])
        }
        
        # Good diff analysis (expected positive)
        if len(good_diffs) > 0:
            metric_results['good'] = {
                'n': len(good_diffs),
                'mean': float(good_diffs.mean()),
                'std': float(good_diffs.std()),
                'median': float(good_diffs.median()),
                'min': float(good_diffs.min()),
                'max': float(good_diffs.max()),
                'mean_latest': float(metric_data['latest_score'].mean()),
                'mean_good': float(metric_data['good_score'].dropna().mean()) if metric_data['good_score'].notna().any() else None
            }
            
            # One-tailed t-test: H0: mean <= 0, H1: mean > 0
            if len(good_diffs) > 1:
                t_stat, p_value_two_tailed = stats.ttest_1samp(good_diffs, 0)
                p_value_one_tailed = p_value_two_tailed / 2 if t_stat > 0 else 1 - p_value_two_tailed / 2
                
                metric_results['good']['t_statistic'] = float(t_stat)
                metric_results['good']['p_value'] = float(p_value_one_tailed)
                metric_results['good']['significant_at_0.05'] = bool(p_value_one_tailed < 0.05)
                metric_results['good']['significant_at_0.01'] = bool(p_value_one_tailed < 0.01)
                
                # Effect size (Cohen's d)
                cohens_d = good_diffs.mean() / good_diffs.std() if good_diffs.std() > 0 else 0
                metric_results['good']['cohens_d'] = float(cohens_d)
        else:
            metric_results['good'] = None
        
        # Bad diff analysis (expected negative)
        if len(bad_diffs) > 0:
            metric_results['bad'] = {
                'n': len(bad_diffs),
                'mean': float(bad_diffs.mean()),
                'std': float(bad_diffs.std()),
                'median': float(bad_diffs.median()),
                'min': float(bad_diffs.min()),
                'max': float(bad_diffs.max()),
                'mean_latest': float(metric_data['latest_score'].mean()),
                'mean_bad': float(metric_data['bad_score'].dropna().mean()) if metric_data['bad_score'].notna().any() else None
            }
            
            # One-tailed t-test: H0: mean >= 0, H1: mean < 0
            if len(bad_diffs) > 1:
                t_stat, p_value_two_tailed = stats.ttest_1samp(bad_diffs, 0)
                p_value_one_tailed = p_value_two_tailed / 2 if t_stat < 0 else 1 - p_value_two_tailed / 2
                
                metric_results['bad']['t_statistic'] = float(t_stat)
                metric_results['bad']['p_value'] = float(p_value_one_tailed)
                metric_results['bad']['significant_at_0.05'] = bool(p_value_one_tailed < 0.05)
                metric_results['bad']['significant_at_0.01'] = bool(p_value_one_tailed < 0.01)
                
                # Effect size (Cohen's d)
                cohens_d = bad_diffs.mean() / bad_diffs.std() if bad_diffs.std() > 0 else 0
                metric_results['bad']['cohens_d'] = float(cohens_d)
        else:
            metric_results['bad'] = None
        
        results[metric] = metric_results
    
    return results

def create_comparison_plots(df_diffs: pd.DataFrame, stats_results: Dict, output_dir: Path):
    """
    Create comprehensive visualization plots.
    """
    metrics = ['soundness', 'presentation', 'contribution', 'rating']
    metric_labels = {
        'soundness': 'Soundness',
        'presentation': 'Presentation',
        'contribution': 'Contribution',
        'rating': 'Overall Rating'
    }
    
    # 1. Box plots: Score distributions for all three versions
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Score Distributions: Original vs Good vs Bad Author/Affiliation', 
                 fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        metric_data = df_diffs[df_diffs['metric'] == metric]
        
        # Prepare data for box plot
        plot_data = []
        labels = []
        
        latest_scores = metric_data['latest_score'].dropna()
        if len(latest_scores) > 0:
            plot_data.append(latest_scores)
            labels.append('Original')
        
        good_scores = metric_data['good_score'].dropna()
        if len(good_scores) > 0:
            plot_data.append(good_scores)
            labels.append('Good Author')
        
        bad_scores = metric_data['bad_score'].dropna()
        if len(bad_scores) > 0:
            plot_data.append(bad_scores)
            labels.append('Bad Author')
        
        if plot_data:
            bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
            colors = ['#3498db', '#2ecc71', '#e74c3c'][:len(plot_data)]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Add significance markers if available
            stat = stats_results.get(metric, {})
            if stat.get('good') and stat['good'].get('significant_at_0.05'):
                y_max = max([s.max() for s in plot_data])
                ax.plot([1, 2], [y_max * 1.1, y_max * 1.1], 'k-', linewidth=1)
                sig = '***' if stat['good']['p_value'] < 0.001 else ('**' if stat['good']['p_value'] < 0.01 else '*')
                ax.text(1.5, y_max * 1.12, sig, ha='center', fontsize=10, fontweight='bold')
            
            if stat.get('bad') and stat['bad'].get('significant_at_0.05'):
                y_max = max([s.max() for s in plot_data])
                ax.plot([1, 3], [y_max * 1.15, y_max * 1.15], 'k-', linewidth=1)
                sig = '***' if stat['bad']['p_value'] < 0.001 else ('**' if stat['bad']['p_value'] < 0.01 else '*')
                ax.text(2, y_max * 1.17, sig, ha='center', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Score')
        ax.set_title(metric_labels[metric])
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = output_dir / "score_distributions_comparison.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")
    
    # 2. Violin plots: Difference distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Score Differences: Good Author vs Original, Bad Author vs Original', 
                 fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        metric_data = df_diffs[df_diffs['metric'] == metric]
        
        plot_data = []
        labels = []
        
        good_diffs = metric_data['good_diff'].dropna()
        if len(good_diffs) > 0:
            plot_data.append(good_diffs)
            labels.append('Good - Original\n(Expected: +)')
        
        bad_diffs = metric_data['bad_diff'].dropna()
        if len(bad_diffs) > 0:
            plot_data.append(bad_diffs)
            labels.append('Bad - Original\n(Expected: -)')
        
        if plot_data:
            parts = ax.violinplot(plot_data, positions=range(len(plot_data)), 
                                 showmeans=True, showmedians=True)
            
            # Color the violins
            colors = ['#2ecc71', '#e74c3c'][:len(plot_data)]
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)
            
            # Zero line
            ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
            
            # Add statistics
            stat = stats_results.get(metric, {})
            stats_text = ""
            if stat.get('good'):
                g = stat['good']
                stats_text += f"Good: μ={g['mean']:.3f}"
                if g.get('significant_at_0.05'):
                    stats_text += f"* (p={g['p_value']:.4f})"
                stats_text += "\n"
            if stat.get('bad'):
                b = stat['bad']
                stats_text += f"Bad: μ={b['mean']:.3f}"
                if b.get('significant_at_0.05'):
                    stats_text += f"* (p={b['p_value']:.4f})"
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5), fontsize=9)
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylabel('Score Difference')
            ax.set_title(metric_labels[metric])
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = output_dir / "difference_distributions.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")
    
    # 3. Scatter plots: Good vs Bad differences
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Comparison: Good Author Effect vs Bad Author Effect', 
                 fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        metric_data = df_diffs[
            (df_diffs['metric'] == metric) &
            (df_diffs['good_diff'].notna()) &
            (df_diffs['bad_diff'].notna())
        ]
        
        if len(metric_data) == 0:
            ax.text(0.5, 0.5, 'No complete data', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(metric_labels[metric])
            continue
        
        good_diffs = metric_data['good_diff'].values
        bad_diffs = metric_data['bad_diff'].values
        
        # Scatter plot
        ax.scatter(good_diffs, bad_diffs, alpha=0.6, s=50, color='#3498db', edgecolors='black')
        
        # Reference lines
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Diagonal line (good = -bad)
        min_val = min(good_diffs.min(), -bad_diffs.max())
        max_val = max(good_diffs.max(), -bad_diffs.min())
        ax.plot([min_val, max_val], [-min_val, -max_val], 'r--', 
               linewidth=1, alpha=0.5, label='Symmetric effect')
        
        # Add correlation
        if len(good_diffs) > 1:
            corr = np.corrcoef(good_diffs, bad_diffs)[0, 1]
            ax.text(0.02, 0.98, f'r = {corr:.3f}', transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5), fontsize=9)
        
        ax.set_xlabel('Good Author Effect (Good - Original)')
        ax.set_ylabel('Bad Author Effect (Bad - Original)')
        ax.set_title(metric_labels[metric])
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "good_vs_bad_scatter.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")
    
    # 4. Heatmap: Effect sizes by metric
    fig, ax = plt.subplots(figsize=(10, 6))
    
    effect_data = []
    for metric in metrics:
        stat = stats_results.get(metric, {})
        if stat.get('good') and stat.get('bad'):
            effect_data.append({
                'Metric': metric_labels[metric],
                'Good Effect\n(mean diff)': stat['good']['mean'],
                'Bad Effect\n(mean diff)': stat['bad']['mean'],
                'Good Cohen\'s d': stat['good'].get('cohens_d', 0),
                'Bad Cohen\'s d': stat['bad'].get('cohens_d', 0)
            })
    
    if effect_data:
        effect_df = pd.DataFrame(effect_data)
        effect_df = effect_df.set_index('Metric')
        
        # Plot heatmap
        sns.heatmap(effect_df[['Good Effect\n(mean diff)', 'Bad Effect\n(mean diff)']], 
                   annot=True, fmt='.3f', cmap='RdYlGn', center=0, 
                   cbar_kws={'label': 'Score Difference'}, ax=ax, 
                   linewidths=0.5, linecolor='gray')
        
        ax.set_title('Author/Affiliation Effect by Metric', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_path = output_dir / "effect_heatmap.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plot_path}")
    
    # 5. Bar plot: Mean differences with confidence intervals
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Mean Score Differences with 95% Confidence Intervals', 
                 fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        stat = stats_results.get(metric, {})
        
        categories = []
        means = []
        errors_lower = []
        errors_upper = []
        colors_list = []
        
        if stat.get('good'):
            g = stat['good']
            categories.append('Good - Original')
            means.append(g['mean'])
            # 95% CI
            n = g['n']
            if n > 1:
                sem = g['std'] / np.sqrt(n)
                ci = stats.t.interval(0.95, n-1, loc=g['mean'], scale=sem)
                errors_lower.append(g['mean'] - ci[0])
                errors_upper.append(ci[1] - g['mean'])
            else:
                errors_lower.append(0)
                errors_upper.append(0)
            colors_list.append('#2ecc71')
        
        if stat.get('bad'):
            b = stat['bad']
            categories.append('Bad - Original')
            means.append(b['mean'])
            # 95% CI
            n = b['n']
            if n > 1:
                sem = b['std'] / np.sqrt(n)
                ci = stats.t.interval(0.95, n-1, loc=b['mean'], scale=sem)
                errors_lower.append(b['mean'] - ci[0])
                errors_upper.append(ci[1] - b['mean'])
            else:
                errors_lower.append(0)
                errors_upper.append(0)
            colors_list.append('#e74c3c')
        
        if categories:
            x_pos = np.arange(len(categories))
            bars = ax.bar(x_pos, means, yerr=[errors_lower, errors_upper], 
                         capsize=10, color=colors_list, alpha=0.7, edgecolor='black')
            
            # Add significance stars
            for i, cat in enumerate(categories):
                if 'Good' in cat and stat.get('good', {}).get('significant_at_0.05'):
                    p_val = stat['good']['p_value']
                    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else '*')
                    y_pos = means[i] + errors_upper[i] + 0.1
                    ax.text(i, y_pos, sig, ha='center', fontsize=12, fontweight='bold')
                elif 'Bad' in cat and stat.get('bad', {}).get('significant_at_0.05'):
                    p_val = stat['bad']['p_value']
                    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else '*')
                    y_pos = means[i] + errors_upper[i] + 0.1
                    ax.text(i, y_pos, sig, ha='center', fontsize=12, fontweight='bold')
            
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(categories, rotation=45, ha='right')
            ax.set_ylabel('Score Difference')
            ax.set_title(metric_labels[metric])
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = output_dir / "mean_differences_with_ci.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the effect of author/affiliation modifications on review scores"
    )
    parser.add_argument(
        "--reviews_dir",
        type=str,
        default="../sampled_data/reviews/ICLR2024",
        help="Directory containing review folders (default: ../sampled_data/reviews/ICLR2024)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: reviews_dir/../evaluation_results)"
    )
    parser.add_argument(
        "--folders",
        type=str,
        nargs="+",
        default=["latest", "authors_affiliation_good", "authors_affiliation_bad"],
        help="Folders to analyze (default: latest authors_affiliation_good authors_affiliation_bad)"
    )
    
    args = parser.parse_args()
    
    reviews_dir = Path(args.reviews_dir)
    
    # Set default output directory
    if args.output_dir is None:
        output_dir = reviews_dir.parent / "evaluation_results"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Author/Affiliation Effect Evaluation")
    print("="*80)
    print(f"Reviews directory: {reviews_dir}")
    print(f"Folders to analyze: {args.folders}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Collect scores from all folders
    print("Collecting scores from review files...")
    all_scores = []
    
    for folder in args.folders:
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
    df_diffs = calculate_differences(df_all)
    
    if len(df_diffs) == 0:
        print("Error: Could not calculate differences. Check that all three versions exist for papers.")
        return
    
    print(f"Calculated differences for {df_diffs['paper_id'].nunique()} papers")
    print(f"  Metrics: {df_diffs['metric'].unique()}")
    
    # Compute statistics
    print("\nComputing statistical tests...")
    stats_results = compute_statistics(df_diffs)
    
    # Print results
    print("\n" + "="*80)
    print("STATISTICAL RESULTS")
    print("="*80)
    
    for metric in ['soundness', 'presentation', 'contribution', 'rating']:
        stat = stats_results.get(metric, {})
        print(f"\n{metric.upper()}:")
        print(f"  Papers: {stat.get('n_papers', 0)}")
        print(f"  Complete triplets: {stat.get('n_complete_triplets', 0)}")
        
        if stat.get('good'):
            g = stat['good']
            print(f"\n  Good Author Effect (Good - Original):")
            print(f"    N: {g['n']}")
            print(f"    Mean difference: {g['mean']:.3f} ± {g['std']:.3f}")
            print(f"    Median: {g['median']:.3f}")
            print(f"    Range: [{g['min']:.3f}, {g['max']:.3f}]")
            if g.get('t_statistic') is not None:
                print(f"    t-statistic: {g['t_statistic']:.3f}")
                print(f"    p-value (one-tailed): {g['p_value']:.4f}")
                print(f"    Significant (p<0.05): {'Yes' if g['significant_at_0.05'] else 'No'}")
                print(f"    Cohen's d: {g['cohens_d']:.3f}")
                print(f"    Interpretation: {'✅ Good author improves scores' if g['mean'] > 0 and g['significant_at_0.05'] else '❌ No significant improvement'}")
        
        if stat.get('bad'):
            b = stat['bad']
            print(f"\n  Bad Author Effect (Bad - Original):")
            print(f"    N: {b['n']}")
            print(f"    Mean difference: {b['mean']:.3f} ± {b['std']:.3f}")
            print(f"    Median: {b['median']:.3f}")
            print(f"    Range: [{b['min']:.3f}, {b['max']:.3f}]")
            if b.get('t_statistic') is not None:
                print(f"    t-statistic: {b['t_statistic']:.3f}")
                print(f"    p-value (one-tailed): {b['p_value']:.4f}")
                print(f"    Significant (p<0.05): {'Yes' if b['significant_at_0.05'] else 'No'}")
                print(f"    Cohen's d: {b['cohens_d']:.3f}")
                print(f"    Interpretation: {'✅ Bad author decreases scores' if b['mean'] < 0 and b['significant_at_0.05'] else '❌ No significant decrease'}")
    
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
    for metric in ['soundness', 'presentation', 'contribution', 'rating']:
        stat = stats_results.get(metric, {})
        row = {'metric': metric}
        
        if stat.get('good'):
            g = stat['good']
            row.update({
                'good_mean_diff': g['mean'],
                'good_std': g['std'],
                'good_p_value': g.get('p_value'),
                'good_significant': g.get('significant_at_0.05'),
                'good_cohens_d': g.get('cohens_d')
            })
        
        if stat.get('bad'):
            b = stat['bad']
            row.update({
                'bad_mean_diff': b['mean'],
                'bad_std': b['std'],
                'bad_p_value': b.get('p_value'),
                'bad_significant': b.get('significant_at_0.05'),
                'bad_cohens_d': b.get('cohens_d')
            })
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / "summary_statistics.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"✅ Saved summary to: {summary_csv_path}")
    
    # Generate plots
    print("\n" + "="*80)
    print("Generating plots...")
    print("="*80)
    
    try:
        create_comparison_plots(df_diffs, stats_results, output_dir)
        print("\n✅ All plots generated successfully!")
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
    print(f"  - score_distributions_comparison.png")
    print(f"  - difference_distributions.png")
    print(f"  - good_vs_bad_scatter.png")
    print(f"  - effect_heatmap.png")
    print(f"  - mean_differences_with_ci.png")

if __name__ == "__main__":
    main()

