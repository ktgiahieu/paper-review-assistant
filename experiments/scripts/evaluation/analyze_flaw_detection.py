#!/usr/bin/env python3
"""
Analyze Flaw Detection Results with Paired t-Tests

This script performs statistical analysis on flaw detection results,
comparing v1 vs latest versions to see if paper revisions improved
the AI's ability to detect consensus flaws.

Usage:
    python analyze_flaw_detection.py \
      --results_file ./flaw_detection_results/flaw_detection_detailed.json \
      --output_dir ./flaw_detection_analysis/
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def load_results(results_file: Path) -> List[Dict]:
    """Load flaw detection results from JSON file."""
    with open(results_file) as f:
        return json.load(f)

def perform_paired_ttest(df_paired: pd.DataFrame) -> Dict:
    """
    Perform paired t-test comparing v1 vs latest recall.
    
    Args:
        df_paired: DataFrame with columns [paper_id, run_id, v1_recall, latest_recall]
    
    Returns:
        Dict with statistical results
    """
    v1_recalls = df_paired['v1_recall'].values
    latest_recalls = df_paired['latest_recall'].values
    
    # Paired t-test
    t_statistic, p_value = stats.ttest_rel(latest_recalls, v1_recalls)
    
    # Differences
    differences = latest_recalls - v1_recalls
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    
    # Confidence interval
    n = len(differences)
    se = std_diff / np.sqrt(n)
    ci_95 = stats.t.interval(0.95, n-1, loc=mean_diff, scale=se)
    
    # Cohen's d effect size
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0
    
    # Interpretation
    def interpret_cohens_d(d):
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def interpret_results(mean_diff, p_value, cohens_d):
        sig = "highly significant" if p_value < 0.01 else ("significant" if p_value < 0.05 else "not significant")
        direction = "higher" if mean_diff > 0 else "lower"
        effect = interpret_cohens_d(cohens_d)
        
        return f"Latest version had {abs(mean_diff):.3f} {'higher' if mean_diff > 0 else 'lower'} recall than v1 ({sig}, p={p_value:.4f}). Effect size: {effect} (Cohen's d={cohens_d:.3f})."
    
    return {
        'n_pairs': int(n),
        'n_papers': int(df_paired['paper_id'].nunique()),
        'v1_mean': float(np.mean(v1_recalls)),
        'v1_std': float(np.std(v1_recalls, ddof=1)),
        'latest_mean': float(np.mean(latest_recalls)),
        'latest_std': float(np.std(latest_recalls, ddof=1)),
        'mean_difference': float(mean_diff),
        'std_difference': float(std_diff),
        't_statistic': float(t_statistic),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'ci_95_lower': float(ci_95[0]),
        'ci_95_upper': float(ci_95[1]),
        'significant_at_0.05': bool(p_value < 0.05),
        'significant_at_0.01': bool(p_value < 0.01),
        'interpretation': interpret_results(mean_diff, p_value, cohens_d)
    }

def create_comparison_plots(df_summary: pd.DataFrame, results: Dict, output_dir: Path):
    """Create visualization plots for flaw detection comparison."""
    
    # Prepare data for plotting
    df_v1 = df_summary[df_summary['version'] == 'v1'].copy()
    df_latest = df_summary[df_summary['version'] == 'latest'].copy()
    
    # Aggregate across runs
    df_v1_agg = df_v1.groupby('paper_id')['recall'].mean().reset_index()
    df_v1_agg.columns = ['paper_id', 'v1_recall']
    
    df_latest_agg = df_latest.groupby('paper_id')['recall'].mean().reset_index()
    df_latest_agg.columns = ['paper_id', 'latest_recall']
    
    df_paired = df_v1_agg.merge(df_latest_agg, on='paper_id', how='inner')
    
    if len(df_paired) == 0:
        print("Warning: No paired data for plotting")
        return
    
    # 1. Bar plot: v1 vs latest mean recall
    fig, ax = plt.subplots(figsize=(8, 6))
    
    means = [results['v1_mean'], results['latest_mean']]
    stds = [results['v1_std'], results['latest_std']]
    labels = ['v1', 'latest']
    colors = ['#3498db', '#2ecc71']
    
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    
    # Add significance stars
    sig_text = '***' if results['p_value'] < 0.001 else ('**' if results['p_value'] < 0.01 else ('*' if results['p_value'] < 0.05 else 'ns'))
    y_max = max(means[0] + stds[0], means[1] + stds[1])
    ax.plot([0, 1], [y_max + 0.05, y_max + 0.05], 'k-', linewidth=1)
    ax.text(0.5, y_max + 0.08, sig_text, ha='center', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Mean Flaw Detection Recall')
    ax.set_title('Flaw Detection: v1 vs Latest\n(Higher is better)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, min(1.0, y_max + 0.15))
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add sample size
    ax.text(0.02, 0.98, f'n = {results["n_pairs"]} pairs', 
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plot_path = output_dir / "flaw_detection_bar_comparison.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")
    
    # 2. Scatter plot: v1 vs latest recall
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(df_paired['v1_recall'], df_paired['latest_recall'], 
              alpha=0.6, s=50, color='#3498db', edgecolors='black', linewidth=0.5)
    
    # Diagonal line (perfect agreement)
    lims = [0, 1]
    ax.plot(lims, lims, 'r--', linewidth=2, label='No change', zorder=1)
    
    # Regression line
    if len(df_paired) > 1:
        z = np.polyfit(df_paired['v1_recall'], df_paired['latest_recall'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, 1, 100)
        ax.plot(x_line, p(x_line), 'g-', linewidth=2, alpha=0.7, label='Regression')
    
    # Add statistics
    corr = np.corrcoef(df_paired['v1_recall'], df_paired['latest_recall'])[0, 1]
    
    stats_text = f"n = {len(df_paired)}\n"
    stats_text += f"r = {corr:.3f}\n"
    stats_text += f"Mean Δ = {results['mean_difference']:.3f}\n"
    stats_text += f"p = {results['p_value']:.4f}"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.5), fontsize=10)
    
    ax.set_xlabel('v1 Recall')
    ax.set_ylabel('Latest Recall')
    ax.set_title('Flaw Detection Recall: v1 vs Latest', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plot_path = output_dir / "flaw_detection_scatter.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")
    
    # 3. Difference distribution (key for paired t-test)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    differences = df_paired['latest_recall'] - df_paired['v1_recall']
    
    # Histogram + KDE
    ax.hist(differences, bins=20, alpha=0.6, color='#3498db', edgecolor='black', density=True)
    
    if len(differences) > 1:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(differences)
        x_range = np.linspace(differences.min(), differences.max(), 100)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    
    # Zero line (null hypothesis)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='H₀: Δ=0')
    
    # Mean difference
    mean_diff = differences.mean()
    ax.axvline(x=mean_diff, color='red', linestyle='-', linewidth=2, 
              label=f'Mean Δ={mean_diff:.3f}')
    
    # Confidence interval
    ci_lower = results['ci_95_lower']
    ci_upper = results['ci_95_upper']
    ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='green', label='95% CI')
    
    # Statistics box
    stats_text = f"n = {len(differences)}\n"
    if results['p_value'] < 0.001:
        stats_text += "p < 0.001***\n"
    elif results['p_value'] < 0.01:
        stats_text += f"p = {results['p_value']:.3f}**\n"
    elif results['p_value'] < 0.05:
        stats_text += f"p = {results['p_value']:.3f}*\n"
    else:
        stats_text += f"p = {results['p_value']:.3f} (ns)\n"
    stats_text += f"d = {results['cohens_d']:.3f}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.5), fontsize=10)
    
    ax.set_xlabel('Recall Difference (Latest - v1)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Flaw Detection Recall Differences', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = output_dir / "flaw_detection_difference_distribution.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")
    
    # 4. Violin plot: distribution comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    
    plot_data = pd.DataFrame({
        'Recall': list(df_paired['v1_recall']) + list(df_paired['latest_recall']),
        'Version': ['v1'] * len(df_paired) + ['latest'] * len(df_paired)
    })
    
    parts = ax.violinplot(
        [df_paired['v1_recall'], df_paired['latest_recall']],
        positions=[0, 1],
        showmeans=True,
        showmedians=True,
        widths=0.7
    )
    
    # Color the violins
    colors = ['#3498db', '#2ecc71']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['v1', 'latest'])
    ax.set_ylabel('Flaw Detection Recall')
    ax.set_title('Distribution Comparison: v1 vs Latest', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add significance
    sig_text = '***' if results['p_value'] < 0.001 else ('**' if results['p_value'] < 0.01 else ('*' if results['p_value'] < 0.05 else 'ns'))
    ax.text(0.5, 0.95, sig_text, ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plot_path = output_dir / "flaw_detection_violin.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze flaw detection results with paired t-tests"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to flaw_detection_detailed.json from evaluate_flaw_detection.py"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./flaw_detection_analysis/",
        help="Output directory for analysis results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Flaw Detection Analysis - Paired t-Test")
    print("="*80)
    print(f"\nResults file: {args.results_file}")
    print(f"Output directory: {output_dir}")
    
    # Load results
    print("\nLoading flaw detection results...")
    results_data = load_results(Path(args.results_file))
    print(f"Loaded {len(results_data)} evaluation results")
    
    # Convert to DataFrame
    df_summary = pd.DataFrame([
        {
            'paper_id': r['paper_id'],
            'title': r['title'],
            'version': r['version'],
            'run_id': r['run_id'],
            'model_type': r['model_type'],
            'num_flaws': r['num_flaws'],
            'num_detected': r['num_detected'],
            'recall': r['recall']
        }
        for r in results_data
    ])
    
    print(f"\nUnique papers: {df_summary['paper_id'].nunique()}")
    print(f"Versions: {df_summary['version'].unique()}")
    print(f"Model types: {df_summary['model_type'].unique()}")
    
    # Check if we have both versions
    versions_present = set(df_summary['version'].unique())
    if 'v1' not in versions_present or 'latest' not in versions_present:
        print("\n❌ Error: Need both v1 and latest versions for paired t-test!")
        print(f"   Found versions: {versions_present}")
        return
    
    # Aggregate across runs (take mean recall per paper-version)
    df_agg = df_summary.groupby(['paper_id', 'version']).agg({
        'recall': 'mean',
        'num_flaws': 'first',
        'num_detected': 'mean'
    }).reset_index()
    
    # Pivot for paired comparison
    df_v1 = df_agg[df_agg['version'] == 'v1'][['paper_id', 'recall']].copy()
    df_v1.columns = ['paper_id', 'v1_recall']
    
    df_latest = df_agg[df_agg['version'] == 'latest'][['paper_id', 'recall']].copy()
    df_latest.columns = ['paper_id', 'latest_recall']
    
    df_paired = df_v1.merge(df_latest, on='paper_id', how='inner')
    
    print(f"\nPaired samples: {len(df_paired)}")
    
    if len(df_paired) == 0:
        print("\n❌ Error: No paired samples found!")
        print("   Make sure the same papers have reviews for both v1 and latest versions.")
        return
    
    # Perform paired t-test
    print("\n" + "="*80)
    print("Performing Paired t-Test...")
    print("="*80)
    
    # Add run_id column for compatibility (using 0 since we aggregated)
    df_paired['run_id'] = 0
    
    results = perform_paired_ttest(df_paired)
    
    # Print results
    print(f"\n{'='*80}")
    print("RESULTS")
    print("="*80)
    
    print(f"\nPaired samples: {results['n_pairs']}")
    print(f"Unique papers: {results['n_papers']}")
    
    print(f"\nv1 Recall:")
    print(f"  Mean: {results['v1_mean']:.3f}")
    print(f"  Std:  {results['v1_std']:.3f}")
    
    print(f"\nLatest Recall:")
    print(f"  Mean: {results['latest_mean']:.3f}")
    print(f"  Std:  {results['latest_std']:.3f}")
    
    print(f"\nDifference (Latest - v1):")
    print(f"  Mean: {results['mean_difference']:.3f}")
    print(f"  Std:  {results['std_difference']:.3f}")
    print(f"  95% CI: [{results['ci_95_lower']:.3f}, {results['ci_95_upper']:.3f}]")
    
    print(f"\nStatistical Test:")
    print(f"  t-statistic: {results['t_statistic']:.3f}")
    print(f"  p-value: {results['p_value']:.4f}")
    print(f"  Cohen's d: {results['cohens_d']:.3f}")
    
    sig_stars = '***' if results['p_value'] < 0.001 else ('**' if results['p_value'] < 0.01 else ('*' if results['p_value'] < 0.05 else ''))
    print(f"  Significant at α=0.05: {'Yes' + sig_stars if results['significant_at_0.05'] else 'No'}")
    print(f"  Significant at α=0.01: {'Yes' + sig_stars if results['significant_at_0.01'] else 'No'}")
    
    print(f"\nInterpretation:")
    print(f"  {results['interpretation']}")
    
    # Save results
    print("\n" + "="*80)
    print("Saving results...")
    print("="*80)
    
    # Save statistical results
    results_json_path = output_dir / "flaw_detection_ttest_results.json"
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved t-test results to: {results_json_path}")
    
    # Save summary CSV
    summary_csv_path = output_dir / "flaw_detection_comparison_summary.csv"
    summary_data = {
        'metric': ['flaw_detection_recall'],
        'n_pairs': [results['n_pairs']],
        'v1_mean': [results['v1_mean']],
        'v1_std': [results['v1_std']],
        'latest_mean': [results['latest_mean']],
        'latest_std': [results['latest_std']],
        'mean_difference': [results['mean_difference']],
        't_statistic': [results['t_statistic']],
        'p_value': [results['p_value']],
        'cohens_d': [results['cohens_d']],
        'significant': ['***' if results['p_value'] < 0.001 else ('**' if results['p_value'] < 0.01 else ('*' if results['p_value'] < 0.05 else 'ns'))]
    }
    pd.DataFrame(summary_data).to_csv(summary_csv_path, index=False)
    print(f"✅ Saved summary to: {summary_csv_path}")
    
    # Save paired data
    paired_csv_path = output_dir / "flaw_detection_paired_data.csv"
    df_paired.to_csv(paired_csv_path, index=False)
    print(f"✅ Saved paired data to: {paired_csv_path}")
    
    # Generate plots
    print("\n" + "="*80)
    print("Generating plots...")
    print("="*80)
    
    try:
        create_comparison_plots(df_summary, results, output_dir)
        print("\n✅ Plots generated successfully!")
    except Exception as e:
        print(f"\n⚠️  Warning: Failed to generate some plots: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    
    # Final interpretation
    if results['significant_at_0.05']:
        direction = "improved" if results['mean_difference'] > 0 else "decreased"
        print(f"\n✅ CONCLUSION: Paper revisions significantly {direction} the AI's ability")
        print(f"   to detect consensus flaws (p={results['p_value']:.4f}).")
    else:
        print(f"\n⚠️  CONCLUSION: No significant difference in flaw detection between")
        print(f"   v1 and latest versions (p={results['p_value']:.4f}).")

if __name__ == "__main__":
    main()

