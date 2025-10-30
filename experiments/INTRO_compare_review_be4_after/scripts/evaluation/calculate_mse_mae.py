#!/usr/bin/env python3
"""
Calculate MSE and MAE between AI-generated and Human Review Scores

This script compares AI-generated review scores with human review scores from OpenReview
and calculates Mean Squared Error (MSE) and Mean Absolute Error (MAE) for each metric.

Usage:
    python calculate_mse_mae.py --csv_file ./data/ICLR2024_pairs/filtered_pairs_with_human_scores.csv \
                                 --reviews_dir ./reviews_vllm_Llama3-1_70B_3_runs/
"""

import argparse
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def extract_numerical_value(score_str) -> Optional[float]:
    """
    Extract numerical value from score string.
    
    Examples:
        "3 good" -> 3.0
        "2 fair" -> 2.0
        "6: marginally above the acceptance threshold" -> 6.0
        "3.5"生肖 -> 3.5
    
    Args:
        score_str: String or number containing numerical score
        
    Returns:
        Extracted numerical value or None if not found
    """
    if score_str is None:
        return None
    
    # If already a number, return it
    if isinstance(score_str, (int, float)):
        return float(score_str)
    
    if not isinstance(score_str, str):
        return None
    
    # Try to extract number from various formats
    # Pattern 1: "3 good", "2 fair"
    match = re.match(r'^(\d+(?:\.\d+)?)', str(score_str).strip())
    if match:
        return float(match.group(1))
    
    return None

def load_ai_reviews(reviews_dir: Path, paper_id: str, version: str, run_id: int = None) -> List[Dict]:
    """
    Load AI-generated reviews for a paper.
    
    Args:
        reviews_dir: Directory containing review subdirectories
        paper_id: Paper ID
        version: 'v1' or 'latest'
        run_id: Specific run ID (None = load all runs)
        
    Returns:
        List of review dicts
    """
    paper_dir = reviews_dir / paper_id
    
    if not paper_dir.exists():
        return []
    
    reviews = []
    
    if run_id is not None:
        # Load specific run
        review_file = paper_dir / f"{version}_review_run{run_id}.json"
        if review_file.exists():
            try:
                with open(review_file) as f:
                    review_data = json.load(f)
                    if review_data.get('success', False):
                        reviews.append(review_data)
            except Exception as e:
                pass
    else:
        # Load all runs for this version
        for review_file in paper_dir.glob(f"{version}_review_run*.json"):
            try:
                with open(review_file) as f:
                    review_data = json.load(f)
                    if review_data.get('success', False):
                        reviews.append(review_data)
            except Exception as e:
                pass
    
    return reviews

def extract_ai_scores(review_data: Dict) -> Dict:
    """
    Extract numerical scores from AI review JSON.
    
    Handles different formats (SEA-E, CycleReviewer, GenericStructured, Anthropic)
    """
    scores = {
        'soundness': None,
        'presentation': None,
        'contribution': None,
        'rating': None
    }
    
    model_type = review_data.get('model_type', 'default')
    
    if model_type == 'CycleReviewer':
        # Average scores across reviewers
        reviewers = review_data.get('reviewers', [])
        if reviewers:
            for metric in ['soundness', 'presentation', 'contribution', 'rating']:
                values = [r.get(metric) for r in reviewers if r.get(metric) is not None]
                if values:
                    scores[metric] = np.mean(values)
    else:
        # Direct extraction for other formats
        for metric in ['soundness', 'presentation', 'contribution', 'rating']:
            if metric in review_data:
                value = review_data[metric]
                # Handle string parsing if needed
                if isinstance(value, str):
                    try:
                        # Try to extract number from strings like "3 good" or "8: accept"
                        scores[metric] = float(value.split()[0].split(':')[0])
                    except:
                        pass
                elif isinstance(value, (int, float)):
                    scores[metric] = float(value)
    
    return scores

def calculate_errors(human_scores: pd.DataFrame, ai_scores: pd.DataFrame) -> Dict:
    """
    Calculate MSE and MAE between human and AI scores.
    
    Args:
        human_scores: DataFrame with columns [paper_id, version, run_id, metric, human_score]
        ai_scores: DataFrame with columns [paper_id, version, run_id, metric, ai_score]
        
    Returns:
        Dict with MSE and MAE per metric
    """
    # Merge human and AI scores
    merged = human_scores.merge(
        ai_scores,
        on=['paper_id', 'version', 'run_id', 'metric'],
        how='inner'
    )
    
    if len(merged) == 0:
        return {}
    
    results = {}
    
    for metric in merged['metric'].unique():
        metric_data = merged[merged['metric'] == metric]
        
        human_vals = metric_data['human_score'].values
        ai_vals = metric_data['ai_score'].values
        
        # Calculate errors
        mse = np.mean((human_vals - ai_vals) ** 2)
        mae = np.mean(np.abs(human_vals - ai_vals))
        rmse = np.sqrt(mse)
        
        # Pearson correlation
        correlation = np.corrcoef(human_vals, ai_vals)[0, 1] if len(human_vals) > 1 else 0.0
        
        results[metric] = {
            'n_pairs': len(metric_data),
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'correlation': float(correlation),
            'human_mean': float(np.mean(human_vals)),
            'human_std': float(np.std(human_vals)),
            'ai_mean': float(np.mean(ai_vals)),
            'ai_std': float(np.std(ai_vals))
        }
    
    return results

def create_comparison_plots(merged_df: pd.DataFrame, output_dir: Path, model_type: str):
    """
    Create visualization plots comparing AI vs Human scores.
    """
    metrics = merged_df['metric'].unique()
    
    # 1. Scatter plots: AI vs Human scores
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{model_type}: AI vs Human Review Scores', fontsize=14, fontweight='bold')
    
    metric_labels = {
        'soundness': 'Soundness',
        'presentation': 'Presentation',
        'contribution': 'Contribution',
        'rating': 'Overall Rating'
    }
    
    for idx, metric in enumerate(['soundness', 'presentation', 'contribution', 'rating']):
        ax = axes[idx // 2, idx % 2]
        
        if metric not in metrics:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(metric_labels[metric])
            continue
        
        metric_data = merged_df[merged_df['metric'] == metric]
        
        human_scores = metric_data['human_score'].values
        ai_scores = metric_data['ai_score'].values
        
        # Scatter plot
        ax.scatter(human_scores, ai_scores, alpha=0.5, s=30, color='#3498db')
        
        # Perfect agreement line
        min_val = min(human_scores.min(), ai_scores.min())
        max_val = max(human_scores.max(), ai_scores.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect agreement')
        
        # Regression line
        if len(human_scores) > 1:
            z = np.polyfit(human_scores, ai_scores, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min_val, max_val, 100)
            ax.plot(x_line, p(x_line), 'g-', linewidth=2, alpha=0.7, label='Regression')
        
        # Correlation
        if len(human_scores) > 1:
            corr = np.corrcoef(human_scores, ai_scores)[0, 1]
            mae = np.mean(np.abs(human_scores - ai_scores))
            rmse = np.sqrt(np.mean((human_scores - ai_scores) ** 2))
            
            ax.text(0.05, 0.95, f'r = {corr:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Human Score')
        ax.set_ylabel('AI Score')
        ax.set_title(metric_labels[metric])
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / f"{model_type}_ai_vs_human_scatter.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")
    
    # 2. Error distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{model_type}: Distribution of Prediction Errors (AI - Human)', 
                fontsize=14, fontweight='bold')
    
    for idx, metric in enumerate(['soundness', 'presentation', 'contribution', 'rating']):
        ax = axes[idx // 2, idx % 2]
        
        if metric not in metrics:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(metric_labels[metric])
            continue
        
        metric_data = merged_df[merged_df['metric'] == metric]
        
        errors = metric_data['ai_score'].values - metric_data['human_score'].values
        
        # Histogram + KDE
        ax.hist(errors, bins=20, alpha=0.6, color='#3498db', edgecolor='black', density=True)
        
        if len(errors) > 1:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(errors)
            x_range = np.linspace(errors.min(), errors.max(), 100)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        # Zero line
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Zero error')
        
        # Mean error
        mean_error = errors.mean()
        ax.axvline(x=mean_error, color='red', linestyle='-', linewidth=2, 
                  label=f'Mean = {mean_error:.3f}')
        
        # Stats box
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        
        stats_text = f"MAE = {mae:.3f}\nRMSE = {rmse:.3f}\nn = {len(errors)}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.5), fontsize=9)
        
        ax.set_xlabel('Prediction Error (AI - Human)')
        ax.set_ylabel('Density')
        ax.set_title(metric_labels[metric])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = output_dir / f"{model_type}_error_distributions.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")
    
    # 3. Bar plot: MAE comparison across metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_list = []
    mae_list = []
    rmse_list = []
    
    for metric in ['soundness', 'presentation', 'contribution', 'rating']:
        if metric in metrics:
            metric_data = merged_df[merged_df['metric'] == metric]
            errors = metric_data['ai_score'].values - metric_data['human_score'].values
            
            metrics_list.append(metric_labels[metric])
            mae_list.append(np.mean(np.abs(errors)))
            rmse_list.append(np.sqrt(np.mean(errors ** 2)))
    
    x = np.arange(len(metrics_list))
    width = 0.35
    
    ax.bar(x - width/2, mae_list, width, label='MAE', color='#3498db')
    ax.bar(x + width/2, rmse_list, width, label='RMSE', color='#e74c3c')
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Error')
    ax.set_title(f'{model_type}: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_list)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = output_dir / f"{model_type}_mae_rmse_comparison.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Calculate MSE and MAE between AI and human review scores"
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Path to CSV file with human scores (output from fetch_human_scores.py)"
    )
    parser.add_argument(
        "--reviews_dir",
        type=str,
        required=True,
        help="Directory containing AI-generated reviews"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ai_vs_human_evaluation/",
        help="Output directory for results"
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=['v1', 'latest', 'both'],
        default='latest',
        help="Which paper version to evaluate"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read CSV with human scores
    print("="*80)
    print("AI vs Human Review Score Evaluation")
    print("="*80)
    print(f"\nReading human scores from: {args.csv_file}")
    df_papers = pd.read_csv(args.csv_file)
    
    # Filter papers with human scores
    df_papers_with_human = df_papers[df_papers['num_reviews'] > 0].copy()
    print(f"Papers with human reviews: {len(df_papers_with_human)}/{len(df_papers)}")
    
    reviews_dir = Path(args.reviews_dir)
    print(f"Reading AI reviews from: {reviews_dir}")
    
    # Determine which versions to evaluate
    versions = ['v1', 'latest'] if args.version == 'both' else [args.version]
    
    # Collect all human and AI scores
    print("\n" + "="*80)
    print("Collecting scores...")
    print("="*80)
    
    human_scores_list = []
    ai_scores_list = []
    
    for idx, row in tqdm(df_papers_with_human.iterrows(), total=len(df_papers_with_human), 
                         desc="Processing papers"):
        paper_id = row['paperid']
        
        for version in versions:
            # Get AI reviews
            ai_reviews = load_ai_reviews(reviews_dir, paper_id, version, run_id=None)
            
            if not ai_reviews:
                continue
            
            for run_id, ai_review in enumerate(ai_reviews):
                ai_scores = extract_ai_scores(ai_review)
                
                # Add AI scores
                for metric, ai_score in ai_scores.items():
                    if ai_score is not None:
                        human_score = row.get(f'human_{metric}_mean')
                        
                        if pd.notna(human_score):
                            human_scores_list.append({
                                'paper_id': paper_id,
                                'version': version,
                                'run_id': run_id,
                                'metric': metric,
                                'human_score': float(human_score)
                            })
                            
                            ai_scores_list.append({
                                'paper_id': paper_id,
                                'version': version,
                                'run_id': run_id,
                                'metric': metric,
                                'ai_score': float(ai_score)
                            })
    
    df_human = pd.DataFrame(human_scores_list)
    df_ai = pd.DataFrame(ai_scores_list)
    
    print(f"\nCollected {len(df_human)} human scores")
    print(f"Collected {len(df_ai)} AI scores")
    
    # Merge and calculate errors
    print("\n" + "="*80)
    print("Calculating MSE and MAE...")
    print("="*80)
    
    results = calculate_errors(df_human, df_ai)
    
    if not results:
        print("❌ No matching scores found between human and AI reviews!")
        return
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    results_rows = []
    
    for metric, stats in results.items():
        print(f"\n{metric.upper()}:")
        print(f"  Pairs: {stats['n_pairs']}")
        print(f"  Human: {stats['human_mean']:.3f} ± {stats['human_std']:.3f}")
        print(f"  AI:    {stats['ai_mean']:.3f} ± {stats['ai_std']:.3f}")
        print(f"  MAE:   {stats['mae']:.3f}")
        print(f"  RMSE:  {stats['rmse']:.3f}")
        print(f"  MSE:   {stats['mse']:.3f}")
        print(f"  Correlation: {stats['correlation']:.3f}")
        
        results_rows.append({
            'metric': metric,
            'n_pairs': stats['n_pairs'],
            'human_mean': stats['human_mean'],
            'human_std': stats['human_std'],
            'ai_mean': stats['ai_mean'],
            'ai_std': stats['ai_std'],
            'mae': stats['mae'],
            'rmse': stats['rmse'],
            'mse': stats['mse'],
            'correlation': stats['correlation']
        })
    
    # Save results
    df_results = pd.DataFrame(results_rows)
    results_csv_path = output_dir / "ai_vs_human_results.csv"
    df_results.to_csv(results_csv_path, index=False)
    print(f"\n✅ Saved summary results to: {results_csv_path}")
    
    # Save detailed comparison
    merged_df = df_human.merge(df_ai, on=['paper_id', 'version', 'run_id', 'metric'], how='inner')
    detailed_csv_path = output_dir / "ai_vs_human_detailed.csv"
    merged_df.to_csv(detailed_csv_path, index=False)
    print(f"✅ Saved detailed comparison to: {detailed_csv_path}")
    
    # Save full results JSON
    results_json_path = output_dir / "ai_vs_human_results.json"
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved full results to: {results_json_path}")
    
    # Generate plots
    print("\n" + "="*80)
    print("Generating visualization plots...")
    print("="*80)
    
    model_type = reviews_dir.name.replace('reviews_', '').replace('_', ' ').title()
    
    try:
        create_comparison_plots(merged_df, output_dir, model_type)
        print("\n✅ Plots generated successfully!")
    except Exception as e:
        print(f"\n⚠️  Warning: Failed to generate some plots: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)

if __name__ == "__main__":
    main()

