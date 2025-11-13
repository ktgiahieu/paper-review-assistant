#!/usr/bin/env python3
"""
Run all evaluation scripts automatically.

This script runs:
1. evaluate_numerical_scores.py - v1 vs latest comparison
2. evaluate_good_bad_plant_effect.py - good/bad manipulation effect analysis (author/affiliation, abstract, etc.)
3. calculate_mse_mae.py - AI vs human score comparison (if human scores CSV provided)

Usage:
    python run_all_evaluations.py --reviews_dir ../sampled_data/reviews_gemini_2-0_flash_lite/ICLR2024
    
    # With human scores comparison
    python run_all_evaluations.py --reviews_dir ../sampled_data/reviews_gemini_2-0_flash_lite/ICLR2024 \
                                  --human_scores_csv ../sampled_data/ICLR2024/filtered_pairs_with_human_scores.csv
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional
import os


def run_command(cmd: list, description: str, verbose: bool = False) -> bool:
    """
    Run a command and return success status.
    
    Args:
        cmd: Command as list of arguments
        description: Description of what this command does
        verbose: Print command before running
        
    Returns:
        True if successful, False otherwise
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=not verbose,  # Show output if verbose
            text=True
        )
        
        if not verbose and result.stdout:
            print(result.stdout)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {description}")
        print(f"Command: {' '.join(cmd)}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error running {description}: {e}")
        return False


def run_all_evaluations(
    reviews_dir: Path,
    output_base_dir: Optional[Path] = None,
    human_scores_csv: Optional[Path] = None,
    dataset: str = "ICLR2024",
    num_runs: int = 3,
    verbose: bool = False,
    skip_numerical_scores: bool = False,
    skip_author_affiliation: bool = False,
    skip_abstract: bool = False,
    skip_human_comparison: bool = False
):
    """
    Run all evaluation scripts.
    
    Args:
        reviews_dir: Directory containing review folders
        output_base_dir: Base directory for all outputs (default: reviews_dir.parent / "evaluation_results")
        human_scores_csv: Path to CSV with human scores (for calculate_mse_mae.py)
        dataset: Dataset name (for organizing outputs)
        num_runs: Number of runs expected (for evaluate_numerical_scores.py)
        verbose: Verbose output
        skip_numerical_scores: Skip v1 vs latest evaluation
        skip_author_affiliation: Skip author/affiliation effect evaluation
        skip_human_comparison: Skip AI vs human comparison
    """
    reviews_dir = Path(reviews_dir)
    
    if not reviews_dir.exists():
        print(f"❌ Error: Reviews directory does not exist: {reviews_dir}")
        return False
    
    # Set default output directory
    if output_base_dir is None:
        output_base_dir = reviews_dir.parent / "evaluation_results"
    else:
        output_base_dir = Path(output_base_dir)
    
    # Create base output directory
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create all subdirectories upfront (even if evaluations are skipped)
    # This ensures consistent folder structure
    output_dirs = {
        'v1_latest': output_base_dir / "v1_latest",
        'author_affiliation': output_base_dir / "author_affiliation",
        'abstract': output_base_dir / "abstract",
        'v1_human': output_base_dir / "v1_human"
    }
    
    for dir_name, dir_path in output_dirs.items():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get script directory (where this script is located)
    script_dir = Path(__file__).parent
    
    # Track success/failure
    results = {
        'numerical_scores': None,
        'author_affiliation': None,
        'abstract': None,
        'human_comparison': None
    }
    
    print("="*80)
    print("Running All Evaluations")
    print("="*80)
    print(f"Reviews directory: {reviews_dir}")
    print(f"Output base directory: {output_base_dir}")
    print(f"Dataset: {dataset}")
    print(f"\nOutput folders created:")
    for dir_name, dir_path in output_dirs.items():
        print(f"  - {dir_name}/ → {dir_path}")
    print()
    
    # 1. Run evaluate_numerical_scores.py (v1 vs latest)
    if not skip_numerical_scores:
        print("\n" + "="*80)
        print("1. Running v1 vs Latest Evaluation")
        print("="*80)
        
        output_dir = output_dirs['v1_latest']
        
        cmd = [
            sys.executable,
            str(script_dir / "evaluate_numerical_scores.py"),
            "--reviews_dir", str(reviews_dir),
            "--output_dir", str(output_dir),
            "--output_prefix", dataset,
            "--num_runs", str(num_runs)
        ]
        
        success = run_command(cmd, "v1 vs Latest Evaluation", verbose)
        results['numerical_scores'] = success
        
        if success:
            print(f"✅ v1 vs Latest evaluation completed")
            print(f"   Results saved to: {output_dir}")
        else:
            print(f"❌ v1 vs Latest evaluation failed")
    else:
        print("\n⏭️  Skipping v1 vs Latest evaluation")
        results['numerical_scores'] = None
    
    # 2. Run evaluate_good_bad_plant_effect.py for author/affiliation
    if not skip_author_affiliation:
        print("\n" + "="*80)
        print("2. Running Author/Affiliation Effect Evaluation")
        print("="*80)
        
        output_dir = output_dirs['author_affiliation']
        
        # Check which folders exist
        baseline_folder = "latest"
        good_folder = "authors_affiliation_good"
        bad_folder = "authors_affiliation_bad"
        
        baseline_exists = (reviews_dir / baseline_folder).exists()
        good_exists = (reviews_dir / good_folder).exists()
        bad_exists = (reviews_dir / bad_folder).exists()
        
        if not baseline_exists or (not good_exists and not bad_exists):
            print(f"❌ No author/affiliation folders found in {reviews_dir}")
            print(f"   Expected: {baseline_folder}, {good_folder}, {bad_folder}")
            results['author_affiliation'] = False
        else:
            cmd = [
                sys.executable,
                str(script_dir / "evaluate_good_bad_plant_effect.py"),
                "--reviews_dir", str(reviews_dir),
                "--output_dir", str(output_dir),
                "--pattern", "authors_affiliation",
                "--baseline", baseline_folder
            ]
            
            success = run_command(cmd, "Author/Affiliation Effect Evaluation", verbose)
            results['author_affiliation'] = success
            
            if success:
                print(f"✅ Author/Affiliation effect evaluation completed")
                print(f"   Results saved to: {output_dir}")
            else:
                print(f"❌ Author/Affiliation effect evaluation failed")
    else:
        print("\n⏭️  Skipping Author/Affiliation effect evaluation")
        results['author_affiliation'] = None
    
    # 2b. Run evaluate_good_bad_plant_effect.py for abstract manipulation
    if not skip_author_affiliation and not skip_abstract:
        print("\n" + "="*80)
        print("2b. Running Abstract Manipulation Effect Evaluation")
        print("="*80)
        
        output_dir = output_dirs['abstract']
        
        # Check which folders exist
        baseline_folder = "latest"
        good_folder = "abstract_good"
        bad_folder = "abstract_bad"
        
        baseline_exists = (reviews_dir / baseline_folder).exists()
        good_exists = (reviews_dir / good_folder).exists()
        bad_exists = (reviews_dir / bad_folder).exists()
        
        if not baseline_exists or (not good_exists and not bad_exists):
            print(f"⚠️  No abstract manipulation folders found in {reviews_dir}")
            print(f"   Expected: {baseline_folder}, {good_folder}, {bad_folder}")
            print(f"   Skipping abstract manipulation evaluation")
            results['abstract'] = None
        else:
            cmd = [
                sys.executable,
                str(script_dir / "evaluate_good_bad_plant_effect.py"),
                "--reviews_dir", str(reviews_dir),
                "--output_dir", str(output_dir),
                "--pattern", "abstract",
                "--baseline", baseline_folder
            ]
            
            success = run_command(cmd, "Abstract Manipulation Effect Evaluation", verbose)
            results['abstract'] = success
            
            if success:
                print(f"✅ Abstract manipulation effect evaluation completed")
                print(f"   Results saved to: {output_dir}")
            else:
                print(f"❌ Abstract manipulation effect evaluation failed")
    else:
        print("\n⏭️  Skipping Abstract manipulation effect evaluation")
        results['abstract'] = None
    
    # 3. Run calculate_mse_mae.py (if human scores CSV provided)
    if not skip_human_comparison and human_scores_csv:
        print("\n" + "="*80)
        print("3. Running AI vs Human Score Comparison")
        print("="*80)
        
        human_scores_csv = Path(human_scores_csv)
        if not human_scores_csv.exists():
            print(f"❌ Error: Human scores CSV not found: {human_scores_csv}")
            results['human_comparison'] = False
        else:
            output_dir = output_dirs['v1_human']
            
            cmd = [
                sys.executable,
                str(script_dir / "calculate_mse_mae.py"),
                "--csv_file", str(human_scores_csv),
                "--reviews_dir", str(reviews_dir),
                "--output_dir", str(output_dir),
                "--version", "v1"  # Compare with human scores for v1 version
            ]
            
            success = run_command(cmd, "AI vs Human Score Comparison", verbose)
            results['human_comparison'] = success
            
            if success:
                print(f"✅ AI vs Human comparison completed")
                print(f"   Results saved to: {output_dir}")
            else:
                print(f"❌ AI vs Human comparison failed")
    elif not skip_human_comparison and not human_scores_csv:
        print("\n⏭️  Skipping AI vs Human comparison (no human scores CSV provided)")
        print("   Use --human_scores_csv to enable this evaluation")
        results['human_comparison'] = None
    else:
        print("\n⏭️  Skipping AI vs Human comparison")
        results['human_comparison'] = None
    
    # Summary
    print("\n" + "="*80)
    print("Evaluation Summary")
    print("="*80)
    
    for evaluation_name, success in results.items():
        if success is True:
            status = "✅ Completed"
        elif success is False:
            status = "❌ Failed"
        else:
            status = "⏭️  Skipped"
        
        print(f"  {evaluation_name.replace('_', ' ').title()}: {status}")
    
    # Count successes
    completed = sum(1 for s in results.values() if s is True)
    failed = sum(1 for s in results.values() if s is False)
    skipped = sum(1 for s in results.values() if s is None)
    
    print(f"\nCompleted: {completed}, Failed: {failed}, Skipped: {skipped}")
    print(f"\nAll results saved to: {output_base_dir}")
    print("="*80)
    
    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run all evaluation scripts automatically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all evaluations (skip human comparison if no CSV)
  python run_all_evaluations.py --reviews_dir ../sampled_data/reviews_gemini_2-0_flash_lite/ICLR2024
  
  # With human scores comparison
  python run_all_evaluations.py \\
      --reviews_dir ../sampled_data/reviews_gemini_2-0_flash_lite/ICLR2024 \\
      --human_scores_csv ../sampled_data/ICLR2024/filtered_pairs_with_human_scores.csv
  
  # Custom output directory
  python run_all_evaluations.py \\
      --reviews_dir ../sampled_data/reviews_gemini_2-0_flash_lite/ICLR2024 \\
      --output_dir ../sampled_data/custom_evaluation_results
  
  # Skip specific evaluations
  python run_all_evaluations.py \\
      --reviews_dir ../sampled_data/reviews_gemini_2-0_flash_lite/ICLR2024 \\
      --skip-author-affiliation
        """
    )
    
    parser.add_argument(
        "--reviews_dir",
        type=str,
        required=True,
        help="Directory containing review folders (e.g., reviews_gemini_2-0_flash_lite/ICLR2024)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Base output directory for all evaluation results (default: reviews_dir.parent/evaluation_results)"
    )
    parser.add_argument(
        "--human_scores_csv",
        type=str,
        default=None,
        help="Path to CSV file with human scores (for AI vs human comparison). "
             "If not provided, AI vs human comparison will be skipped."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ICLR2024",
        help="Dataset name (for organizing outputs, default: ICLR2024)"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Number of runs expected per paper (default: 3)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output from each evaluation script"
    )
    parser.add_argument(
        "--skip-numerical-scores",
        action="store_true",
        help="Skip v1 vs latest evaluation"
    )
    parser.add_argument(
        "--skip-author-affiliation",
        action="store_true",
        help="Skip author/affiliation and abstract manipulation effect evaluations"
    )
    parser.add_argument(
        "--skip-abstract",
        action="store_true",
        help="Skip abstract manipulation effect evaluation (author/affiliation will still run if available)"
    )
    parser.add_argument(
        "--skip-human-comparison",
        action="store_true",
        help="Skip AI vs human score comparison"
    )
    
    args = parser.parse_args()
    
    success = run_all_evaluations(
        reviews_dir=Path(args.reviews_dir),
        output_base_dir=Path(args.output_dir) if args.output_dir else None,
        human_scores_csv=Path(args.human_scores_csv) if args.human_scores_csv else None,
        dataset=args.dataset,
        num_runs=args.num_runs,
        verbose=args.verbose,
        skip_numerical_scores=args.skip_numerical_scores,
        skip_author_affiliation=args.skip_author_affiliation,
        skip_abstract=args.skip_abstract,
        skip_human_comparison=args.skip_human_comparison
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

