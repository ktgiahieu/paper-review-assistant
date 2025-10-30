#!/usr/bin/env python3
"""
Retry Failed Reviews Script

Identifies review files with errors and re-runs only those reviews.
Useful for recovering from parsing failures, API errors, or other issues.

Usage:
    python retry_failed_reviews.py --reviews_dir ./reviews_output --csv_file ./data/filtered_pairs.csv [other review args]
"""

import json
import argparse
from pathlib import Path
import subprocess
import sys

def find_failed_reviews(reviews_dir: Path) -> list:
    """
    Find all review files that failed.
    
    Returns list of tuples: (paper_id, version, run_id, reason)
    """
    failed = []
    
    paper_dirs = [d for d in reviews_dir.iterdir() if d.is_dir()]
    
    for paper_dir in paper_dirs:
        paper_id = paper_dir.name
        
        # Find all review JSON files
        review_files = list(paper_dir.glob("*_review_run*.json"))
        
        for review_file in review_files:
            try:
                with open(review_file, 'r', encoding='utf-8') as f:
                    review_data = json.load(f)
                
                # Check if review was successful
                if not review_data.get('success', False):
                    # Extract version and run_id from filename
                    # Format: v1_review_run0.json or latest_review_run0.json
                    filename = review_file.stem  # Remove .json
                    parts = filename.split('_')
                    
                    if 'latest' in filename:
                        version = 'latest'
                        run_id = int(parts[-1].replace('run', ''))
                    else:  # v1
                        version = 'v1'
                        run_id = int(parts[-1].replace('run', ''))
                    
                    error_msg = review_data.get('error', 'Unknown error')
                    
                    failed.append({
                        'paper_id': paper_id,
                        'version': version,
                        'run_id': run_id,
                        'file': str(review_file),
                        'error': error_msg
                    })
            
            except Exception as e:
                print(f"Error reading {review_file}: {e}")
    
    return failed

def find_missing_reviews(reviews_dir: Path, expected_papers: list, expected_versions: list, num_runs: int) -> list:
    """
    Find reviews that should exist but don't.
    
    Args:
        expected_papers: List of paper IDs that should have reviews
        expected_versions: List of versions to check ("v1", "latest", or both)
        num_runs: Number of runs expected per version
    """
    missing = []
    
    for paper_id in expected_papers:
        paper_dir = reviews_dir / paper_id
        
        if not paper_dir.exists():
            # Entire paper directory missing
            for version in expected_versions:
                for run_id in range(num_runs):
                    missing.append({
                        'paper_id': paper_id,
                        'version': version,
                        'run_id': run_id,
                        'reason': 'Paper directory does not exist'
                    })
            continue
        
        # Check for each expected review file
        for version in expected_versions:
            for run_id in range(num_runs):
                review_file = paper_dir / f"{version}_review_run{run_id}.json"
                
                if not review_file.exists():
                    missing.append({
                        'paper_id': paper_id,
                        'version': version,
                        'run_id': run_id,
                        'reason': 'Review file does not exist'
                    })
    
    return missing

def create_retry_csv(failed_reviews: list, missing_reviews: list, output_file: Path, original_csv: Path):
    """Create a CSV with only the papers that need to be retried."""
    import pandas as pd
    
    # Get unique paper IDs
    paper_ids = set()
    for item in failed_reviews + missing_reviews:
        paper_ids.add(item['paper_id'])
    
    # Load original CSV
    df_original = pd.read_csv(original_csv)
    
    # Filter to only failed/missing papers
    df_retry = df_original[df_original['paperid'].isin(paper_ids)]
    
    # Save
    df_retry.to_csv(output_file, index=False)
    
    return len(df_retry)

def main():
    parser = argparse.ArgumentParser(
        description="Identify and retry failed review generations"
    )
    
    # Required arguments
    parser.add_argument("--reviews_dir", type=str, required=True,
                       help="Directory containing review outputs")
    parser.add_argument("--csv_file", type=str, required=True,
                       help="Original CSV file with paper pairs")
    
    # Review script arguments (passed through)
    parser.add_argument("--vllm_endpoint", type=str, required=True,
                       help="vLLM server endpoint")
    parser.add_argument("--model_name", type=str, required=True,
                       help="Model name")
    parser.add_argument("--format", type=str, default=None,
                       help="Format override")
    parser.add_argument("--max_figures", type=int, default=5,
                       help="Max figures to include")
    parser.add_argument("--num_runs", type=int, default=1,
                       help="Number of runs per paper")
    parser.add_argument("--max_workers", type=int, default=3,
                       help="Max worker threads")
    parser.add_argument("--version", type=str, default="both",
                       choices=["v1", "latest", "both"],
                       help="Which versions to check")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    # Retry-specific arguments
    parser.add_argument("--check_only", action="store_true",
                       help="Only check for failures, don't retry")
    parser.add_argument("--retry_output", type=str, default=None,
                       help="Where to save retry CSV (default: same as reviews_dir)")
    
    args = parser.parse_args()
    
    reviews_dir = Path(args.reviews_dir)
    csv_file = Path(args.csv_file)
    
    if not reviews_dir.exists():
        print(f"Error: Reviews directory does not exist: {reviews_dir}")
        return 1
    
    if not csv_file.exists():
        print(f"Error: CSV file does not exist: {csv_file}")
        return 1
    
    print("="*80)
    print("Failed Review Detector")
    print("="*80)
    print(f"Reviews directory: {reviews_dir}")
    print(f"CSV file: {csv_file}")
    print()
    
    # Step 1: Find failed reviews
    print("Step 1: Checking for failed reviews...")
    failed = find_failed_reviews(reviews_dir)
    
    if failed:
        print(f"\n❌ Found {len(failed)} failed reviews:")
        for item in failed[:10]:  # Show first 10
            print(f"  - {item['paper_id']} ({item['version']}, run {item['run_id']}): {item['error'][:80]}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    else:
        print("✅ No failed reviews found")
    
    # Step 2: Find missing reviews
    print("\nStep 2: Checking for missing reviews...")
    
    # Load expected papers from CSV
    import pandas as pd
    df = pd.read_csv(csv_file)
    expected_papers = df['paperid'].tolist()
    
    # Determine expected versions
    expected_versions = []
    if args.version in ["v1", "both"]:
        expected_versions.append("v1")
    if args.version in ["latest", "both"]:
        expected_versions.append("latest")
    
    missing = find_missing_reviews(reviews_dir, expected_papers, expected_versions, args.num_runs)
    
    if missing:
        print(f"\n❌ Found {len(missing)} missing reviews:")
        for item in missing[:10]:
            print(f"  - {item['paper_id']} ({item['version']}, run {item['run_id']}): {item['reason']}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    else:
        print("✅ No missing reviews")
    
    # Summary
    total_issues = len(failed) + len(missing)
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Failed reviews: {len(failed)}")
    print(f"Missing reviews: {len(missing)}")
    print(f"Total issues: {total_issues}")
    
    if total_issues == 0:
        print("\n✅ All reviews completed successfully!")
        return 0
    
    # Step 3: Create retry CSV
    unique_papers = set()
    for item in failed + missing:
        unique_papers.add(item['paper_id'])
    
    print(f"\nUnique papers needing retry: {len(unique_papers)}")
    
    if args.check_only:
        print("\n--check_only flag set. Not retrying.")
        return 0
    
    # Create retry CSV
    retry_output = args.retry_output if args.retry_output else str(reviews_dir / "retry_papers.csv")
    retry_csv_path = Path(retry_output)
    
    print(f"\nCreating retry CSV: {retry_csv_path}")
    num_retry_papers = create_retry_csv(failed, missing, retry_csv_path, csv_file)
    print(f"Retry CSV contains {num_retry_papers} papers")
    
    # Step 4: Run review script on retry CSV
    print("\n" + "="*80)
    print("Retrying Failed Reviews")
    print("="*80)
    
    # Build command
    cmd = [
        sys.executable,
        "review_paper_pairs_vllm.py",
        "--csv_file", str(retry_csv_path),
        "--output_dir", str(reviews_dir),
        "--vllm_endpoint", args.vllm_endpoint,
        "--model_name", args.model_name,
        "--max_figures", str(args.max_figures),
        "--num_runs", str(args.num_runs),
        "--max_workers", str(args.max_workers),
        "--version", args.version,
    ]
    
    if args.format:
        cmd.extend(["--format", args.format])
    
    if args.verbose:
        cmd.append("--verbose")
    
    # Don't skip existing - we want to overwrite the failed ones
    # But add a flag to indicate this is a retry
    
    print(f"\nRunning command:")
    print(" ".join(cmd))
    print()
    
    # Run the command
    result = subprocess.run(cmd, cwd=reviews_dir.parent)
    
    if result.returncode == 0:
        print("\n✅ Retry completed successfully!")
    else:
        print(f"\n❌ Retry failed with exit code {result.returncode}")
        return result.returncode
    
    # Step 5: Check again
    print("\n" + "="*80)
    print("Verification")
    print("="*80)
    print("Checking for remaining failures...")
    
    failed_after = find_failed_reviews(reviews_dir)
    missing_after = find_missing_reviews(reviews_dir, expected_papers, expected_versions, args.num_runs)
    
    total_after = len(failed_after) + len(missing_after)
    
    print(f"Failed reviews: {len(failed_after)}")
    print(f"Missing reviews: {len(missing_after)}")
    print(f"Total issues: {total_after}")
    
    if total_after == 0:
        print("\n✅ All reviews completed successfully!")
    else:
        print(f"\n⚠️  Still have {total_after} issues remaining.")
        print("You may need to retry again or investigate individual failures.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

