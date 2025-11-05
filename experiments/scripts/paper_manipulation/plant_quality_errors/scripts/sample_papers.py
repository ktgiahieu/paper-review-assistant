#!/usr/bin/env python3
"""
Script to sample a subset of papers from the data directory.

This script:
1. Samples N papers from the latest/ folder
2. Copies them to sampled_data/latest/
3. Optionally creates a sampled CSV file with the corresponding rows
"""

import os
import random
import argparse
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def sample_papers(
    base_dir: Path,
    output_dir: Path,
    n_samples: int = 25,
    random_seed: int = 42,
    csv_file: Path = None,
    output_csv: Path = None
):
    """
    Sample papers from the latest/ folder and copy them to sampled_data/.
    
    Args:
        base_dir: Base directory containing the latest/ folder
        output_dir: Output directory (will create sampled_data/ inside)
        n_samples: Number of papers to sample
        random_seed: Random seed for reproducibility
        csv_file: Optional CSV file to filter/sample from
        output_csv: Optional output CSV file path
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    latest_dir = base_dir / "latest"
    if not latest_dir.exists():
        print(f"Error: {latest_dir} does not exist")
        return
    
    # Get all paper directories
    paper_dirs = [d for d in latest_dir.iterdir() if d.is_dir()]
    print(f"Found {len(paper_dirs)} papers in {latest_dir}")
    
    # Filter papers if CSV is provided
    if csv_file and csv_file.exists():
        try:
            df = pd.read_csv(csv_file)
            # Get paper IDs from CSV (first column or paperid column)
            first_col = df.columns[0]
            if first_col.lower() in ['paperid', 'paper_id', 'id']:
                paper_ids = set(df[first_col].astype(str).str.strip())
            else:
                # Try to find paperid column
                paperid_col = None
                for col in df.columns:
                    if 'paperid' in col.lower() or 'paper_id' in col.lower():
                        paperid_col = col
                        break
                if paperid_col:
                    paper_ids = set(df[paperid_col].astype(str).str.strip())
                else:
                    paper_ids = set(df.iloc[:, 0].astype(str).str.strip())
            
            # Filter paper directories to only those in CSV
            paper_dirs = [d for d in paper_dirs if d.name.split('_')[0] in paper_ids]
            print(f"Filtered to {len(paper_dirs)} papers from CSV")
        except Exception as e:
            print(f"Warning: Could not read CSV file {csv_file}: {e}")
            print("Proceeding with all papers...")
    
    # Check if we have enough papers
    if len(paper_dirs) < n_samples:
        print(f"Warning: Only {len(paper_dirs)} papers available, sampling all of them")
        n_samples = len(paper_dirs)
    
    # Sample papers
    sampled_dirs = random.sample(paper_dirs, n_samples)
    print(f"Sampled {len(sampled_dirs)} papers")
    
    # Create output directory
    output_latest_dir = output_dir / "latest"
    output_latest_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy sampled papers
    print(f"\nCopying papers to {output_latest_dir}...")
    copied_count = 0
    sampled_paper_ids = []
    
    for paper_dir in tqdm(sampled_dirs, desc="Copying papers"):
        paper_id = paper_dir.name.split('_')[0]
        sampled_paper_ids.append(paper_id)
        
        try:
            # Copy entire directory
            dest_dir = output_latest_dir / paper_dir.name
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(paper_dir, dest_dir)
            copied_count += 1
        except Exception as e:
            print(f"\nError copying {paper_dir.name}: {e}")
    
    print(f"\nSuccessfully copied {copied_count} papers to {output_latest_dir}")
    
    # Create sampled CSV if requested
    if csv_file and csv_file.exists() and output_csv:
        try:
            df = pd.read_csv(csv_file)
            # Find paper ID column
            first_col = df.columns[0]
            if first_col.lower() in ['paperid', 'paper_id', 'id']:
                paperid_col = first_col
            else:
                paperid_col = None
                for col in df.columns:
                    if 'paperid' in col.lower() or 'paper_id' in col.lower():
                        paperid_col = col
                        break
                if not paperid_col:
                    paperid_col = df.columns[0]
            
            # Filter to sampled papers
            sampled_df = df[df[paperid_col].astype(str).str.strip().isin(sampled_paper_ids)]
            sampled_df.to_csv(output_csv, index=False)
            print(f"Created sampled CSV with {len(sampled_df)} rows: {output_csv}")
        except Exception as e:
            print(f"Warning: Could not create sampled CSV: {e}")
    
    return sampled_paper_ids


def main():
    parser = argparse.ArgumentParser(
        description="Sample a subset of papers from the data directory"
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        required=True,
        help='Base directory containing the latest/ folder (e.g., data/ICLR2024)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: ../sampled_data/ICLR2024 relative to script location)'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=25,
        help='Number of papers to sample (default: 25)'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--csv_file',
        type=str,
        default=None,
        help='Optional CSV file to filter papers from (e.g., filtered_pairs.csv)'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default=None,
        help='Optional output CSV file path (default: output_dir/filtered_pairs.csv)'
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    # Set default output directory
    if args.output_dir is None:
        # Create sampled_data outside the scripts folder (relative to script location)
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "sampled_data" / base_dir.name
    else:
        output_dir = Path(args.output_dir)
    
    # Set default CSV output path
    output_csv_path = None
    if args.output_csv:
        output_csv_path = Path(args.output_csv)
    elif args.csv_file:
        output_csv_path = output_dir / Path(args.csv_file).name
    
    csv_file_path = None
    if args.csv_file:
        csv_file_path = Path(args.csv_file)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Sampling {args.n_samples} papers from {base_dir}")
    print(f"Output directory: {output_dir}")
    if csv_file_path:
        print(f"Filtering from CSV: {csv_file_path}")
    if output_csv_path:
        print(f"Output CSV: {output_csv_path}")
    print()
    
    # Sample papers
    sampled_ids = sample_papers(
        base_dir=base_dir,
        output_dir=output_dir,
        n_samples=args.n_samples,
        random_seed=args.random_seed,
        csv_file=csv_file_path,
        output_csv=output_csv_path
    )
    
    print(f"\n{'='*60}")
    print(f"Sampling complete!")
    print(f"  Sampled {len(sampled_ids)} papers")
    print(f"  Output directory: {output_dir}")
    if output_csv_path:
        print(f"  Output CSV: {output_csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

