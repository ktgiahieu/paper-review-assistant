#!/usr/bin/env python3
"""
Utility script to extract matching human scores from a sampled CSV.

This script:
1. Reads paper IDs from a sampled CSV (e.g., sampled_data/ICLR2024/filtered_pairs.csv)
2. Finds matching rows in the full human scores CSV (e.g., data/ICLR2024_pairs/filtered_pairs_with_human_scores.csv)
3. Creates a filtered CSV with only the matching rows

Usage:
    python3 get_matching_human_scores.py \
        --sampled_csv ../../sampled_data/ICLR2024/filtered_pairs.csv \
        --human_scores_csv ../../data/ICLR2024_pairs/filtered_pairs_with_human_scores.csv \
        --output_csv ../../sampled_data/ICLR2024/filtered_pairs_with_human_scores.csv
"""

import argparse
import pandas as pd
from pathlib import Path


def get_matching_human_scores(
    sampled_csv: Path,
    human_scores_csv: Path,
    output_csv: Path,
    verbose: bool = False
) -> dict:
    """
    Extract matching rows from human scores CSV based on sampled paper IDs.
    
    Args:
        sampled_csv: Path to sampled filtered_pairs.csv
        human_scores_csv: Path to full filtered_pairs_with_human_scores.csv
        output_csv: Output path for filtered human scores CSV
        verbose: Enable verbose output
        
    Returns:
        Dictionary with matching statistics
    """
    sampled_csv = Path(sampled_csv)
    human_scores_csv = Path(human_scores_csv)
    output_csv = Path(output_csv)
    
    if not sampled_csv.exists():
        raise ValueError(f"Sampled CSV does not exist: {sampled_csv}")
    
    if not human_scores_csv.exists():
        raise ValueError(f"Human scores CSV does not exist: {human_scores_csv}")
    
    # Read sampled CSV to get paper IDs
    if verbose:
        print(f"Reading sampled CSV: {sampled_csv}")
    try:
        sampled_df = pd.read_csv(sampled_csv)
        # Find paper ID column
        paperid_col = None
        for col in sampled_df.columns:
            if 'paperid' in col.lower() or 'paper_id' in col.lower():
                paperid_col = col
                break
        if not paperid_col:
            paperid_col = sampled_df.columns[0]
        
        sampled_paper_ids = set(sampled_df[paperid_col].astype(str).str.strip())
        if verbose:
            print(f"  Found {len(sampled_paper_ids)} paper IDs")
    except Exception as e:
        raise ValueError(f"Error reading sampled CSV: {e}")
    
    # Read human scores CSV
    if verbose:
        print(f"\nReading human scores CSV: {human_scores_csv}")
    try:
        human_df = pd.read_csv(human_scores_csv)
        # Find paper ID column
        paperid_col_human = None
        for col in human_df.columns:
            if 'paperid' in col.lower() or 'paper_id' in col.lower():
                paperid_col_human = col
                break
        if not paperid_col_human:
            paperid_col_human = human_df.columns[0]
        
        if verbose:
            print(f"  Total rows in human scores CSV: {len(human_df)}")
        
        # Filter to matching papers
        human_df[paperid_col_human] = human_df[paperid_col_human].astype(str).str.strip()
        matched_df = human_df[human_df[paperid_col_human].isin(sampled_paper_ids)]
        
        if verbose:
            print(f"  Found {len(matched_df)} matching rows")
        
        # Save filtered CSV
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        matched_df.to_csv(output_csv, index=False)
        
        if verbose:
            print(f"\nSaved filtered CSV to: {output_csv}")
        
        not_found = sampled_paper_ids - set(matched_df[paperid_col_human])
        
        return {
            'matched': len(matched_df),
            'total_sampled': len(sampled_paper_ids),
            'not_found': list(not_found),
            'total_in_human_csv': len(human_df)
        }
    except Exception as e:
        raise ValueError(f"Error processing human scores CSV: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract matching human scores from a sampled CSV"
    )
    parser.add_argument(
        "--sampled_csv",
        type=str,
        required=True,
        help="Path to sampled CSV file (e.g., sampled_data/ICLR2024/filtered_pairs.csv)"
    )
    parser.add_argument(
        "--human_scores_csv",
        type=str,
        required=True,
        help="Path to full human scores CSV (e.g., data/ICLR2024_pairs/filtered_pairs_with_human_scores.csv)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output path for filtered human scores CSV (default: same directory as sampled_csv with _with_human_scores suffix)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    sampled_csv = Path(args.sampled_csv)
    human_scores_csv = Path(args.human_scores_csv)
    
    # Set default output path if not provided
    if args.output_csv:
        output_csv = Path(args.output_csv)
    else:
        # Default: same directory as sampled_csv, with _with_human_scores suffix
        output_csv = sampled_csv.parent / f"{sampled_csv.stem}_with_human_scores.csv"
    
    print("="*80)
    print("Get Matching Human Scores")
    print("="*80)
    print(f"Sampled CSV: {sampled_csv}")
    print(f"Human scores CSV: {human_scores_csv}")
    print(f"Output CSV: {output_csv}")
    print()
    
    try:
        stats = get_matching_human_scores(
            sampled_csv=sampled_csv,
            human_scores_csv=human_scores_csv,
            output_csv=output_csv,
            verbose=args.verbose
        )
        
        print("\n" + "="*80)
        print("Summary")
        print("="*80)
        print(f"Total papers in sampled CSV: {stats['total_sampled']}")
        print(f"Total rows in human scores CSV: {stats['total_in_human_csv']}")
        print(f"Matching rows found: {stats['matched']}")
        print(f"Papers not found in human scores CSV: {len(stats['not_found'])}")
        print(f"Output CSV: {output_csv}")
        print("="*80)
        
        if stats['not_found']:
            print(f"\n⚠️  Papers not found in human scores CSV ({len(stats['not_found'])}):")
            for paper_id in stats['not_found'][:10]:
                print(f"  - {paper_id}")
            if len(stats['not_found']) > 10:
                print(f"  ... and {len(stats['not_found']) - 10} more")
        
        if stats['matched'] == 0:
            print("\n❌ No matching rows found!")
            return 1
        else:
            print(f"\n✅ Successfully extracted {stats['matched']} matching rows")
            return 0
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

