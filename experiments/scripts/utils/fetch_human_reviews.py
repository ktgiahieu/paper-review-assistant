#!/usr/bin/env python3
"""
Fetch Human Reviews from OpenReview API

This script fetches human review data from OpenReview for papers in a CSV file.
You can choose to fetch:
- Aggregated scores only (mean/std for rating, soundness, presentation, contribution, confidence)
- Raw review text only
- Both scores and raw reviews

Usage:
    # Fetch both scores and reviews (default)
    python fetch_human_reviews.py --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv
    
    # Fetch only scores
    python fetch_human_reviews.py --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv --scores_only
    
    # Fetch only raw reviews
    python fetch_human_reviews.py --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv --reviews_only
"""

import argparse
import pandas as pd
import requests
import time
from pathlib import Path
from typing import Optional, Dict, List
import json
from tqdm import tqdm
import numpy as np

# OpenReview API base URL
OPENREVIEW_API_BASE = "https://api2.openreview.net"
VENUE_ID = "ICLR.cc/2024/Conference"


def get_submission_with_reviews(paper_id: str, max_retries: int = 3) -> Optional[Dict]:
    """
    Fetch submission note with its reviews (replies) from OpenReview API.
    
    Based on: https://docs.openreview.net/how-to-guides/data-retrieval-and-modification/how-to-get-all-notes-for-submissions-reviews-rebuttals-etc
    
    Args:
        paper_id: OpenReview forum ID (same as paperid in CSV)
        max_retries: Number of retry attempts
        
    Returns:
        Dict containing submission note with details.replies, or None if failed
    """
    url = f"{OPENREVIEW_API_BASE}/notes"
    params = {
        "id": paper_id,
        "details": "replies"
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'notes' in data and len(data['notes']) > 0:
                submission = data['notes'][0]
                return submission
            else:
                # Try alternative: get by forum ID
                params_alt = {
                    "forum": paper_id,
                    "details": "replies"
                }
                response = requests.get(url, params=params_alt, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if 'notes' in data and len(data['notes']) > 0:
                    submission = data['notes'][0]
                    return submission
                else:
                    if attempt == max_retries - 1:
                        print(f"Warning: No submission found for {paper_id}")
                    return None
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"Retry {attempt + 1}/{max_retries} for {paper_id}: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed to fetch submission for {paper_id}: {e}")
                return None
    
    return None


def extract_scores_from_review(review_note: Dict, debug: bool = False) -> Dict:
    """
    Extract numerical scores from a review note.
    
    OpenReview ICLR 2024 reviews typically have:
    - rating: Overall rating (1-10 scale)
    - confidence: Reviewer confidence (1-5 scale)
    - soundness: Technical soundness (1-4 scale)
    - presentation: Presentation quality (1-4 scale)
    - contribution: Contribution (1-4 scale)
    
    Args:
        review_note: OpenReview note dict
        debug: Print debug information
        
    Returns:
        Dict with extracted scores
    """
    content = review_note.get('content', {})
    
    if debug:
        print(f"\n    DEBUG: Review content keys: {list(content.keys())}")
        # Print first few fields for inspection
        for key in list(content.keys())[:5]:
            value = content[key]
            if isinstance(value, dict):
                print(f"      {key}: {value.get('value', value)[:100] if isinstance(value.get('value', value), str) else value.get('value', value)}")
            else:
                print(f"      {key}: {str(value)[:100]}")
    
    scores = {
        'soundness': None,
        'presentation': None,
        'contribution': None,
        'rating': None,
        'confidence': None
    }
    
    # Extract scores (try different field name variations)
    # Rating
    if 'rating' in content:
        rating_str = content['rating']
        if isinstance(rating_str, dict):
            rating_str = rating_str.get('value', '')
        # Parse "8: accept, good paper" -> 8
        if isinstance(rating_str, str):
            try:
                scores['rating'] = float(rating_str.split(':')[0].strip())
            except:
                pass
        elif isinstance(rating_str, (int, float)):
            scores['rating'] = float(rating_str)
    
    # Soundness
    for field_name in ['soundness', 'technical_soundness', 'correctness']:
        if field_name in content:
            soundness_str = content[field_name]
            if isinstance(soundness_str, dict):
                soundness_str = soundness_str.get('value', '')
            if isinstance(soundness_str, str):
                try:
                    # Parse "3: good" -> 3
                    scores['soundness'] = float(soundness_str.split(':')[0].strip())
                    break
                except:
                    pass
            elif isinstance(soundness_str, (int, float)):
                scores['soundness'] = float(soundness_str)
                break
    
    # Presentation
    for field_name in ['presentation', 'clarity']:
        if field_name in content:
            pres_str = content[field_name]
            if isinstance(pres_str, dict):
                pres_str = pres_str.get('value', '')
            if isinstance(pres_str, str):
                try:
                    scores['presentation'] = float(pres_str.split(':')[0].strip())
                    break
                except:
                    pass
            elif isinstance(pres_str, (int, float)):
                scores['presentation'] = float(pres_str)
                break
    
    # Contribution
    for field_name in ['contribution', 'novelty']:
        if field_name in content:
            contrib_str = content[field_name]
            if isinstance(contrib_str, dict):
                contrib_str = contrib_str.get('value', '')
            if isinstance(contrib_str, str):
                try:
                    scores['contribution'] = float(contrib_str.split(':')[0].strip())
                    break
                except:
                    pass
            elif isinstance(contrib_str, (int, float)):
                scores['contribution'] = float(contrib_str)
                break
    
    # Confidence
    if 'confidence' in content:
        conf_str = content['confidence']
        if isinstance(conf_str, dict):
            conf_str = conf_str.get('value', '')
        if isinstance(conf_str, str):
            try:
                scores['confidence'] = float(conf_str.split(':')[0].strip())
            except:
                pass
        elif isinstance(conf_str, (int, float)):
            scores['confidence'] = float(conf_str)
    
    return scores


def extract_review_text(review_note: Dict) -> str:
    """
    Extract review text from a review note.
    
    Args:
        review_note: OpenReview note dict
        
    Returns:
        Review text as string, or empty string if not found
    """
    content = review_note.get('content', {})
    
    # Extract review text - try common field names
    review_text = None
    for field_name in ['review', 'summary', 'comment', 'review_text', 'content']:
        if field_name in content:
            value = content[field_name]
            if isinstance(value, dict):
                value = value.get('value', '')
            if isinstance(value, str) and value.strip():
                review_text = value.strip()
                break
    
    # If no review text found, try to get all text fields
    if not review_text:
        text_parts = []
        for key, value in content.items():
            if isinstance(value, str) and len(value) > 50:  # Likely review text
                text_parts.append(f"{key}:\n{value}")
            elif isinstance(value, dict) and 'value' in value:
                val = value['value']
                if isinstance(val, str) and len(val) > 50:
                    text_parts.append(f"{key}:\n{val}")
        if text_parts:
            review_text = "\n\n".join(text_parts)
    
    return review_text or ''


def fetch_reviews_for_paper(
    paper_id: str,
    fetch_scores: bool = True,
    fetch_reviews: bool = True,
    verbose: bool = False,
    debug: bool = False
) -> Dict:
    """
    Fetch review data for a single paper from OpenReview API.
    
    Args:
        paper_id: OpenReview forum ID
        fetch_scores: If True, fetch and aggregate scores
        fetch_reviews: If True, fetch raw review text
        verbose: Enable verbose output
        debug: Print debug information
        
    Returns:
        Dictionary with fetched data:
        - If fetch_scores: num_reviews, human_*_mean, human_*_std columns
        - If fetch_reviews: raw_human_reviews (JSON), num_raw_reviews
    """
    submission = get_submission_with_reviews(paper_id)
    
    if not submission:
        result = {}
        if fetch_scores:
            result.update({
                'num_reviews': 0,
                'human_soundness_mean': None,
                'human_soundness_std': None,
                'human_presentation_mean': None,
                'human_presentation_std': None,
                'human_contribution_mean': None,
                'human_contribution_std': None,
                'human_rating_mean': None,
                'human_rating_std': None,
                'human_confidence_mean': None,
                'human_confidence_std': None
            })
        if fetch_reviews:
            result.update({
                'raw_human_reviews': json.dumps([]),
                'num_raw_reviews': 0
            })
        return result
    
    # Get replies (reviews, comments, etc.) from submission.details.replies
    replies = submission.get('details', {}).get('replies', [])
    
    if not replies:
        if verbose:
            print(f"  No replies found for {paper_id}")
        result = {}
        if fetch_scores:
            result.update({
                'num_reviews': 0,
                'human_soundness_mean': None,
                'human_soundness_std': None,
                'human_presentation_mean': None,
                'human_presentation_std': None,
                'human_contribution_mean': None,
                'human_contribution_std': None,
                'human_rating_mean': None,
                'human_rating_std': None,
                'human_confidence_mean': None,
                'human_confidence_std': None
            })
        if fetch_reviews:
            result.update({
                'raw_human_reviews': json.dumps([]),
                'num_raw_reviews': 0
            })
        return result
    
    # Filter for official reviews
    # Per OpenReview docs: reviews use invitations ending with "Official_Review"
    review_notes = []
    for reply in replies:
        invitations = reply.get('invitations', [])
        # Check if any invitation ends with "Official_Review"
        if any(inv.endswith('Official_Review') for inv in invitations):
            review_notes.append(reply)
    
    if verbose:
        print(f"  Found {len(replies)} total replies, {len(review_notes)} official reviews for {paper_id}")
    
    result = {}
    
    # Extract scores if requested
    if fetch_scores:
        scores_list = []
        for idx, review_note in enumerate(review_notes):
            if debug:
                print(f"\n  DEBUG: Processing review {idx + 1}/{len(review_notes)}")
                print(f"    Invitations: {review_note.get('invitations', [])}")
            
            scores = extract_scores_from_review(review_note, debug=debug)
            
            if debug:
                print(f"    Extracted scores: {scores}")
            
            # Only include if at least one score is present
            if any(v is not None for v in scores.values()):
                scores_list.append(scores)
            elif debug:
                print(f"    ⚠️  No scores extracted from this review")
        
        if scores_list:
            result['num_reviews'] = len(scores_list)
            
            for metric in ['soundness', 'presentation', 'contribution', 'rating', 'confidence']:
                values = [r[metric] for r in scores_list if r[metric] is not None]
                
                if values:
                    result[f'human_{metric}_mean'] = float(np.mean(values))
                    result[f'human_{metric}_std'] = float(np.std(values)) if len(values) > 1 else 0.0
                else:
                    result[f'human_{metric}_mean'] = None
                    result[f'human_{metric}_std'] = None
        else:
            result.update({
                'num_reviews': 0,
                'human_soundness_mean': None,
                'human_soundness_std': None,
                'human_presentation_mean': None,
                'human_presentation_std': None,
                'human_contribution_mean': None,
                'human_contribution_std': None,
                'human_rating_mean': None,
                'human_rating_std': None,
                'human_confidence_mean': None,
                'human_confidence_std': None
            })
    
    # Extract raw reviews if requested
    if fetch_reviews:
        raw_reviews = []
        for review_note in review_notes:
            review_text = extract_review_text(review_note)
            scores = extract_scores_from_review(review_note, debug=False)
            
            # Only include if we have at least review text or scores
            if review_text or any(v is not None for v in scores.values()):
                review_dict = {
                    'review_text': review_text,
                    **{k: v for k, v in scores.items() if v is not None}
                }
                raw_reviews.append(review_dict)
        
        result['raw_human_reviews'] = json.dumps(raw_reviews)
        result['num_raw_reviews'] = len(raw_reviews)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Fetch human review data from OpenReview API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch both scores and reviews (default)
  python fetch_human_reviews.py --csv_file ./data/filtered_pairs.csv
  
  # Fetch only scores
  python fetch_human_reviews.py --csv_file ./data/filtered_pairs.csv --scores_only
  
  # Fetch only raw reviews
  python fetch_human_reviews.py --csv_file ./data/filtered_pairs.csv --reviews_only
        """
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Path to CSV file with paper IDs (must have 'paperid' column)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output CSV file path (default: adds _with_human_reviews to input filename)"
    )
    
    # Mode selection - mutually exclusive
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--scores_only",
        action="store_true",
        help="Fetch only aggregated scores (mean/std for rating, soundness, etc.)"
    )
    mode_group.add_argument(
        "--reviews_only",
        action="store_true",
        help="Fetch only raw review text"
    )
    # Default is both, so no flag needed
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N papers (for testing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information (API responses, score extraction)"
    )
    
    args = parser.parse_args()
    
    # Determine what to fetch
    fetch_scores = not args.reviews_only  # True if not reviews_only
    fetch_reviews = not args.scores_only  # True if not scores_only
    
    # Read CSV
    print(f"Reading {args.csv_file}...")
    df = pd.read_csv(args.csv_file)
    
    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to first {args.limit} papers for testing")
    
    print(f"Found {len(df)} papers")
    
    # Check for paperid column
    paperid_col = None
    for col in df.columns:
        if 'paperid' in col.lower() or 'paper_id' in col.lower():
            paperid_col = col
            break
    
    if not paperid_col:
        raise ValueError("CSV file must have a 'paperid' or 'paper_id' column")
    
    # Check if columns already exist
    if fetch_scores:
        human_cols = [col for col in df.columns if col.startswith('human_') or col == 'num_reviews']
        if human_cols:
            print(f"Warning: Found existing human score columns: {human_cols}")
            print("These will be overwritten.")
    
    if fetch_reviews:
        if 'raw_human_reviews' in df.columns or 'num_raw_reviews' in df.columns:
            print("Warning: Found existing raw review columns. These will be overwritten.")
    
    # Print what we're fetching
    print("\n" + "="*80)
    print("Fetch Configuration")
    print("="*80)
    if fetch_scores and fetch_reviews:
        print("Mode: Fetching both scores and raw reviews")
    elif fetch_scores:
        print("Mode: Fetching scores only")
    elif fetch_reviews:
        print("Mode: Fetching raw reviews only")
    print("="*80)
    
    # Fetch data for each paper
    print("\nFetching human review data from OpenReview API...")
    print("(This may take several minutes depending on API rate limits)")
    
    review_data_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Fetching reviews", disable=args.debug):
        paper_id = row[paperid_col]
        
        if args.debug:
            print(f"\n{'='*80}")
            print(f"Processing paper: {paper_id}")
            print(f"Title: {row.get('title', 'N/A')[:80]}")
            print('='*80)
        
        data = fetch_reviews_for_paper(
            paper_id,
            fetch_scores=fetch_scores,
            fetch_reviews=fetch_reviews,
            verbose=args.verbose,
            debug=args.debug
        )
        review_data_list.append(data)
        
        if args.debug:
            if fetch_scores and data.get('num_reviews', 0) > 0:
                print(f"\n✅ Successfully extracted {data['num_reviews']} reviews")
                for metric in ['soundness', 'presentation', 'contribution', 'rating']:
                    mean_val = data.get(f'human_{metric}_mean')
                    if mean_val is not None:
                        print(f"  {metric}: {mean_val:.2f}")
            if fetch_reviews and data.get('num_raw_reviews', 0) > 0:
                print(f"  Raw reviews: {data['num_raw_reviews']}")
            elif args.debug and (not fetch_scores or data.get('num_reviews', 0) == 0):
                print(f"\n⚠️  No reviews found")
        
        # Rate limiting - be respectful to OpenReview API
        time.sleep(0.5)
    
    # Create DataFrame from fetched data
    review_data_df = pd.DataFrame(review_data_list)
    
    # Merge with original DataFrame
    df_with_reviews = pd.concat([df, review_data_df], axis=1)
    
    # Determine output file path
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        input_path = Path(args.csv_file)
        suffix = "_with_human_reviews"
        if args.scores_only:
            suffix = "_with_human_scores"
        elif args.reviews_only:
            suffix = "_with_raw_reviews"
        output_path = input_path.parent / f"{input_path.stem}{suffix}{input_path.suffix}"
    
    # Save
    df_with_reviews.to_csv(output_path, index=False)
    print(f"\n✅ Saved results to: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("Summary Statistics:")
    print("="*80)
    
    print(f"\nTotal papers: {len(df_with_reviews)}")
    
    if fetch_scores:
        papers_with_reviews = (df_with_reviews['num_reviews'] > 0).sum() if 'num_reviews' in df_with_reviews.columns else 0
        print(f"Papers with reviews: {papers_with_reviews}")
        print(f"Papers without reviews: {len(df_with_reviews) - papers_with_reviews}")
        
        if papers_with_reviews > 0:
            print(f"\nAverage number of reviews per paper: {df_with_reviews[df_with_reviews['num_reviews'] > 0]['num_reviews'].mean():.2f}")
            
            print("\nHuman score availability:")
            for metric in ['soundness', 'presentation', 'contribution', 'rating']:
                col = f'human_{metric}_mean'
                if col in df_with_reviews.columns:
                    available = df_with_reviews[col].notna().sum()
                    print(f"  {metric.capitalize()}: {available}/{len(df_with_reviews)} papers ({available/len(df_with_reviews)*100:.1f}%)")
            
            print("\nHuman score statistics (mean ± std):")
            for metric in ['soundness', 'presentation', 'contribution', 'rating']:
                col = f'human_{metric}_mean'
                if col in df_with_reviews.columns and df_with_reviews[col].notna().any():
                    mean = df_with_reviews[col].mean()
                    std = df_with_reviews[col].std()
                    print(f"  {metric.capitalize()}: {mean:.2f} ± {std:.2f}")
    
    if fetch_reviews:
        if 'num_raw_reviews' in df_with_reviews.columns:
            papers_with_raw = (df_with_reviews['num_raw_reviews'] > 0).sum()
            total_raw_reviews = df_with_reviews['num_raw_reviews'].sum()
            print(f"\nPapers with raw reviews: {papers_with_raw}")
            print(f"Total raw reviews fetched: {total_raw_reviews}")
            if papers_with_raw > 0:
                print(f"Average raw reviews per paper: {total_raw_reviews / papers_with_raw:.2f}")
    
    print("\n" + "="*80)
    if fetch_scores:
        print("Next steps:")
        print("  1. Use the output CSV with calculate_mse_mae.py to compute AI vs Human score differences")
        print("  2. Run: python calculate_mse_mae.py --csv_file", str(output_path))
    print("="*80)


if __name__ == "__main__":
    main()

