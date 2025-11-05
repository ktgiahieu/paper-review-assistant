#!/usr/bin/env python3
"""
Fetch Human Review Scores from OpenReview API

This script fetches official human review scores from OpenReview for ICLR 2024 papers
and adds them to the filtered_pairs.csv file.

Usage:
    python fetch_human_scores.py --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv
"""

import argparse
import pandas as pd
import requests
import time
from pathlib import Path
from typing import Optional, Dict, List
import json
from tqdm import tqdm

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
    # Get the submission note with replies included
    url = f"{OPENREVIEW_API_BASE}/notes"
    params = {
        "id": paper_id,  # Get specific submission by ID
        "details": "replies"  # Include all replies (reviews, comments, etc.)
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

def aggregate_review_scores(paper_id: str, verbose: bool = False, debug: bool = False) -> Dict:
    """
    Fetch and aggregate all human review scores for a paper.
    
    Based on OpenReview API v2 guide:
    https://docs.openreview.net/how-to-guides/data-retrieval-and-modification/how-to-get-all-notes-for-submissions-reviews-rebuttals-etc
    
    Returns dict with:
    - human_soundness_mean, human_soundness_std
    - human_presentation_mean, human_presentation_std
    - human_contribution_mean, human_contribution_std
    - human_rating_mean, human_rating_std
    - human_confidence_mean, human_confidence_std
    - num_reviews
    """
    submission = get_submission_with_reviews(paper_id)
    
    if not submission:
        return {
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
        }
    
    # Get replies (reviews, comments, etc.) from submission.details.replies
    replies = submission.get('details', {}).get('replies', [])
    
    if not replies:
        if verbose:
            print(f"  No replies found for {paper_id}")
        return {
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
        }
    
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
    
    # Extract scores from review notes
    reviews = []
    for idx, review_note in enumerate(review_notes):
        if debug:
            print(f"\n  DEBUG: Processing review {idx + 1}/{len(review_notes)}")
            print(f"    Invitations: {review_note.get('invitations', [])}")
        
        scores = extract_scores_from_review(review_note, debug=debug)
        
        if debug:
            print(f"    Extracted scores: {scores}")
        
        # Only include if at least one score is present
        if any(v is not None for v in scores.values()):
            reviews.append(scores)
        elif debug:
            print(f"    ⚠️  No scores extracted from this review")
    
    if not reviews:
        return {
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
        }
    
    # Aggregate scores
    import numpy as np
    
    result = {'num_reviews': len(reviews)}
    
    for metric in ['soundness', 'presentation', 'contribution', 'rating', 'confidence']:
        values = [r[metric] for r in reviews if r[metric] is not None]
        
        if values:
            result[f'human_{metric}_mean'] = float(np.mean(values))
            result[f'human_{metric}_std'] = float(np.std(values)) if len(values) > 1 else 0.0
        else:
            result[f'human_{metric}_mean'] = None
            result[f'human_{metric}_std'] = None
    
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Fetch human review scores from OpenReview API"
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Path to filtered_pairs.csv"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output CSV file path (default: adds _with_human_scores to input filename)"
    )
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
    
    # Read CSV
    print(f"Reading {args.csv_file}...")
    df = pd.read_csv(args.csv_file)
    
    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to first {args.limit} papers for testing")
    
    print(f"Found {len(df)} papers")
    
    # Check if human scores already exist
    human_cols = [col for col in df.columns if col.startswith('human_')]
    if human_cols:
        print(f"Warning: Found existing human score columns: {human_cols}")
        print("These will be overwritten.")
    
    # Fetch human scores for each paper
    print("\nFetching human review scores from OpenReview API...")
    print("(This may take several minutes depending on API rate limits)")
    
    human_scores_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Fetching reviews", disable=args.debug):
        paper_id = row['paperid']
        
        if args.debug:
            print(f"\n{'='*80}")
            print(f"Processing paper: {paper_id}")
            print(f"Title: {row.get('title', 'N/A')[:80]}")
            print('='*80)
        
        scores = aggregate_review_scores(paper_id, verbose=args.verbose, debug=args.debug)
        human_scores_list.append(scores)
        
        if args.debug and scores['num_reviews'] > 0:
            print(f"\n✅ Successfully extracted {scores['num_reviews']} reviews")
            for metric in ['soundness', 'presentation', 'contribution', 'rating']:
                mean_val = scores.get(f'human_{metric}_mean')
                if mean_val is not None:
                    print(f"  {metric}: {mean_val:.2f}")
        elif args.debug:
            print(f"\n⚠️  No reviews found")
        
        # Rate limiting - be respectful to OpenReview API
        time.sleep(0.5)
    
    # Create DataFrame from scores
    human_scores_df = pd.DataFrame(human_scores_list)
    
    # Merge with original DataFrame
    df_with_human = pd.concat([df, human_scores_df], axis=1)
    
    # Determine output file path
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        input_path = Path(args.csv_file)
        output_path = input_path.parent / f"{input_path.stem}_with_human_scores{input_path.suffix}"
    
    # Save
    df_with_human.to_csv(output_path, index=False)
    print(f"\n✅ Saved results to: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("Summary Statistics:")
    print("="*80)
    
    print(f"\nTotal papers: {len(df_with_human)}")
    print(f"Papers with reviews: {(df_with_human['num_reviews'] > 0).sum()}")
    print(f"Papers without reviews: {(df_with_human['num_reviews'] == 0).sum()}")
    
    if (df_with_human['num_reviews'] > 0).any():
        print(f"\nAverage number of reviews per paper: {df_with_human[df_with_human['num_reviews'] > 0]['num_reviews'].mean():.2f}")
        
        print("\nHuman score availability:")
        for metric in ['soundness', 'presentation', 'contribution', 'rating']:
            col = f'human_{metric}_mean'
            if col in df_with_human.columns:
                available = df_with_human[col].notna().sum()
                print(f"  {metric.capitalize()}: {available}/{len(df_with_human)} papers ({available/len(df_with_human)*100:.1f}%)")
        
        print("\nHuman score statistics (mean ± std):")
        for metric in ['soundness', 'presentation', 'contribution', 'rating']:
            col = f'human_{metric}_mean'
            if col in df_with_human.columns and df_with_human[col].notna().any():
                mean = df_with_human[col].mean()
                std = df_with_human[col].std()
                print(f"  {metric.capitalize()}: {mean:.2f} ± {std:.2f}")
    
    print("\n" + "="*80)
    print("Next steps:")
    print("  1. Use the output CSV with calculate_mse_mae.py to compute AI vs Human score differences")
    print("  2. Run: python calculate_mse_mae.py --csv_file", str(output_path))
    print("="*80)

if __name__ == "__main__":
    main()

