#!/usr/bin/env python3
"""
Evaluate Flaw Detection in AI-Generated Reviews

This script uses an evaluator LLM to check whether AI-generated reviews
mention the consensus flaws (from flaw_descriptions in the CSV).

It calculates recall: how many ground truth flaws are detected in the weaknesses section.

Usage:
    python evaluate_flaw_detection.py \
      --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
      --reviews_dir ./reviews_vllm_Llama3-1_70B_3_runs/ \
      --evaluator_endpoint "http://localhost:8000" \
      --output_dir ./flaw_detection_results/
"""

import argparse
import pandas as pd
import numpy as np
import json
import requests
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import ast

def sanitize_json_string(json_str: str) -> str:
    """
    Sanitize a JSON string to fix common issues with LLM-generated JSON.
    
    Handles:
    - Invalid escape sequences (e.g., \e in "escape")
    - Unescaped backslashes before quotes
    - Truncated JSON (adds closing braces/quotes)
    """
    # First, try to detect if JSON is truncated and complete it
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    if open_braces > close_braces:
        # JSON is likely truncated
        # Try to close the current string if we're in one
        if json_str.count('"') % 2 == 1:
            json_str += '"'
        # Add missing closing braces
        json_str += '}' * (open_braces - close_braces)
    
    # Fix common invalid escape sequences
    # Pattern: find backslash followed by invalid escape character
    # Valid escapes are: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
    def fix_escapes(match):
        escaped_char = match.group(1)
        # If it's already a valid escape, leave it
        if escaped_char in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']:
            return match.group(0)
        # Otherwise, escape the backslash
        return '\\\\' + escaped_char
    
    # Find all backslash-letter combinations and fix invalid ones
    json_str = re.sub(r'\\(.)', fix_escapes, json_str)
    
    # Fix unescaped quotes within string values (heuristic)
    # This is tricky, but we can try to fix cases where we have an odd number of quotes
    # between key-value pairs
    
    return json_str

def parse_flaw_descriptions(flaw_desc_str: str) -> List[str]:
    """
    Parse the flaw_descriptions string from CSV into a list of individual flaws.
    
    The CSV stores flaws as a string representation of a Python list.
    """
    if pd.isna(flaw_desc_str) or flaw_desc_str == '':
        return []
    
    try:
        # Try to parse as Python literal
        flaws = ast.literal_eval(flaw_desc_str)
        if isinstance(flaws, list):
            return [f.strip() for f in flaws if f.strip()]
    except:
        pass
    
    # Fallback: treat as single flaw
    return [flaw_desc_str.strip()]

def load_ai_review(reviews_dir: Path, paper_id: str, version: str, run_id: int = 0) -> Optional[Dict]:
    """
    Load a single AI review JSON file.
    """
    paper_dir = reviews_dir / paper_id
    review_file = paper_dir / f"{version}_review_run{run_id}.json"
    
    if not review_file.exists():
        return None
    
    try:
        with open(review_file) as f:
            review_data = json.load(f)
            if review_data.get('success', False):
                return review_data
    except Exception as e:
        print(f"Error loading {review_file}: {e}")
    
    return None

def extract_weaknesses(review_data: Dict) -> str:
    """
    Extract the weaknesses section from a review.
    Handles different formats (SEA-E, CycleReviewer, GenericStructured, Anthropic).
    """
    model_type = review_data.get('model_type', 'default')
    
    weaknesses_text = ""
    
    if model_type == 'CycleReviewer':
        # Combine weaknesses from all reviewers
        reviewers = review_data.get('reviewers', [])
        all_weaknesses = []
        for reviewer in reviewers:
            weak_list = reviewer.get('weaknesses', [])
            if isinstance(weak_list, list):
                all_weaknesses.extend(weak_list)
            elif isinstance(weak_list, str):
                all_weaknesses.append(weak_list)
        weaknesses_text = "\n".join(all_weaknesses)
    
    elif 'weaknesses' in review_data:
        weaknesses = review_data['weaknesses']
        if isinstance(weaknesses, list):
            weaknesses_text = "\n".join(weaknesses)
        elif isinstance(weaknesses, str):
            weaknesses_text = weaknesses
    
    return weaknesses_text.strip()

def check_flaw_detection(
    flaw: str,
    weaknesses: str,
    evaluator_endpoint: str,
    evaluator_model: str,
    max_retries: int = 3,
    timeout: int = 60
) -> Tuple[bool, str]:
    """
    Use the evaluator LLM to check if a specific flaw is mentioned in the weaknesses.
    
    Returns:
        (is_detected, reasoning)
    """
    system_prompt = """You are an expert evaluator tasked with determining whether a specific flaw or weakness mentioned in a paper review matches a ground truth flaw description.

You will be given:
1. A GROUND TRUTH FLAW - a known issue with the paper
2. A WEAKNESSES SECTION - from an AI-generated review

Your task: Determine if the weakness section mentions or identifies the ground truth flaw, even if worded differently.

Consider a match if:
- The weakness directly mentions the same issue
- The weakness describes the same problem with different wording
- The weakness implies or relates to the ground truth flaw

Do NOT consider a match if:
- The weaknesses are completely unrelated to the ground truth flaw
- The weaknesses only tangentially touch on different aspects

Respond in JSON format:
{
  "detected": true/false,
  "reasoning": "Brief explanation of your decision"
}"""

    user_prompt = f"""GROUND TRUTH FLAW:
{flaw}

WEAKNESSES SECTION FROM REVIEW:
{weaknesses}

Is this ground truth flaw detected/mentioned in the weaknesses section?"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    request_data = {
        "model": evaluator_model,
        "messages": messages,
        "temperature": 0.0,  # Deterministic for consistency
        "max_tokens": 1000  # Increased to reduce truncation
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{evaluator_endpoint}/v1/chat/completions",
                json=request_data,
                timeout=timeout
            )
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            # Parse JSON response
            # Try to extract JSON from markdown code blocks if present
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            # Sanitize JSON before parsing
            content_sanitized = sanitize_json_string(content)
            
            try:
                result_json = json.loads(content_sanitized)
            except json.JSONDecodeError:
                # If sanitization didn't work, try the original
                result_json = json.loads(content)
            
            is_detected = result_json.get('detected', False)
            reasoning = result_json.get('reasoning', '')
            
            return (is_detected, reasoning)
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"  Timeout, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(2)
            else:
                print(f"  Failed after {max_retries} attempts (timeout)")
                return (False, "Evaluator timeout")
        
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"  JSON decode error, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(2)
            else:
                print(f"  Failed to parse evaluator response: {e}")
                print(f"  Raw content (first 300 chars): {content[:300]}")
                if 'content_sanitized' in locals():
                    print(f"  Sanitized content (first 300 chars): {content_sanitized[:300]}")
                return (False, f"JSON parse error: {str(e)}")
        
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Error: {e}, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(2)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return (False, f"Evaluator error: {str(e)}")
    
    return (False, "Unknown error")

def evaluate_paper_review(
    paper_id: str,
    flaws: List[str],
    weaknesses: str,
    evaluator_endpoint: str,
    evaluator_model: str,
    verbose: bool = False
) -> Dict:
    """
    Evaluate a single review against ground truth flaws.
    
    Returns:
        Dict with recall and detailed flaw detection results
    """
    if not flaws:
        return {
            'num_flaws': 0,
            'num_detected': 0,
            'recall': None,
            'flaws_detailed': []
        }
    
    if not weaknesses:
        return {
            'num_flaws': len(flaws),
            'num_detected': 0,
            'recall': 0.0,
            'flaws_detailed': [
                {'flaw': flaw, 'detected': False, 'reasoning': 'No weaknesses section'}
                for flaw in flaws
            ]
        }
    
    detected_count = 0
    flaws_detailed = []
    
    for flaw_idx, flaw in enumerate(flaws):
        if verbose:
            print(f"    Checking flaw {flaw_idx + 1}/{len(flaws)}: {flaw[:80]}...")
        
        is_detected, reasoning = check_flaw_detection(
            flaw,
            weaknesses,
            evaluator_endpoint,
            evaluator_model
        )
        
        if is_detected:
            detected_count += 1
        
        flaws_detailed.append({
            'flaw': flaw,
            'detected': is_detected,
            'reasoning': reasoning
        })
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.2)
    
    recall = detected_count / len(flaws) if len(flaws) > 0 else 0.0
    
    return {
        'num_flaws': len(flaws),
        'num_detected': detected_count,
        'recall': recall,
        'flaws_detailed': flaws_detailed
    }

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate flaw detection in AI-generated reviews"
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Path to filtered_pairs.csv with flaw_descriptions"
    )
    parser.add_argument(
        "--reviews_dir",
        type=str,
        required=True,
        help="Directory containing AI-generated reviews"
    )
    parser.add_argument(
        "--evaluator_endpoint",
        type=str,
        default="http://localhost:8000",
        help="vLLM endpoint for evaluator model"
    )
    parser.add_argument(
        "--evaluator_model",
        type=str,
        default="Qwen3-30B-A3B-Instruct-2507-FP8",
        help="Evaluator model name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./flaw_detection_results/",
        help="Output directory for results"
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=['v1', 'latest', 'both'],
        default='both',
        help="Which version(s) to evaluate"
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
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Flaw Detection Evaluation")
    print("="*80)
    print(f"\nCSV file: {args.csv_file}")
    print(f"Reviews directory: {args.reviews_dir}")
    print(f"Evaluator endpoint: {args.evaluator_endpoint}")
    print(f"Evaluator model: {args.evaluator_model}")
    print(f"Output directory: {output_dir}")
    
    # Load CSV
    print("\nLoading paper data...")
    df_papers = pd.read_csv(args.csv_file)
    
    if args.limit:
        df_papers = df_papers.head(args.limit)
        print(f"Limited to first {args.limit} papers for testing")
    
    print(f"Found {len(df_papers)} papers")
    
    # Filter papers with flaws
    df_papers['has_flaws'] = df_papers['flaw_descriptions'].apply(
        lambda x: pd.notna(x) and x != '' and x != '[]'
    )
    df_with_flaws = df_papers[df_papers['has_flaws']].copy()
    print(f"Papers with flaw descriptions: {len(df_with_flaws)}")
    
    if len(df_with_flaws) == 0:
        print("\n❌ No papers with flaw descriptions found!")
        return
    
    reviews_dir = Path(args.reviews_dir)
    versions = ['v1', 'latest'] if args.version == 'both' else [args.version]
    
    # Collect all evaluations
    print("\n" + "="*80)
    print("Evaluating flaw detection in reviews...")
    print("="*80)
    print("(This may take a while depending on the number of papers and flaws)")
    
    all_results = []
    
    for idx, row in tqdm(df_with_flaws.iterrows(), total=len(df_with_flaws), 
                         desc="Evaluating papers"):
        paper_id = row['paperid']
        title = row['title']
        flaws = parse_flaw_descriptions(row['flaw_descriptions'])
        
        if not flaws:
            continue
        
        if args.verbose:
            print(f"\n{'='*80}")
            print(f"Paper: {paper_id} - {title[:60]}")
            print(f"Number of ground truth flaws: {len(flaws)}")
        
        for version in versions:
            # Load all runs for this version
            run_id = 0
            while True:
                review_data = load_ai_review(reviews_dir, paper_id, version, run_id)
                
                if review_data is None:
                    break
                
                if args.verbose:
                    print(f"\n  Evaluating {version} review (run {run_id})...")
                
                # Extract weaknesses
                weaknesses = extract_weaknesses(review_data)
                
                if args.verbose:
                    print(f"  Weaknesses length: {len(weaknesses)} chars")
                
                # Evaluate against each flaw
                evaluation_result = evaluate_paper_review(
                    paper_id,
                    flaws,
                    weaknesses,
                    args.evaluator_endpoint,
                    args.evaluator_model,
                    verbose=args.verbose
                )
                
                # Store result
                result_entry = {
                    'paper_id': paper_id,
                    'title': title,
                    'version': version,
                    'run_id': run_id,
                    'model_type': review_data.get('model_type', 'unknown'),
                    'num_flaws': evaluation_result['num_flaws'],
                    'num_detected': evaluation_result['num_detected'],
                    'recall': evaluation_result['recall'],
                    'flaws_detailed': evaluation_result['flaws_detailed']
                }
                
                all_results.append(result_entry)
                
                if args.verbose:
                    print(f"  Recall: {evaluation_result['recall']:.2f} "
                          f"({evaluation_result['num_detected']}/{evaluation_result['num_flaws']})")
                
                run_id += 1
    
    if not all_results:
        print("\n❌ No results collected!")
        return
    
    # Save detailed results
    print("\n" + "="*80)
    print("Saving results...")
    print("="*80)
    
    # Save full detailed JSON
    detailed_json_path = output_dir / "flaw_detection_detailed.json"
    with open(detailed_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Saved detailed results to: {detailed_json_path}")
    
    # Create summary DataFrame
    summary_rows = []
    for result in all_results:
        summary_rows.append({
            'paper_id': result['paper_id'],
            'title': result['title'],
            'version': result['version'],
            'run_id': result['run_id'],
            'model_type': result['model_type'],
            'num_flaws': result['num_flaws'],
            'num_detected': result['num_detected'],
            'recall': result['recall']
        })
    
    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / "flaw_detection_summary.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"✅ Saved summary to: {summary_csv_path}")
    
    # Create per-flaw details CSV
    flaw_detail_rows = []
    for result in all_results:
        for flaw_detail in result['flaws_detailed']:
            flaw_detail_rows.append({
                'paper_id': result['paper_id'],
                'version': result['version'],
                'run_id': result['run_id'],
                'flaw': flaw_detail['flaw'],
                'detected': flaw_detail['detected'],
                'reasoning': flaw_detail['reasoning']
            })
    
    df_flaw_details = pd.DataFrame(flaw_detail_rows)
    flaw_details_csv_path = output_dir / "flaw_detection_per_flaw.csv"
    df_flaw_details.to_csv(flaw_details_csv_path, index=False)
    print(f"✅ Saved per-flaw details to: {flaw_details_csv_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal evaluations: {len(df_summary)}")
    print(f"Unique papers: {df_summary['paper_id'].nunique()}")
    print(f"Versions: {df_summary['version'].unique()}")
    print(f"Model types: {df_summary['model_type'].unique()}")
    
    print("\nOverall Recall Statistics:")
    print(f"  Mean recall: {df_summary['recall'].mean():.3f}")
    print(f"  Std recall: {df_summary['recall'].std():.3f}")
    print(f"  Min recall: {df_summary['recall'].min():.3f}")
    print(f"  Max recall: {df_summary['recall'].max():.3f}")
    
    print("\nBy Version:")
    for version in versions:
        version_data = df_summary[df_summary['version'] == version]
        print(f"  {version}:")
        print(f"    Mean recall: {version_data['recall'].mean():.3f}")
        print(f"    Std recall: {version_data['recall'].std():.3f}")
        print(f"    N: {len(version_data)}")
    
    print("\nFlaw Detection Rate:")
    total_flaws = df_summary['num_flaws'].sum()
    total_detected = df_summary['num_detected'].sum()
    print(f"  Total flaws: {total_flaws}")
    print(f"  Total detected: {total_detected}")
    print(f"  Detection rate: {total_detected/total_flaws*100:.1f}%")
    
    print("\n" + "="*80)
    print("Next steps:")
    print("  Run: python analyze_flaw_detection.py \\")
    print(f"         --results_file {detailed_json_path}")
    print("="*80)

if __name__ == "__main__":
    main()

