#!/usr/bin/env python3
"""
Script to filter flaws by category with strict matching.

This script filters flaws from categorized_flaw_cleaned.csv to only include
rows where the category_ids exactly matches the target category (not part of
a comma-separated list).

For example:
- If filtering for "2a", it will include rows with category_ids="2a"
- It will NOT include rows with category_ids="2a,1a" or "1a,2a"

The script can optionally:
1. Merge with flaw descriptions
2. Merge with metareview data (is_flaw_mentioned, mention_reasoning)
3. Merge with LLM review data
4. Sample from the filtered results
5. Copy corresponding paper directories (latest versions) to organized output structure
6. Copy flawed paper versions to planted_error/ subdirectory
"""

import json
import os
import pandas as pd
import re
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


def load_json_file(filepath):
    """Safely loads a single JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Warning: Could not read or parse {filepath}: {e}")
        return None


def extract_single_categories(category_str):
    """
    Extract single categories from a category string (handles quoted values).
    
    Args:
        category_str: Category string (e.g., "2a" or "1b,3b")
    
    Returns:
        List of single category strings
    """
    if pd.isna(category_str) or str(category_str) == 'nan':
        return []
    
    # Convert to string and remove all types of quotes
    category_str = str(category_str).strip()
    while category_str and (category_str[0] in ['"', "'"] and category_str[-1] in ['"', "'"]):
        category_str = category_str[1:-1].strip()
    
    # Split by comma and strip whitespace
    categories = [cat.strip() for cat in category_str.split(',') if cat.strip()]
    return categories


def get_all_unique_categories(categorized_csv_path):
    """
    Extract all unique single categories from the categorized CSV.
    
    Args:
        categorized_csv_path: Path to categorized_flaw_cleaned.csv
    
    Returns:
        Sorted list of unique category IDs
    """
    try:
        df = pd.read_csv(categorized_csv_path)
        df['category_ids'] = df['category_ids'].astype(str)
        
        all_categories = set()
        for category_str in df['category_ids']:
            categories = extract_single_categories(category_str)
            all_categories.update(categories)
        
        return sorted(list(all_categories))
    except Exception as e:
        print(f"Error reading categorized CSV: {e}")
        return []


def strict_category_filter(df, category_id):
    """
    Filter DataFrame to only include rows where category_ids exactly matches
    the target category (not part of a comma-separated list).
    
    Args:
        df: DataFrame with 'category_ids' column
        category_id: Target category ID (e.g., '2a', '1b')
    
    Returns:
        Filtered DataFrame
    """
    # Ensure category_ids is a string
    df = df.copy()
    df['category_ids'] = df['category_ids'].astype(str)
    
    def is_exact_match(category_str):
        """Check if category_str exactly matches category_id (not part of a list)."""
        categories = extract_single_categories(category_str)
        # Check if category_id is the only category
        return len(categories) == 1 and categories[0] == category_id
    
    # Apply strict filtering
    filtered_df = df[df['category_ids'].apply(is_exact_match)]
    
    return filtered_df


def collect_metareview_data(metareviews_path, venue_folder_name, model_name='o3'):
    """
    Collects 'is_flaw_mentioned' and 'mention_reasoning' from metareview JSON files.
    """
    mention_data = {}
    print(f"Scanning for metareview data in: {metareviews_path}")

    if not os.path.exists(metareviews_path):
        print(f"Warning: Metareviews directory not found at {metareviews_path}")
        return mention_data

    model_path = os.path.join(metareviews_path, model_name)
    if os.path.isdir(model_path):
        venue_path = os.path.join(model_path, venue_folder_name)
        if os.path.isdir(venue_path):
            for status in ['accepted', 'rejected']:
                status_path = os.path.join(venue_path, status)
                if os.path.isdir(status_path):
                    for filename in os.listdir(status_path):
                        if filename.endswith(".json"):
                            data = load_json_file(os.path.join(status_path, filename))
                            if not data:
                                continue
                            for paper_key, flaws in data.items():
                                openreview_id = paper_key.split('_')[0]
                                for flaw in flaws:
                                    flaw_id = flaw.get('flaw_id')
                                    if openreview_id and flaw_id:
                                        key = (openreview_id, flaw_id)
                                        mention_data[key] = {
                                            'is_flaw_mentioned': flaw.get('is_flaw_mentioned'),
                                            'mention_reasoning': flaw.get('mention_reasoning')
                                        }
    print(f"Collected mention data for {len(mention_data)} flaws.")
    return mention_data


def copy_papers_for_category(
    openreview_ids,
    papers_source_dir,
    output_base_dir,
    category_id,
    venue_name='NeurIPS2024',
    subdirectory='latest'
):
    """
    Copy paper directories for the given openreview_ids to the output structure.
    
    Args:
        openreview_ids: Set or list of openreview_ids to copy papers for
        papers_source_dir: Source directory containing papers
        output_base_dir: Base output directory
        category_id: Category ID (e.g., '2a')
        venue_name: Venue name (default: 'NeurIPS2024')
        subdirectory: Subdirectory name (default: 'latest', can be 'planted_error')
    
    Returns:
        Tuple of (copied_count, not_found_ids)
    """
    papers_source_dir = Path(papers_source_dir)
    output_base_dir = Path(output_base_dir)
    
    if not papers_source_dir.exists():
        print(f"  Warning: Papers source directory not found: {papers_source_dir}")
        return 0, list(openreview_ids)
    
    # Create output directory structure: {output_base}/NeurIPS2024/{category_id}/{subdirectory}/
    output_dir = output_base_dir / venue_name / category_id / subdirectory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    not_found_ids = []
    
    print(f"  Copying papers to: {output_dir}")
    print(f"  Looking for {len(openreview_ids)} unique papers...")
    
    # Get all paper directories in source
    source_paper_dirs = {d.name.split('_')[0]: d for d in papers_source_dir.iterdir() 
                         if d.is_dir() and not d.name.startswith('.')}
    
    for openreview_id in tqdm(openreview_ids, desc=f"  Copying papers to {subdirectory}"):
        # Find matching paper directory (starts with openreview_id)
        matching_dirs = [d for name, d in source_paper_dirs.items() 
                        if name == openreview_id]
        
        if not matching_dirs:
            not_found_ids.append(openreview_id)
            continue
        
        # Use the first matching directory
        source_dir = matching_dirs[0]
        dest_dir = output_dir / source_dir.name
        
        try:
            # Remove destination if it exists
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            # Copy the entire directory
            shutil.copytree(source_dir, dest_dir)
            copied_count += 1
        except Exception as e:
            print(f"\n  Error copying {source_dir.name}: {e}")
            not_found_ids.append(openreview_id)
    
    print(f"  Successfully copied {copied_count} papers to {subdirectory}/")
    if not_found_ids:
        print(f"  Warning: {len(not_found_ids)} papers not found in source directory")
    
    return copied_count, not_found_ids


def copy_flawed_papers_for_category(
    filtered_df,
    flawed_papers_source_dir,
    output_base_dir,
    category_id,
    venue_name='NeurIPS2024',
    copy_csv_only=False
):
    """
    Copy only the flawed paper versions that match the category.
    
    For each paper, this function:
    1. Reads the modifications_summary.csv file
    2. Filters it to only include flaws matching the category
    3. Copies the filtered CSV and optionally the corresponding .md files
    
    Args:
        filtered_df: DataFrame with columns 'openreview_id' and 'flaw_id' for flaws in this category
        flawed_papers_source_dir: Source directory containing flawed papers
        output_base_dir: Base output directory
        category_id: Category ID (e.g., '2a')
        venue_name: Venue name (default: 'NeurIPS2024')
        copy_csv_only: If True, only copy CSV files, not .md files (default: False)
    
    Returns:
        Tuple of (copied_count, not_found_ids, total_flaws_copied)
    """
    flawed_papers_source_dir = Path(flawed_papers_source_dir)
    output_base_dir = Path(output_base_dir)
    
    if not flawed_papers_source_dir.exists():
        print(f"  Warning: Flawed papers source directory not found: {flawed_papers_source_dir}")
        return 0, [], 0
    
    # Create output directory structure: {output_base}/NeurIPS2024/{category_id}/planted_error/
    output_dir = output_base_dir / venue_name / category_id / 'planted_error'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a set of (openreview_id, flaw_id) pairs for quick lookup
    category_flaws = set()
    for _, row in filtered_df.iterrows():
        openreview_id = row.get('openreview_id')
        flaw_id = row.get('flaw_id')
        if pd.notna(openreview_id) and pd.notna(flaw_id):
            category_flaws.add((str(openreview_id), str(flaw_id)))
    
    if not category_flaws:
        print(f"  Warning: No flaws found in filtered data for category {category_id}")
        return 0, [], 0
    
    # Get unique openreview_ids
    unique_openreview_ids = set(filtered_df['openreview_id'].dropna().unique())
    
    copied_count = 0
    not_found_ids = []
    total_flaws_copied = 0
    
    print(f"  Copying flawed papers to: {output_dir}")
    print(f"  Looking for {len(unique_openreview_ids)} unique papers with {len(category_flaws)} flaws...")
    
    # Get all paper directories in source
    source_paper_dirs = {d.name.split('_')[0]: d for d in flawed_papers_source_dir.iterdir() 
                         if d.is_dir() and not d.name.startswith('.')}
    
    for openreview_id in tqdm(unique_openreview_ids, desc="  Copying flawed papers"):
        openreview_id = str(openreview_id)
        
        # Find matching paper directory
        matching_dirs = [d for name, d in source_paper_dirs.items() 
                        if name == openreview_id]
        
        if not matching_dirs:
            not_found_ids.append(openreview_id)
            continue
        
        source_dir = matching_dirs[0]
        dest_dir = output_dir / source_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Look for modifications_summary.csv in the source directory
            csv_filename = f"{openreview_id}_modifications_summary.csv"
            csv_path = source_dir / csv_filename
            
            if not csv_path.exists():
                # Try alternative naming patterns
                csv_files = list(source_dir.glob("*_modifications_summary.csv"))
                if csv_files:
                    csv_path = csv_files[0]
                else:
                    print(f"\n  Warning: No modifications_summary.csv found for {openreview_id}")
                    continue
            
            # Read the CSV
            try:
                modifications_df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"\n  Error reading CSV for {openreview_id}: {e}")
                continue
            
            # Filter to only include flaws in this category
            filtered_modifications = modifications_df[
                modifications_df['flaw_id'].apply(
                    lambda flaw_id: (openreview_id, str(flaw_id)) in category_flaws
                )
            ]
            
            if filtered_modifications.empty:
                # No flaws for this category in this paper
                continue
            
            # Copy the filtered CSV
            dest_csv_path = dest_dir / csv_filename
            filtered_modifications.to_csv(dest_csv_path, index=False)
            
            # Copy corresponding .md files only if copy_csv_only is False
            if not copy_csv_only:
                # Look for flawed_papers subdirectory or .md files directly in the source directory
                flawed_papers_dir = source_dir / 'flawed_papers'
                if flawed_papers_dir.exists() and flawed_papers_dir.is_dir():
                    # Create flawed_papers subdirectory in destination
                    dest_flawed_papers_dir = dest_dir / 'flawed_papers'
                    dest_flawed_papers_dir.mkdir(exist_ok=True)
                    
                    # Copy each .md file for the filtered flaws
                    for flaw_id in filtered_modifications['flaw_id']:
                        flaw_id = str(flaw_id)
                        md_file = flawed_papers_dir / f"{flaw_id}.md"
                        if md_file.exists():
                            shutil.copy2(md_file, dest_flawed_papers_dir / f"{flaw_id}.md")
                            total_flaws_copied += 1
                        else:
                            print(f"\n  Warning: .md file not found for flaw {flaw_id} in {openreview_id}")
                else:
                    # Try to find .md files directly in source directory
                    for flaw_id in filtered_modifications['flaw_id']:
                        flaw_id = str(flaw_id)
                        md_file = source_dir / f"{flaw_id}.md"
                        if md_file.exists():
                            shutil.copy2(md_file, dest_dir / f"{flaw_id}.md")
                            total_flaws_copied += 1
            else:
                # Just count the flaws in the CSV
                total_flaws_copied += len(filtered_modifications)
            
            copied_count += 1
            
        except Exception as e:
            print(f"\n  Error processing {openreview_id}: {e}")
            not_found_ids.append(openreview_id)
    
    if copy_csv_only:
        print(f"  Successfully copied {copied_count} CSV files with {total_flaws_copied} flaws to planted_error/")
    else:
        print(f"  Successfully copied {copied_count} papers with {total_flaws_copied} flaws to planted_error/")
    if not_found_ids:
        print(f"  Warning: {len(not_found_ids)} papers not found in source directory")
    
    return copied_count, not_found_ids, total_flaws_copied


def collect_llm_review_data(reviews_path, venue_folder_name, model_name='o3'):
    """
    Collects the full LLM review content from individual review JSON files.
    """
    review_data = {}
    print(f"Scanning for LLM review data in: {reviews_path}")

    if not os.path.exists(reviews_path):
        print(f"Warning: Reviews directory not found at {reviews_path}")
        return review_data

    model_path = os.path.join(reviews_path, model_name)
    if os.path.isdir(model_path):
        venue_path = os.path.join(model_path, venue_folder_name)
        if os.path.isdir(venue_path):
            for status in ['accepted', 'rejected']:
                status_path = os.path.join(venue_path, status)
                if not os.path.isdir(status_path):
                    continue
                for paper_folder in os.listdir(status_path):
                    paper_folder_path = os.path.join(status_path, paper_folder)
                    if not os.path.isdir(paper_folder_path):
                        continue
                    
                    openreview_id = paper_folder.split('_')[0]
                    for filename in os.listdir(paper_folder_path):
                        if filename.endswith("_review.json"):
                            # Extract flaw_id from filename
                            match = re.match(r'(.+?)_(\d+_\d+)_(.+)_review\.json', filename)
                            if match:
                                flaw_id = match.group(3)
                            else:
                                # Fallback for different naming
                                base_name = filename.replace('_review.json', '')
                                flaw_id = '_'.join(base_name.split('_')[3:])

                            if not flaw_id:
                                continue
                            
                            data = load_json_file(os.path.join(paper_folder_path, filename))
                            if data is not None:
                                key = (openreview_id, flaw_id)
                                review_data[key] = {'llm_review': json.dumps(data, indent=2)}
    
    print(f"Collected LLM reviews for {len(review_data)} flaws.")
    return review_data


def filter_and_aggregate(
    categorized_csv_path,
    category_id,
    output_path,
    venue_folder_name=None,
    base_data_dir=None,
    include_descriptions=True,
    include_metareviews=True,
    include_reviews=True,
    model_name='o3',
    n_samples=None,
    random_seed=42,
    copy_papers=False,
    papers_source_dir=None,
    copy_flawed_papers=False,
    flawed_papers_source_dir=None,
    output_base_dir=None,
    venue_name='NeurIPS2024',
    copy_csv_only=False
):
    """
    Main function to filter flaws by category and optionally aggregate additional data.
    
    Args:
        categorized_csv_path: Path to categorized_flaw_cleaned.csv
        category_id: Category ID to filter for (e.g., '2a')
        output_path: Path to save the filtered output CSV
        venue_folder_name: Venue folder name (e.g., 'NeurIPS2024_latest_flawed_papers_v1')
        base_data_dir: Base data directory (e.g., '../data')
        include_descriptions: Whether to merge flaw descriptions
        include_metareviews: Whether to merge metareview data
        include_reviews: Whether to merge LLM review data
        model_name: Model name for metareviews/reviews (default: 'o3')
        n_samples: Optional number of samples to take (None = no sampling)
        random_seed: Random seed for sampling
        copy_papers: Whether to copy corresponding paper directories
        papers_source_dir: Source directory for accepted papers (required if copy_papers=True)
        copy_flawed_papers: Whether to copy flawed paper versions
        flawed_papers_source_dir: Source directory for flawed papers (required if copy_flawed_papers=True)
        output_base_dir: Base output directory for papers (default: same as CSV output directory)
        venue_name: Venue name for paper output structure (default: 'NeurIPS2024')
        copy_csv_only: If True, only copy CSV files, not .md files (default: False)
    """
    # --- Step 1: Load and filter categorized flaws ---
    print(f"Step 1: Loading and filtering categorized flaws...")
    print(f"  Reading from: {categorized_csv_path}")
    print(f"  Filtering for category: {category_id} (strict match only)")
    
    try:
        categories_df = pd.read_csv(categorized_csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at {categorized_csv_path}")
        return
    
    print(f"  Loaded {len(categories_df)} total flaws")
    
    # Apply strict category filter
    filtered_df = strict_category_filter(categories_df, category_id)
    print(f"  Found {len(filtered_df)} flaws with exact category match '{category_id}'")
    
    if len(filtered_df) == 0:
        print(f"Warning: No flaws found for category '{category_id}'")
        return
    
    # --- Step 2: Merge flaw descriptions (optional) ---
    if include_descriptions and base_data_dir and venue_folder_name:
        print(f"\nStep 2: Merging flaw descriptions...")
        flawed_papers_dir = os.path.join(base_data_dir, 'flawed_papers', venue_folder_name)
        descriptions_path = os.path.join(flawed_papers_dir, 'flawed_papers_global_summary.csv')
        
        try:
            descriptions_df = pd.read_csv(descriptions_path)[['openreview_id', 'flaw_id', 'flaw_description']]
            filtered_df = pd.merge(filtered_df, descriptions_df, on=['openreview_id', 'flaw_id'], how='left')
            print(f"  Merged descriptions. Shape: {filtered_df.shape}")
        except FileNotFoundError:
            print(f"  Warning: Flaw description file not found at {descriptions_path}")
            filtered_df['flaw_description'] = None
    else:
        filtered_df['flaw_description'] = None
    
    # --- Step 3: Merge metareview data (optional) ---
    if include_metareviews and base_data_dir and venue_folder_name:
        print(f"\nStep 3: Collecting and merging metareview data...")
        metareviews_dir = os.path.join(base_data_dir, 'metareviews')
        mention_data = collect_metareview_data(metareviews_dir, venue_folder_name, model_name)
        
        if mention_data:
            mention_df = pd.DataFrame.from_dict(mention_data, orient='index').reset_index()
            mention_df.rename(columns={'level_0': 'openreview_id', 'level_1': 'flaw_id'}, inplace=True)
            filtered_df = pd.merge(filtered_df, mention_df, on=['openreview_id', 'flaw_id'], how='left')
            print(f"  Merged metareview data. Shape: {filtered_df.shape}")
        else:
            filtered_df['is_flaw_mentioned'] = None
            filtered_df['mention_reasoning'] = None
    else:
        filtered_df['is_flaw_mentioned'] = None
        filtered_df['mention_reasoning'] = None
    
    # --- Step 4: Merge LLM review data (optional) ---
    if include_reviews and base_data_dir and venue_folder_name:
        print(f"\nStep 4: Collecting and merging LLM review data...")
        reviews_dir = os.path.join(base_data_dir, 'reviews')
        review_data = collect_llm_review_data(reviews_dir, venue_folder_name, model_name)
        
        if review_data:
            review_df = pd.DataFrame.from_dict(review_data, orient='index').reset_index()
            review_df.rename(columns={'level_0': 'openreview_id', 'level_1': 'flaw_id'}, inplace=True)
            filtered_df = pd.merge(filtered_df, review_df, on=['openreview_id', 'flaw_id'], how='left')
            print(f"  Merged LLM review data. Shape: {filtered_df.shape}")
        else:
            filtered_df['llm_review'] = None
    else:
        filtered_df['llm_review'] = None
    
    # --- Step 5: Sample if requested ---
    if n_samples is not None and n_samples > 0:
        print(f"\nStep 5: Sampling {n_samples} flaws...")
        if len(filtered_df) <= n_samples:
            print(f"  Only {len(filtered_df)} flaws available, using all of them")
        else:
            filtered_df = filtered_df.sample(n=min(n_samples, len(filtered_df)), random_state=random_seed)
            print(f"  Sampled {len(filtered_df)} flaws")
    
    # --- Step 6: Reorder columns and save ---
    print(f"\nStep 6: Saving filtered data...")
    final_columns = [
        'openreview_id', 'flaw_id', 'category_ids', 'flaw_description',
        'llm_review', 'is_flaw_mentioned', 'mention_reasoning'
    ]
    
    # Ensure all columns exist
    for col in final_columns:
        if col not in filtered_df.columns:
            filtered_df[col] = None
    
    # Reorder columns
    final_df = filtered_df[final_columns]
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    
    # --- Step 7: Copy papers if requested ---
    copied_count = 0
    flawed_copied_count = 0
    total_flaws_copied = 0
    
    # Determine output base directory
    if (copy_papers and papers_source_dir) or (copy_flawed_papers and flawed_papers_source_dir):
        if output_base_dir is None:
            # Use the parent directory of the CSV output
            output_base_dir = output_path.parent
        else:
            output_base_dir = Path(output_base_dir)
        
        # Get unique openreview_ids from the filtered data
        unique_openreview_ids = set(final_df['openreview_id'].dropna().unique())
    
    if copy_papers and papers_source_dir:
        print(f"\nStep 7: Copying corresponding papers (latest versions)...")
        copied_count, not_found_ids = copy_papers_for_category(
            openreview_ids=unique_openreview_ids,
            papers_source_dir=papers_source_dir,
            output_base_dir=output_base_dir,
            category_id=category_id,
            venue_name=venue_name,
            subdirectory='latest'
        )
        
        if not_found_ids:
            print(f"  Papers not found: {len(not_found_ids)}")
    
    # --- Step 8: Copy flawed papers if requested ---
    if copy_flawed_papers and flawed_papers_source_dir:
        if copy_csv_only:
            print(f"\nStep 8: Copying flawed paper CSV files only (filtered by category)...")
        else:
            print(f"\nStep 8: Copying flawed paper versions (filtered by category)...")
        flawed_copied_count, flawed_not_found_ids, total_flaws_copied = copy_flawed_papers_for_category(
            filtered_df=final_df,
            flawed_papers_source_dir=flawed_papers_source_dir,
            output_base_dir=output_base_dir,
            category_id=category_id,
            venue_name=venue_name,
            copy_csv_only=copy_csv_only
        )
        
        if flawed_not_found_ids:
            print(f"  Flawed papers not found: {len(flawed_not_found_ids)}")
    
    print(f"\n{'='*60}")
    print(f"Successfully created filtered dataset!")
    print(f"  Category: {category_id} (strict match)")
    print(f"  Total rows: {len(final_df)}")
    print(f"  Output file: {output_path}")
    if copy_papers and papers_source_dir:
        print(f"  Latest papers copied: {copied_count} to {output_base_dir}/{venue_name}/{category_id}/latest/")
    if copy_flawed_papers and flawed_papers_source_dir:
        print(f"  Flawed papers copied: {flawed_copied_count} papers with {total_flaws_copied} flaws to {output_base_dir}/{venue_name}/{category_id}/planted_error/")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter flaws by category with strict matching (exact category only, not comma-separated lists)"
    )
    parser.add_argument(
        '--categorized_csv',
        type=str,
        required=True,
        help='Path to categorized_flaw_cleaned.csv'
    )
    parser.add_argument(
        '--category_id',
        type=str,
        required=True,
        help='Category ID to filter for (e.g., "2a", "1b"). Use "all" to process all categories found in the CSV.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (e.g., "2a_sampled_flaws.csv"). If not provided, will auto-generate from category_id. If category_id is "all", must be a directory path.'
    )
    parser.add_argument(
        '--venue_folder_name',
        type=str,
        default=None,
        help='Venue folder name (e.g., "NeurIPS2024_latest_flawed_papers_v1"). Required if including descriptions/metareviews/reviews.'
    )
    parser.add_argument(
        '--base_data_dir',
        type=str,
        default=None,
        help='Base data directory (e.g., "../data"). Required if including descriptions/metareviews/reviews.'
    )
    parser.add_argument(
        '--no_descriptions',
        action='store_true',
        help='Skip merging flaw descriptions'
    )
    parser.add_argument(
        '--no_metareviews',
        action='store_true',
        help='Skip merging metareview data'
    )
    parser.add_argument(
        '--no_reviews',
        action='store_true',
        help='Skip merging LLM review data'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='o3',
        help='Model name for metareviews/reviews (default: "o3")'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=None,
        help='Optional: Number of samples to take (default: None = no sampling)'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )
    parser.add_argument(
        '--copy_papers',
        action='store_true',
        help='Copy corresponding paper directories to output structure'
    )
    parser.add_argument(
        '--papers_source_dir',
        type=str,
        default=None,
        help='Source directory for accepted papers (e.g., "data/original_papers/NeurIPS2024_latest/accepted"). Required if --copy_papers is used.'
    )
    parser.add_argument(
        '--copy_flawed_papers',
        action='store_true',
        help='Copy flawed paper versions to planted_error/ subdirectory'
    )
    parser.add_argument(
        '--flawed_papers_source_dir',
        type=str,
        default=None,
        help='Source directory for flawed papers (e.g., "data/flawed_papers/NeurIPS2024_latest_flawed_papers_v1/accepted"). Required if --copy_flawed_papers is used.'
    )
    parser.add_argument(
        '--copy_csv_only',
        action='store_true',
        help='When copying flawed papers, only copy CSV files, not .md files'
    )
    parser.add_argument(
        '--output_base_dir',
        type=str,
        default=None,
        help='Base output directory for papers (default: same as CSV output directory). Papers will be copied to {output_base_dir}/NeurIPS2024/{category_id}/latest/ and planted_error/'
    )
    parser.add_argument(
        '--venue_name',
        type=str,
        default='NeurIPS2024',
        help='Venue name for paper output structure (default: "NeurIPS2024")'
    )
    
    args = parser.parse_args()
    
    # Validate copy_papers arguments
    if args.copy_papers and not args.papers_source_dir:
        parser.error("--papers_source_dir is required when --copy_papers is used")
    
    # Validate copy_flawed_papers arguments
    if args.copy_flawed_papers and not args.flawed_papers_source_dir:
        parser.error("--flawed_papers_source_dir is required when --copy_flawed_papers is used")
    
    # Handle "all" category_id
    if args.category_id.lower() == 'all':
        print("Processing all categories...")
        all_categories = get_all_unique_categories(args.categorized_csv)
        
        if not all_categories:
            print("Error: Could not extract categories from CSV")
            return
        
        print(f"Found {len(all_categories)} unique categories: {', '.join(all_categories)}")
        
        # Determine output directory
        if args.output:
            output_dir = Path(args.output)
        else:
            # Default to current directory
            output_dir = Path(".")
        
        # Ensure output is a directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each category
        for category_id in all_categories:
            print(f"\n{'='*60}")
            print(f"Processing category: {category_id}")
            print(f"{'='*60}")
            
            # Auto-generate output filename
            output_file = output_dir / f"{category_id}_sampled_flaws.csv"
            
            filter_and_aggregate(
                categorized_csv_path=args.categorized_csv,
                category_id=category_id,
                output_path=str(output_file),
                venue_folder_name=args.venue_folder_name,
                base_data_dir=args.base_data_dir,
                include_descriptions=not args.no_descriptions,
                include_metareviews=not args.no_metareviews,
                include_reviews=not args.no_reviews,
                model_name=args.model_name,
                n_samples=args.n_samples,
                random_seed=args.random_seed,
                copy_papers=args.copy_papers,
                papers_source_dir=args.papers_source_dir,
                copy_flawed_papers=args.copy_flawed_papers,
                flawed_papers_source_dir=args.flawed_papers_source_dir,
                output_base_dir=args.output_base_dir,
                venue_name=args.venue_name,
                copy_csv_only=args.copy_csv_only
            )
        
        print(f"\n{'='*60}")
        print(f"Completed processing all {len(all_categories)} categories!")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")
    else:
        # Single category processing
        # Auto-generate output filename if not provided
        if args.output is None:
            output_path = f"{args.category_id}_sampled_flaws.csv"
        else:
            output_path = args.output
        
        filter_and_aggregate(
            categorized_csv_path=args.categorized_csv,
            category_id=args.category_id,
            output_path=output_path,
            venue_folder_name=args.venue_folder_name,
            base_data_dir=args.base_data_dir,
            include_descriptions=not args.no_descriptions,
            include_metareviews=not args.no_metareviews,
            include_reviews=not args.no_reviews,
            model_name=args.model_name,
            n_samples=args.n_samples,
            random_seed=args.random_seed,
            copy_papers=args.copy_papers,
            papers_source_dir=args.papers_source_dir,
            copy_flawed_papers=args.copy_flawed_papers,
            flawed_papers_source_dir=args.flawed_papers_source_dir,
            output_base_dir=args.output_base_dir,
            venue_name=args.venue_name,
            copy_csv_only=args.copy_csv_only
        )


if __name__ == '__main__':
    main()

