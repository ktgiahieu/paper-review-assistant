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
import numpy as np
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_json_file(filepath):
    """Safely loads a single JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Warning: Could not read or parse {filepath}: {e}")
        return None


def prune_appendix_from_markdown(md_content: str) -> str:
    """
    Remove appendix sections from a markdown paper.
    
    This function finds the "# References" section and truncates everything
    after the last reference entry (bibliography entry with citation marker).
    
    Args:
        md_content: The full markdown content of the paper
    
    Returns:
        The markdown content with everything after the last reference removed
    """
    lines = md_content.split('\n')
    
    # Find "# References" section
    references_idx = None
    for i, line in enumerate(lines):
        # Match "# References" with optional reference anchor
        if re.match(r'^#\s+References\s*(\[.*\])?', line, re.IGNORECASE):
            references_idx = i
            break
    
    # If we found References, find where the References section actually ends
    # by looking for the end of bibliography entries (lines with citation markers)
    if references_idx is not None:
        # Pattern to match bibliography citation markers like (@author2024title) or (@Author2024Title)
        citation_pattern = r'\(@[a-zA-Z0-9_]+\)'
        
        # Find the last line that contains a citation marker
        # This indicates where the bibliography entries end
        last_citation_idx = None
        for i in range(references_idx + 1, len(lines)):
            line = lines[i]
            # Check if this line contains a citation marker
            if re.search(citation_pattern, line):
                last_citation_idx = i
        
        # If we found citations, truncate after the last one
        if last_citation_idx is not None:
            pruned_lines = lines[:last_citation_idx + 1]
            result = '\n'.join(pruned_lines)
            if not result.endswith('\n'):
                result += '\n'
            return result
        else:
            # References section found but no citations - just return up to References heading
            pruned_lines = lines[:references_idx + 1]
            result = '\n'.join(pruned_lines)
            if not result.endswith('\n'):
                result += '\n'
            return result
    
    # If References not found, return original content unchanged
    return md_content


def prune_appendix_from_file(file_path: Path, prune_appendix: bool = True) -> bool:
    """
    Prune appendix from a markdown file in place.
    
    Args:
        file_path: Path to the markdown file
        prune_appendix: Whether to prune the appendix (if False, does nothing)
    
    Returns:
        True if pruning was successful or not needed, False otherwise
    """
    if not prune_appendix:
        return True
    
    if not file_path.exists() or not file_path.is_file():
        return False
    
    if not file_path.suffix == '.md':
        return False
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Prune the appendix
        pruned_content = prune_appendix_from_markdown(content)
        
        # Only write if content actually changed
        if pruned_content != content:
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(pruned_content)
            return True
        else:
            # No changes needed
            return True
    except Exception as e:
        print(f"  Warning: Could not prune appendix from {file_path.name}: {e}")
        return False


def analyze_file_sizes(directory, pattern='paper.md', recursive=True):
    """
    Analyze file sizes (word/character counts) for markdown files in a directory.
    
    Args:
        directory: Directory to search for files
        pattern: Filename pattern to match (default: 'paper.md')
        recursive: Whether to search recursively (default: True)
    
    Returns:
        List of dictionaries with file info: {'path': str, 'chars': int, 'words': int, 'lines': int}
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    file_stats = []
    
    # Find all matching files
    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            char_count = len(content)
            word_count = len(content.split())
            line_count = len(content.splitlines())
            
            # Get relative path for cleaner output
            try:
                rel_path = str(file_path.relative_to(directory))
            except ValueError:
                rel_path = str(file_path)
            
            file_stats.append({
                'path': rel_path,
                'full_path': str(file_path),
                'chars': char_count,
                'words': word_count,
                'lines': line_count
            })
        except Exception as e:
            print(f"  Warning: Could not analyze {file_path}: {e}")
    
    return file_stats


def plot_file_size_histogram(file_stats, metric='words', output_path=None, show_outliers=True):
    """
    Plot histogram of file sizes and identify outliers.
    
    Args:
        file_stats: List of dictionaries with file stats (from analyze_file_sizes)
        metric: Metric to plot ('words', 'chars', or 'lines', default: 'words')
        output_path: Path to save the plot (default: None, displays instead)
        show_outliers: Whether to print outlier information (default: True)
    
    Returns:
        List of outlier file paths
    """
    if not file_stats:
        print("  No files found to analyze")
        return []
    
    # Extract the metric values
    values = [stat[metric] for stat in file_stats]
    
    if not values:
        print(f"  No {metric} data found")
        return []
    
    # Calculate statistics
    mean_val = np.mean(values)
    median_val = np.median(values)
    std_val = np.std(values)
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    
    # Identify outliers using IQR method (values beyond 1.5 * IQR from Q1/Q3)
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = [stat for stat in file_stats if stat[metric] > upper_bound]
    
    # Print statistics
    print(f"\n  File size statistics ({metric}):")
    print(f"    Total files: {len(file_stats)}")
    print(f"    Mean: {mean_val:.1f}")
    print(f"    Median: {median_val:.1f}")
    print(f"    Std Dev: {std_val:.1f}")
    print(f"    Q1: {q1:.1f}, Q3: {q3:.1f}, IQR: {iqr:.1f}")
    print(f"    Outlier threshold: {upper_bound:.1f} (upper bound)")
    print(f"    Found {len(outliers)} outliers")
    
    if show_outliers and outliers:
        print(f"\n  Outlier files ({metric} > {upper_bound:.1f}):")
        # Sort by metric value (descending)
        outliers_sorted = sorted(outliers, key=lambda x: x[metric], reverse=True)
        for i, outlier in enumerate(outliers_sorted[:20], 1):  # Show top 20
            print(f"    {i}. {outlier['path']}: {outlier[metric]:,} {metric}")
        if len(outliers) > 20:
            print(f"    ... and {len(outliers) - 20} more outliers")
    
    # Plot histogram if matplotlib is available
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f}')
        plt.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.1f}')
        plt.axvline(upper_bound, color='orange', linestyle='--', label=f'Outlier threshold: {upper_bound:.1f}')
        
        # Mark outliers
        outlier_values = [stat[metric] for stat in outliers]
        if outlier_values:
            plt.scatter(outlier_values, [0] * len(outlier_values), 
                       color='red', marker='x', s=100, zorder=5, label=f'Outliers ({len(outliers)})')
        
        plt.xlabel(f'File size ({metric})')
        plt.ylabel('Number of files')
        plt.title(f'Distribution of file sizes ({metric})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  Histogram saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    else:
        print("  Matplotlib not available, skipping histogram plot")
    
    return [out['full_path'] for out in outliers]


def should_filter_limitation_only_csv(csv_path, flaw_id=None):
    """
    Check if a modifications_summary.csv should be filtered out.
    
    Filters out CSVs where:
    - num_modifications == 1
    - The single modification's target_heading contains "limitation" (case-insensitive)
    
    Args:
        csv_path: Path to the modifications_summary.csv file
        flaw_id: Optional flaw_id to filter to (if None, checks all flaws in CSV)
    
    Returns:
        True if the CSV should be filtered out, False otherwise
    """
    if not csv_path.exists():
        return False
    
    try:
        modifications_df = pd.read_csv(csv_path)
        
        # Filter to specific flaw_id if provided
        if flaw_id is not None:
            modifications_df = modifications_df[modifications_df['flaw_id'] == flaw_id]
            if modifications_df.empty:
                return False
        
        # Check each row
        for _, row in modifications_df.iterrows():
            num_modifications = row.get('num_modifications', 0)
            
            # Only filter if there's exactly 1 modification
            if num_modifications != 1:
                continue
            
            # Parse the llm_generated_modifications JSON
            try:
                modifications_json = row.get('llm_generated_modifications', '[]')
                if pd.isna(modifications_json) or modifications_json == '':
                    continue
                
                modifications = json.loads(modifications_json)
                
                # Check if there's exactly one modification and it targets "limitation"
                if len(modifications) == 1:
                    target_heading = modifications[0].get('target_heading', '')
                    if 'limitation' in str(target_heading).lower():
                        return True
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                # If we can't parse, don't filter it out (safer to include)
                continue
        
        return False
    except Exception as e:
        # If we can't read the CSV, don't filter it out (safer to include)
        print(f"  Warning: Could not check CSV {csv_path.name}: {e}")
        return False


def filter_limitation_only_entries(filtered_df, flawed_papers_source_dir):
    """
    Filter out entries from filtered_df where the corresponding CSV has only 1 modification
    targeting a "limitation" heading.
    
    Args:
        filtered_df: DataFrame with columns 'openreview_id' and 'flaw_id'
        flawed_papers_source_dir: Source directory containing flawed papers with modifications_summary.csv files
    
    Returns:
        Filtered DataFrame with limitation-only entries removed
    """
    flawed_papers_source_dir = Path(flawed_papers_source_dir)
    
    if not flawed_papers_source_dir.exists():
        print("  Warning: Flawed papers source directory not found, skipping limitation-only filter")
        return filtered_df
    
    print(f"  Filtering out limitation-only modifications...")
    original_count = len(filtered_df)
    
    # Group by openreview_id to read each CSV only once
    filtered_out_count = 0
    rows_to_keep = []
    
    for openreview_id, group_df in filtered_df.groupby('openreview_id'):
        # Find the source directory for this paper
        source_dirs = [d for d in flawed_papers_source_dir.iterdir() 
                      if d.is_dir() and d.name.startswith(openreview_id)]
        
        if not source_dirs:
            # If we can't find the directory, keep all rows for this paper
            rows_to_keep.extend(group_df.index.tolist())
            continue
        
        source_dir = source_dirs[0]
        
        # Look for modifications_summary.csv
        csv_filename = f"{openreview_id}_modifications_summary.csv"
        csv_path = source_dir / csv_filename
        
        if not csv_path.exists():
            # Try alternative naming patterns
            csv_files = list(source_dir.glob("*_modifications_summary.csv"))
            if csv_files:
                csv_path = csv_files[0]
            else:
                # If CSV not found, keep all rows for this paper
                rows_to_keep.extend(group_df.index.tolist())
                continue
        
        # Check each flaw_id in this group
        for idx, row in group_df.iterrows():
            flaw_id = row['flaw_id']
            
            # Check if this CSV should be filtered out
            if should_filter_limitation_only_csv(csv_path, flaw_id=flaw_id):
                filtered_out_count += 1
            else:
                rows_to_keep.append(idx)
    
    # Create filtered DataFrame
    filtered_df = filtered_df.loc[rows_to_keep].copy()
    
    print(f"  Filtered out {filtered_out_count} limitation-only entries")
    print(f"  Remaining: {len(filtered_df)} entries (from {original_count} original)")
    
    return filtered_df


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
    subdirectory='latest',
    prune_appendix=False
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
        prune_appendix: If True, prune appendix sections from paper.md files (default: False)
    
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
            
            # Prune appendix if requested
            if prune_appendix:
                # Look for paper.md in common locations
                paper_md_paths = [
                    dest_dir / 'structured_paper_output' / 'paper.md',
                    dest_dir / 'paper.md'
                ]
                for paper_md_path in paper_md_paths:
                    if paper_md_path.exists():
                        prune_appendix_from_file(paper_md_path, prune_appendix=True)
                        break
            
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
    copy_csv_only=False,
    prune_appendix=False
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
        prune_appendix: If True, prune appendix sections from .md files (default: False)
    
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
                            dest_md_file = dest_flawed_papers_dir / f"{flaw_id}.md"
                            shutil.copy2(md_file, dest_md_file)
                            # Prune appendix if requested
                            if prune_appendix:
                                prune_appendix_from_file(dest_md_file, prune_appendix=True)
                            total_flaws_copied += 1
                        else:
                            print(f"\n  Warning: .md file not found for flaw {flaw_id} in {openreview_id}")
                else:
                    # Try to find .md files directly in source directory
                    for flaw_id in filtered_modifications['flaw_id']:
                        flaw_id = str(flaw_id)
                        md_file = source_dir / f"{flaw_id}.md"
                        if md_file.exists():
                            dest_md_file = dest_dir / f"{flaw_id}.md"
                            shutil.copy2(md_file, dest_md_file)
                            # Prune appendix if requested
                            if prune_appendix:
                                prune_appendix_from_file(dest_md_file, prune_appendix=True)
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
    copy_csv_only=False,
    prune_appendix=False,
    filter_limitation_only=False
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
        prune_appendix: If True, prune appendix sections from paper.md files (default: False)
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
    
    # --- Step 5: Filter out limitation-only modifications (optional) ---
    if filter_limitation_only and flawed_papers_source_dir:
        print(f"\nStep 5: Filtering out limitation-only modifications...")
        filtered_df = filter_limitation_only_entries(filtered_df, flawed_papers_source_dir)
    
    # --- Step 6: Sample if requested ---
    if n_samples is not None and n_samples > 0:
        print(f"\nStep 6: Sampling {n_samples} flaws...")
        if len(filtered_df) <= n_samples:
            print(f"  Only {len(filtered_df)} flaws available, using all of them")
        else:
            filtered_df = filtered_df.sample(n=min(n_samples, len(filtered_df)), random_state=random_seed)
            print(f"  Sampled {len(filtered_df)} flaws")
    
    # --- Step 7: Reorder columns and save ---
    print(f"\nStep 7: Saving filtered data...")
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
    
    # --- Step 8: Copy papers if requested ---
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
        if prune_appendix:
            print(f"  Note: Pruning appendix sections from paper.md files")
        copied_count, not_found_ids = copy_papers_for_category(
            openreview_ids=unique_openreview_ids,
            papers_source_dir=papers_source_dir,
            output_base_dir=output_base_dir,
            category_id=category_id,
            venue_name=venue_name,
            subdirectory='latest',
            prune_appendix=prune_appendix
        )
        
        if not_found_ids:
            print(f"  Papers not found: {len(not_found_ids)}")
    
    # --- Step 8: Copy flawed papers if requested ---
    if copy_flawed_papers and flawed_papers_source_dir:
        if copy_csv_only:
            print(f"\nStep 8: Copying flawed paper CSV files only (filtered by category)...")
        else:
            print(f"\nStep 8: Copying flawed paper versions (filtered by category)...")
            if prune_appendix:
                print(f"  Note: Pruning appendix sections from .md files")
        flawed_copied_count, flawed_not_found_ids, total_flaws_copied = copy_flawed_papers_for_category(
            filtered_df=final_df,
            flawed_papers_source_dir=flawed_papers_source_dir,
            output_base_dir=output_base_dir,
            category_id=category_id,
            venue_name=venue_name,
            copy_csv_only=copy_csv_only,
            prune_appendix=prune_appendix
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
    parser.add_argument(
        '--prune_appendix',
        action='store_true',
        help='Prune appendix sections from paper.md files when copying papers. Removes everything after "# References" or common appendix markers like "# Hyperparameter setting", "# Details of evaluation metrics", etc.'
    )
    parser.add_argument(
        '--filter_limitation_only',
        action='store_true',
        help='Before sampling, filter out entries where the modifications_summary.csv has only 1 modification targeting a "limitation" heading. This helps exclude cases where authors only made minor adjustments by mentioning the flaw in the Limitations section. Requires --flawed_papers_source_dir to be set.'
    )
    parser.add_argument(
        '--analyze_file_sizes',
        action='store_true',
        help='After copying/pruning papers, analyze file sizes (words/characters) and plot histogram to identify outliers that may have unpruned appendices'
    )
    parser.add_argument(
        '--size_metric',
        type=str,
        default='words',
        choices=['words', 'chars', 'lines'],
        help='Metric to use for file size analysis (default: "words")'
    )
    parser.add_argument(
        '--size_plot_output',
        type=str,
        default=None,
        help='Path to save the file size histogram plot (default: None, displays instead)'
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
                copy_csv_only=args.copy_csv_only,
                prune_appendix=args.prune_appendix,
                filter_limitation_only=args.filter_limitation_only
            )
        
        # Analyze file sizes if requested
        if args.analyze_file_sizes and args.output_base_dir:
            output_base = Path(args.output_base_dir)
            venue_dir = output_base / args.venue_name
            
            if venue_dir.exists():
                print(f"\n{'='*60}")
                print("Analyzing file sizes...")
                print(f"{'='*60}")
                
                # Analyze each category's directories
                for category_id in all_categories:
                    category_dir = venue_dir / category_id
                    if category_dir.exists():
                        print(f"\nAnalyzing category {category_id}...")
                        
                        # Analyze both latest and planted_error directories
                        for subdir in ['latest', 'planted_error']:
                            subdir_path = category_dir / subdir
                            if subdir_path.exists():
                                print(f"  Analyzing {subdir} directory...")
                                file_stats = analyze_file_sizes(subdir_path, pattern='paper.md', recursive=True)
                                
                                if file_stats:
                                    # Determine plot output path
                                    plot_path = None
                                    if args.size_plot_output:
                                        plot_dir = Path(args.size_plot_output).parent
                                        plot_dir.mkdir(parents=True, exist_ok=True)
                                        plot_name = Path(args.size_plot_output).stem
                                        plot_ext = Path(args.size_plot_output).suffix or '.png'
                                        plot_path = plot_dir / f"{plot_name}_{category_id}_{subdir}{plot_ext}"
                                    else:
                                        # Auto-generate plot path in output directory
                                        plot_dir = category_dir
                                        plot_dir.mkdir(parents=True, exist_ok=True)
                                        plot_path = plot_dir / f"file_size_histogram_{args.size_metric}_{subdir}.png"
                                    
                                    outliers = plot_file_size_histogram(
                                        file_stats,
                                        metric=args.size_metric,
                                        output_path=plot_path,
                                        show_outliers=True
                                    )
                                else:
                                    print(f"    No paper.md files found in {subdir_path}")
        
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
            copy_csv_only=args.copy_csv_only,
            prune_appendix=args.prune_appendix,
            filter_limitation_only=args.filter_limitation_only
        )
        
        # Analyze file sizes if requested
        if args.analyze_file_sizes and args.output_base_dir:
            output_base = Path(args.output_base_dir)
            venue_dir = output_base / args.venue_name / args.category_id
            
            if venue_dir.exists():
                print(f"\n{'='*60}")
                print("Analyzing file sizes...")
                print(f"{'='*60}")
                
                # Analyze both latest and planted_error directories
                for subdir in ['latest', 'planted_error']:
                    subdir_path = venue_dir / subdir
                    if subdir_path.exists():
                        print(f"\nAnalyzing {subdir} directory...")
                        file_stats = analyze_file_sizes(subdir_path, pattern='paper.md', recursive=True)
                        
                        if file_stats:
                            # Determine plot output path
                            plot_path = None
                            if args.size_plot_output:
                                plot_dir = Path(args.size_plot_output).parent
                                plot_dir.mkdir(parents=True, exist_ok=True)
                                plot_name = Path(args.size_plot_output).stem
                                plot_ext = Path(args.size_plot_output).suffix or '.png'
                                plot_path = plot_dir / f"{plot_name}_{subdir}{plot_ext}"
                            elif args.output_base_dir:
                                # Auto-generate plot path in output directory
                                plot_dir = output_base / args.venue_name / args.category_id
                                plot_dir.mkdir(parents=True, exist_ok=True)
                                plot_path = plot_dir / f"file_size_histogram_{args.size_metric}_{subdir}.png"
                            
                            outliers = plot_file_size_histogram(
                                file_stats,
                                metric=args.size_metric,
                                output_path=plot_path,
                                show_outliers=True
                            )
                        else:
                            print(f"  No paper.md files found in {subdir_path}")


if __name__ == '__main__':
    main()

