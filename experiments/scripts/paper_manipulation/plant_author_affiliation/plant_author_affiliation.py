#!/usr/bin/env python3
"""
Script to plant author and affiliation modifications in papers.

This script creates three versions of each paper:
1. Original (unchanged)
2. Good version: Adds a prestigious author/affiliation from ICLR
3. Bad version: Adds a random unknown author/affiliation with no significant academic presence

The modified papers are saved in:
- authors_affiliation_good/
- authors_affiliation_bad/
"""

import os
import random
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
from tqdm import tqdm


# Prestigious authors and affiliations from ICLR (examples - customize as needed)
PRESTIGIOUS_AUTHORS = [
    ("Yoshua Bengio", "Mila, Université de Montréal"),
    ("Geoffrey Hinton", "University of Toronto, Google"),
    ("Yann LeCun", "New York University, Meta AI"),
    ("Pieter Abbeel", "UC Berkeley, Covariant"),
    ("Sergey Levine", "UC Berkeley"),
    ("Ruslan Salakhutdinov", "Carnegie Mellon University, Apple"),
    ("Ian Goodfellow", "Google Brain"),
    ("Quoc Le", "Google DeepMind"),
    ("Raia Hadsell", "DeepMind"),
    ("Oriol Vinyals", "Google DeepMind"),
    ("David Silver", "DeepMind"),
    ("Drew Bagnell", "Carnegie Mellon University"),
    ("Michael I. Jordan", "UC Berkeley"),
    ("Andrew Ng", "Stanford University"),
    ("Fei-Fei Li", "Stanford University"),
    ("Jitendra Malik", "UC Berkeley"),
    ("Trevor Darrell", "UC Berkeley"),
    ("Trevor Hastie", "Stanford University"),
    ("Zoubin Ghahramani", "University of Cambridge, Uber AI"),
    ("Max Welling", "University of Amsterdam, Microsoft Research"),
]

# Unknown authors - people with no significant academic presence or citations
# These are deliberately uncommon name combinations to ensure they are truly unknown
UNKNOWN_AUTHORS = [
    ("Zephyr Kumbirai", "University of Gaborone, Botswana"),
    ("Quincy Ntuli", "University of Mbabane, Eswatini"),
    ("Xolani Dlamini", "University of Lobamba, Eswatini"),
    ("Yasmin Al-Rashid", "University of Nuuk, Greenland"),
    ("Zara Kovalenko", "University of Pristina, Kosovo"),
    ("Kairos Mwangi", "University of Nuku'alofa, Tonga"),
    ("Thalia Andrianos", "University of Apia, Samoa"),
    ("Zenon Petrovic", "University of Podgorica, Montenegro"),
    ("Quinn O'Brien", "University of Palikir, Micronesia"),
    ("Xara Nkomo", "University of Moroni, Comoros"),
    ("Yuki Tanaka", "University of Majuro, Marshall Islands"),
    ("Zephyr Mbeki", "University of Tarawa, Kiribati"),
    ("Quincy Okafor", "University of Port Vila, Vanuatu"),
    ("Xolani Ndlovu", "University of Funafuti, Tuvalu"),
    ("Yara Suleiman", "University of Yaren, Nauru"),
    ("Zara Mkhize", "University of Ngerulmud, Palau"),
    ("Kairos Okonkwo", "University of Honiara, Solomon Islands"),
    ("Thalia Stavros", "University of Dili, East Timor"),
    ("Zenon Petrov", "University of Basseterre, Saint Kitts and Nevis"),
    ("Quinn O'Sullivan", "University of Castries, Saint Lucia"),
]


def get_random_prestigious_author() -> Tuple[str, str]:
    """Return a random prestigious author and affiliation."""
    return random.choice(PRESTIGIOUS_AUTHORS)


def get_random_unknown_author() -> Tuple[str, str]:
    """Return a random unknown author with no significant academic presence."""
    return random.choice(UNKNOWN_AUTHORS)


def add_author_section_to_paper(paper_content: str, author_name: str, affiliation: str) -> str:
    """
    Add author and affiliation section to the paper markdown.
    
    Inserts after the title and before the Abstract section.
    If authors already exist, appends the new author.
    """
    lines = paper_content.split('\n')
    new_lines = []
    
    # Find the title (first # heading) and Abstract section
    title_idx = None
    abstract_idx = None
    author_section_start = None
    author_section_end = None
    
    for i, line in enumerate(lines):
        # Find the first # heading (title)
        if title_idx is None and line.strip().startswith('# ') and not line.strip().startswith('## '):
            title_idx = i
        # Find the Abstract section
        if line.strip().startswith('## Abstract'):
            abstract_idx = i
            break
        # Check if there's already an author section (look for patterns like "Author Name" or affiliations with *)
        if title_idx is not None and abstract_idx is None:
            # Look for author-like patterns: bold text, italic text (affiliations), or email patterns
            if (line.strip().startswith('**') and line.strip().endswith('**')) or \
               (line.strip().startswith('*') and '@' not in line and 'http' not in line):
                if author_section_start is None:
                    author_section_start = i
                author_section_end = i + 1
    
    # If we found both title and abstract, insert author section
    if title_idx is not None and abstract_idx is not None:
        # Add title
        new_lines.extend(lines[:title_idx + 1])
        
        # Check if there's already an author section
        if author_section_start is not None and author_section_end is not None:
            # Append new author to existing author section
            # First, add everything up to the author section
            new_lines.extend(lines[title_idx + 1:author_section_end])
            # Add the new author as a new entry (typical format: author on one line, affiliation on next)
            new_lines.append(f"**{author_name}**")
            new_lines.append(f"*{affiliation}*")
            # Add any blank lines that might exist before abstract
            # Find where the author section ends and abstract begins
            remaining_lines = lines[author_section_end:abstract_idx]
            # Add remaining lines (should be blank lines or similar)
            new_lines.extend(remaining_lines)
            # Ensure we have the abstract section
            new_lines.extend(lines[abstract_idx:])
        else:
            # No existing author section, add new one
            new_lines.append('')
            new_lines.append(f"**{author_name}**")
            new_lines.append(f"*{affiliation}*")
            new_lines.append('')
            # Add the rest (Abstract onwards)
            new_lines.extend(lines[abstract_idx:])
    else:
        # Fallback: if structure is different, try to add after first line
        if lines:
            new_lines.append(lines[0])  # Title
            new_lines.append('')
            new_lines.append(f"**{author_name}**")
            new_lines.append(f"*{affiliation}*")
            new_lines.append('')
            new_lines.extend(lines[1:])
        else:
            # Empty file, just add the author section
            new_lines.append(f"**{author_name}**")
            new_lines.append(f"*{affiliation}*")
            new_lines.append('')
    
    return '\n'.join(new_lines)


def create_modified_paper(
    source_paper_path: Path,
    output_paper_path: Path,
    author_name: str,
    affiliation: str
) -> bool:
    """
    Create a modified version of the paper with added author/affiliation.
    
    Also copies the entire directory structure including figures.
    """
    try:
        # Read the original paper
        with open(source_paper_path, 'r', encoding='utf-8') as f:
            paper_content = f.read()
        
        # Add author section
        modified_content = add_author_section_to_paper(paper_content, author_name, affiliation)
        
        # Create output directory
        output_paper_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write modified paper
        with open(output_paper_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        # Copy the entire directory structure (including figures, etc.)
        source_dir = source_paper_path.parent
        output_dir = output_paper_path.parent
        
        # Copy all files except paper.md (which we already modified)
        for item in source_dir.iterdir():
            if item.name == 'paper.md':
                continue  # Skip, we already wrote the modified version
            if item.is_dir():
                shutil.copytree(item, output_dir / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, output_dir / item.name)
        
        return True
    except Exception as e:
        print(f"Error processing {source_paper_path}: {e}")
        return False


def process_papers(
    base_dir: Path,
    output_good_dir: Path,
    output_bad_dir: Path,
    paper_ids: List[str] = None,
    random_seed: int = None
):
    """
    Process all papers and create good/bad versions.
    
    Args:
        base_dir: Directory containing the latest/ folder with papers
        output_good_dir: Output directory for good author/affiliation versions
        output_bad_dir: Output directory for bad author/affiliation versions
        paper_ids: Optional list of paper IDs to process. If None, processes all.
        random_seed: Random seed for reproducibility
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    latest_dir = base_dir / "latest"
    if not latest_dir.exists():
        print(f"Error: {latest_dir} does not exist")
        return
    
    # Find all paper directories
    paper_dirs = [d for d in latest_dir.iterdir() if d.is_dir()]
    
    if paper_ids:
        # Filter to only specified paper IDs
        # Convert to set for faster lookup and normalize paper IDs
        paper_ids_set = {str(pid).strip() for pid in paper_ids}
        paper_dirs = [d for d in paper_dirs if d.name.split('_')[0] in paper_ids_set]
    
    print(f"Processing {len(paper_dirs)} papers...")
    
    # Track statistics
    stats = {
        'total': len(paper_dirs),
        'good_success': 0,
        'bad_success': 0,
        'good_failed': 0,
        'bad_failed': 0,
    }
    
    # Process each paper
    for paper_dir in tqdm(paper_dirs, desc="Processing papers"):
        # Extract paper ID from directory name (format: paperid_arxiv_id or similar)
        paper_id = paper_dir.name.split('_')[0]
        paper_md = paper_dir / "structured_paper_output" / "paper.md"
        
        if not paper_md.exists():
            print(f"Warning: {paper_md} not found, skipping {paper_id}")
            continue
        
        # Get random authors
        good_author, good_affiliation = get_random_prestigious_author()
        bad_author, bad_affiliation = get_random_unknown_author()
        
        # Create good version
        good_output_md = output_good_dir / paper_dir.name / "structured_paper_output" / "paper.md"
        if create_modified_paper(paper_md, good_output_md, good_author, good_affiliation):
            stats['good_success'] += 1
        else:
            stats['good_failed'] += 1
        
        # Create bad version
        bad_output_md = output_bad_dir / paper_dir.name / "structured_paper_output" / "paper.md"
        if create_modified_paper(paper_md, bad_output_md, bad_author, bad_affiliation):
            stats['bad_success'] += 1
        else:
            stats['bad_failed'] += 1
    
    # Print statistics
    print("\n" + "="*60)
    print("Processing Statistics:")
    print(f"  Total papers: {stats['total']}")
    print(f"  Good versions - Success: {stats['good_success']}, Failed: {stats['good_failed']}")
    print(f"  Bad versions - Success: {stats['bad_success']}, Failed: {stats['bad_failed']}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Plant author and affiliation modifications in papers"
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        required=True,
        help='Base directory containing the latest/ folder (e.g., data/ICLR2024)'
    )
    parser.add_argument(
        '--output_good_dir',
        type=str,
        default=None,
        help='Output directory for good author/affiliation versions (default: base_dir/authors_affiliation_good)'
    )
    parser.add_argument(
        '--output_bad_dir',
        type=str,
        default=None,
        help='Output directory for bad author/affiliation versions (default: base_dir/authors_affiliation_bad)'
    )
    parser.add_argument(
        '--paper_ids',
        type=str,
        nargs='+',
        default=None,
        help='Optional: specific paper IDs to process (e.g., ViNe1fjGME 0akLDTFR9x)'
    )
    parser.add_argument(
        '--paper_ids_file',
        type=str,
        default=None,
        help='Optional: CSV file with paper IDs in first column (e.g., filtered_pairs.csv)'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    # Set default output directories
    if args.output_good_dir is None:
        output_good_dir = base_dir / "authors_affiliation_good"
    else:
        output_good_dir = Path(args.output_good_dir)
    
    if args.output_bad_dir is None:
        output_bad_dir = base_dir / "authors_affiliation_bad"
    else:
        output_bad_dir = Path(args.output_bad_dir)
    
    # Get paper IDs if specified
    paper_ids = args.paper_ids
    if args.paper_ids_file:
        try:
            df = pd.read_csv(args.paper_ids_file)
            # Assume first column contains paper IDs (usually 'paperid')
            first_col = df.columns[0]
            if first_col.lower() in ['paperid', 'paper_id', 'id']:
                paper_ids = df[first_col].astype(str).tolist()
            else:
                # Try to find paperid column
                paperid_col = None
                for col in df.columns:
                    if 'paperid' in col.lower() or 'paper_id' in col.lower():
                        paperid_col = col
                        break
                if paperid_col:
                    paper_ids = df[paperid_col].astype(str).tolist()
                else:
                    # Fallback to first column
                    paper_ids = df.iloc[:, 0].astype(str).tolist()
            print(f"Loaded {len(paper_ids)} paper IDs from {args.paper_ids_file}")
        except Exception as e:
            print(f"Error reading paper IDs file: {e}")
            return
    
    # Create output directories
    output_good_dir.mkdir(parents=True, exist_ok=True)
    output_bad_dir.mkdir(parents=True, exist_ok=True)
    
    # Process papers
    process_papers(
        base_dir=base_dir,
        output_good_dir=output_good_dir,
        output_bad_dir=output_bad_dir,
        paper_ids=paper_ids,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()

