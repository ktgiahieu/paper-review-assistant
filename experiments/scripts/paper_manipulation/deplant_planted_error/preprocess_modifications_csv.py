#!/usr/bin/env python3
"""
Preprocessing script to update modifications_summary.csv files.

This script:
1. Renames "new_content" to "planted_error_content" in the CSV
2. Extracts "camera_ready_content" by comparing flawed papers with original papers
3. Updates the CSV with the new fields

Usage:
    python preprocess_modifications_csv.py \\
        --data_dir path/to/with_appendix \\
        --conference NeurIPS2024 \\
        [--categories 1a,1b,1c]
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import re
import difflib
import shutil


def clean_heading_text_aggressively(text: str) -> str:
    """Clean heading text for matching."""
    if not text:
        return ""
    # Remove markdown formatting
    text = re.sub(r'^#+\s*', '', text)
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'\*', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()


def find_heading_in_lines(lines, target_heading: str) -> int:
    """Find heading in lines using multiple matching strategies."""
    target_clean = target_heading.strip()
    
    # Strategy 1: Exact match
    for i, line in enumerate(lines):
        if line.strip() == target_clean:
            return i
    
    # Strategy 2: Match after stripping markdown
    semi_cleaned_target = target_clean.strip('#* \t')
    for i, line in enumerate(lines):
        semi_cleaned_line = line.strip().strip('#* \t')
        if semi_cleaned_line == semi_cleaned_target and semi_cleaned_line:
            return i
    
    # Strategy 3: Aggressive cleaning
    aggressively_cleaned_target = clean_heading_text_aggressively(target_clean)
    for i, line in enumerate(lines):
        aggressively_cleaned_line = clean_heading_text_aggressively(line)
        if aggressively_cleaned_line and aggressively_cleaned_target:
            if aggressively_cleaned_line.lower().startswith(aggressively_cleaned_target.lower()):
                return i
    
    return -1


def find_section_by_heading(paper_content: str, target_heading: str) -> tuple:
    """
    Find a section by its heading. Returns (start_line, end_line) or (None, None) if not found.
    """
    if not target_heading or not target_heading.strip():
        return None, None
    
    lines = paper_content.split('\n')
    match_index = find_heading_in_lines(lines, target_heading)
    
    if match_index == -1:
        return None, None
    
    # Find the end of the section by looking for the next heading
    start_line = match_index
    end_line = len(lines)
    
    for i in range(start_line + 1, len(lines)):
        line_to_check = lines[i].strip()
        # A line is considered a heading if it starts with '#' or is fully bolded/italicized
        is_hash_heading = line_to_check.startswith('#')
        is_bold_heading = line_to_check.startswith('**') and line_to_check.endswith('**')
        is_italic_heading = line_to_check.startswith('*') and line_to_check.endswith('*') and not is_bold_heading
        
        if is_hash_heading or is_bold_heading or is_italic_heading:
            end_line = i
            break
    
    return start_line, end_line


def extract_section_content(paper_content: str, target_heading: str) -> str:
    """Extract section content by heading."""
    start_line, end_line = find_section_by_heading(paper_content, target_heading)
    
    if start_line is None or end_line is None:
        return ""
    
    lines = paper_content.split('\n')
    section_lines = lines[start_line:end_line]
    return '\n'.join(section_lines).strip()


def find_content_in_paper(paper_content: str, search_content: str, context_lines: int = 50) -> tuple:
    """
    Find content in paper by searching for a substring.
    Returns (start_line, end_line) of the section containing the content.
    """
    if not search_content or not search_content.strip():
        return None, None
    
    lines = paper_content.split('\n')
    paper_text = paper_content
    
    # Try to find the content
    # First, try to find a unique substring from the search_content
    search_lines = search_content.split('\n')
    
    # Try to find the first substantial line
    first_substantial_line = None
    for line in search_lines[:10]:  # Check first 10 lines
        line_stripped = line.strip()
        if len(line_stripped) > 20:  # Substantial line
            first_substantial_line = line_stripped
            break
    
    if not first_substantial_line:
        return None, None
    
    # Find this line in the paper
    match_line_idx = None
    for i, line in enumerate(lines):
        if first_substantial_line in line or line in first_substantial_line:
            match_line_idx = i
            break
    
    if match_line_idx is None:
        # Try fuzzy matching
        for i, line in enumerate(lines):
            if len(line.strip()) > 20:
                similarity = difflib.SequenceMatcher(None, first_substantial_line.lower(), line.lower()).ratio()
                if similarity > 0.7:
                    match_line_idx = i
                    break
    
    if match_line_idx is None:
        return None, None
    
    # Now find the section boundaries - look backwards for a heading
    start_line = match_line_idx
    for i in range(match_line_idx, max(0, match_line_idx - context_lines), -1):
        line = lines[i].strip()
        if line.startswith('#') or (line.startswith('**') and line.endswith('**')):
            start_line = i
            break
    
    # Find the end - look forward for the next heading
    end_line = len(lines)
    for i in range(match_line_idx + 1, min(len(lines), match_line_idx + context_lines * 2)):
        line = lines[i].strip()
        if line.startswith('#') or (line.startswith('**') and line.endswith('**')):
            end_line = i
            break
    
    return start_line, end_line


def extract_section_using_content_hint(paper_content: str, target_heading: str, content_hint: str = None) -> str:
    """
    Extract section content using heading and optionally a content hint.
    Falls back to content-based search if heading search fails.
    """
    # First try heading-based extraction
    result = extract_section_content(paper_content, target_heading)
    if result:
        return result
    
    # If heading search failed and we have a content hint, try content-based search
    if content_hint:
        start_line, end_line = find_content_in_paper(paper_content, content_hint)
        if start_line is not None and end_line is not None:
            lines = paper_content.split('\n')
            section_lines = lines[start_line:end_line]
            return '\n'.join(section_lines).strip()
    
    return ""


def find_modified_sections_by_diff(flawed_paper: str, original_paper: str, target_heading: str) -> tuple:
    """
    Find modified sections by comparing the two papers using diff.
    Returns (planted_error_content, camera_ready_content) or (None, None) if not found.
    """
    flawed_lines = flawed_paper.split('\n')
    original_lines = original_paper.split('\n')
    
    # Use difflib to find differences
    diff = list(difflib.unified_diff(original_lines, flawed_lines, lineterm='', n=0))
    
    # Find sections that differ
    # First, try to find the heading in both papers
    flawed_start, flawed_end = find_section_by_heading(flawed_paper, target_heading)
    original_start, original_end = find_section_by_heading(original_paper, target_heading)
    
    if flawed_start is not None and flawed_end is not None:
        planted_content = '\n'.join(flawed_lines[flawed_start:flawed_end]).strip()
    else:
        planted_content = None
    
    if original_start is not None and original_end is not None:
        camera_ready_content = '\n'.join(original_lines[original_start:original_end]).strip()
    else:
        camera_ready_content = None
    
    return planted_content, camera_ready_content


def process_modifications_csv(
    csv_path: Path,
    flawed_paper_path: Path,
    original_paper_path: Path
) -> tuple[bool, bool]:
    """
    Process a single modifications_summary.csv file.
    
    Returns:
        (success, is_empty): 
        - success: Whether processing succeeded
        - is_empty: Whether the CSV ended up empty (all modifications removed)
    """
    if not csv_path.exists():
        print(f"  ⚠️ CSV not found: {csv_path}")
        return False, False
    
    if not flawed_paper_path.exists():
        print(f"  ⚠️ Flawed paper not found: {flawed_paper_path}")
        return False, False
    
    if not original_paper_path.exists():
        print(f"  ⚠️ Original paper not found: {original_paper_path}")
        return False, False
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Read papers
    flawed_paper = flawed_paper_path.read_text(encoding='utf-8')
    original_paper = original_paper_path.read_text(encoding='utf-8')
    
    # Process each row
    rows_to_remove = []
    
    for idx, row in df.iterrows():
        flaw_id = row['flaw_id']
        flaw_description = row['flaw_description']
        num_modifications = row['num_modifications']
        
        # Parse the JSON modifications
        try:
            modifications = json.loads(row['llm_generated_modifications'])
        except (json.JSONDecodeError, TypeError):
            tqdm.write(f"  ⚠️ Failed to parse modifications JSON for {flaw_id}, marking row for removal")
            rows_to_remove.append(idx)
            continue
        
        # Update each modification
        updated_mods = []
        for mod in modifications:
            target_heading = mod.get('target_heading', '')
            new_content = mod.get('new_content', '')
            reasoning = mod.get('reasoning', '')
            
            # Strategy 1: Try diff-based approach first (most reliable)
            planted_error_content, camera_ready_content = find_modified_sections_by_diff(
                flawed_paper, original_paper, target_heading
            )
            
            # Strategy 2: If diff approach didn't work, try heading-based extraction
            if not planted_error_content:
                planted_error_content = extract_section_using_content_hint(
                    flawed_paper, 
                    target_heading, 
                    new_content if new_content else None
                )
            
            if not camera_ready_content:
                camera_ready_content = extract_section_content(original_paper, target_heading)
            
            # Strategy 3: If we still don't have planted_error_content and new_content exists, use it
            if not planted_error_content and new_content:
                # new_content might be the actual planted_error_content
                planted_error_content = new_content
            
            # Strategy 4: If we found planted_error_content but not camera_ready_content,
            # try to find a similar section in the original paper using various heading formats
            if planted_error_content and not camera_ready_content:
                heading_text = target_heading.strip('#* \t')
                if heading_text:
                    # Try various heading formats
                    for heading_variant in [
                        target_heading,
                        f"# {heading_text}",
                        f"## {heading_text}",
                        f"### {heading_text}",
                        f"**{heading_text}**",
                        heading_text
                    ]:
                        camera_ready_content = extract_section_content(original_paper, heading_variant)
                        if camera_ready_content:
                            break
                    
                    # If still not found, try content-based search in original paper
                    if not camera_ready_content and planted_error_content:
                        # Try to find similar content in original paper
                        start_line, end_line = find_content_in_paper(original_paper, planted_error_content[:500])
                        if start_line is not None and end_line is not None:
                            original_lines = original_paper.split('\n')
                            camera_ready_content = '\n'.join(original_lines[start_line:end_line]).strip()
            
            # Skip modifications where camera_ready_content is empty
            if not camera_ready_content or not camera_ready_content.strip():
                tqdm.write(f"  ⚠️ Skipping modification with target_heading '{target_heading[:50]}...' (camera_ready_content is empty)")
                continue
            
            # Create updated modification
            updated_mod = {
                'target_heading': target_heading,
                'planted_error_content': planted_error_content,
                'camera_ready_content': camera_ready_content,
                'reasoning': reasoning
            }
            
            # Keep any other fields that might exist
            for key, value in mod.items():
                if key not in ['new_content', 'target_heading', 'reasoning']:
                    updated_mod[key] = value
            
            updated_mods.append(updated_mod)
        
        # Update the row with filtered modifications and update num_modifications
        num_updated_mods = len(updated_mods)
        df.at[idx, 'llm_generated_modifications'] = json.dumps(updated_mods, ensure_ascii=False, indent=2)
        df.at[idx, 'num_modifications'] = num_updated_mods
        
        # If all modifications were removed, mark this row for removal
        if num_updated_mods == 0:
            tqdm.write(f"  ⚠️ All modifications removed for flaw_id '{flaw_id}', marking row for removal")
            rows_to_remove.append(idx)
    
    # Remove rows with no modifications
    if rows_to_remove:
        df = df.drop(index=rows_to_remove)
        df = df.reset_index(drop=True)
    
    # Check if CSV is now empty
    is_empty = df.empty or len(df) == 0
    
    if is_empty:
        tqdm.write(f"  ⚠️ CSV became empty after processing, will delete paper folders")
        # Don't save the empty CSV, just return
        return True, True
    
    # Save updated CSV
    df.to_csv(csv_path, index=False)
    return True, False


def process_category(
    data_dir: Path,
    conference: str,
    category: str
):
    """Process all papers in a category."""
    category_dir = data_dir / conference / category
    planted_error_dir = category_dir / 'planted_error'
    latest_dir = category_dir / 'latest'
    
    if not planted_error_dir.exists():
        print(f"  ⚠️ Planted error directory not found: {planted_error_dir}")
        return
    
    # Find all paper folders
    paper_folders = [d for d in planted_error_dir.iterdir() if d.is_dir()]
    
    for paper_folder in tqdm(paper_folders, desc=f"Processing {category}"):
        paper_name = paper_folder.name
        
        # Find modifications CSV
        csv_files = list(paper_folder.glob("*_modifications_summary.csv"))
        if not csv_files:
            tqdm.write(f"  ⚠️ No modifications CSV found in {paper_folder}")
            continue
        
        csv_path = csv_files[0]
        
        # Read CSV to get flaw_id
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            
            # Get flaw_id from first row
            flaw_id = df.iloc[0]['flaw_id']
        except Exception as e:
            tqdm.write(f"  ⚠️ Error reading CSV {csv_path}: {e}")
            continue
        
        # Find flawed paper
        flawed_paper_path = paper_folder / 'flawed_papers' / f"{flaw_id}.md"
        if not flawed_paper_path.exists():
            # Try alternative location
            flawed_paper_path = paper_folder / f"{flaw_id}.md"
        
        # Find original paper
        original_paper_path = latest_dir / paper_name / 'structured_paper_output' / 'paper.md'
        if not original_paper_path.exists():
            # Try alternative location
            original_paper_path = latest_dir / paper_name / 'paper.md'
        
        if not flawed_paper_path.exists() or not original_paper_path.exists():
            tqdm.write(f"  ⚠️ Papers not found for {paper_name}/{flaw_id}")
            continue
        
        # Process the CSV
        success, is_empty = process_modifications_csv(csv_path, flawed_paper_path, original_paper_path)
        
        if is_empty:
            # CSV became empty - delete paper folders from both planted_error and latest
            tqdm.write(f"  🗑️  Deleting paper folders for {paper_name} (CSV became empty)")
            
            # Delete from planted_error
            try:
                if paper_folder.exists():
                    shutil.rmtree(paper_folder)
                    tqdm.write(f"  ✅ Deleted {paper_folder}")
            except Exception as e:
                tqdm.write(f"  ⚠️ Error deleting {paper_folder}: {e}")
            
            # Delete from latest
            latest_paper_folder = latest_dir / paper_name
            try:
                if latest_paper_folder.exists():
                    shutil.rmtree(latest_paper_folder)
                    tqdm.write(f"  ✅ Deleted {latest_paper_folder}")
            except Exception as e:
                tqdm.write(f"  ⚠️ Error deleting {latest_paper_folder}: {e}")
            
            # Continue to next paper
            continue
        elif success:
            tqdm.write(f"  ✅ Updated {paper_name}/{csv_path.name}")
        else:
            tqdm.write(f"  ❌ Failed to update {paper_name}/{csv_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess modifications_summary.csv files to add camera_ready_content and rename new_content."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the data structure"
    )
    parser.add_argument(
        "--conference",
        type=str,
        required=True,
        help="Conference name (e.g., NeurIPS2024)"
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated list of categories to process (e.g., '1a,1b,1c'). If not provided, processes all categories."
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    conference_dir = data_dir / args.conference
    
    if not conference_dir.exists():
        print(f"❌ Conference directory not found: {conference_dir}")
        return
    
    # Get categories to process
    if args.categories:
        categories = [c.strip() for c in args.categories.split(',')]
    else:
        # Find all category directories
        categories = [d.name for d in conference_dir.iterdir() if d.is_dir() and (d / 'planted_error').exists()]
        categories.sort()
    
    print(f"\n📂 Processing categories: {', '.join(categories)}")
    
    for category in categories:
        print(f"\n{'='*80}")
        print(f"Processing category: {category}")
        print(f"{'='*80}")
        
        process_category(
            data_dir=data_dir,
            conference=args.conference,
            category=category
        )
    
    print("\n✅ Preprocessing complete!")


if __name__ == "__main__":
    main()

