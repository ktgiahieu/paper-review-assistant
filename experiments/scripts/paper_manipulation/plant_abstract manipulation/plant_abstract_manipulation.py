#!/usr/bin/env python3
"""
Script to plant abstract manipulations in papers.

This script creates two versions of each paper:
1. Good version: Rewrites abstract with buzzwords and superlatives, makes it twice as long,
   and adds a code availability statement with a GitHub link.
2. Bad version: Removes buzzwords and superlatives, makes it concise (75% length),
   uses old scientific writing style (2000s), and removes any code mentions.

The modified papers are saved in:
- abstract_manipulation_good/
- abstract_manipulation_bad/
"""

import os
import json
import argparse
import shutil
import time
import threading
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("WARNING: google-generativeai not installed. Install with: pip install google-generativeai")

# --- Environment & API Configuration ---
load_dotenv()

# Load multiple Gemini API keys (GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.)
GEMINI_API_KEYS: Dict[str, str] = {}
for i in range(1, 10):  # Support up to 9 keys
    key = os.getenv(f'GEMINI_API_KEY_{i}')
    if key:
        GEMINI_API_KEYS[str(i)] = key

# Fallback to single key if no numbered keys found
if not GEMINI_API_KEYS:
    single_key = os.getenv('GEMINI_API_KEY')
    if single_key:
        GEMINI_API_KEYS = {'SINGLE': single_key}
        print("⚠️ Only single API key found, falling back to sequential processing")
    else:
        raise ValueError("No Gemini API keys found in environment variables (GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.)")

print(f"✅ Loaded {len(GEMINI_API_KEYS)} Gemini API keys: {list(GEMINI_API_KEYS.keys())}")

# --- Constants ---
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2
GEMINI_MODEL = "gemini-2.0-flash-lite"

# Gemini model RPM limits (Requests Per Minute)
GEMINI_MODEL_RPM_LIMITS = {
    "gemini-2.0-flash-lite": 30,
    "gemini-2.0-flash-exp": 10,
    "gemini-2.0-flash-preview-image-generation": 10,
    "gemini-2.0-flash": 15,
    "gemini-2.5-flash-lite": 15,
    "gemini-2.5-flash-tts": 3,
    "gemini-2.5-flash": 10,
    "gemini-2.5-pro": 2,
}

def get_request_delay_for_model(model_name: str) -> float:
    """Calculate request delay in seconds based on model's RPM limit."""
    rpm_limit = GEMINI_MODEL_RPM_LIMITS.get(model_name, 30)
    return 60.0 / rpm_limit

GEMINI_REQUEST_DELAY = get_request_delay_for_model(GEMINI_MODEL)

# Rate limiting tracking
key_last_used: Dict[str, float] = {}
key_lock = threading.Lock()

# --- Pydantic Models for API Response ---
class AbstractRewriteResponse(BaseModel):
    """Response model for abstract rewrite."""
    rewritten_abstract: str = Field(description="The rewritten abstract")
    github_link: Optional[str] = Field(default=None, description="GitHub repository link (only for good version)")


def get_api_key_for_task(task_idx: int) -> Tuple[str, str]:
    """Get API key for a task using round-robin assignment."""
    key_names = list(GEMINI_API_KEYS.keys())
    key_name = key_names[task_idx % len(key_names)]
    return key_name, GEMINI_API_KEYS[key_name]


def wait_for_rate_limit(key_name: str, request_delay: float = None):
    """Wait if necessary to respect rate limits."""
    if request_delay is None:
        request_delay = GEMINI_REQUEST_DELAY
    
    with key_lock:
        if key_name in key_last_used:
            elapsed = time.time() - key_last_used[key_name]
            if elapsed < request_delay:
                sleep_time = request_delay - elapsed
                time.sleep(sleep_time)
        key_last_used[key_name] = time.time()


def extract_abstract(paper_content: str) -> Optional[str]:
    """Extract the abstract section from paper markdown."""
    lines = paper_content.split('\n')
    abstract_lines = []
    in_abstract = False
    abstract_end_patterns = ['## ', '# ', '##Introduction', '## Introduction', '##1.', '## 1.']
    
    for i, line in enumerate(lines):
        # Check if we're entering the abstract section
        if line.strip().lower().startswith('## abstract') or line.strip().lower() == '##abstract':
            in_abstract = True
            continue
        
        # Check if we're leaving the abstract section
        if in_abstract:
            # Check if this line starts a new section
            if any(line.strip().startswith(pattern) for pattern in abstract_end_patterns):
                break
            abstract_lines.append(line)
    
    abstract_text = '\n'.join(abstract_lines).strip()
    return abstract_text if abstract_text else None


def rewrite_abstract_good(abstract: str, api_key: str, key_name: str, model_name: str, verbose: bool = False) -> Optional[Tuple[str, str]]:
    """
    Rewrite abstract with buzzwords and superlatives, make it twice as long, and add code availability.
    
    Returns:
        Tuple of (rewritten_abstract, github_link) or None if failed
    """
    prompt = f"""You are an expert at rewriting scientific abstracts to make them more appealing and impactful.

Rewrite the following abstract to:
1. Add buzzwords and superlatives (e.g., "state-of-the-art", "groundbreaking", "revolutionary", "unprecedented", "cutting-edge", "remarkable", "significant", "substantial", "dramatically", "significantly", "extensively", "comprehensive")
2. Make it approximately TWICE as long as the original (expand descriptions, add emphasis, elaborate on contributions)
3. Use more engaging and marketing-style language while maintaining scientific accuracy
4. At the end, add a line: "Code is available at [a GitHub repository link]"
   - Generate a realistic GitHub repository URL (format: https://github.com/username/repo-name)
   - The username should be academic-sounding (e.g., "research-lab", "paper-author", "ml-research")
   - The repo-name should relate to the paper topic

Original abstract:
{abstract}

Your response MUST be a valid JSON object with this structure:
{{
  "rewritten_abstract": "The rewritten abstract text here",
  "github_link": "https://github.com/username/repo-name"
}}

Return ONLY the JSON object, no additional text."""

    for attempt in range(MAX_RETRIES):
        try:
            wait_for_rate_limit(key_name)
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean JSON response
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            result = json.loads(response_text)
            rewritten_abstract = result.get('rewritten_abstract', '')
            github_link = result.get('github_link', '')
            
            if rewritten_abstract:
                return (rewritten_abstract, github_link)
            else:
                if verbose:
                    print(f"Warning: Empty abstract returned for key {key_name}, attempt {attempt + 1}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(INITIAL_BACKOFF_SECONDS * (2 ** attempt))
                    continue
                return None
                
        except Exception as e:
            if verbose:
                print(f"Error rewriting abstract (good) for key {key_name}, attempt {attempt + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(INITIAL_BACKOFF_SECONDS * (2 ** attempt))
                continue
            return None
    
    return None


def rewrite_abstract_bad(abstract: str, api_key: str, key_name: str, model_name: str, verbose: bool = False) -> Optional[str]:
    """
    Rewrite abstract to remove buzzwords, make it concise (75% length), use old scientific style, remove code mentions.
    
    Returns:
        Rewritten abstract or None if failed
    """
    prompt = f"""You are an expert at rewriting scientific abstracts in the style of academic papers from the early 2000s.

Rewrite the following abstract to:
1. Remove all buzzwords and superlatives (e.g., remove "state-of-the-art", "groundbreaking", "revolutionary", "unprecedented", "cutting-edge", "remarkable", "dramatically", "significantly")
2. Make it concise - approximately 75% of the original length
3. Use direct, formal, old-fashioned scientific writing style (like papers from 2000-2005):
   - Use passive voice more often
   - Avoid marketing language
   - Be direct and factual
   - Use simple, clear statements
   - Avoid excessive emphasis
4. Remove any mentions of code availability, GitHub links, or code repositories
5. Focus on the core contribution and results without embellishment

Original abstract:
{abstract}

Return ONLY the rewritten abstract text, no additional explanation or formatting."""

    for attempt in range(MAX_RETRIES):
        try:
            wait_for_rate_limit(key_name)
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            rewritten_abstract = response.text.strip()
            
            # Clean up response
            if rewritten_abstract.startswith('```'):
                # Remove markdown code blocks if present
                lines = rewritten_abstract.split('\n')
                rewritten_abstract = '\n'.join([l for l in lines if not l.strip().startswith('```')]).strip()
            
            if rewritten_abstract:
                return rewritten_abstract
            else:
                if verbose:
                    print(f"Warning: Empty abstract returned (bad) for key {key_name}, attempt {attempt + 1}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(INITIAL_BACKOFF_SECONDS * (2 ** attempt))
                    continue
                return None
                
        except Exception as e:
            if verbose:
                print(f"Error rewriting abstract (bad) for key {key_name}, attempt {attempt + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(INITIAL_BACKOFF_SECONDS * (2 ** attempt))
                continue
            return None
    
    return None


def replace_abstract_in_paper(paper_content: str, new_abstract: str) -> str:
    """Replace the abstract section in paper markdown with new abstract."""
    lines = paper_content.split('\n')
    new_lines = []
    in_abstract = False
    abstract_start_idx = None
    abstract_end_idx = None
    
    for i, line in enumerate(lines):
        # Find abstract section start
        if line.strip().lower().startswith('## abstract') or line.strip().lower() == '##abstract':
            abstract_start_idx = i
            in_abstract = True
            new_lines.append(line)  # Keep the ## Abstract header
            continue
        
        # Find abstract section end
        if in_abstract:
            abstract_end_patterns = ['## ', '# ', '##Introduction', '## Introduction', '##1.', '## 1.']
            if any(line.strip().startswith(pattern) for pattern in abstract_end_patterns):
                abstract_end_idx = i
                break
        
        if not in_abstract:
            new_lines.append(line)
    
    # Add new abstract
    if abstract_start_idx is not None:
        new_lines.append('')  # Blank line after header
        new_lines.append(new_abstract)
        new_lines.append('')  # Blank line before next section
        
        # Add remaining lines after abstract
        if abstract_end_idx is not None:
            new_lines.extend(lines[abstract_end_idx:])
        else:
            # If we didn't find an end, add remaining lines after abstract start
            remaining_start = abstract_start_idx + 1
            while remaining_start < len(lines):
                # Skip old abstract lines (they're between abstract_start_idx+1 and abstract_end_idx)
                if abstract_end_idx and remaining_start < abstract_end_idx:
                    remaining_start += 1
                    continue
                new_lines.append(lines[remaining_start])
                remaining_start += 1
        
        return '\n'.join(new_lines)
    else:
        # Abstract section not found, return original
        return paper_content


def create_modified_paper(
    source_paper_path: Path,
    output_paper_path: Path,
    version_type: str,  # "good" or "bad"
    api_key: str,
    key_name: str,
    model_name: str,
    verbose: bool = False
) -> bool:
    """
    Create a modified version of the paper with rewritten abstract.
    
    Args:
        version_type: "good" or "bad"
    """
    try:
        # Read the original paper
        with open(source_paper_path, 'r', encoding='utf-8') as f:
            paper_content = f.read()
        
        # Extract abstract
        original_abstract = extract_abstract(paper_content)
        if not original_abstract:
            if verbose:
                print(f"Warning: Could not extract abstract from {source_paper_path}")
            return False
        
        # Rewrite abstract
        if version_type == "good":
            result = rewrite_abstract_good(original_abstract, api_key, key_name, model_name, verbose)
            if result is None:
                if verbose:
                    print(f"Failed to rewrite abstract (good) for {source_paper_path}")
                return False
            rewritten_abstract, github_link = result
        else:  # bad
            rewritten_abstract = rewrite_abstract_bad(original_abstract, api_key, key_name, model_name, verbose)
            if rewritten_abstract is None:
                if verbose:
                    print(f"Failed to rewrite abstract (bad) for {source_paper_path}")
                return False
        
        # Replace abstract in paper
        modified_content = replace_abstract_in_paper(paper_content, rewritten_abstract)
        
        # Create output directory
        output_paper_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write modified paper
        with open(output_paper_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        # Copy the entire directory structure (including figures, etc.)
        source_dir = source_paper_path.parent.parent  # Go up from structured_paper_output
        output_dir = output_paper_path.parent.parent  # Go up from structured_paper_output
        
        # Copy all files except paper.md (which we already modified)
        for item in source_dir.iterdir():
            if item.name == 'structured_paper_output':
                continue  # We'll handle this separately
            if item.is_dir():
                shutil.copytree(item, output_dir / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, output_dir / item.name)
        
        # Copy structured_paper_output directory except paper.md
        source_structured = source_dir / 'structured_paper_output'
        output_structured = output_dir / 'structured_paper_output'
        if source_structured.exists():
            output_structured.mkdir(parents=True, exist_ok=True)
            for item in source_structured.iterdir():
                if item.name == 'paper.md':
                    continue  # Skip, we already wrote the modified version
                if item.is_dir():
                    shutil.copytree(item, output_structured / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, output_structured / item.name)
        
        return True
    except Exception as e:
        if verbose:
            print(f"Error processing {source_paper_path}: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        return False


def process_papers(
    base_dir: Path,
    output_good_dir: Path,
    output_bad_dir: Path,
    paper_ids: List[str] = None,
    model_name: str = GEMINI_MODEL,
    max_workers: int = None,
    verbose: bool = False
):
    """
    Process all papers and create good/bad abstract versions.
    
    Args:
        base_dir: Directory containing the latest/ folder with papers
        output_good_dir: Output directory for good abstract versions
        output_bad_dir: Output directory for bad abstract versions
        paper_ids: Optional list of paper IDs to process. If None, processes all.
        model_name: Gemini model name to use
        max_workers: Maximum number of worker threads (default: number of API keys)
        verbose: Enable verbose output
    """
    latest_dir = base_dir / "latest"
    if not latest_dir.exists():
        print(f"Error: {latest_dir} does not exist")
        return
    
    # Find all paper directories
    paper_dirs = [d for d in latest_dir.iterdir() if d.is_dir()]
    
    if paper_ids:
        # Filter to only specified paper IDs
        paper_ids_set = {str(pid).strip() for pid in paper_ids}
        paper_dirs = [d for d in paper_dirs if d.name.split('_')[0] in paper_ids_set]
    
    print(f"Processing {len(paper_dirs)} papers...")
    print(f"Model: {model_name}")
    print(f"Number of API keys: {len(GEMINI_API_KEYS)}")
    
    # Set max_workers based on number of API keys if not specified
    if max_workers is None:
        max_workers = len(GEMINI_API_KEYS)
    
    # Prepare tasks
    tasks = []
    key_names = list(GEMINI_API_KEYS.keys())
    request_delay = get_request_delay_for_model(model_name)
    
    for idx, paper_dir in enumerate(paper_dirs):
        paper_id = paper_dir.name.split('_')[0]
        paper_md = paper_dir / "structured_paper_output" / "paper.md"
        
        if not paper_md.exists():
            if verbose:
                print(f"Warning: {paper_md} not found, skipping {paper_id}")
            continue
        
        # Assign API key (round-robin)
        key_name = key_names[idx % len(key_names)]
        api_key = GEMINI_API_KEYS[key_name]
        
        # Good version
        good_output_md = output_good_dir / paper_dir.name / "structured_paper_output" / "paper.md"
        tasks.append((paper_md, good_output_md, "good", api_key, key_name, model_name, request_delay))
        
        # Bad version
        bad_output_md = output_bad_dir / paper_dir.name / "structured_paper_output" / "paper.md"
        tasks.append((paper_md, bad_output_md, "bad", api_key, key_name, model_name, request_delay))
    
    # Track statistics
    stats = {
        'total': len(paper_dirs),
        'good_success': 0,
        'bad_success': 0,
        'good_failed': 0,
        'bad_failed': 0,
    }
    
    # Process tasks
    def process_task(task):
        source_md, output_md, version_type, api_key, key_name, task_model_name, task_delay = task
        paper_id = source_md.parent.parent.name.split('_')[0]
        
        success = create_modified_paper(
            source_paper_path=source_md,
            output_paper_path=output_md,
            version_type=version_type,
            api_key=api_key,
            key_name=key_name,
            model_name=task_model_name,
            verbose=verbose
        )
        
        return paper_id, version_type, success
    
    # Process with thread pool if multiple keys, otherwise sequential
    if len(GEMINI_API_KEYS) > 1:
        print(f"Processing {len(tasks)} tasks in parallel using {len(GEMINI_API_KEYS)} API keys (max_workers={max_workers})...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_task, task): task for task in tasks}
            
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing abstracts"):
                try:
                    paper_id, version_type, success = future.result()
                    if version_type == "good":
                        if success:
                            stats['good_success'] += 1
                        else:
                            stats['good_failed'] += 1
                    else:
                        if success:
                            stats['bad_success'] += 1
                        else:
                            stats['bad_failed'] += 1
                except Exception as e:
                    task = futures[future]
                    paper_id = task[0].parent.parent.name.split('_')[0]
                    if verbose:
                        print(f"Error processing {paper_id}: {e}")
                    version_type = task[2]
                    if version_type == "good":
                        stats['good_failed'] += 1
                    else:
                        stats['bad_failed'] += 1
    else:
        # Sequential processing
        print(f"Processing {len(tasks)} tasks sequentially...")
        for task in tqdm(tasks, desc="Processing abstracts"):
            try:
                paper_id, version_type, success = process_task(task)
                if version_type == "good":
                    if success:
                        stats['good_success'] += 1
                    else:
                        stats['good_failed'] += 1
                else:
                    if success:
                        stats['bad_success'] += 1
                    else:
                        stats['bad_failed'] += 1
            except Exception as e:
                paper_id = task[0].parent.parent.name.split('_')[0]
                if verbose:
                    print(f"Error processing {paper_id}: {e}")
                version_type = task[2]
                if version_type == "good":
                    stats['good_failed'] += 1
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
        description="Plant abstract manipulations in papers"
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
        help='Output directory for good abstract versions (default: base_dir/abstract_manipulation_good)'
    )
    parser.add_argument(
        '--output_bad_dir',
        type=str,
        default=None,
        help='Output directory for bad abstract versions (default: base_dir/abstract_manipulation_bad)'
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
        '--model_name',
        type=str,
        default=GEMINI_MODEL,
        help=f'Gemini model name (default: {GEMINI_MODEL})'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=None,
        help='Maximum number of worker threads (default: number of API keys)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Check if Gemini is available
    if not GENAI_AVAILABLE:
        print("Error: google-generativeai not installed. Install with: pip install google-generativeai")
        return
    
    if not GEMINI_API_KEYS:
        print("Error: No Gemini API keys found in environment variables or .env file.")
        return
    
    base_dir = Path(args.base_dir)
    
    # Set default output directories
    if args.output_good_dir is None:
        output_good_dir = base_dir / "abstract_manipulation_good"
    else:
        output_good_dir = Path(args.output_good_dir)
    
    if args.output_bad_dir is None:
        output_bad_dir = base_dir / "abstract_manipulation_bad"
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
        model_name=args.model_name,
        max_workers=args.max_workers,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()

