#!/usr/bin/env python3
"""
Script to generate placebo/sham surgery versions for all categories in a venue directory.

This script iterates through all category_id folders (e.g., 1a, 1b, 2a, etc.) in a venue directory,
and generates placebo versions for each category's planted_error files.

Usage:
    python generate_placebos_for_all_categories.py \
        --venue_dir experiments/category_sampled_data/NeurIPS2024 \
        --model_name gemini-2.0-flash-lite
"""

import argparse
import sys
import importlib.util
from pathlib import Path
from tqdm import tqdm

# Import the generate_placebo_only module
script_dir = Path(__file__).parent
generate_placebo_script = script_dir / 'generate_placebo_only.py'

# Load the generate_placebo_only module
spec = importlib.util.spec_from_file_location("generate_placebo_only", generate_placebo_script)
generate_placebo_module = importlib.util.module_from_spec(spec)
sys.modules["generate_placebo_only"] = generate_placebo_module
spec.loader.exec_module(generate_placebo_module)

# Import necessary functions
load_planted_error_data = generate_placebo_module.load_planted_error_data
process_planted_error = generate_placebo_module.process_planted_error
get_api_key_for_task = generate_placebo_module.get_api_key_for_task
GEMINI_API_KEYS = generate_placebo_module.GEMINI_API_KEYS
DEFAULT_GEMINI_MODEL = generate_placebo_module.DEFAULT_GEMINI_MODEL
GEMINI_MODEL_RPM_LIMITS = generate_placebo_module.GEMINI_MODEL_RPM_LIMITS
GEMINI_MODEL_TPM_LIMITS = generate_placebo_module.GEMINI_MODEL_TPM_LIMITS
get_request_delay_for_model = generate_placebo_module.get_request_delay_for_model
GEMINI_MODEL = generate_placebo_module.GEMINI_MODEL
GEMINI_REQUEST_DELAY = generate_placebo_module.GEMINI_REQUEST_DELAY

import concurrent.futures
import pandas as pd


def get_category_folders(venue_dir: Path) -> list:
    """
    Get all category_id folders from the venue directory.
    Expected structure: venue_dir/{category_id}/planted_error/
    """
    category_folders = []
    
    if not venue_dir.exists():
        return category_folders
    
    for item in venue_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if this folder has a planted_error subdirectory
            planted_error_dir = item / 'planted_error'
            if planted_error_dir.exists() and planted_error_dir.is_dir():
                # Check if there are any .md files in the planted_error directory
                # (either directly or in subdirectories)
                has_md_files = any(
                    f.suffix == '.md' 
                    for f in planted_error_dir.rglob('*.md')
                )
                if has_md_files:
                    category_folders.append(item.name)
    
    return sorted(category_folders)


def process_category(
    venue_dir: Path,
    category_id: str,
    model_name: str,
    max_workers: int = None,
    task_counter: list = None
) -> dict:
    """
    Process a single category to generate placebos.
    """
    category_dir = venue_dir / category_id
    planted_error_dir = category_dir / 'planted_error'
    original_papers_dir = category_dir / 'latest'
    output_dir = category_dir
    
    # Set model and request delay
    GEMINI_MODEL = model_name
    GEMINI_REQUEST_DELAY = get_request_delay_for_model(GEMINI_MODEL)
    model_rpm = GEMINI_MODEL_RPM_LIMITS.get(GEMINI_MODEL, 30)
    model_tpm = GEMINI_MODEL_TPM_LIMITS.get(GEMINI_MODEL, 1000000)
    
    # Load planted error files for this category
    planted_errors = load_planted_error_data(planted_error_dir)
    
    if not planted_errors:
        print(f"  ⚠️ No planted error files found for category {category_id}")
        return {
            'category_id': category_id,
            'total_flaws': 0,
            'successful_placebo': 0,
            'success_rate': 0.0
        }
    
    print(f"\n  📂 Category {category_id}: Found {len(planted_errors)} planted error files")
    
    # Set max_workers
    if max_workers is None:
        max_workers = len(GEMINI_API_KEYS)
    
    # Process planted errors
    all_results = []
    local_task_counter = [0]
    
    def process_with_counter(planted_error_data):
        idx = local_task_counter[0]
        local_task_counter[0] += 1
        # Use global task counter if provided for cross-category task distribution
        if task_counter is not None:
            global_idx = task_counter[0]
            task_counter[0] += 1
            idx = global_idx
        return process_planted_error(
            planted_error_data,
            original_papers_dir,
            output_dir,
            idx,
            request_delay=GEMINI_REQUEST_DELAY,
            tpm_limit=model_tpm,
            rpm_limit=model_rpm
        )
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_with_counter, pe): pe
            for pe in planted_errors
        }
        
        progress_bar = tqdm(
            concurrent.futures.as_completed(futures),
            total=len(planted_errors),
            desc=f"  Category {category_id}",
            leave=False
        )
        for future in progress_bar:
            try:
                result = future.result()
                if result:
                    all_results.append(result)
            except Exception as e:
                pe = futures[future]
                tqdm.write(f"  ❌ Error processing {pe['paper_folder']}/{pe['flaw_id']}: {e}")
    
    # Calculate statistics
    total_flaws = len(all_results)
    successful_placebo = sum(1 for r in all_results if r.get('success', False))
    success_rate = (successful_placebo / total_flaws * 100) if total_flaws > 0 else 0.0
    
    return {
        'category_id': category_id,
        'total_flaws': total_flaws,
        'successful_placebo': successful_placebo,
        'success_rate': success_rate
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate placebo/sham surgery versions for all categories in a venue directory"
    )
    parser.add_argument(
        "--venue_dir",
        type=str,
        required=True,
        help="Directory containing category folders (e.g., experiments/category_sampled_data/NeurIPS2024)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_GEMINI_MODEL,
        help=f"Gemini model name to use (default: {DEFAULT_GEMINI_MODEL})"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Max worker threads per category (default: number of API keys)"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs='+',
        default=None,
        help="Specific categories to process (e.g., '2a' '2b'). If not provided, processes all categories."
    )
    parser.add_argument(
        "--skip_existing",
        action='store_true',
        help="Skip categories that already have sham_surgery directory with files"
    )
    
    args = parser.parse_args()
    
    venue_dir = Path(args.venue_dir)
    
    if not venue_dir.exists():
        print(f"❌ Venue directory not found: {venue_dir}")
        return
    
    # Get all category folders
    all_categories = get_category_folders(venue_dir)
    
    if not all_categories:
        print(f"❌ No category folders with planted_error subdirectories found in {venue_dir}")
        return
    
    # Filter to specific categories if provided
    if args.categories:
        categories_to_process = [c for c in all_categories if c in args.categories]
        if not categories_to_process:
            print(f"❌ None of the specified categories {args.categories} were found")
            return
    else:
        categories_to_process = all_categories
    
    # Filter out categories with existing sham_surgery if skip_existing is set
    if args.skip_existing:
        categories_to_process = [
            cat for cat in categories_to_process
            if not (venue_dir / cat / 'sham_surgery').exists() or
            len(list((venue_dir / cat / 'sham_surgery').rglob('*.md'))) == 0
        ]
    
    print(f"✅ Found {len(all_categories)} total categories")
    print(f"✅ Will process {len(categories_to_process)} categories: {', '.join(categories_to_process)}")
    print(f"✅ Using Gemini model: {args.model_name}")
    
    # Get model limits
    model_rpm = GEMINI_MODEL_RPM_LIMITS.get(args.model_name, 30)
    model_tpm = GEMINI_MODEL_TPM_LIMITS.get(args.model_name, 1000000)
    request_delay = get_request_delay_for_model(args.model_name)
    
    max_workers = args.max_workers if args.max_workers else len(GEMINI_API_KEYS)
    print(f"✅ Using {max_workers} worker threads per category")
    print(f"✅ Model RPM limit: {model_rpm} requests/minute per key")
    print(f"✅ Model TPM limit: {model_tpm:,} tokens/minute per key")
    print()
    
    # Process each category
    all_category_results = []
    global_task_counter = [0]  # Shared task counter across all categories
    
    for category_id in categories_to_process:
        print(f"\n{'='*80}")
        print(f"Processing category: {category_id}")
        print(f"{'='*80}")
        
        result = process_category(
            venue_dir=venue_dir,
            category_id=category_id,
            model_name=args.model_name,
            max_workers=max_workers,
            task_counter=global_task_counter
        )
        
        all_category_results.append(result)
        
        print(f"\n  ✅ Category {category_id} complete:")
        print(f"     Total flaws: {result['total_flaws']}")
        print(f"     Successful placebos: {result['successful_placebo']}")
        print(f"     Success rate: {result['success_rate']:.1f}%")
    
    # Print summary
    print(f"\n{'='*80}")
    print("✅ ALL CATEGORIES PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"\n📊 Summary Statistics:")
    
    total_flaws_all = sum(r['total_flaws'] for r in all_category_results)
    total_successful_all = sum(r['successful_placebo'] for r in all_category_results)
    overall_success_rate = (total_successful_all / total_flaws_all * 100) if total_flaws_all > 0 else 0.0
    
    print(f"   Total categories processed: {len(all_category_results)}")
    print(f"   Total flaws processed: {total_flaws_all}")
    print(f"   Total successful placebos: {total_successful_all}")
    print(f"   Overall success rate: {overall_success_rate:.1f}%")
    
    print(f"\n📊 Per-Category Breakdown:")
    for result in all_category_results:
        print(f"   {result['category_id']:>3s}: {result['successful_placebo']:>3d}/{result['total_flaws']:>3d} ({result['success_rate']:>5.1f}%)")
    
    print(f"\n📁 Sham surgery files saved to: {venue_dir}/{{category_id}}/sham_surgery/")
    print(f"{'='*80}")
    
    # Save summary CSV
    summary_csv_path = venue_dir / 'placebo_generation_summary.csv'
    summary_df = pd.DataFrame(all_category_results)
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n📄 Summary saved to: {summary_csv_path}")


if __name__ == "__main__":
    main()

