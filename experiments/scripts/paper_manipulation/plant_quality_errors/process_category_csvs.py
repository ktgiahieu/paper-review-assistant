#!/usr/bin/env python3
"""
Script to process category CSV files and generate both planted_error and sham_surgery files.

This script:
1. Reads CSV files like `1a_sampled_flaws.csv` from category_sampled_data/
2. For each category, finds papers in {venue_dir}/{category}/latest/
3. Generates planted_error files (if they don't exist)
4. Generates sham_surgery files from planted_error files

Usage:
    python process_category_csvs.py \
        --category_sampled_data_dir experiments/category_sampled_data \
        --venue_dir experiments/category_sampled_data/NeurIPS2024 \
        --model_name gemini-2.0-flash-lite \
        --skip_existing_planted \
        --skip_existing_sham
"""

import argparse
import sys
import importlib.util
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from typing import Optional

# Import the plant_errors_and_placebo module
script_dir = Path(__file__).parent
plant_errors_script = script_dir / 'plant_errors_and_placebo.py'

# Load the plant_errors_and_placebo module
spec = importlib.util.spec_from_file_location("plant_errors_and_placebo", plant_errors_script)
plant_errors_module = importlib.util.module_from_spec(spec)
sys.modules["plant_errors_and_placebo"] = plant_errors_module
spec.loader.exec_module(plant_errors_module)

# Import necessary functions and variables
GEMINI_API_KEYS = plant_errors_module.GEMINI_API_KEYS
DEFAULT_GEMINI_MODEL = plant_errors_module.DEFAULT_GEMINI_MODEL
GEMINI_MODEL_RPM_LIMITS = plant_errors_module.GEMINI_MODEL_RPM_LIMITS
GEMINI_MODEL_TPM_LIMITS = plant_errors_module.GEMINI_MODEL_TPM_LIMITS
get_request_delay_for_model = plant_errors_module.get_request_delay_for_model
get_api_key_for_task = plant_errors_module.get_api_key_for_task
plant_error = plant_errors_module.plant_error
generate_placebo = plant_errors_module.generate_placebo
Modification = plant_errors_module.Modification
call_gemini_with_retries = plant_errors_module.call_gemini_with_retries
find_heading_in_lines = plant_errors_module.find_heading_in_lines
try_apply_modifications = plant_errors_module.try_apply_modifications

# Import from generate_placebo_only for loading planted_error data and reconstructing modifications
generate_placebo_script = script_dir / 'generate_placebo_only.py'
spec2 = importlib.util.spec_from_file_location("generate_placebo_only", generate_placebo_script)
generate_placebo_module = importlib.util.module_from_spec(spec2)
sys.modules["generate_placebo_only"] = generate_placebo_module
spec2.loader.exec_module(generate_placebo_module)

load_planted_error_data = generate_placebo_module.load_planted_error_data
reconstruct_modifications_from_metadata = generate_placebo_module.reconstruct_modifications_from_metadata

# Import Pydantic for marker extraction
from pydantic import BaseModel, Field

class MarkerExtraction(BaseModel):
    start_marker: str = Field(..., description="A unique text marker (3-10 words) that appears at the START of the section in the original paper. This should be the exact beginning text of the section, including the heading line.")
    end_marker: str = Field(..., description="A unique text marker (3-10 words) that appears at the END of the section in the original paper, just before the next section starts. This should be the exact ending text of the section.")

class MarkerExtractionResponse(BaseModel):
    markers: MarkerExtraction = Field(..., description="The extracted start and end markers for the section in the original paper.")


def find_paper_folder(openreview_id: str, category_dir: Path) -> Path:
    """
    Find the paper folder in latest/ directory matching the openreview_id.
    Returns the full path to the paper folder.
    """
    latest_dir = category_dir / 'latest'
    if not latest_dir.exists():
        return None
    
    # Try exact match first
    matching_folders = list(latest_dir.glob(f"{openreview_id}*"))
    if matching_folders:
        return matching_folders[0]
    
    return None


def extract_markers_from_original_paper(
    original_paper: str,
    target_heading: str,
    new_content: str,
    api_key: str,
    key_name: str,
    model_name: str,
    request_delay: float = None,
    tpm_limit: int = 1000000,
    rpm_limit: int = 30
) -> Optional[MarkerExtraction]:
    """
    Use LLM to extract start_marker and end_marker from the original paper
    given a target_heading and new_content.
    """
    # First, try to find the heading in the original paper
    lines = original_paper.split('\n')
    heading_index = find_heading_in_lines(lines, target_heading)
    
    if heading_index == -1:
        # If heading not found, we'll need LLM to help
        heading_context = "Heading not found in paper"
    else:
        # Extract a reasonable context around the heading (50 lines before and 200 lines after)
        start_context = max(0, heading_index - 50)
        end_context = min(len(lines), heading_index + 200)
        heading_context = '\n'.join(lines[start_context:end_context])
    
    # Create prompt for marker extraction
    prompt = f"""You are analyzing a research paper to extract precise text markers for a section that needs to be modified.

**Target Heading:** {target_heading}

**New Content (what the section should become):**
{new_content[:500]}...

**Original Paper Context (around the target heading):**
{heading_context[:3000]}

Your task is to find the EXACT start and end markers in the ORIGINAL paper that correspond to the section with the target heading.

**Instructions:**
1. Find the section in the original paper that starts with the target heading
2. Extract the START marker: The exact beginning text (3-10 words) including the heading line
3. Extract the END marker: The exact ending text (3-10 words) just before the next section starts
4. The markers must be EXACT text that appears in the original paper (not the new_content)

Return the markers as JSON with "start_marker" and "end_marker" fields.
"""
    
    try:
        response = call_gemini_with_retries(
            api_key=api_key,
            key_name=key_name,
            prompt=prompt,
            response_model=MarkerExtractionResponse,
            max_retries=3,
            request_delay=request_delay,
            tpm_limit=tpm_limit,
            rpm_limit=rpm_limit
        )
        
        if response and response.markers:
            return response.markers
    except Exception as e:
        tqdm.write(f"  ⚠️ Error extracting markers: {e}")
    
    return None


def convert_csv_modifications_to_marker_format(
    csv_path: Path,
    original_paper: str,
    api_key: str,
    key_name: str,
    model_name: str,
    request_delay: float = None,
    tpm_limit: int = 1000000,
    rpm_limit: int = 30
) -> bool:
    """
    Convert modifications in CSV from target_heading/new_content format to start_marker/end_marker format.
    Updates the CSV file in place.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        tqdm.write(f"  ⚠️ Error reading CSV {csv_path}: {e}")
        return False
    
    updated = False
    
    for idx, row in df.iterrows():
        modifications_json = row.get('llm_generated_modifications', '')
        if not modifications_json:
            continue
        
        try:
            if isinstance(modifications_json, str):
                modifications = json.loads(modifications_json)
            else:
                modifications = modifications_json
            
            # Check if already has start_marker and end_marker
            has_markers = all(
                mod.get('start_marker') and mod.get('end_marker')
                for mod in modifications
            )
            
            if has_markers:
                continue  # Already converted
            
            # Convert each modification
            converted_modifications = []
            for mod in modifications:
                if mod.get('start_marker') and mod.get('end_marker'):
                    # Already has markers
                    converted_modifications.append(mod)
                    continue
                
                target_heading = mod.get('target_heading', '')
                new_content = mod.get('new_content', '')
                
                if not target_heading or not new_content:
                    # Can't convert without target_heading
                    converted_modifications.append(mod)
                    continue
                
                # Extract markers using LLM
                markers = extract_markers_from_original_paper(
                    original_paper=original_paper,
                    target_heading=target_heading,
                    new_content=new_content,
                    api_key=api_key,
                    key_name=key_name,
                    model_name=model_name,
                    request_delay=request_delay,
                    tpm_limit=tpm_limit,
                    rpm_limit=rpm_limit
                )
                
                if markers:
                    # Update modification with markers
                    new_mod = mod.copy()
                    new_mod['start_marker'] = markers.start_marker
                    new_mod['end_marker'] = markers.end_marker
                    converted_modifications.append(new_mod)
                    updated = True
                    tqdm.write(f"  ✅ Extracted markers for modification in {csv_path.name}")
                else:
                    # Keep original if extraction failed
                    converted_modifications.append(mod)
                    tqdm.write(f"  ⚠️ Failed to extract markers for modification in {csv_path.name}")
            
            if updated:
                # Update the row
                df.at[idx, 'llm_generated_modifications'] = json.dumps(converted_modifications, ensure_ascii=False)
        
        except Exception as e:
            tqdm.write(f"  ⚠️ Error processing modification in CSV {csv_path.name}: {e}")
            continue
    
    if updated:
        # Save updated CSV
        try:
            df.to_csv(csv_path, index=False)
            return True
        except Exception as e:
            tqdm.write(f"  ⚠️ Error saving updated CSV {csv_path}: {e}")
            return False
    
    return False


def find_original_paper_md(paper_folder: Path) -> Path:
    """
    Find the paper.md file in a paper folder.
    Tries structured_paper_output/paper.md first, then paper.md directly.
    """
    possible_paths = [
        paper_folder / 'structured_paper_output' / 'paper.md',
        paper_folder / 'paper.md',
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def process_flaw_from_csv(
    openreview_id: str,
    flaw_id: str,
    flaw_description: str,
    category_id: str,
    venue_dir: Path,
    model_name: str,
    task_idx: int,
    skip_existing_planted: bool = False,
    skip_existing_sham: bool = False,
    request_delay: float = None,
    tpm_limit: int = 1000000,
    rpm_limit: int = 30
) -> dict:
    """
    Process a single flaw from CSV file:
    1. Generate planted_error file if it doesn't exist
    2. Generate sham_surgery file from planted_error if it doesn't exist
    """
    
    category_dir = venue_dir / category_id
    paper_folder = find_paper_folder(openreview_id, category_dir)
    
    if not paper_folder:
        return {
            'category_id': category_id,
            'openreview_id': openreview_id,
            'flaw_id': flaw_id,
            'success': False,
            'error': 'Paper folder not found'
        }
    
    paper_folder_name = paper_folder.name
    original_paper_path = find_original_paper_md(paper_folder)
    
    if not original_paper_path:
        return {
            'category_id': category_id,
            'openreview_id': openreview_id,
            'flaw_id': flaw_id,
            'success': False,
            'error': 'Original paper.md not found'
        }
    
    # Read original paper
    try:
        with open(original_paper_path, 'r', encoding='utf-8') as f:
            original_paper = f.read()
    except Exception as e:
        return {
            'category_id': category_id,
            'openreview_id': openreview_id,
            'flaw_id': flaw_id,
            'success': False,
            'error': f'Error reading original paper: {e}'
        }
    
    # Get API key for this task
    key_name, api_key = get_api_key_for_task(task_idx)
    
    result = {
        'category_id': category_id,
        'openreview_id': openreview_id,
        'flaw_id': flaw_id,
        'paper_folder': paper_folder_name,
        'planted_error_generated': False,
        'sham_surgery_generated': False,
        'success': False
    }
    
    # Step 1: Generate planted_error file if needed
    planted_error_dir = category_dir / 'planted_error' / paper_folder_name
    flawed_papers_dir = planted_error_dir / 'flawed_papers'
    planted_error_md_path = flawed_papers_dir / f"{flaw_id}.md"
    csv_path = planted_error_dir / f"{openreview_id}_modifications_summary.csv"
    
    planted_error_exists = planted_error_md_path.exists()
    modifications = None
    flawed_paper = None
    metadata_list = None  # For storing metadata from try_apply_modifications
    
    # Try to load modifications from CSV first
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            flaw_row = df[df['flaw_id'] == flaw_id]
            if not flaw_row.empty:
                modifications_json = flaw_row.iloc[0].get('llm_generated_modifications', '')
                if modifications_json:
                    if isinstance(modifications_json, str):
                        modifications_data = json.loads(modifications_json)
                    else:
                        modifications_data = modifications_json
                    
                    # Check if modifications have start_marker and end_marker
                    has_markers = all(
                        mod.get('start_marker') and mod.get('end_marker')
                        for mod in modifications_data
                    )
                    
                    if has_markers:
                        # Convert to Modification objects
                        modifications = []
                        for mod_data in modifications_data:
                            modifications.append(Modification(
                                start_marker=mod_data.get('start_marker', ''),
                                end_marker=mod_data.get('end_marker', ''),
                                new_content=mod_data.get('new_content', ''),
                                reasoning=mod_data.get('reasoning', '')
                            ))
        except Exception as e:
            tqdm.write(f"  ⚠️ Error reading CSV for {openreview_id}/{flaw_id}: {e}")
    
    # If we have modifications from CSV, apply them directly (no API call)
    if modifications:
        if not planted_error_exists or not skip_existing_planted:
            if not planted_error_exists:
                tqdm.write(f"  ✂️  Applying modifications from CSV for {category_id}/{openreview_id}/{flaw_id}...")
            else:
                tqdm.write(f"  🔄 Re-applying modifications from CSV for {category_id}/{openreview_id}/{flaw_id}...")
            
            # Apply modifications directly (no API call - just apply the modifications from CSV)
            flawed_paper, success, error_msg, metadata_list = try_apply_modifications(
                original_paper,
                modifications
            )
            
            if not success or not flawed_paper:
                result['error'] = f'Failed to apply modifications: {error_msg}'
                return result
            
            # Save planted error version
            flawed_papers_dir.mkdir(parents=True, exist_ok=True)
            with open(planted_error_md_path, 'w', encoding='utf-8') as f:
                f.write(flawed_paper)
            
            result['planted_error_generated'] = True
            result['planted_error_path'] = str(planted_error_md_path)
        else:
            # Load existing flawed paper
            try:
                with open(planted_error_md_path, 'r', encoding='utf-8') as f:
                    flawed_paper = f.read()
                # Re-apply modifications to get metadata_list (for consistency)
                if modifications:
                    _, _, _, metadata_list = try_apply_modifications(
                        original_paper,
                        modifications
                    )
            except Exception as e:
                result['error'] = f'Error reading existing flawed paper: {e}'
                return result
            result['planted_error_path'] = str(planted_error_md_path)
            tqdm.write(f"  ⏭️  Skipping planted_error (already exists) for {category_id}/{openreview_id}/{flaw_id}")
    else:
        # No CSV modifications found - need to generate from scratch (expensive)
        if not planted_error_exists or not skip_existing_planted:
            if not planted_error_exists:
                tqdm.write(f"  🌱 Generating planted_error (no CSV found) for {category_id}/{openreview_id}/{flaw_id}...")
            else:
                tqdm.write(f"  🔄 Regenerating planted_error for {category_id}/{openreview_id}/{flaw_id}...")
            
            # Plant error using API (expensive) - only if CSV doesn't exist
            flawed_paper, modifications, success, error_msg, metadata = plant_error(
                original_paper,
                flaw_description,
                api_key,
                key_name,
                request_delay=request_delay,
                tpm_limit=tpm_limit,
                rpm_limit=rpm_limit
            )
            
            if not success or not flawed_paper:
                result['error'] = f'Failed to plant error: {error_msg}'
                return result
            
            # Convert metadata to metadata_list format for consistency
            # plant_error returns metadata as List[dict], which is the same format as metadata_list
            if metadata and isinstance(metadata, list):
                metadata_list = metadata
            else:
                metadata_list = []
            
            # Save planted error version
            flawed_papers_dir.mkdir(parents=True, exist_ok=True)
            with open(planted_error_md_path, 'w', encoding='utf-8') as f:
                f.write(flawed_paper)
            
            # Save metadata to CSV (modifications_summary.csv)
            # Read existing CSV if it exists
            if csv_path.exists():
                try:
                    existing_df = pd.read_csv(csv_path)
                except:
                    existing_df = pd.DataFrame(columns=['flaw_id', 'flaw_description', 'num_modifications', 'llm_generated_modifications'])
            else:
                existing_df = pd.DataFrame(columns=['flaw_id', 'flaw_description', 'num_modifications', 'llm_generated_modifications'])
            
            # Prepare modifications data for CSV
            modifications_data = []
            for mod in modifications:
                mod_dict = {
                    'start_marker': mod.start_marker,
                    'end_marker': mod.end_marker,
                    'new_content': mod.new_content,
                    'reasoning': mod.reasoning
                }
                modifications_data.append(mod_dict)
            
            # Check if this flaw_id already exists in CSV
            if 'flaw_id' in existing_df.columns:
                existing_df = existing_df[existing_df['flaw_id'] != flaw_id]
            
            # Add new row
            new_row = pd.DataFrame([{
                'flaw_id': flaw_id,
                'flaw_description': flaw_description,
                'num_modifications': len(modifications),
                'llm_generated_modifications': json.dumps(modifications_data, ensure_ascii=False)
            }])
            
            existing_df = pd.concat([existing_df, new_row], ignore_index=True)
            existing_df.to_csv(csv_path, index=False)
            
            result['planted_error_generated'] = True
            result['planted_error_path'] = str(planted_error_md_path)
        else:
            # Load existing flawed paper and modifications
            try:
                with open(planted_error_md_path, 'r', encoding='utf-8') as f:
                    flawed_paper = f.read()
            except Exception as e:
                result['error'] = f'Error reading existing flawed paper: {e}'
                return result
            
            # Try to load modifications from CSV
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    flaw_row = df[df['flaw_id'] == flaw_id]
                    if not flaw_row.empty:
                        modifications_json = flaw_row.iloc[0].get('llm_generated_modifications', '')
                        if modifications_json:
                            if isinstance(modifications_json, str):
                                modifications_data = json.loads(modifications_json)
                            else:
                                modifications_data = modifications_json
                            
                            modifications = []
                            for mod_data in modifications_data:
                                modifications.append(Modification(
                                    start_marker=mod_data.get('start_marker', ''),
                                    end_marker=mod_data.get('end_marker', ''),
                                    new_content=mod_data.get('new_content', ''),
                                    reasoning=mod_data.get('reasoning', '')
                                ))
                except Exception as e:
                    tqdm.write(f"  ⚠️ Error loading modifications from CSV: {e}")
            
            result['planted_error_path'] = str(planted_error_md_path)
            tqdm.write(f"  ⏭️  Skipping planted_error (already exists) for {category_id}/{openreview_id}/{flaw_id}")
    
    # Step 2: Generate sham_surgery file if needed
    # Ensure we have flawed_paper and modifications
    if not flawed_paper:
        # Try to load existing flawed paper
        if planted_error_md_path.exists():
            try:
                with open(planted_error_md_path, 'r', encoding='utf-8') as f:
                    flawed_paper = f.read()
            except Exception as e:
                result['error'] = f'Error reading flawed paper: {e}'
                return result
        else:
            result['error'] = 'Flawed paper not found and not generated'
            return result
    
    if not modifications:
        # Try to load modifications from CSV
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                flaw_row = df[df['flaw_id'] == flaw_id]
                if not flaw_row.empty:
                    modifications_json = flaw_row.iloc[0].get('llm_generated_modifications', '')
                    if modifications_json:
                        if isinstance(modifications_json, str):
                            modifications_data = json.loads(modifications_json)
                        else:
                            modifications_data = modifications_json
                        
                        modifications = []
                        for mod_data in modifications_data:
                            modifications.append(Modification(
                                start_marker=mod_data.get('start_marker', ''),
                                end_marker=mod_data.get('end_marker', ''),
                                new_content=mod_data.get('new_content', ''),
                                reasoning=mod_data.get('reasoning', '')
                            ))
            except Exception as e:
                tqdm.write(f"  ⚠️ Error loading modifications from CSV: {e}")
        
        # If still no modifications, try loading from planted_error_data
        if not modifications:
            planted_errors = load_planted_error_data(category_dir / 'planted_error')
            planted_error_data = None
            for pe in planted_errors:
                if pe['paper_folder'] == paper_folder_name and pe['flaw_id'] == flaw_id:
                    planted_error_data = pe
                    break
            
            if planted_error_data:
                metadata = planted_error_data['metadata']
                modifications = reconstruct_modifications_from_metadata(
                    metadata,
                    flawed_paper=flawed_paper,
                    original_paper=original_paper
                )
    
    if not modifications:
        result['error'] = 'Could not load modifications for placebo generation'
        return result
    
    sham_surgery_dir = category_dir / 'sham_surgery' / paper_folder_name
    sham_surgery_md_path = sham_surgery_dir / f"{flaw_id}.md"
    
    sham_surgery_exists = sham_surgery_md_path.exists()
    
    if not sham_surgery_exists or not skip_existing_sham:
        
        if not sham_surgery_exists:
            tqdm.write(f"  💊 Generating sham_surgery for {category_id}/{openreview_id}/{flaw_id}...")
        else:
            tqdm.write(f"  🔄 Regenerating sham_surgery for {category_id}/{openreview_id}/{flaw_id}...")
        
        # Generate placebo
        placebo_paper = generate_placebo(
            original_paper,
            flawed_paper,
            modifications,
            api_key,
            key_name,
            openreview_id,
            flaw_id,
            request_delay=request_delay,
            tpm_limit=tpm_limit,
            rpm_limit=rpm_limit
        )
        
        if not placebo_paper:
            result['error'] = 'Failed to generate placebo'
            return result
        
        # Save sham surgery version
        sham_surgery_dir.mkdir(parents=True, exist_ok=True)
        with open(sham_surgery_md_path, 'w', encoding='utf-8') as f:
            f.write(placebo_paper)
        
        # Save JSON metadata for sham surgery
        sham_surgery_metadata = {
            'paperid': openreview_id,
            'flaw_id': flaw_id,
            'flaw_description': flaw_description,
            'type': 'sham_surgery',
            'note': 'This is a placebo version where original sections were rewritten with the learned style but without introducing the flaw',
            'modifications': [
                {
                    'index': idx,
                    'start_marker': mod.start_marker,
                    'end_marker': mod.end_marker,
                    'reasoning': mod.reasoning,
                    'note': 'Original section was rewritten with learned style, preserving all tables/figures'
                }
                for idx, mod in enumerate(modifications)
            ],
            'metadata': metadata_list if metadata_list else []
        }
        sham_surgery_json_path = sham_surgery_dir / f"{flaw_id}.json"
        with open(sham_surgery_json_path, 'w', encoding='utf-8') as f:
            json.dump(sham_surgery_metadata, f, indent=2, ensure_ascii=False)
        
        result['sham_surgery_generated'] = True
        result['sham_surgery_path'] = str(sham_surgery_md_path)
    else:
        result['sham_surgery_path'] = str(sham_surgery_md_path)
        tqdm.write(f"  ⏭️  Skipping sham_surgery (already exists) for {category_id}/{openreview_id}/{flaw_id}")
    
    result['success'] = True
    return result


def process_category_from_planted_error_csvs(
    category_id: str,
    venue_dir: Path,
    model_name: str,
    max_workers: int = None,
    skip_existing_planted: bool = False,
    skip_existing_sham: bool = False,
    extract_markers: bool = False,
    task_counter: list = None
) -> dict:
    """
    Process a category by reading CSV files from planted_error directories.
    """
    category_dir = venue_dir / category_id
    planted_error_dir = category_dir / 'planted_error'
    
    if not planted_error_dir.exists():
        return {
            'category_id': category_id,
            'total_flaws': 0,
            'successful': 0,
            'error': f'Planted error directory not found: {planted_error_dir}'
        }
    
    # Find all modifications_summary.csv files
    csv_files = []
    # Look in subdirectories (paper folders)
    for paper_dir in planted_error_dir.iterdir():
        if paper_dir.is_dir():
            csv_files.extend(list(paper_dir.glob("*_modifications_summary.csv")))
    # Also look directly in planted_error_dir
    csv_files.extend(list(planted_error_dir.glob("*_modifications_summary.csv")))
    
    if not csv_files:
        return {
            'category_id': category_id,
            'total_flaws': 0,
            'successful': 0,
            'error': f'No modifications_summary.csv files found in {planted_error_dir}'
        }
    
    # Collect all flaws from CSV files
    all_flaws = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            paper_folder_name = csv_path.parent.name
            
            # Extract openreview_id from paper folder name or CSV filename
            if '_' in paper_folder_name:
                openreview_id = paper_folder_name.split('_')[0]
            else:
                # Try to extract from CSV filename
                csv_name = csv_path.stem
                if '_modifications_summary' in csv_name:
                    openreview_id = csv_name.replace('_modifications_summary', '')
                else:
                    openreview_id = paper_folder_name
            
            for _, row in df.iterrows():
                flaw_id = row.get('flaw_id', '')
                flaw_description = row.get('flaw_description', '')
                
                if flaw_id and flaw_description:
                    all_flaws.append({
                        'openreview_id': openreview_id,
                        'flaw_id': flaw_id,
                        'flaw_description': flaw_description,
                        'paper_folder': paper_folder_name,
                        'csv_path': csv_path
                    })
        except Exception as e:
            print(f"  ⚠️ Error reading {csv_path}: {e}")
            continue
    
    if not all_flaws:
        return {
            'category_id': category_id,
            'total_flaws': 0,
            'successful': 0,
            'error': 'No flaws found in CSV files'
        }
    
    print(f"\n  📂 Category {category_id}: Processing {len(all_flaws)} flaws from {len(csv_files)} CSV files")
    
    # Step 0: Extract markers from CSV files if requested
    if extract_markers:
        print(f"  🔍 Extracting markers from CSV files...")
        model_rpm = GEMINI_MODEL_RPM_LIMITS.get(model_name, 30)
        model_tpm = GEMINI_MODEL_TPM_LIMITS.get(model_name, 1000000)
        request_delay = get_request_delay_for_model(model_name)
        
        marker_task_idx = 0
        for csv_path in csv_files:
            paper_folder_name = csv_path.parent.name
            
            # Extract openreview_id
            if '_' in paper_folder_name:
                openreview_id = paper_folder_name.split('_')[0]
            else:
                csv_name = csv_path.stem
                if '_modifications_summary' in csv_name:
                    openreview_id = csv_name.replace('_modifications_summary', '')
                else:
                    openreview_id = paper_folder_name
            
            # Find original paper
            paper_folder = find_paper_folder(openreview_id, category_dir)
            if not paper_folder:
                continue
            
            original_paper_path = find_original_paper_md(paper_folder)
            if not original_paper_path:
                continue
            
            # Read original paper
            try:
                with open(original_paper_path, 'r', encoding='utf-8') as f:
                    original_paper = f.read()
            except Exception as e:
                tqdm.write(f"  ⚠️ Error reading original paper for {openreview_id}: {e}")
                continue
            
            # Get API key
            key_name, api_key = get_api_key_for_task(marker_task_idx)
            marker_task_idx += 1
            
            # Convert CSV
            converted = convert_csv_modifications_to_marker_format(
                csv_path=csv_path,
                original_paper=original_paper,
                api_key=api_key,
                key_name=key_name,
                model_name=model_name,
                request_delay=request_delay,
                tpm_limit=model_tpm,
                rpm_limit=model_rpm
            )
            
            if converted:
                tqdm.write(f"  ✅ Converted CSV: {csv_path.name}")
        
        # Re-read CSV files after conversion to get updated flaw data
        all_flaws = []
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
                paper_folder_name = csv_path.parent.name
                
                if '_' in paper_folder_name:
                    openreview_id = paper_folder_name.split('_')[0]
                else:
                    csv_name = csv_path.stem
                    if '_modifications_summary' in csv_name:
                        openreview_id = csv_name.replace('_modifications_summary', '')
                    else:
                        openreview_id = paper_folder_name
                
                for _, row in df.iterrows():
                    flaw_id = row.get('flaw_id', '')
                    flaw_description = row.get('flaw_description', '')
                    
                    if flaw_id and flaw_description:
                        all_flaws.append({
                            'openreview_id': openreview_id,
                            'flaw_id': flaw_id,
                            'flaw_description': flaw_description,
                            'paper_folder': paper_folder_name,
                            'csv_path': csv_path
                        })
            except Exception as e:
                print(f"  ⚠️ Error re-reading {csv_path}: {e}")
                continue
    
    # Set model limits
    model_rpm = GEMINI_MODEL_RPM_LIMITS.get(model_name, 30)
    model_tpm = GEMINI_MODEL_TPM_LIMITS.get(model_name, 1000000)
    request_delay = get_request_delay_for_model(model_name)
    
    # Set max_workers
    if max_workers is None:
        max_workers = len(GEMINI_API_KEYS)
    
    # Process flaws
    all_results = []
    local_task_counter = [0]
    
    def process_with_counter(flaw_data):
        idx = local_task_counter[0]
        local_task_counter[0] += 1
        # Use global task counter if provided for cross-category task distribution
        if task_counter is not None:
            global_idx = task_counter[0]
            task_counter[0] += 1
            idx = global_idx
        return process_flaw_from_csv(
            flaw_data['openreview_id'],
            flaw_data['flaw_id'],
            flaw_data['flaw_description'],
            category_id,
            venue_dir,
            model_name,
            idx,
            skip_existing_planted=skip_existing_planted,
            skip_existing_sham=skip_existing_sham,
            request_delay=request_delay,
            tpm_limit=model_tpm,
            rpm_limit=model_rpm
        )
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_with_counter, flaw_data): flaw_data
            for flaw_data in all_flaws
        }
        
        progress_bar = tqdm(
            concurrent.futures.as_completed(futures),
            total=len(all_flaws),
            desc=f"  Category {category_id}",
            leave=False
        )
        for future in progress_bar:
            try:
                result = future.result()
                if result:
                    all_results.append(result)
            except Exception as e:
                flaw_data = futures[future]
                tqdm.write(f"  ❌ Error processing {category_id}/{flaw_data['openreview_id']}/{flaw_data['flaw_id']}: {e}")
    
    # Calculate statistics
    total_flaws = len(all_results)
    successful = sum(1 for r in all_results if r.get('success', False))
    planted_generated = sum(1 for r in all_results if r.get('planted_error_generated', False))
    sham_generated = sum(1 for r in all_results if r.get('sham_surgery_generated', False))
    
    return {
        'category_id': category_id,
        'total_flaws': total_flaws,
        'successful': successful,
        'planted_error_generated': planted_generated,
        'sham_surgery_generated': sham_generated,
        'success_rate': (successful / total_flaws * 100) if total_flaws > 0 else 0.0,
        'results': all_results
    }


def main():
    parser = argparse.ArgumentParser(
        description="Process category CSV files to generate planted_error and sham_surgery files"
    )
    parser.add_argument(
        "--category_sampled_data_dir",
        type=str,
        required=False,
        help="[DEPRECATED] This argument is no longer used. The script now reads from planted_error CSV files."
    )
    parser.add_argument(
        "--venue_dir",
        type=str,
        required=True,
        help="Venue directory containing category folders (e.g., experiments/category_sampled_data/NeurIPS2024)"
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
        help="Specific categories to process (e.g., '1a' '1b'). If not provided, processes all categories."
    )
    parser.add_argument(
        "--skip_existing_planted",
        action='store_true',
        help="Skip generating planted_error files if they already exist"
    )
    parser.add_argument(
        "--skip_existing_sham",
        action='store_true',
        help="Skip generating sham_surgery files if they already exist"
    )
    parser.add_argument(
        "--extract_markers",
        action='store_true',
        help="Extract start_marker and end_marker from existing CSV files using LLM. This converts target_heading/new_content format to marker format without regenerating the entire flawed paper."
    )
    
    args = parser.parse_args()
    
    venue_dir = Path(args.venue_dir)
    
    if not venue_dir.exists():
        print(f"❌ Venue directory not found: {venue_dir}")
        return
    
    # Find all category directories
    category_dirs = [d for d in venue_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not category_dirs:
        print(f"❌ No category directories found in {venue_dir}")
        return
    
    # Filter to specific categories if provided
    if args.categories:
        category_dirs = [d for d in category_dirs if d.name in args.categories]
        if not category_dirs:
            print(f"❌ None of the specified categories {args.categories} were found")
            return
    
    print(f"✅ Found {len(category_dirs)} category directories")
    print(f"✅ Venue directory: {venue_dir}")
    print(f"✅ Using Gemini model: {args.model_name}")
    print(f"✅ Skip existing planted_error: {args.skip_existing_planted}")
    print(f"✅ Skip existing sham_surgery: {args.skip_existing_sham}")
    print(f"✅ Extract markers from CSV: {args.extract_markers}")
    
    # Get model limits
    model_rpm = GEMINI_MODEL_RPM_LIMITS.get(args.model_name, 30)
    model_tpm = GEMINI_MODEL_TPM_LIMITS.get(args.model_name, 1000000)
    
    max_workers = args.max_workers if args.max_workers else len(GEMINI_API_KEYS)
    print(f"✅ Using {max_workers} worker threads per category")
    print(f"✅ Model RPM limit: {model_rpm} requests/minute per key")
    print(f"✅ Model TPM limit: {model_tpm:,} tokens/minute per key")
    print()
    
    # Process each category
    all_category_results = []
    global_task_counter = [0]  # Shared task counter across all categories
    
    for category_dir in sorted(category_dirs):
        category_id = category_dir.name
        print(f"\n{'='*80}")
        print(f"Processing category: {category_id}")
        print(f"{'='*80}")
        
        result = process_category_from_planted_error_csvs(
            category_id=category_id,
            venue_dir=venue_dir,
            model_name=args.model_name,
            max_workers=max_workers,
            skip_existing_planted=args.skip_existing_planted,
            skip_existing_sham=args.skip_existing_sham,
            extract_markers=args.extract_markers,
            task_counter=global_task_counter
        )
        
        all_category_results.append(result)
        
        print(f"\n  ✅ Category {category_id} complete:")
        print(f"     Total flaws: {result['total_flaws']}")
        print(f"     Successful: {result['successful']}")
        print(f"     Planted errors generated: {result.get('planted_error_generated', 0)}")
        print(f"     Sham surgery generated: {result.get('sham_surgery_generated', 0)}")
        print(f"     Success rate: {result['success_rate']:.1f}%")
    
    # Print summary
    print(f"\n{'='*80}")
    print("✅ ALL CATEGORIES PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"\n📊 Summary Statistics:")
    
    total_flaws_all = sum(r['total_flaws'] for r in all_category_results)
    total_successful_all = sum(r['successful'] for r in all_category_results)
    total_planted_generated = sum(r.get('planted_error_generated', 0) for r in all_category_results)
    total_sham_generated = sum(r.get('sham_surgery_generated', 0) for r in all_category_results)
    overall_success_rate = (total_successful_all / total_flaws_all * 100) if total_flaws_all > 0 else 0.0
    
    print(f"   Total categories processed: {len(all_category_results)}")
    print(f"   Total flaws processed: {total_flaws_all}")
    print(f"   Total successful: {total_successful_all}")
    print(f"   Overall success rate: {overall_success_rate:.1f}%")
    print(f"   Planted errors generated: {total_planted_generated}")
    print(f"   Sham surgery generated: {total_sham_generated}")
    
    print(f"\n📊 Per-Category Breakdown:")
    for result in all_category_results:
        print(f"   {result['category_id']:>3s}: {result['successful']:>3d}/{result['total_flaws']:>3d} "
              f"({result['success_rate']:>5.1f}%) | "
              f"Planted: {result.get('planted_error_generated', 0):>3d} | "
              f"Sham: {result.get('sham_surgery_generated', 0):>3d}")
    
    print(f"\n📁 Files saved to:")
    print(f"   Planted errors: {venue_dir}/{{category_id}}/planted_error/{{paper_folder}}/flawed_papers/")
    print(f"   Sham surgery: {venue_dir}/{{category_id}}/sham_surgery/{{paper_folder}}/")
    print(f"{'='*80}")
    
    # Save summary CSV
    summary_data = []
    for result in all_category_results:
        summary_data.append({
            'category_id': result['category_id'],
            'total_flaws': result['total_flaws'],
            'successful': result['successful'],
            'planted_error_generated': result.get('planted_error_generated', 0),
            'sham_surgery_generated': result.get('sham_surgery_generated', 0),
            'success_rate': result['success_rate']
        })
    
    summary_csv_path = venue_dir / 'category_processing_summary.csv'
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n📄 Summary saved to: {summary_csv_path}")


if __name__ == "__main__":
    main()

