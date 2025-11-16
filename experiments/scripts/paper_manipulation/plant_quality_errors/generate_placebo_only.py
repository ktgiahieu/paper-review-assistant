#!/usr/bin/env python3
"""
Script to generate placebo/sham surgery versions from existing planted_error files.

This script reads planted_error files (both .md and .json) that were previously created,
and generates placebo versions by learning the writing style from the flawed papers
and applying it to rewrite the original sections without introducing the flaw.

The script can also extract start_marker and end_marker from CSV files that contain
target_heading and new_content, using an LLM to convert them to marker format before
generating placebos.

Usage:
    python generate_placebo_only.py \
        --planted_error_dir path/to/planted_error \
        --original_papers_dir path/to/original/papers \
        --output_dir path/to/output \
        [--extract_markers] \
        [--model_name gemini-2.0-flash]
"""

import os
import json
import argparse
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import threading
from collections import deque
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple, Dict
import google.generativeai as genai

# Import shared functions from the original script
# Use importlib to import from the script file
import sys
import importlib.util
script_dir = Path(__file__).parent
original_script_path = script_dir / 'plant_errors_and_placebo.py'

# Load the original script as a module
spec = importlib.util.spec_from_file_location("plant_errors_and_placebo", original_script_path)
plant_errors_module = importlib.util.module_from_spec(spec)
sys.modules["plant_errors_and_placebo"] = plant_errors_module
spec.loader.exec_module(plant_errors_module)

# Import necessary items from the module
GEMINI_API_KEYS = plant_errors_module.GEMINI_API_KEYS
DEFAULT_GEMINI_MODEL = plant_errors_module.DEFAULT_GEMINI_MODEL
GEMINI_MODEL_RPM_LIMITS = plant_errors_module.GEMINI_MODEL_RPM_LIMITS
GEMINI_MODEL_TPM_LIMITS = plant_errors_module.GEMINI_MODEL_TPM_LIMITS
get_request_delay_for_model = plant_errors_module.get_request_delay_for_model
GEMINI_MODEL = plant_errors_module.GEMINI_MODEL
GEMINI_REQUEST_DELAY = plant_errors_module.GEMINI_REQUEST_DELAY
key_request_times = plant_errors_module.key_request_times
key_lock = plant_errors_module.key_lock
key_dynamic_delays = plant_errors_module.key_dynamic_delays
key_delay_lock = plant_errors_module.key_delay_lock
key_last_429_time = plant_errors_module.key_last_429_time
key_429_lock = plant_errors_module.key_429_lock
key_token_usage = plant_errors_module.key_token_usage
tpm_lock = plant_errors_module.tpm_lock
Modification = plant_errors_module.Modification
StyleAnalysis = plant_errors_module.StyleAnalysis
RewrittenSection = plant_errors_module.RewrittenSection
PlaceboRewritingResponse = plant_errors_module.PlaceboRewritingResponse
get_api_key_for_task = plant_errors_module.get_api_key_for_task
estimate_tokens = plant_errors_module.estimate_tokens
wait_for_rate_limit = plant_errors_module.wait_for_rate_limit
find_marker_in_text = plant_errors_module.find_marker_in_text
extract_heading_from_content = plant_errors_module.extract_heading_from_content
find_heading_in_lines = plant_errors_module.find_heading_in_lines
extract_section_by_heading = plant_errors_module.extract_section_by_heading
clean_json_schema_for_gemini = plant_errors_module.clean_json_schema_for_gemini
call_gemini_with_retries = plant_errors_module.call_gemini_with_retries
generate_placebo = plant_errors_module.generate_placebo

# --- Marker Extraction Models and Functions ---

class MarkerExtraction(BaseModel):
    start_marker: str = Field(..., description="A unique text marker (3-10 words) that appears at the START of the section in the original paper. This should be the exact beginning text of the section, including the heading line.")
    end_marker: str = Field(..., description="A unique text marker (3-10 words) that appears at the END of the section in the original paper, just before the next section starts. This should be the exact ending text of the section.")

class MarkerExtractionResponse(BaseModel):
    markers: MarkerExtraction = Field(..., description="The extracted start and end markers for the section in the original paper.")


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

# --- Main Processing Functions ---

def load_planted_error_data(planted_error_dir: Path) -> List[Dict]:
    """
    Load all planted error files from a directory.
    Returns a list of dicts with 'paper_folder', 'flaw_id', 'flawed_paper_path', 'metadata_path', 'metadata'
    
    Handles multiple directory structures:
    1. planted_error_dir/{paper_folder}/{flaw_id}.md and {flaw_id}.json
    2. planted_error_dir/{paper_folder}/flawed_papers/{flaw_id}.md and {flaw_id}.json (in same dir or parent)
    3. planted_error_dir/{paper_folder}/flawed_papers/{flaw_id}.md with metadata in CSV file
    """
    planted_errors = []
    
    if not planted_error_dir.exists():
        return planted_errors
    
    # First, try to load from CSV if it exists (for structure with flawed_papers/)
    csv_files = list(planted_error_dir.rglob("*_modifications_summary.csv"))
    csv_metadata = {}
    
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            paper_folder_name = csv_path.parent.name
            
            for _, row in df.iterrows():
                flaw_id = row.get('flaw_id', '')
                flaw_description = row.get('flaw_description', '')
                modifications_json = row.get('llm_generated_modifications', '')
                
                if flaw_id and modifications_json:
                    try:
                        # Parse the JSON string from CSV
                        if isinstance(modifications_json, str):
                            modifications = json.loads(modifications_json)
                        else:
                            modifications = modifications_json
                        
                        # Create metadata structure similar to JSON files
                        metadata = {
                            'paperid': paper_folder_name.split('_')[0],
                            'flaw_id': flaw_id,
                            'flaw_description': flaw_description,
                            'modifications': modifications
                        }
                        
                        key = (paper_folder_name, flaw_id)
                        csv_metadata[key] = metadata
                    except Exception as e:
                        tqdm.write(f"  ⚠️ Error parsing modifications from CSV for {paper_folder_name}/{flaw_id}: {e}")
        except Exception as e:
            tqdm.write(f"  ⚠️ Error reading CSV {csv_path}: {e}")
    
    # Walk through the directory structure
    for paper_folder in planted_error_dir.iterdir():
        if not paper_folder.is_dir():
            continue
        
        paper_folder_name = paper_folder.name
        
        # Strategy 1: Look for .md files directly in paper_folder
        for file_path in paper_folder.iterdir():
            if file_path.is_file() and file_path.suffix == '.md':
                flaw_id = file_path.stem
                json_path = paper_folder / f"{flaw_id}.json"
                
                metadata = None
                metadata_path = None
                
                # Try JSON file first
                if json_path.exists():
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            metadata_path = json_path
                    except Exception as e:
                        tqdm.write(f"  ⚠️ Error loading metadata from {json_path}: {e}")
                
                # If no JSON, try CSV metadata
                if metadata is None:
                    key = (paper_folder_name, flaw_id)
                    if key in csv_metadata:
                        metadata = csv_metadata[key]
                        metadata_path = None  # No JSON file, metadata from CSV
                
                if metadata:
                    planted_errors.append({
                        'paper_folder': paper_folder_name,
                        'flaw_id': flaw_id,
                        'flawed_paper_path': file_path,
                        'metadata_path': metadata_path,
                        'metadata': metadata
                    })
        
        # Strategy 2: Look for .md files in flawed_papers subdirectory
        flawed_papers_dir = paper_folder / 'flawed_papers'
        if flawed_papers_dir.exists() and flawed_papers_dir.is_dir():
            for file_path in flawed_papers_dir.iterdir():
                if file_path.is_file() and file_path.suffix == '.md':
                    flaw_id = file_path.stem
                    
                    metadata = None
                    metadata_path = None
                    
                    # Try JSON in same directory first
                    json_path = flawed_papers_dir / f"{flaw_id}.json"
                    if json_path.exists():
                        try:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                                metadata_path = json_path
                        except Exception as e:
                            tqdm.write(f"  ⚠️ Error loading metadata from {json_path}: {e}")
                    
                    # If not found, try in parent directory
                    if metadata is None:
                        json_path = paper_folder / f"{flaw_id}.json"
                        if json_path.exists():
                            try:
                                with open(json_path, 'r', encoding='utf-8') as f:
                                    metadata = json.load(f)
                                    metadata_path = json_path
                            except Exception as e:
                                tqdm.write(f"  ⚠️ Error loading metadata from {json_path}: {e}")
                    
                    # If still no JSON, try CSV metadata
                    if metadata is None:
                        key = (paper_folder_name, flaw_id)
                        if key in csv_metadata:
                            metadata = csv_metadata[key]
                            metadata_path = None  # No JSON file, metadata from CSV
                    
                    if metadata:
                        planted_errors.append({
                            'paper_folder': paper_folder_name,
                            'flaw_id': flaw_id,
                            'flawed_paper_path': file_path,
                            'metadata_path': metadata_path,
                            'metadata': metadata
                        })
    
    return planted_errors

def find_original_paper(paper_folder_name: str, original_papers_dir: Path) -> Optional[Path]:
    """
    Find the original paper.md file for a given paper folder name.
    """
    # Try different possible locations
    possible_paths = [
        original_papers_dir / paper_folder_name / 'structured_paper_output' / 'paper.md',
        original_papers_dir / paper_folder_name / 'paper.md',
        original_papers_dir / f"{paper_folder_name.split('_')[0]}*" / 'structured_paper_output' / 'paper.md',
    ]
    
    # Try exact matches first
    for path in possible_paths[:2]:
        if path.exists():
            return path
    
    # Try glob pattern for the third option
    if '*' in str(possible_paths[2]):
        matching_folders = list(original_papers_dir.glob(f"{paper_folder_name.split('_')[0]}*"))
        for folder in matching_folders:
            paper_md = folder / 'structured_paper_output' / 'paper.md'
            if paper_md.exists():
                return paper_md
            paper_md = folder / 'paper.md'
            if paper_md.exists():
                return paper_md
    
    return None

def extract_markers_from_target_heading(flawed_paper: str, original_paper: str, target_heading: str, new_content: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract start_marker and end_marker from ORIGINAL paper using target_heading.
    This is critical because markers must match the original paper to find sections for rewriting.
    Returns (start_marker, end_marker) or (None, None) if not found.
    
    Priority:
    1. Extract from original_paper (preferred - markers must match original)
    2. Extract from new_content (fallback - from CSV)
    3. Extract from flawed_paper (last resort)
    """
    if not target_heading:
        return None, None
    
    # Strategy 1: Extract from original_paper (CRITICAL - markers must match original)
    if original_paper:
        lines = original_paper.split('\n')
        heading_index = find_heading_in_lines(lines, target_heading)
        
        if heading_index != -1:
            # Find the end of this section (next heading or end of document)
            start_line = heading_index
            end_line = len(lines)
            
            for i in range(start_line + 1, len(lines)):
                line = lines[i].strip()
                # Check if this is a new heading
                if line.startswith('#') or (line.startswith('**') and line.endswith('**') and len(line) > 4):
                    end_line = i
                    break
            
            # Extract the section from original paper
            section_lines = lines[start_line:end_line]
            
            if len(section_lines) > 0:
                heading_line = section_lines[0]
                # Get first 2-3 words from the first non-empty content line
                content_idx = 1
                while content_idx < len(section_lines) and not section_lines[content_idx].strip():
                    content_idx += 1
                
                if content_idx < len(section_lines):
                    first_content_line = section_lines[content_idx]
                    first_words = ' '.join(first_content_line.split()[:3])
                    start_marker = f"{heading_line}\n{first_words}"
                else:
                    start_marker = heading_line
                
                # Use last 3-5 words from the last non-empty line as end_marker
                for i in range(len(section_lines) - 1, -1, -1):
                    if section_lines[i].strip():
                        last_line = section_lines[i]
                        last_words = ' '.join(last_line.split()[-5:])
                        end_marker = last_words
                        return start_marker, end_marker
                else:
                    end_marker = section_lines[-1] if section_lines else ""
                    return start_marker, end_marker
    
    # Strategy 2: Extract from new_content (from CSV)
    if new_content:
        new_content_lines = new_content.split('\n')
        heading_index_in_content = find_heading_in_lines(new_content_lines, target_heading)
        
        if heading_index_in_content != -1 and len(new_content_lines) > heading_index_in_content:
            heading_line = new_content_lines[heading_index_in_content]
            
            # Find end of section in new_content
            end_line_in_content = len(new_content_lines)
            for i in range(heading_index_in_content + 1, len(new_content_lines)):
                line = new_content_lines[i].strip()
                if line.startswith('#') or (line.startswith('**') and line.endswith('**') and len(line) > 4):
                    end_line_in_content = i
                    break
            
            # Get first words for start_marker
            if len(new_content_lines) > heading_index_in_content + 1:
                content_idx = heading_index_in_content + 1
                while content_idx < end_line_in_content and not new_content_lines[content_idx].strip():
                    content_idx += 1
                if content_idx < end_line_in_content:
                    first_content_line = new_content_lines[content_idx]
                    first_words = ' '.join(first_content_line.split()[:3])
                    start_marker = f"{heading_line}\n{first_words}"
                else:
                    start_marker = heading_line
            else:
                start_marker = heading_line
            
            # Get last words for end_marker
            for i in range(end_line_in_content - 1, heading_index_in_content, -1):
                if new_content_lines[i].strip():
                    last_line = new_content_lines[i]
                    last_words = ' '.join(last_line.split()[-5:])
                    end_marker = last_words
                    return start_marker, end_marker
    
    # Strategy 3: Extract from flawed_paper (last resort)
    if flawed_paper:
        lines = flawed_paper.split('\n')
        heading_index = find_heading_in_lines(lines, target_heading)
        
        if heading_index != -1:
            start_line = heading_index
            end_line = len(lines)
            
            for i in range(start_line + 1, len(lines)):
                line = lines[i].strip()
                if line.startswith('#') or (line.startswith('**') and line.endswith('**') and len(line) > 4):
                    end_line = i
                    break
            
            section_lines = lines[start_line:end_line]
            
            if len(section_lines) > 0:
                heading_line = section_lines[0]
                content_idx = 1
                while content_idx < len(section_lines) and not section_lines[content_idx].strip():
                    content_idx += 1
                
                if content_idx < len(section_lines):
                    first_content_line = section_lines[content_idx]
                    first_words = ' '.join(first_content_line.split()[:3])
                    start_marker = f"{heading_line}\n{first_words}"
                else:
                    start_marker = heading_line
                
                for i in range(len(section_lines) - 1, -1, -1):
                    if section_lines[i].strip():
                        last_line = section_lines[i]
                        last_words = ' '.join(last_line.split()[-5:])
                        end_marker = last_words
                        return start_marker, end_marker
                else:
                    end_marker = section_lines[-1] if section_lines else ""
                    return start_marker, end_marker
    
    return None, None


def reconstruct_modifications_from_metadata(metadata: Dict, flawed_paper: str = None, original_paper: str = None) -> List[Modification]:
    """
    Reconstruct Modification objects from JSON metadata.
    If flawed_paper is provided, will extract full new_content from it.
    Handles both formats:
    1. Standard format: start_marker, end_marker, new_content
    2. CSV format: target_heading, new_content (will extract markers from original_paper)
    
    IMPORTANT: For CSV format, markers are extracted from original_paper (not flawed_paper)
    to ensure they match the original sections that need to be rewritten.
    """
    modifications = []
    
    # The metadata should have a 'modifications' list
    mods_data = metadata.get('modifications', [])
    
    for mod_data in mods_data:
        # Try standard format first
        start_marker = mod_data.get('start_marker', '')
        end_marker = mod_data.get('end_marker', '')
        
        # If markers are missing, try CSV format with target_heading
        if not start_marker or not end_marker:
            target_heading = mod_data.get('target_heading', '')
            # We need at least target_heading and either original_paper or new_content
            if target_heading and (original_paper or flawed_paper or mod_data.get('new_content')):
                # Extract markers - prioritize original_paper, then new_content, then flawed_paper
                extracted_start, extracted_end = extract_markers_from_target_heading(
                    flawed_paper, original_paper, target_heading, mod_data.get('new_content', '')
                )
                if extracted_start and extracted_end:
                    start_marker = extracted_start
                    end_marker = extracted_end
                    # Debug: show which source was used
                    if original_paper and find_heading_in_lines(original_paper.split('\n'), target_heading) != -1:
                        tqdm.write(f"  ✅ Extracted markers from original paper for: {target_heading[:50]}...")
                    elif mod_data.get('new_content'):
                        tqdm.write(f"  ✅ Extracted markers from CSV new_content for: {target_heading[:50]}...")
                    elif flawed_paper:
                        tqdm.write(f"  ⚠️ Extracted markers from flawed paper (may not match original) for: {target_heading[:50]}...")
                else:
                    # If extraction fails, skip this modification
                    tqdm.write(f"  ⚠️ Could not extract markers for heading: {target_heading[:50]}...")
                    continue
            else:
                # No markers and no target_heading, skip
                if not target_heading:
                    tqdm.write(f"  ⚠️ Modification missing both markers and target_heading, skipping")
                continue
        
        # Try to get new_content from metadata first
        new_content = mod_data.get('new_content', '')
        
        # If not available or too short, try to extract from flawed_paper using markers
        if flawed_paper and (not new_content or len(new_content) < 100):
            if start_marker and end_marker:
                extracted = extract_new_content_from_flawed_paper(
                    flawed_paper, start_marker, end_marker
                )
                if extracted:
                    new_content = extracted
        
        # If still no content, try to extract using target_heading
        if (not new_content or len(new_content) < 100) and flawed_paper:
            target_heading = mod_data.get('target_heading', '')
            if target_heading:
                # Extract section by heading
                extracted_section = extract_section_by_heading(flawed_paper, target_heading)
                if extracted_section:
                    new_content = extracted_section
        
        # Final fallback
        if not new_content:
            new_content = mod_data.get('new_content', mod_data.get('new_content_preview', ''))
        
        modification = Modification(
            start_marker=start_marker,
            end_marker=end_marker,
            new_content=new_content,
            reasoning=mod_data.get('reasoning', '')
        )
        modifications.append(modification)
    
    return modifications

def extract_new_content_from_flawed_paper(flawed_paper: str, start_marker: str, end_marker: str) -> Optional[str]:
    """
    Extract the new_content section from flawed_paper using start and end markers.
    """
    lines = flawed_paper.split('\n')
    start_line = find_marker_in_text(flawed_paper, start_marker)
    
    if start_line is None:
        return None
    
    lines_after_start = lines[start_line+1:]
    text_after_start = '\n'.join(lines_after_start)
    end_line_relative = find_marker_in_text(text_after_start, end_marker)
    
    if end_line_relative is None:
        return None
    
    end_line = start_line + 1 + end_line_relative
    new_content = '\n'.join(lines[start_line:end_line+1])
    return new_content

def process_planted_error(
    planted_error_data: Dict,
    original_papers_dir: Path,
    output_dir: Path,
    task_idx: int,
    request_delay: float = None,
    tpm_limit: int = 1000000,
    rpm_limit: int = 30
) -> Optional[Dict]:
    """
    Process a single planted error to generate placebo.
    """
    paper_folder_name = planted_error_data['paper_folder']
    flaw_id = planted_error_data['flaw_id']
    flawed_paper_path = planted_error_data['flawed_paper_path']
    metadata = planted_error_data['metadata']
    
    # Read flawed paper
    try:
        with open(flawed_paper_path, 'r', encoding='utf-8') as f:
            flawed_paper = f.read()
    except Exception as e:
        tqdm.write(f"  ❌ Error reading flawed paper {flawed_paper_path}: {e}")
        return None
    
    # Find original paper
    original_paper_path = find_original_paper(paper_folder_name, original_papers_dir)
    if not original_paper_path:
        tqdm.write(f"  ❌ Original paper not found for {paper_folder_name}")
        return None
    
    try:
        with open(original_paper_path, 'r', encoding='utf-8') as f:
            original_paper = f.read()
    except Exception as e:
        tqdm.write(f"  ❌ Error reading original paper {original_paper_path}: {e}")
        return None
    
    # Reconstruct modifications from metadata, extracting new_content from flawed_paper
    modifications = reconstruct_modifications_from_metadata(metadata, flawed_paper=flawed_paper, original_paper=original_paper)
    
    if not modifications:
        tqdm.write(f"  ❌ No modifications found in metadata for {paper_folder_name}/{flaw_id}")
        return None
    
    # Get API key for this task
    key_name, api_key = get_api_key_for_task(task_idx)
    
    paperid = metadata.get('paperid', paper_folder_name.split('_')[0])
    flaw_description = metadata.get('flaw_description', '')
    
    # Generate placebo
    tqdm.write(f"  💊 Generating placebo for {paperid}, flaw {flaw_id}...")
    placebo_paper = generate_placebo(
        original_paper,
        flawed_paper,
        modifications,
        api_key,
        key_name,
        paperid,
        flaw_id,
        request_delay=request_delay,
        tpm_limit=tpm_limit,
        rpm_limit=rpm_limit
    )
    
    if not placebo_paper:
        tqdm.write(f"  ❌ Failed to generate placebo for {paperid}, flaw {flaw_id}")
        return None
    
    # Save placebo version
    sham_surgery_dir = output_dir / 'sham_surgery' / paper_folder_name
    sham_surgery_dir.mkdir(parents=True, exist_ok=True)
    sham_surgery_path = sham_surgery_dir / f"{flaw_id}.md"
    
    with open(sham_surgery_path, 'w', encoding='utf-8') as f:
        f.write(placebo_paper)
    
    # Save JSON metadata for sham surgery
    sham_surgery_metadata = {
        'paperid': paperid,
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
        'metadata': metadata.get('metadata', [])
    }
    sham_surgery_json_path = sham_surgery_dir / f"{flaw_id}.json"
    with open(sham_surgery_json_path, 'w', encoding='utf-8') as f:
        json.dump(sham_surgery_metadata, f, indent=2, ensure_ascii=False)
    
    tqdm.write(f"  ✅ Successfully generated placebo for {paperid}, flaw {flaw_id}")
    
    return {
        'paperid': paperid,
        'paper_folder': paper_folder_name,
        'flaw_id': flaw_id,
        'flaw_description': flaw_description,
        'sham_surgery_path': str(sham_surgery_path),
        'modifications_count': len(modifications),
        'success': True
    }

def main():
    global GEMINI_MODEL, GEMINI_REQUEST_DELAY
    
    parser = argparse.ArgumentParser(
        description="Generate placebo/sham surgery versions from existing planted_error files using Gemini API."
    )
    parser.add_argument(
        "--planted_error_dir",
        type=str,
        required=True,
        help="Directory containing planted_error files (with structure: {paper_folder}/{flaw_id}.md and {flaw_id}.json)"
    )
    parser.add_argument(
        "--original_papers_dir",
        type=str,
        required=True,
        help="Directory containing original paper folders (with paper.md files)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as planted_error_dir parent)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Max worker threads (default: calculated based on model and API keys)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_GEMINI_MODEL,
        help=f"Gemini model name to use (default: {DEFAULT_GEMINI_MODEL})"
    )
    parser.add_argument(
        "--extract_markers",
        action='store_true',
        help="Extract start_marker and end_marker from existing CSV files using LLM. This converts target_heading/new_content format to marker format before generating placebos."
    )
    args = parser.parse_args()
    
    # Set the global model name and request delay
    GEMINI_MODEL = args.model_name
    GEMINI_REQUEST_DELAY = get_request_delay_for_model(GEMINI_MODEL)
    
    # Get model RPM and TPM limits
    model_rpm = GEMINI_MODEL_RPM_LIMITS.get(GEMINI_MODEL, 30)
    model_tpm = GEMINI_MODEL_TPM_LIMITS.get(GEMINI_MODEL, 1000000)
    
    print(f"✅ Using Gemini model: {GEMINI_MODEL}")
    print(f"✅ Model RPM limit: {model_rpm} requests/minute per key")
    print(f"✅ Model TPM limit: {model_tpm:,} tokens/minute per key")
    print(f"✅ Request delay: {GEMINI_REQUEST_DELAY:.2f} seconds per key")
    print(f"✅ Extract markers from CSV: {args.extract_markers}")
    
    planted_error_dir = Path(args.planted_error_dir)
    original_papers_dir = Path(args.original_papers_dir)
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = planted_error_dir.parent
    output_dir = Path(args.output_dir)
    
    # Step 0: Extract markers from CSV files if requested
    if args.extract_markers:
        print(f"\n🔍 Extracting markers from CSV files...")
        csv_files = list(planted_error_dir.rglob("*_modifications_summary.csv"))
        
        if csv_files:
            print(f"✅ Found {len(csv_files)} CSV files to process")
            
            marker_task_idx = 0
            for csv_path in tqdm(csv_files, desc="Extracting markers"):
                paper_folder_name = csv_path.parent.name
                
                # Extract openreview_id from paper folder name
                if '_' in paper_folder_name:
                    openreview_id = paper_folder_name.split('_')[0]
                else:
                    # Try to extract from CSV filename
                    csv_name = csv_path.stem
                    if '_modifications_summary' in csv_name:
                        openreview_id = csv_name.replace('_modifications_summary', '')
                    else:
                        openreview_id = paper_folder_name
                
                # Find original paper
                original_paper_path = find_original_paper(paper_folder_name, original_papers_dir)
                if not original_paper_path:
                    tqdm.write(f"  ⚠️ Original paper not found for {paper_folder_name}, skipping CSV: {csv_path.name}")
                    continue
                
                # Read original paper
                try:
                    with open(original_paper_path, 'r', encoding='utf-8') as f:
                        original_paper = f.read()
                except Exception as e:
                    tqdm.write(f"  ⚠️ Error reading original paper for {paper_folder_name}: {e}")
                    continue
                
                # Get API key
                key_name, api_key = get_api_key_for_task(marker_task_idx)
                marker_task_idx = (marker_task_idx + 1) % len(GEMINI_API_KEYS)
                
                # Convert CSV
                converted = convert_csv_modifications_to_marker_format(
                    csv_path=csv_path,
                    original_paper=original_paper,
                    api_key=api_key,
                    key_name=key_name,
                    model_name=GEMINI_MODEL,
                    request_delay=GEMINI_REQUEST_DELAY,
                    tpm_limit=model_tpm,
                    rpm_limit=model_rpm
                )
                
                if converted:
                    tqdm.write(f"  ✅ Converted CSV: {csv_path.name}")
            
            print(f"✅ Marker extraction complete")
        else:
            print(f"⚠️ No CSV files found in {planted_error_dir}")
    
    # Load all planted error files
    print(f"\n📂 Loading planted error files from: {planted_error_dir}")
    planted_errors = load_planted_error_data(planted_error_dir)
    print(f"✅ Found {len(planted_errors)} planted error files")
    
    if not planted_errors:
        print("❌ No planted error files found. Exiting.")
        return
    
    # Set max_workers
    if args.max_workers is not None:
        max_workers = args.max_workers
    else:
        max_workers = len(GEMINI_API_KEYS)
    
    # Calculate theoretical throughput
    total_throughput = len(GEMINI_API_KEYS) * model_rpm
    estimated_tpm_per_request = 50000  # Conservative estimate
    estimated_tpm_throughput = max_workers * estimated_tpm_per_request
    print(f"\n✅ Using {max_workers} worker threads (one per API key)")
    print(f"✅ Total theoretical throughput: {total_throughput} RPM ({len(GEMINI_API_KEYS)} keys × {model_rpm} RPM)")
    print(f"✅ Rate limiting: Sliding window tracking with dynamic backoff on 429 errors")
    print(f"⚠️ Estimated TPM usage: ~{estimated_tpm_throughput:,} tokens/min per key (if all workers active)")
    print(f"⚠️ TPM limit per key: {model_tpm:,} tokens/min")
    if estimated_tpm_throughput > model_tpm * 0.8:
        print(f"⚠️ WARNING: Estimated TPM usage ({estimated_tpm_throughput:,}) may exceed limit ({model_tpm:,})")
    print()
    
    # Process planted errors
    all_results = []
    task_counter = [0]
    
    def process_with_counter(planted_error_data):
        idx = task_counter[0]
        task_counter[0] += 1
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
            desc="Generating Placebos"
        )
        for future in progress_bar:
            try:
                result = future.result()
                if result:
                    all_results.append(result)
            except Exception as e:
                pe = futures[future]
                tqdm.write(f"Error processing {pe['paper_folder']}/{pe['flaw_id']}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save results CSV
    if all_results:
        results_csv_path = output_dir / 'placebo_generation_results.csv'
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(results_csv_path, index=False)
        
        # Calculate statistics
        total_flaws = len(all_results)
        successful_placebo = sum(1 for r in all_results if r.get('success', False))
        
        print(f"\n{'='*80}")
        print("✅ PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"📊 Statistics:")
        print(f"   Total flaws processed: {total_flaws}")
        print(f"   Successfully generated placebos: {successful_placebo} ({successful_placebo/total_flaws*100:.1f}%)")
        print(f"\n📁 Results saved to: {results_csv_path}")
        print(f"📁 Sham surgery files: {output_dir / 'sham_surgery'}")
        print(f"{'='*80}")
    else:
        print("\n⚠️ No results to save - no placebos were successfully generated")

if __name__ == "__main__":
    main()

