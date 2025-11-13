import os
import csv
import json
import argparse
import time
import re
import pandas as pd
import ast
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import threading
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple
import google.generativeai as genai

# --- Environment & API Configuration ---
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent.parent / '.env')

# Load multiple Gemini API keys
GEMINI_API_KEYS = {}
for i in range(1, 10):  # Support up to 9 keys
    key = os.getenv(f'GEMINI_API_KEY_{i}')
    if key:
        GEMINI_API_KEYS[str(i)] = key

if not GEMINI_API_KEYS:
    raise ValueError("No Gemini API keys found in environment variables (GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.)")

print(f"✅ Loaded {len(GEMINI_API_KEYS)} Gemini API keys: {list(GEMINI_API_KEYS.keys())}")

GEMINI_MODEL = "gemini-2.0-flash-lite"
# Rate limit: 30 RPM = 1 request per 2 seconds per key
GEMINI_REQUEST_DELAY = 2.0  # seconds between requests per API key

# Rate limiting tracking
key_last_used = {}
key_lock = threading.Lock()

# --- Pydantic Models for Structured Responses ---

class Modification(BaseModel):
    target_heading: str = Field(..., description="The exact markdown heading of the section to replace.")
    new_content: str = Field(..., description="The complete, rewritten text for the entire section, including the heading.")
    reasoning: str = Field(..., description="A brief explanation for the modification.")

class ModificationGenerationResponse(BaseModel):
    modifications: List[Modification] = Field(..., description="A list of sections to modify to introduce the flaw.")

class StyleAnalysis(BaseModel):
    style_description: str = Field(..., description="A detailed description of the writing style observed in the modified paragraphs.")
    key_characteristics: List[str] = Field(..., description="List of key writing style characteristics (e.g., sentence structure, terminology, tone).")

class RewrittenSection(BaseModel):
    original_heading: str = Field(..., description="The original heading of the section.")
    new_content: str = Field(..., description="The rewritten content for the section.")

class PlaceboRewritingResponse(BaseModel):
    rewritten_section: RewrittenSection = Field(..., description="The rewritten section.")

# --- Helper Functions ---

def get_api_key_for_task(task_idx: int) -> Tuple[str, str]:
    """Get API key for a task using round-robin assignment."""
    key_names = list(GEMINI_API_KEYS.keys())
    key_name = key_names[task_idx % len(key_names)]
    return key_name, GEMINI_API_KEYS[key_name]

def wait_for_rate_limit(key_name: str):
    """Wait if necessary to respect rate limits."""
    with key_lock:
        if key_name in key_last_used:
            elapsed = time.time() - key_last_used[key_name]
            if elapsed < GEMINI_REQUEST_DELAY:
                sleep_time = GEMINI_REQUEST_DELAY - elapsed
                time.sleep(sleep_time)
        key_last_used[key_name] = time.time()

def clean_heading_text_aggressively(text: str) -> str:
    """Aggressively clean heading text for matching."""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\[[^\]]*?\]', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'\\[a-zA-Z@]+({.*?})?|[\{\}\$\(\)\\]', '', text)
    text = text.strip().strip('#*').strip()
    text = text.rstrip('.,;:')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def try_apply_modifications(original_markdown: str, modifications: List[Modification]) -> Tuple[str, bool, Optional[str]]:
    """Apply modifications to markdown using tiered matching strategy."""
    current_markdown = original_markdown
    lines = original_markdown.split('\n')
    
    for mod in modifications:
        target_heading = mod.target_heading.strip()
        if not target_heading:
            continue

        match_index = -1

        # Tier 1: Exact match
        for i, line in enumerate(lines):
            if line.strip() == target_heading:
                match_index = i
                break
        
        # Tier 2: Match after stripping whitespace and markdown
        if match_index == -1:
            semi_cleaned_target = target_heading.strip('#* \t')
            for i, line in enumerate(lines):
                semi_cleaned_line = line.strip().strip('#* \t')
                if semi_cleaned_line == semi_cleaned_target and semi_cleaned_line:
                    match_index = i
                    break

        # Tier 3: Aggressive cleaning
        if match_index == -1:
            aggressively_cleaned_target = clean_heading_text_aggressively(target_heading)
            for i, line in enumerate(lines):
                aggressively_cleaned_line = clean_heading_text_aggressively(line)
                
                if not aggressively_cleaned_line or not aggressively_cleaned_target:
                    continue

                if aggressively_cleaned_line.lower().startswith(aggressively_cleaned_target.lower()):
                    match_index = i
                    break

        if match_index == -1:
            return original_markdown, False, target_heading

        # Apply modification
        start_line = match_index
        
        # Find end of section (next heading)
        end_line = len(lines)
        for i in range(start_line + 1, len(lines)):
            line_to_check = lines[i].strip()
            is_hash_heading = line_to_check.startswith('#')
            is_bold_heading = line_to_check.startswith('**') and line_to_check.endswith('**')
            is_italic_heading = line_to_check.startswith('*') and line_to_check.endswith('*') and not is_bold_heading

            if is_hash_heading or is_bold_heading or is_italic_heading:
                end_line = i
                break

        # Reconstruct markdown
        pre_section_lines = lines[:start_line]
        post_section_lines = lines[end_line:]
        new_content_lines = mod.new_content.split('\n')
        lines = pre_section_lines + new_content_lines + post_section_lines
        current_markdown = '\n'.join(lines)
            
    return current_markdown, True, None

def extract_section_by_heading(markdown: str, heading: str) -> Optional[str]:
    """Extract a section from markdown by its heading."""
    lines = markdown.split('\n')
    
    # Find heading
    match_index = -1
    for i, line in enumerate(lines):
        if line.strip() == heading or line.strip().strip('#* \t') == heading.strip('#* \t'):
            match_index = i
            break
    
    if match_index == -1:
        # Try aggressive cleaning
        cleaned_heading = clean_heading_text_aggressively(heading)
        for i, line in enumerate(lines):
            if clean_heading_text_aggressively(line).lower().startswith(cleaned_heading.lower()):
                match_index = i
                break
    
    if match_index == -1:
        return None
    
    # Find end of section
    start_line = match_index
    end_line = len(lines)
    for i in range(start_line + 1, len(lines)):
        line_to_check = lines[i].strip()
        is_hash_heading = line_to_check.startswith('#')
        is_bold_heading = line_to_check.startswith('**') and line_to_check.endswith('**')
        is_italic_heading = line_to_check.startswith('*') and line_to_check.endswith('*') and not is_bold_heading

        if is_hash_heading or is_bold_heading or is_italic_heading:
            end_line = i
            break
    
    return '\n'.join(lines[start_line:end_line])

def clean_json_schema_for_gemini(schema: dict) -> dict:
    """Remove unsupported fields and inline definitions to make schema compatible with Gemini."""
    import copy
    
    # Fields that Gemini Schema proto doesn't support
    UNSUPPORTED_FIELDS = {'$defs', 'title', 'description', '$schema', 'definitions'}
    
    # Make a deep copy to avoid modifying the original
    schema = copy.deepcopy(schema)
    defs = schema.pop('$defs', {})
    
    def clean_and_inline_refs(obj):
        """Recursively remove unsupported fields and inline $ref references."""
        if isinstance(obj, dict):
            # Remove unsupported top-level fields
            cleaned = {k: v for k, v in obj.items() if k not in UNSUPPORTED_FIELDS}
            
            # Handle $ref references
            if '$ref' in obj:
                ref_path = obj['$ref']
                if ref_path.startswith('#/$defs/'):
                    def_name = ref_path.replace('#/$defs/', '')
                    if def_name in defs:
                        # Inline the definition (deep copy to avoid circular refs)
                        inlined = copy.deepcopy(defs[def_name])
                        # Continue cleaning and inlining any nested refs
                        return clean_and_inline_refs(inlined)
                # If ref doesn't point to $defs, keep it as is (but clean it)
                return cleaned
            
            # Recursively process all values
            return {k: clean_and_inline_refs(v) for k, v in cleaned.items()}
        elif isinstance(obj, list):
            return [clean_and_inline_refs(item) for item in obj]
        return obj
    
    return clean_and_inline_refs(schema)

def call_gemini_with_retries(api_key: str, key_name: str, prompt: str, response_model: type, max_retries: int = 3) -> Optional[BaseModel]:
    """Call Gemini API with retries and structured output parsing."""
    wait_for_rate_limit(key_name)
    
    genai.configure(api_key=api_key)
    
    # Get JSON schema from Pydantic model and clean it for Gemini
    json_schema = response_model.model_json_schema()
    json_schema = clean_json_schema_for_gemini(json_schema)
    
    model = genai.GenerativeModel(
        GEMINI_MODEL,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": json_schema,
        }
    )
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            json_text = response.text.strip()
            
            # Clean JSON if needed
            json_text = re.sub(r'```json\s*', '', json_text)
            json_text = re.sub(r'```\s*$', '', json_text)
            json_text = json_text.strip()
            
            result = response_model.model_validate_json(json_text)
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                tqdm.write(f"Error calling Gemini API (key {key_name}): {e}")
                import traceback
                traceback.print_exc()
                return None
    
    return None

# --- Main Processing Functions ---

def plant_error(original_paper: str, flaw_description: str, api_key: str, key_name: str) -> Tuple[Optional[str], Optional[List[Modification]], bool]:
    """Plant an error in the paper based on flaw description."""
    
    prompt = f"""You are an AI model creating a benchmark of flawed scholarly papers for a research experiment. Your objective is to modify a published research paper to re-introduce a specific flaw in a way that makes the paper's claims methodologically unsound.

The key is to make the paper *embody* the flaw, not just state it as a limitation. You must adopt the persona of an author who is either unaware of the flaw or actively downplaying it to make their results seem more significant. Stay true to the authors' original material and personas as much as possible, use scientific terms, avoid writing in an obviously different manner to create an illusion that the paper is coherent, well-written and the introduced flaws are as hard to detect as possible.

**The Golden Rule:**
When rewriting, you must follow this golden rule: The flaw should **NOT** be acknowledged as a weakness in the "Limitations" or "Future Work" sections. Instead, the authors should either ignore it completely or misrepresent it as a minor point or even a strength. The conclusions and abstract must be written with unearned confidence, drawing broad claims from the now-flawed, narrow evidence.

---
Your Task:
Generate the new, complete text for only the sections that need to be changed to systemically and confidently embed the flaw. Each rewritten section must begin with its original, exact markdown heading line. The rewritten text must maintain the original paper's style but reflect the overconfident persona.

Your output MUST be a JSON object conforming to the provided schema with a list of modifications.

---
The flaw to re-introduce is:
<BEGIN FLAW>
{flaw_description}
<END FLAW>

The original paper:
<BEGIN PAPER>
{original_paper}
<END PAPER>
"""
    
    response = call_gemini_with_retries(api_key, key_name, prompt, ModificationGenerationResponse)
    
    if not response or not response.modifications:
        return None, None, False
    
    # Apply modifications
    flawed_paper, success, failed_heading = try_apply_modifications(original_paper, response.modifications)
    
    if not success:
        return None, response.modifications, False
    
    return flawed_paper, response.modifications, True

def generate_placebo(original_paper: str, flawed_paper: str, modifications: List[Modification], api_key: str, key_name: str, paperid: str = "", flaw_id: str = "") -> Optional[str]:
    """Generate placebo/sham surgery version by learning style and rewriting original sections."""
    
    # Extract modified sections from flawed paper
    modified_sections = []
    for mod in modifications:
        section = extract_section_by_heading(flawed_paper, mod.target_heading)
        if section:
            modified_sections.append(section)
        else:
            tqdm.write(f"  ⚠️ Could not extract modified section for heading: {mod.target_heading[:50]}... (paper: {paperid}, flaw: {flaw_id})")
    
    if not modified_sections:
        tqdm.write(f"  ❌ No modified sections found for placebo generation (paper: {paperid}, flaw: {flaw_id})")
        return None
    
    modified_text = "\n\n---\n\n".join(modified_sections)
    
    # Truncate if too long (Gemini has context limits)
    if len(modified_text) > 50000:  # Rough estimate for token limit
        modified_text = modified_text[:50000] + "\n\n[Truncated...]"
    
    # Step 1: Analyze writing style
    style_prompt = f"""Analyze the writing style of the following modified sections from a research paper. Identify the key characteristics of the writing style including sentence structure, terminology, tone, and any distinctive patterns.

Modified sections:
<BEGIN MODIFIED SECTIONS>
{modified_text}
<END MODIFIED SECTIONS>

Your output MUST be a JSON object with style_description (detailed description) and key_characteristics (list of key features).
"""
    
    style_response = call_gemini_with_retries(api_key, key_name, style_prompt, StyleAnalysis)
    
    if not style_response:
        tqdm.write(f"  ❌ Style analysis failed (paper: {paperid}, flaw: {flaw_id})")
        return None
    
    # Step 2: Extract original sections before modification
    original_sections = []
    for mod in modifications:
        section = extract_section_by_heading(original_paper, mod.target_heading)
        if section:
            original_sections.append({
                "heading": mod.target_heading,
                "content": section
            })
        else:
            tqdm.write(f"  ⚠️ Could not extract original section for heading: {mod.target_heading[:50]}... (paper: {paperid}, flaw: {flaw_id})")
    
    if not original_sections:
        tqdm.write(f"  ❌ No original sections found for placebo generation (paper: {paperid}, flaw: {flaw_id})")
        return None
    
    # Step 3: Rewrite original sections using the learned style (without dropping info)
    # Collect all rewritten sections first, then apply them
    rewritten_sections = []
    
    for section_info in original_sections:
        # Truncate section if too long
        section_content = section_info['content']
        if len(section_content) > 30000:  # Rough estimate for token limit
            section_content = section_content[:30000] + "\n\n[Truncated...]"
        
        rewrite_prompt = f"""Rewrite the following section from a research paper using the writing style described below. 

**CRITICAL REQUIREMENTS:**
1. Maintain ALL factual information from the original section - do not drop any details
2. Apply the writing style characteristics to rewrite the content
3. Keep the same heading format (exact match as in the original)
4. Preserve all technical terms, numbers, and specific claims
5. Only change the writing style, not the content meaning

Writing style characteristics:
{style_response.style_description}

Key characteristics:
{chr(10).join(f"- {c}" for c in style_response.key_characteristics)}

Original section to rewrite:
<BEGIN ORIGINAL SECTION>
{section_content}
<END ORIGINAL SECTION>

Your output MUST be a JSON object with rewritten_section containing original_heading (must match exactly) and new_content fields.
"""
        
        rewrite_response = call_gemini_with_retries(api_key, key_name, rewrite_prompt, PlaceboRewritingResponse)
        
        if rewrite_response and rewrite_response.rewritten_section:
            rewritten_sections.append({
                'heading': rewrite_response.rewritten_section.original_heading,
                'content': rewrite_response.rewritten_section.new_content,
                'original_heading': section_info['heading']  # Keep original for matching
            })
        else:
            tqdm.write(f"  ⚠️ Failed to rewrite section: {section_info['heading'][:50]}... (paper: {paperid}, flaw: {flaw_id})")
    
    if not rewritten_sections:
        tqdm.write(f"  ❌ No sections were successfully rewritten (paper: {paperid}, flaw: {flaw_id})")
        return None
    
    # Apply all rewritten sections to the paper
    placebo_paper = original_paper
    lines = placebo_paper.split('\n')
    
    # Process sections in reverse order to maintain line indices
    for section in reversed(rewritten_sections):
        heading_to_match = section['original_heading']  # Use original heading for matching
        new_content = section['content']
        
        if not new_content:
            continue
        
        # Find heading
        match_index = -1
        for i, line in enumerate(lines):
            if line.strip() == heading_to_match or line.strip().strip('#* \t') == heading_to_match.strip('#* \t'):
                match_index = i
                break
        
        if match_index == -1:
            # Try aggressive cleaning
            cleaned_heading = clean_heading_text_aggressively(heading_to_match)
            for i, line in enumerate(lines):
                if clean_heading_text_aggressively(line).lower().startswith(cleaned_heading.lower()):
                    match_index = i
                    break
        
        if match_index != -1:
            # Find end of section
            start_line = match_index
            end_line = len(lines)
            for i in range(start_line + 1, len(lines)):
                line_to_check = lines[i].strip()
                is_hash_heading = line_to_check.startswith('#')
                is_bold_heading = line_to_check.startswith('**') and line_to_check.endswith('**')
                is_italic_heading = line_to_check.startswith('*') and line_to_check.endswith('*') and not is_bold_heading

                if is_hash_heading or is_bold_heading or is_italic_heading:
                    end_line = i
                    break
            
            # Reconstruct
            pre_section_lines = lines[:start_line]
            post_section_lines = lines[end_line:]
            new_content_lines = new_content.split('\n')
            lines = pre_section_lines + new_content_lines + post_section_lines
    
    placebo_paper = '\n'.join(lines)
    return placebo_paper

def process_paper(row: pd.Series, base_dir: Path, output_dir: Path, task_idx: int) -> dict:
    """Process a single paper: plant error and generate placebo."""
    paperid = row['paperid']
    flaw_descriptions_str = row.get('flaw_descriptions', '')
    
    if pd.isna(flaw_descriptions_str) or not flaw_descriptions_str:
        return None
    
    # Parse flaw descriptions (it's a string representation of a list)
    try:
        flaw_descriptions = ast.literal_eval(flaw_descriptions_str)
    except:
        # Try as JSON
        try:
            flaw_descriptions = json.loads(flaw_descriptions_str)
        except:
            return None
    
    if not flaw_descriptions or len(flaw_descriptions) == 0:
        return None
    
    # Find paper.md file
    # Handle both relative and absolute paths
    latest_folder_path = row.get('latest_folder_path', '')
    if not latest_folder_path:
        # Fallback: try using paperid to find folder
        folder_name = paperid
    elif '/' in latest_folder_path:
        folder_name = latest_folder_path.split('/')[-1]
    else:
        folder_name = latest_folder_path
    
    # Try to find the paper folder
    paper_folder = base_dir / folder_name
    paper_md_path = paper_folder / 'structured_paper_output' / 'paper.md'
    
    # If not found, try looking in base_dir directly
    if not paper_md_path.exists():
        # Try alternative: paper.md might be directly in folder
        alt_path = paper_folder / 'paper.md'
        if alt_path.exists():
            paper_md_path = alt_path
        else:
            # Try to find any folder matching the paperid
            matching_folders = list(base_dir.glob(f"{paperid}*"))
            if matching_folders:
                paper_folder = matching_folders[0]
                paper_md_path = paper_folder / 'structured_paper_output' / 'paper.md'
                if not paper_md_path.exists():
                    paper_md_path = paper_folder / 'paper.md'
    
    if not paper_md_path.exists():
        tqdm.write(f"Paper not found: {paper_md_path}")
        return None
    
    # Read original paper
    with open(paper_md_path, 'r', encoding='utf-8') as f:
        original_paper = f.read()
    
    # Get API key for this task
    key_name, api_key = get_api_key_for_task(task_idx)
    
    results = []
    
    # Process each flaw description
    for flaw_idx, flaw_description in enumerate(flaw_descriptions):
        flaw_id = f"flaw_{flaw_idx + 1}"
        
        # Plant error
        flawed_paper, modifications, success = plant_error(original_paper, flaw_description, api_key, key_name)
        
        if not success or not flawed_paper:
            tqdm.write(f"Failed to plant error for {paperid}, flaw {flaw_idx + 1}")
            continue
        
        # Save planted error version
        planted_error_dir = output_dir / 'planted_error' / paper_folder.name
        planted_error_dir.mkdir(parents=True, exist_ok=True)
        planted_error_path = planted_error_dir / f"{flaw_id}.md"
        
        with open(planted_error_path, 'w', encoding='utf-8') as f:
            f.write(flawed_paper)
        
        # Generate placebo/sham surgery version
        placebo_paper = generate_placebo(original_paper, flawed_paper, modifications, api_key, key_name, paperid, flaw_id)
        sham_surgery_path = None
        
        if placebo_paper:
            # Save placebo version
            sham_surgery_dir = output_dir / 'sham_surgery' / paper_folder.name
            sham_surgery_dir.mkdir(parents=True, exist_ok=True)
            sham_surgery_path = sham_surgery_dir / f"{flaw_id}.md"
            
            with open(sham_surgery_path, 'w', encoding='utf-8') as f:
                f.write(placebo_paper)
        else:
            tqdm.write(f"Failed to generate placebo for {paperid}, flaw {flaw_idx + 1}")
        
        results.append({
            'paperid': paperid,
            'flaw_id': flaw_id,
            'flaw_description': flaw_description,
            'planted_error_path': str(planted_error_path),
            'sham_surgery_path': str(sham_surgery_path) if sham_surgery_path else None,
            'modifications_count': len(modifications) if modifications else 0,
            'success': True
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Plant errors and generate placebo versions of papers using Gemini API.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to filtered_pairs_with_human_scores.csv")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing paper folders (e.g., data/ICLR2024)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: same as base_dir)")
    parser.add_argument("--max_workers", type=int, default=None, help="Max worker threads (default: number of API keys)")
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = args.base_dir
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    
    # Read CSV
    df = pd.read_csv(args.csv_file)
    print(f"✅ Loaded {len(df)} papers from CSV")
    
    # Filter papers with flaw descriptions
    df = df[df['flaw_descriptions'].notna()].copy()
    df = df[df['flaw_descriptions'] != ''].copy()
    print(f"✅ Found {len(df)} papers with flaw descriptions")
    
    # Set max workers
    max_workers = args.max_workers if args.max_workers else len(GEMINI_API_KEYS)
    
    # Process papers
    all_results = []
    task_counter = [0]  # Use list to allow modification in closure
    
    def process_with_counter(row):
        idx = task_counter[0]
        task_counter[0] += 1
        return process_paper(row, base_dir, output_dir, idx)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_with_counter, row): row for _, row in df.iterrows()}
        
        progress_bar = tqdm(concurrent.futures.as_completed(futures), total=len(df), desc="Processing Papers")
        for future in progress_bar:
            try:
                result = future.result()
                if result:
                    if isinstance(result, list):
                        all_results.extend(result)
                    else:
                        all_results.append(result)
            except Exception as e:
                row = futures[future]
                tqdm.write(f"Error processing {row['paperid']}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save results CSV
    if all_results:
        results_csv_path = output_dir / 'planting_results.csv'
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(results_csv_path, index=False)
        print(f"\n✅ Saved results to {results_csv_path}")
        print(f"   Successfully processed {len(all_results)} flaws")
    else:
        print("\n⚠️ No results to save")

if __name__ == "__main__":
    main()

