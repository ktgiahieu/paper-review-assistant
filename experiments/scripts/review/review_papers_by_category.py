#!/usr/bin/env python3
"""
Script to review papers using category-specific prompts.

This script loads prompts from files (e.g., prompt_2a.txt, prompt_2b.txt) and uses them
to review papers from specified folders (e.g., latest/, planted_error/).

Usage:
    python review_papers_by_category.py \
        --base_dir experiments/category_sampled_data/NeurIPS2024 \
        --categories 2a 2b \
        --folders latest planted_error \
        --prompt_dir experiments/scripts/review \
        --output_dir reviews
"""

import os
import json
import argparse
import time
import re
import threading
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple, Dict, Any

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("WARNING: google-generativeai not installed. Install with: pip install google-generativeai")

# Try to import tiktoken for token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("WARNING: tiktoken not installed. Will use character-based token estimation for truncation.")

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
    "gemma-3-27b-it": 1,
}

def get_request_delay_for_model(model_name: str) -> float:
    """Calculate request delay in seconds based on model's RPM limit."""
    rpm_limit = GEMINI_MODEL_RPM_LIMITS.get(model_name, 30)
    return 60.0 / rpm_limit

GEMINI_REQUEST_DELAY = get_request_delay_for_model(GEMINI_MODEL)

# Rate limiting tracking
key_last_used: Dict[str, float] = {}
key_lock = threading.Lock()

# --- Pydantic Models for Category-Specific Review Schemas ---

class PracticalityAssessmentItem(BaseModel):
    """Individual practicality assessment item."""
    score: int = Field(..., ge=1, le=5, description="Score from 1 to 5")
    reasoning: str = Field(..., description="Reasoning for the score")

class PracticalityAssessment(BaseModel):
    """Practicality assessment for category 2a."""
    input_realism: PracticalityAssessmentItem = Field(..., alias="1_input_realism")
    scalability_and_efficiency: PracticalityAssessmentItem = Field(..., alias="2_scalability_and_efficiency")
    generality_and_scope: PracticalityAssessmentItem = Field(..., alias="3_generality_and_scope")
    mechanism_task_fit: PracticalityAssessmentItem = Field(..., alias="4_mechanism_task_fit")
    
    class Config:
        populate_by_name = True

class Category2aReview(BaseModel):
    """Review schema for category 2a (Practicality & Robustness)."""
    practicality_assessment: PracticalityAssessment
    presentation: int = Field(..., ge=1, le=4)
    contribution: int = Field(..., ge=1, le=4)
    overall_score: int = Field(..., ge=1, le=10)
    confidence: int = Field(..., ge=1, le=5)

class Category2bReview(BaseModel):
    """Review schema for category 2b (Theoretical Rigor)."""
    theoretical_rigor_summary: str
    assumption_justification_score: int = Field(..., ge=1, le=4)
    proof_completeness_score: int = Field(..., ge=1, le=4)
    heuristic_linkage_score: int = Field(..., ge=1, le=4)
    definition_rigor_score: int = Field(..., ge=1, le=4)
    soundness: int = Field(..., ge=1, le=4)
    presentation: int = Field(..., ge=1, le=4)
    contribution: int = Field(..., ge=1, le=4)
    overall_score: int = Field(..., ge=1, le=10)
    confidence: int = Field(..., ge=1, le=5)

# Map category IDs to their Pydantic models
CATEGORY_MODELS = {
    "2a": Category2aReview,
    "2b": Category2bReview,
    # Add more categories as needed
}

def load_prompt_file(prompt_path: Path) -> str:
    """Load a prompt file and return its content."""
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

def prepare_prompt(prompt_template: str, paper_content: str) -> str:
    """Replace {paper_content} placeholder in prompt template with actual paper content."""
    return prompt_template.replace("{paper_content}", paper_content)

def _sanitize_json_string(json_str: str) -> str:
    """Cleans common JSON errors from LLM output."""
    json_str = json_str.strip()
    json_str = re.sub(r'^```json\s*', '', json_str, flags=re.MULTILINE)
    json_str = re.sub(r'^```\s*$', '', json_str, flags=re.MULTILINE)
    json_str = json_str.strip()
    json_str = re.sub(r',\s*(?=[}\]])', '', json_str)
    
    try:
        temp = json.loads(json_str)
        json_str = json.dumps(temp, ensure_ascii=False)
    except json.JSONDecodeError:
        pass
    
    return json_str

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

def find_paper_markdown(paper_folder: Path, specific_file: Optional[str] = None) -> Optional[Path]:
    """
    Finds the paper markdown file.
    
    For 'latest' folder: looks for structured_paper_output/paper.md
    For 'planted_error' and 'sham_surgery': looks for specific flaw files (e.g., flaw_1.md)
    or in flawed_papers/ subdirectory
    """
    # If specific file is provided, use it directly
    if specific_file:
        specific_path = paper_folder / specific_file
        if specific_path.exists():
            return specific_path
        # Also try in flawed_papers/ subdirectory
        specific_path = paper_folder / "flawed_papers" / specific_file
        if specific_path.exists():
            return specific_path
    
    # For 'latest' folder structure: structured_paper_output/paper.md
    paper_md_path = paper_folder / "structured_paper_output" / "paper.md"
    if paper_md_path.exists():
        return paper_md_path
    
    # Fallback: search for any .md file in the folder (not in subdirectories)
    md_files = list(paper_folder.glob("*.md"))
    if md_files:
        return md_files[0]
    
    # Last resort: search recursively
    md_files = list(paper_folder.glob("**/*.md"))
    return md_files[0] if md_files else None

def count_tokens(text: str, model_name: str = None) -> int:
    """Count tokens in text. Uses tiktoken if available, otherwise character-based estimation."""
    if TIKTOKEN_AVAILABLE:
        try:
            if "gemma" in model_name.lower() if model_name else False:
                try:
                    encoding = tiktoken.get_encoding("cl100k_base")
                except:
                    encoding = tiktoken.get_encoding("gpt2")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")
            
            return len(encoding.encode(text, allowed_special="all"))
        except Exception:
            pass
    
    return len(text) // 4

def truncate_to_tokens(text: str, max_tokens: int, model_name: str = None) -> str:
    """Truncate text to fit within token limit."""
    current_tokens = count_tokens(text, model_name)
    
    if current_tokens <= max_tokens:
        return text
    
    if TIKTOKEN_AVAILABLE:
        try:
            if "gemma" in model_name.lower() if model_name else False:
                try:
                    encoding = tiktoken.get_encoding("cl100k_base")
                except:
                    encoding = tiktoken.get_encoding("gpt2")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")
            
            tokens = encoding.encode(text, allowed_special="all")
            if len(tokens) > max_tokens:
                truncated_tokens = tokens[:max_tokens]
                return encoding.decode(truncated_tokens)
        except Exception:
            pass
    
    max_chars = max_tokens * 4
    if len(text) > max_chars:
        return text[:max_chars]
    
    return text

def review_single_paper(
    paper_id: str,
    paper_path: Path,
    category_id: str,
    folder_label: str,
    prompt_template: str,
    review_model: type[BaseModel],
    api_key: str,
    key_name: str,
    model_name: str,
    verbose: bool,
    run_id: int = 0,
    request_delay: float = None,
    specific_file: Optional[str] = None,
    version_id: Optional[str] = None
) -> dict:
    """
    Reviews a single paper using a category-specific prompt.
    
    Args:
        paper_id: Paper identifier
        paper_path: Path to paper directory
        category_id: Category ID (e.g., "2a", "2b")
        folder_label: Label for the folder (e.g., "latest", "planted_error")
        prompt_template: Prompt template with {paper_content} placeholder
        review_model: Pydantic model class for the review schema
        api_key: Gemini API key string
        key_name: Name/identifier of the API key
        model_name: Gemini model name
        verbose: Enable verbose output
        run_id: Run ID for multiple runs
        request_delay: Delay in seconds between requests
        specific_file: Optional specific markdown file to review
        version_id: Optional version identifier for tracking
    """
    worker_id = threading.get_ident() if hasattr(threading, 'get_ident') else os.getpid()
    _print_method = tqdm.write if not verbose else print
    
    if request_delay is None:
        request_delay = get_request_delay_for_model(model_name)
    
    try:
        # Read paper content
        paper_md = find_paper_markdown(paper_path, specific_file)
        if not paper_md:
            return {
                "error": f"Could not find paper markdown for {paper_id} at {paper_path}" + (f" (looking for {specific_file})" if specific_file else ""),
                "paper_id": paper_id,
                "category_id": category_id,
                "folder": folder_label,
                "run_id": run_id,
                "version_id": version_id,
                "success": False
            }
        
        with open(paper_md, 'r', encoding='utf-8') as f:
            paper_content = f.read()
        
        # Truncate paper content if using gemma model (15k token limit)
        if "gemma" in model_name.lower():
            max_tokens = 15000
            available_tokens = max_tokens - 5000
            original_tokens = count_tokens(paper_content, model_name)
            
            if original_tokens > available_tokens:
                if verbose:
                    _print_method(f"Worker {worker_id}: Truncating paper from {original_tokens} to ~{available_tokens} tokens for {model_name}")
                paper_content = truncate_to_tokens(paper_content, available_tokens, model_name)
        
        # Prepare prompt with paper content
        full_prompt = prepare_prompt(prompt_template, paper_content)
        
        response_text = None
        last_exception = None
        parsed_review = None
        
        # Retry loop
        for attempt in range(MAX_RETRIES):
            try:
                if verbose:
                    _print_method(f"Worker {worker_id}: Reviewing {paper_id} (category {category_id}, {folder_label}), attempt {attempt + 1}/{MAX_RETRIES}, key: {key_name}")
                
                wait_for_rate_limit(key_name, request_delay)
                
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                
                response_obj = model.generate_content(full_prompt)
                response_text = response_obj.text
                
                # Parse JSON
                raw_json_content = response_text
                sanitized_json_content = _sanitize_json_string(raw_json_content)
                
                # Try Pydantic validation
                try:
                    parsed_review = review_model.model_validate_json(sanitized_json_content)
                    break
                except Exception as pydantic_error:
                    # Try regular JSON parsing as fallback
                    try:
                        parsed_dict = json.loads(sanitized_json_content)
                        
                        # For category 2a, ensure practicality_assessment has the correct structure
                        if category_id == "2a" and "practicality_assessment" in parsed_dict:
                            pa = parsed_dict["practicality_assessment"]
                            # Ensure all required keys exist with proper structure
                            for key in ["1_input_realism", "2_scalability_and_efficiency", 
                                       "3_generality_and_scope", "4_mechanism_task_fit"]:
                                if key in pa and isinstance(pa[key], dict):
                                    if "score" not in pa[key] or "reasoning" not in pa[key]:
                                        # Try to fix if structure is wrong
                                        if isinstance(pa[key], (int, str)):
                                            # If it's just a score, create proper structure
                                            pa[key] = {"score": pa[key] if isinstance(pa[key], int) else 3, 
                                                      "reasoning": ""}
                        
                        parsed_review = review_model(**parsed_dict)
                        break
                    except (json.JSONDecodeError, ValueError) as json_e:
                        if attempt < MAX_RETRIES - 1:
                            _print_method(f"Worker {worker_id}: JSON parsing failed for {paper_id} (category {category_id}, {folder_label}), attempt {attempt + 1}/{MAX_RETRIES}. Retrying...")
                            response_text = None
                            wait_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                            time.sleep(wait_time)
                            continue
                        else:
                            raise json_e
                
            except Exception as e:
                last_exception = e
                _print_method(f"Worker {worker_id}: Error for {paper_id} (category {category_id}, {folder_label}), attempt {attempt + 1}/{MAX_RETRIES}: {e}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    break
        
        # Check if we successfully parsed the review
        if parsed_review is not None:
            review_data = parsed_review.model_dump()
            
            # Add metadata
            review_data["paper_id"] = paper_id
            review_data["category_id"] = category_id
            review_data["folder"] = folder_label
            review_data["run_id"] = run_id
            if version_id:
                review_data["version_id"] = version_id
            review_data["success"] = True
            
            # Add standard score mappings for compatibility
            review_data["rating"] = review_data.get("overall_score")
            
            if verbose:
                _print_method(f"Worker {worker_id}: Successfully reviewed {paper_id} (category {category_id}, {folder_label})")
            
            return review_data
        
        # All attempts failed
        if response_text is None:
            err_msg = f"All API attempts failed for {paper_id} (category {category_id}, {folder_label})."
            if last_exception:
                err_msg += f" Last error: {type(last_exception).__name__} - {last_exception}"
            return {
                "error": err_msg,
                "paper_id": paper_id,
                "category_id": category_id,
                "folder": folder_label,
                "run_id": run_id,
                "version_id": version_id,
                "success": False
            }
        else:
            sanitized_json_content = _sanitize_json_string(response_text)
            _print_method(f"Worker {worker_id}: Failed to parse JSON after {MAX_RETRIES} attempts for {paper_id} (category {category_id}, {folder_label})")
            _print_method(f"Worker {worker_id}: Raw JSON (first 500 chars): {sanitized_json_content[:500]}...")
            
            # Last resort: try to extract whatever we can
            try:
                fallback_data = json.loads(sanitized_json_content)
                fallback_data["paper_id"] = paper_id
                fallback_data["category_id"] = category_id
                fallback_data["folder"] = folder_label
                fallback_data["run_id"] = run_id
                fallback_data["success"] = True
                fallback_data["__parsing_warning"] = "JSON parsed but Pydantic validation may have failed"
                fallback_data["rating"] = fallback_data.get("overall_score")
                return fallback_data
            except:
                return {
                    "error": f"Failed to parse JSON from LLM after {MAX_RETRIES} attempts",
                    "paper_id": paper_id,
                    "category_id": category_id,
                    "folder": folder_label,
                    "run_id": run_id,
                    "version_id": version_id,
                    "raw_content": response_text[:1000] if response_text else None,
                    "last_exception": str(last_exception) if last_exception else None,
                    "success": False
                }
    
    except Exception as e:
        message = f"FATAL ERROR reviewing {paper_id} (category {category_id}, {folder_label}): {type(e).__name__} - {e}"
        _print_method(f"Worker {worker_id}: {message}")
        import traceback
        _print_method(traceback.format_exc())
        return {
            "error": message,
            "paper_id": paper_id,
            "category_id": category_id,
            "folder": folder_label,
            "run_id": run_id,
            "version_id": version_id,
            "success": False
        }

def review_papers_in_category_folder(
    category_dir: Path,
    category_id: str,
    folder_name: str,
    prompt_template: str,
    review_model: type[BaseModel],
    output_dir: Path,
    model_name: str,
    verbose: bool,
    skip_existing: bool = False,
    num_runs: int = 1,
    max_workers: int = None
) -> List[dict]:
    """
    Review all papers in a specific category folder using parallel processing.
    
    Args:
        category_dir: Category directory (e.g., base_dir/2a)
        category_id: Category ID (e.g., "2a")
        folder_name: Folder name to review (e.g., "latest", "planted_error")
        prompt_template: Prompt template with {paper_content} placeholder
        review_model: Pydantic model class for the review schema
        output_dir: Output directory for reviews
        model_name: Model name
        verbose: Enable verbose output
        skip_existing: Skip papers that already have review files
        num_runs: Number of times to review each paper
        max_workers: Maximum number of worker threads
    """
    folder_path = category_dir / folder_name
    if not folder_path.exists():
        print(f"Warning: Folder {folder_path} does not exist, skipping...")
        return []
    
    # Get all paper directories
    paper_dirs = [d for d in folder_path.iterdir() if d.is_dir()]
    print(f"\nFound {len(paper_dirs)} papers in {category_id}/{folder_name}/")
    
    # Create output directory for this category and folder
    category_output_dir = output_dir / category_id / folder_name
    category_output_dir.mkdir(parents=True, exist_ok=True)
    
    request_delay = get_request_delay_for_model(model_name)
    
    # Prepare tasks
    tasks = []
    key_names = list(GEMINI_API_KEYS.keys())
    
    # Check if this is a multi-version folder (planted_error, sham_surgery)
    is_multi_version = folder_name in ['planted_error', 'sham_surgery']
    
    for paper_dir in paper_dirs:
        paper_id = paper_dir.name.split('_')[0]
        
        if is_multi_version:
            # Look for flawed_papers subdirectory or .md files directly
            flawed_papers_dir = paper_dir / "flawed_papers"
            if flawed_papers_dir.exists():
                flaw_files = sorted(flawed_papers_dir.glob("*.md"))
            else:
                flaw_files = sorted(paper_dir.glob("*.md"))
            
            if not flaw_files:
                if verbose:
                    print(f"Warning: No flaw files found in {paper_dir}")
                continue
            
            for flaw_file in flaw_files:
                version_id = flaw_file.stem
                specific_file = flaw_file.name
                
                for run_id in range(num_runs):
                    review_file = category_output_dir / paper_id / f"{version_id}_review_run{run_id}.json"
                    if skip_existing and review_file.exists():
                        if verbose:
                            print(f"Skipping {paper_id} (category {category_id}, {folder_name}) {version_id} run {run_id} - already exists")
                        continue
                    
                    task_idx = len(tasks)
                    key_name = key_names[task_idx % len(key_names)]
                    api_key = GEMINI_API_KEYS[key_name]
                    
                    tasks.append((paper_id, paper_dir, run_id, api_key, key_name, review_file, request_delay, specific_file, version_id))
        else:
            # Single version folder (latest, etc.)
            for run_id in range(num_runs):
                review_file = category_output_dir / paper_id / f"review_run{run_id}.json"
                if skip_existing and review_file.exists():
                    if verbose:
                        print(f"Skipping {paper_id} (category {category_id}, {folder_name}) run {run_id} - already exists")
                    continue
                
                task_idx = len(tasks)
                key_name = key_names[task_idx % len(key_names)]
                api_key = GEMINI_API_KEYS[key_name]
                
                tasks.append((paper_id, paper_dir, run_id, api_key, key_name, review_file, request_delay, None, None))
    
    if not tasks:
        print(f"No tasks to process for {category_id}/{folder_name}/")
        return []
    
    print(f"Prepared {len(tasks)} review tasks for {category_id}/{folder_name}/")
    
    if max_workers is None:
        max_workers = len(GEMINI_API_KEYS)
    
    reviews = []
    
    def process_task(task):
        paper_id, paper_dir, run_id, api_key, key_name, review_file, task_request_delay, specific_file, version_id = task
        
        review_data = review_single_paper(
            paper_id=paper_id,
            paper_path=paper_dir,
            category_id=category_id,
            folder_label=folder_name,
            prompt_template=prompt_template,
            review_model=review_model,
            api_key=api_key,
            key_name=key_name,
            model_name=model_name,
            verbose=verbose,
            run_id=run_id,
            request_delay=task_request_delay,
            specific_file=specific_file,
            version_id=version_id
        )
        
        if review_data.get("success"):
            paper_output_dir = category_output_dir / paper_id
            paper_output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(review_file, 'w', encoding='utf-8') as f:
                json.dump(review_data, f, indent=2, ensure_ascii=False)
        
        return review_data
    
    if len(GEMINI_API_KEYS) > 1:
        print(f"Processing {len(tasks)} tasks in parallel using {len(GEMINI_API_KEYS)} API keys (max_workers={max_workers})...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_task, task): task for task in tasks}
            
            for future in tqdm(as_completed(futures), total=len(tasks), desc=f"Reviewing {category_id}/{folder_name}"):
                try:
                    review_data = future.result()
                    reviews.append(review_data)
                except Exception as e:
                    task = futures[future]
                    paper_id = task[0]
                    print(f"Error processing {paper_id}: {e}")
                    reviews.append({
                        "error": str(e),
                        "paper_id": paper_id,
                        "category_id": category_id,
                        "folder": folder_name,
                        "success": False
                    })
    else:
        print(f"Processing {len(tasks)} tasks sequentially...")
        for task in tqdm(tasks, desc=f"Reviewing {category_id}/{folder_name}"):
            review_data = process_task(task)
            reviews.append(review_data)
    
    return reviews

def main():
    parser = argparse.ArgumentParser(
        description="Review papers using category-specific prompts"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing category folders (e.g., experiments/category_sampled_data/NeurIPS2024)"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        required=True,
        help="Category IDs to process (e.g., 2a 2b)"
    )
    parser.add_argument(
        "--folders",
        type=str,
        nargs="+",
        required=True,
        help="Folder names to review (e.g., latest planted_error)"
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default="experiments/scripts/review",
        help="Directory containing prompt files (default: experiments/scripts/review)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for reviews (default: base_dir/../reviews_by_category/)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-2.0-flash-lite",
        help="Gemini model name (default: gemini-2.0-flash-lite)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Max worker threads for concurrent processing (default: number of API keys)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of times to review each paper (default: 1)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip papers that already have review files"
    )
    
    args = parser.parse_args()
    
    if not GENAI_AVAILABLE:
        print("Error: google-generativeai not installed. Install with: pip install google-generativeai")
        exit(1)
    
    if not GEMINI_API_KEYS:
        print("Error: No Gemini API keys found in environment variables or .env file.")
        exit(1)
    
    base_dir = Path(args.base_dir)
    prompt_dir = Path(args.prompt_dir)
    
    if args.output_dir is None:
        output_dir = base_dir.parent / "reviews_by_category" / base_dir.name
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Base directory: {base_dir}")
    print(f"Categories: {args.categories}")
    print(f"Folders: {args.folders}")
    print(f"Prompt directory: {prompt_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model_name}")
    print(f"Number of API keys: {len(GEMINI_API_KEYS)}")
    model_rpm = GEMINI_MODEL_RPM_LIMITS.get(args.model_name, 30)
    model_delay = get_request_delay_for_model(args.model_name)
    print(f"RPM limit: {model_rpm} per key")
    print(f"Request delay: {model_delay:.2f} seconds per key")
    print(f"Total throughput: {len(GEMINI_API_KEYS) * model_rpm} RPM ({len(GEMINI_API_KEYS)} keys × {model_rpm} RPM)")
    print(f"Number of runs per paper: {args.num_runs}")
    print()
    
    # Load prompts for each category
    category_prompts = {}
    category_models = {}
    
    for category_id in args.categories:
        # Try to find prompt file (prompt_2a.txt, prompt_2b.txt, etc.)
        prompt_file = prompt_dir / f"prompt_{category_id}.txt"
        
        if not prompt_file.exists():
            print(f"⚠️ Warning: Prompt file not found for category {category_id}: {prompt_file}")
            print(f"   Skipping category {category_id}")
            continue
        
        try:
            prompt_template = load_prompt_file(prompt_file)
            category_prompts[category_id] = prompt_template
            
            # Get the appropriate Pydantic model for this category
            if category_id not in CATEGORY_MODELS:
                print(f"⚠️ Warning: No Pydantic model defined for category {category_id}")
                print(f"   Available models: {list(CATEGORY_MODELS.keys())}")
                print(f"   Skipping category {category_id}")
                continue
            
            category_models[category_id] = CATEGORY_MODELS[category_id]
            print(f"✅ Loaded prompt and model for category {category_id}")
        except Exception as e:
            print(f"❌ Error loading prompt for category {category_id}: {e}")
            continue
    
    if not category_prompts:
        print("❌ No valid prompts loaded. Exiting.")
        exit(1)
    
    # Review papers in each category and folder
    all_reviews = []
    
    for category_id in category_prompts.keys():
        category_dir = base_dir / category_id
        if not category_dir.exists():
            print(f"⚠️ Warning: Category directory not found: {category_dir}")
            continue
        
        prompt_template = category_prompts[category_id]
        review_model = category_models[category_id]
        
        for folder_name in args.folders:
            reviews = review_papers_in_category_folder(
                category_dir=category_dir,
                category_id=category_id,
                folder_name=folder_name,
                prompt_template=prompt_template,
                review_model=review_model,
                output_dir=output_dir,
                model_name=args.model_name,
                verbose=args.verbose,
                skip_existing=args.skip_existing,
                num_runs=args.num_runs,
                max_workers=args.max_workers
            )
            all_reviews.extend(reviews)
    
    # Print summary
    successful = sum(1 for r in all_reviews if r.get("success", False))
    failed = len(all_reviews) - successful
    
    print("\n" + "="*60)
    print("Review Summary:")
    print(f"  Total reviews: {len(all_reviews)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Results saved in: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()

