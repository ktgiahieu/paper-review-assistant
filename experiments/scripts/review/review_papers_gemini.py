#!/usr/bin/env python3
"""
Script to review papers from specified folders using Gemini API.

This script can review papers from multiple folders (e.g., latest/, authors_affiliation_good/, authors_affiliation_bad/)
and save reviews to an output directory with proper structure.
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
from typing import Optional, List, Tuple, Dict

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
# Delay = 60 seconds / RPM_limit
GEMINI_MODEL_RPM_LIMITS = {
    "gemini-2.0-flash-lite": 30,  # 60/30 = 2.0 seconds
    "gemini-2.0-flash-exp": 10,   # 60/10 = 6.0 seconds
    "gemini-2.0-flash-preview-image-generation": 10,  # 60/10 = 6.0 seconds
    "gemini-2.0-flash": 15,       # 60/15 = 4.0 seconds
    "gemini-2.5-flash-lite": 15,  # 60/15 = 4.0 seconds
    "gemini-2.5-flash-tts": 3,    # 60/3 = 20.0 seconds
    "gemini-2.5-flash": 10,       # 60/10 = 6.0 seconds
    "gemini-2.5-pro": 2,          # 60/2 = 30.0 seconds
}

def get_request_delay_for_model(model_name: str) -> float:
    """Calculate request delay in seconds based on model's RPM limit."""
    rpm_limit = GEMINI_MODEL_RPM_LIMITS.get(model_name, 30)  # Default to 30 RPM
    return 60.0 / rpm_limit

# Calculate delay for the current model
GEMINI_REQUEST_DELAY = get_request_delay_for_model(GEMINI_MODEL)

# Rate limiting tracking
key_last_used: Dict[str, float] = {}
key_lock = threading.Lock()

# --- Pydantic Models for Structured Review Output ---
class PaperReview(BaseModel):
    """Pydantic model for the structured review output."""
    summary: str = Field(
        description="A 2-3 sentence summary of the paper's main contribution and approach."
    )
    strengths: List[str] = Field(
        description="A bulleted list of the paper's key strengths (3-5 points)."
    )
    weaknesses: List[str] = Field(
        description="A bulleted list of the paper's key weaknesses and limitations (3-5 points)."
    )
    clarity_score: int = Field(
        description="Clarity and presentation quality score (1-10, where 10 is excellent).",
        ge=1, le=10
    )
    novelty_score: int = Field(
        description="Novelty and originality score (1-10, where 10 is highly novel).",
        ge=1, le=10
    )
    technical_quality_score: int = Field(
        description="Technical quality and correctness score (1-10, where 10 is rigorous and correct).",
        ge=1, le=10
    )
    experimental_rigor_score: int = Field(
        description="Experimental evaluation rigor score (1-10, where 10 is comprehensive).",
        ge=1, le=10
    )
    overall_score: int = Field(
        description="Overall recommendation score (1-10, where 1 is strong reject and 10 is strong accept).",
        ge=1, le=10
    )
    confidence: int = Field(
        description="Reviewer confidence in the assessment (1-5, where 5 is very confident).",
        ge=1, le=5
    )
    recommendation: str = Field(
        description="Final recommendation: one of 'Strong Accept', 'Accept', 'Weak Accept', 'Borderline', 'Weak Reject', 'Reject', or 'Strong Reject'."
    )
    detailed_comments: str = Field(
        description="Detailed comments explaining the scores and recommendation (3-5 sentences)."
    )

class CriticalNeurIPSReview(BaseModel):
    """Pydantic model for a comprehensive, critical NeurIPS-style review."""
    summary: str = Field(
        description="A brief, neutral summary of the paper and its contributions. This should not be a critique or a copy of the abstract."
    )
    strengths_and_weaknesses: str = Field(
        description="A thorough assessment of the paper's strengths and weaknesses, touching on originality, quality, clarity, and significance. Use Markdown for formatting."
    )
    questions: str = Field(
        description="A list of actionable questions and suggestions for the authors (ideally 3-5 key points). Frame questions to clarify points that could change the evaluation."
    )
    limitations_and_societal_impact: str = Field(
        description="Assessment of whether limitations and potential negative societal impacts are adequately addressed. State 'Yes' if adequate; otherwise, provide constructive suggestions for improvement."
    )
    soundness: int = Field(
        description="Numerical rating for the soundness of the technical claims, methodology, and whether claims are supported by evidence (4: excellent, 3: good, 2: fair, 1: poor).",
        ge=1, le=4
    )
    presentation: int = Field(
        description="Numerical rating for the quality of the presentation, including writing style, clarity, and contextualization (4: excellent, 3: good, 2: fair, 1: poor).",
        ge=1, le=4
    )
    contribution: int = Field(
        description="Numerical rating for the quality of the overall contribution, including the importance of the questions asked and the value of the results (4: excellent, 3: good, 2: fair, 1: poor).",
        ge=1, le=4
    )
    overall_score: int = Field(
        description="Overall recommendation score (10: Award quality, 9: Strong accept, 8: Accept, 7: Weak accept, 6: Marginally above acceptance, 5: Borderline, 4: Marginally below acceptance, 3: Reject, 2: Strong reject, 1: Trivial/wrong).",
        ge=1, le=10
    )
    confidence: int = Field(
        description="Confidence in the assessment (5: Certain, 4: Confident, 3: Fairly confident, 2: Willing to defend, 1: Educated guess).",
        ge=1, le=5
    )

class ReviewPrompts:
    @staticmethod
    def get_system_prompt(format_type: str = "default") -> str:
        """Returns the system prompt for paper review based on format type."""
        if format_type == "CriticalNeurIPS":
            return """You are a top-tier academic reviewer for NeurIPS, known for writing exceptionally thorough, incisive, and constructive critiques. Your goal is to synthesize multiple expert perspectives into a single, coherent review that elevates the entire research field.

When reviewing the paper, you must adopt a multi-faceted approach, simultaneously analyzing the work from the following critical angles:

1.  **The Conceptual Critic & Historian**:
    * **Question the Core Concepts**: Do not accept the authors' definitions at face value. Situate the paper within the broader scientific landscape by defining its core concepts from first principles, citing foundational literature.
    * **Re-frame with Evidence**: If the authors' framing is weak, re-organize their ideas into a more insightful structure. Challenge their assumptions by citing counter-examples from published research.
    * **Provide a Roadmap**: Use citations constructively to point authors toward literature they may have missed, helping them build a stronger conceptual foundation.

2.  **The Methodological Skeptic & Forensic Examiner**:
    * **Scrutinize the Methodology**: Forensically examine the experimental design, evaluation metrics, and statistical analysis. Are they appropriate for the claims being made?
    * **Identify Critical Omissions**: What is *absent* from the paper? Look for ignored alternative hypotheses, unacknowledged limitations, or countervailing evidence that is not addressed.
    * **Challenge Unstated Assumptions**: Articulate how unstated assumptions in the methodology could undermine the validity of the results and the paper's central claims.

In short: your review must be a synthesis of these perspectives. You are not just checking for flaws; you are deeply engaging with the paper's ideas, challenging its foundations, questioning its methodology, and providing a clear, evidence-backed path for improvement. Your final review should be a masterclass in scholarly critique."""
        else:
            # Default format
            return """
You are an expert peer reviewer for a top-tier machine learning conference (NeurIPS, ICML, or ICLR). Your task is to provide a thorough, balanced, and constructive review of the submitted research paper.

Your review should assess the paper across multiple dimensions:
1. **Clarity**: How well-written and organized is the paper?
2. **Novelty**: How original and innovative is the contribution?
3. **Technical Quality**: How sound and rigorous is the technical approach?
4. **Experimental Rigor**: How comprehensive and convincing are the experiments?

You must provide your assessment in a specific JSON format with the following fields:
- summary: A 2-3 sentence overview of the paper
- strengths: Bulleted list of key strengths (3-5 points)
- weaknesses: Bulleted list of key weaknesses (3-5 points)
- clarity_score: Score from 1-10
- novelty_score: Score from 1-10
- technical_quality_score: Score from 1-10
- experimental_rigor_score: Score from 1-10
- overall_score: Score from 1-10 (1=strong reject, 10=strong accept)
- confidence: Your confidence level from 1-5
- recommendation: One of 'Strong Accept', 'Accept', 'Weak Accept', 'Borderline', 'Weak Reject', 'Reject', or 'Strong Reject'
- detailed_comments: 3-5 sentences explaining your assessment

Be critical but fair. Provide constructive feedback. Your response MUST be a single, valid JSON object with no additional text.
"""

    @staticmethod
    def get_user_prompt(paper_content: str, paper_version: str, format_type: str = "default") -> str:
        """Constructs the user-facing prompt for review."""
        if format_type == "CriticalNeurIPS":
            return f"""Please review the following research paper with exceptional rigor and depth.

<paper_content>
{paper_content}
</paper_content>

**Instructions:**
1. Thoroughly read the entire paper.
2. Adopt the comprehensive, critical persona described in your system instructions.
3. Generate one complete review that fills all the fields in the required JSON format.
4. Your response MUST be a single, valid JSON object. Do not include any text, markdown, or code formatting before or after the JSON object.

**Required JSON Schema:**
Generate a single JSON object with these keys:
* "summary": (string) A brief, neutral summary of the paper and its contributions. Not a critique or copy of the abstract.
* "strengths_and_weaknesses": (string) A thorough assessment of strengths and weaknesses, touching on originality, quality, clarity, and significance. Use Markdown formatting.
* "questions": (string) A list of actionable questions and suggestions for authors (3-5 key points). Frame questions to clarify points that could change the evaluation.
* "limitations_and_societal_impact": (string) Assessment of whether limitations and societal impacts are adequately addressed. State 'Yes' if adequate; otherwise, provide constructive suggestions.
* "soundness": (integer) Rating for soundness of technical claims. Must be 4, 3, 2, or 1. (4=excellent, 3=good, 2=fair, 1=poor)
* "presentation": (integer) Rating for presentation quality. Must be 4, 3, 2, or 1. (4=excellent, 3=good, 2=fair, 1=poor)
* "contribution": (integer) Rating for overall contribution. Must be 4, 3, 2, or 1. (4=excellent, 3=good, 2=fair, 1=poor)
* "overall_score": (integer) Overall recommendation. Must be 10, 9, 8, 7, 6, 5, 4, 3, 2, or 1. (10=Award quality, 8=Strong Accept, 6=Weak Accept, 5=Borderline, 4=Borderline reject, 2=Strong Reject)
* "confidence": (integer) Confidence in assessment. Must be 5, 4, 3, 2, or 1. (5=Certain, 4=Confident, 3=Fairly confident, 2=Willing to defend, 1=Educated guess)

Provide your complete review as a single JSON object."""
        else:
            return f"""
Please review the following research paper ({paper_version}):

<paper_content>
{paper_content}
</paper_content>

Provide a comprehensive review following the specified JSON format.
"""

def _sanitize_json_string(json_str: str) -> str:
    """Cleans common JSON errors from LLM output."""
    # Remove markdown code blocks
    json_str = json_str.strip()
    json_str = re.sub(r'^```json\s*', '', json_str, flags=re.MULTILINE)
    json_str = re.sub(r'^```\s*$', '', json_str, flags=re.MULTILINE)
    json_str = json_str.strip()
    
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',\s*(?=[}\]])', '', json_str)
    
    # Try to parse and re-serialize to fix escape issues
    # This will properly escape all special characters
    try:
        temp = json.loads(json_str)
        json_str = json.dumps(temp, ensure_ascii=False)
    except json.JSONDecodeError as e:
        # If parsing fails, try to fix common escape sequence issues
        # The error message might tell us about invalid escapes
        error_msg = str(e)
        if "invalid escape" in error_msg.lower():
            # Try to fix invalid escape sequences
            # Replace common problematic patterns
            # But be careful - we don't want to break valid JSON
            pass  # Let it pass through and let the retry logic handle it
    
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

def find_paper_markdown(paper_folder: Path) -> Optional[Path]:
    """Finds the paper.md file in the structured_paper_output directory."""
    paper_md_path = paper_folder / "structured_paper_output" / "paper.md"
    if paper_md_path.exists():
        return paper_md_path
    
    # Fallback: search for any .md file
    md_files = list(paper_folder.glob("**/*.md"))
    return md_files[0] if md_files else None

def review_single_paper(
    paper_id: str,
    paper_path: Path,
    folder_label: str,
    api_key: str,
    key_name: str,
    model_name: str,
    verbose: bool,
    run_id: int = 0,
    request_delay: float = None,
    format_type: str = "default"
) -> dict:
    """
    Reviews a single paper and returns structured results.
    
    Args:
        paper_id: Paper identifier
        paper_path: Path to paper directory
        folder_label: Label for the folder (e.g., "latest", "authors_affiliation_good")
        api_key: Gemini API key string
        key_name: Name/identifier of the API key (for rate limiting)
        model_name: Gemini model name
        verbose: Enable verbose output
        run_id: Run ID for multiple runs
        request_delay: Delay in seconds between requests (default: calculated from model RPM limit)
    """
    worker_id = threading.get_ident() if hasattr(threading, 'get_ident') else os.getpid()
    _print_method = tqdm.write if not verbose else print
    
    # Use provided delay or calculate from model
    if request_delay is None:
        request_delay = get_request_delay_for_model(model_name)
    
    try:
        # Read paper content
        paper_md = find_paper_markdown(paper_path)
        if not paper_md:
            return {
                "error": f"Could not find paper markdown for {paper_id} at {paper_path}",
                "paper_id": paper_id,
                "folder": folder_label,
                "run_id": run_id,
                "success": False
            }
        
        with open(paper_md, 'r', encoding='utf-8') as f:
            paper_content = f.read()
        
        system_prompt = ReviewPrompts.get_system_prompt(format_type)
        user_prompt_text = ReviewPrompts.get_user_prompt(paper_content, folder_label, format_type)
        
        response_text = None
        last_exception = None
        parsed_review = None
        
        # Retry loop: includes both API calls and JSON parsing
        for attempt in range(MAX_RETRIES):
            try:
                if verbose:
                    _print_method(f"Worker {worker_id}: Reviewing {paper_id} ({folder_label}), attempt {attempt + 1}/{MAX_RETRIES}, key: {key_name}")
                
                # Rate limiting - wait if necessary
                wait_for_rate_limit(key_name, request_delay)
                
                # Configure Gemini with the specific API key
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                
                # Gemini API call - combine system and user prompts
                full_prompt = f"{system_prompt}\n\n{user_prompt_text}"
                response_obj = model.generate_content(full_prompt)
                response_text = response_obj.text
                
                # Try to parse JSON immediately
                raw_json_content = response_text
                sanitized_json_content = _sanitize_json_string(raw_json_content)
                
                # Try Pydantic validation first
                try:
                    # Choose the appropriate Pydantic model based on format
                    if format_type == "CriticalNeurIPS":
                        parsed_review = CriticalNeurIPSReview.model_validate_json(sanitized_json_content)
                    else:
                        parsed_review = PaperReview.model_validate_json(sanitized_json_content)
                    # Success! Break out of retry loop
                    break
                except Exception as pydantic_error:
                    # If Pydantic fails, try regular JSON parsing as fallback
                    try:
                        parsed_dict = json.loads(sanitized_json_content)
                        
                        # Fix common format issues: strengths/weaknesses as dicts with "point" key (only for default format)
                        if format_type != "CriticalNeurIPS":
                            for field in ['strengths', 'weaknesses']:
                                if field in parsed_dict and parsed_dict[field]:
                                    # Check if it's a list of dicts with "point" key
                                    if isinstance(parsed_dict[field], list) and len(parsed_dict[field]) > 0:
                                        if isinstance(parsed_dict[field][0], dict) and 'point' in parsed_dict[field][0]:
                                            # Convert [{"point": "..."}, ...] to ["...", ...]
                                            parsed_dict[field] = [item.get('point', str(item)) if isinstance(item, dict) else item for item in parsed_dict[field]]
                        
                        # Create a valid review object from the dict based on format type
                        if format_type == "CriticalNeurIPS":
                            parsed_review = CriticalNeurIPSReview(**parsed_dict)
                        else:
                            parsed_review = PaperReview(**parsed_dict)
                        break
                    except (json.JSONDecodeError, ValueError) as json_e:
                        # JSON parsing failed - this is a retryable error
                        if attempt < MAX_RETRIES - 1:
                            _print_method(f"Worker {worker_id}: JSON parsing failed for {paper_id} ({folder_label}), attempt {attempt + 1}/{MAX_RETRIES}. Retrying...")
                            _print_method(f"Worker {worker_id}: Error: {json_e}")
                            response_text = None  # Reset to trigger retry
                            wait_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                            time.sleep(wait_time)
                            continue
                        else:
                            # Last attempt failed, raise to be caught below
                            raise json_e
                
            except Exception as e:
                last_exception = e
                _print_method(f"Worker {worker_id}: Error for {paper_id} ({folder_label}), attempt {attempt + 1}/{MAX_RETRIES}: {e}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    # Last attempt failed
                    break
        
        # Check if we successfully parsed the review
        if parsed_review is not None:
            # Successfully parsed
            review_data = parsed_review.model_dump()
            
            # Add metadata
            review_data["paper_id"] = paper_id
            review_data["folder"] = folder_label
            review_data["run_id"] = run_id
            review_data["success"] = True
            
            # Add score mappings for compatibility with evaluation scripts
            if format_type == "CriticalNeurIPS":
                # CriticalNeurIPS format already has soundness, presentation, contribution
                review_data["rating"] = review_data.get("overall_score")
            else:
                # Default format - map to standard names
                review_data["soundness"] = review_data.get("technical_quality_score")
                review_data["presentation"] = review_data.get("clarity_score")
                review_data["contribution"] = review_data.get("novelty_score")
                review_data["rating"] = review_data.get("overall_score")
            
            if verbose:
                _print_method(f"Worker {worker_id}: Successfully reviewed {paper_id} ({folder_label})")
            
            return review_data
        
        # If we get here, all attempts failed
        if response_text is None:
            err_msg = f"All API attempts failed for {paper_id} ({folder_label})."
            if last_exception:
                err_msg += f" Last error: {type(last_exception).__name__} - {last_exception}"
            return {
                "error": err_msg,
                "paper_id": paper_id,
                "folder": folder_label,
                "run_id": run_id,
                "success": False
            }
        else:
            # API call succeeded but JSON parsing failed after all retries
            sanitized_json_content = _sanitize_json_string(response_text)
            _print_method(f"Worker {worker_id}: Failed to parse JSON after {MAX_RETRIES} attempts for {paper_id} ({folder_label})")
            _print_method(f"Worker {worker_id}: Raw JSON (first 500 chars): {sanitized_json_content[:500]}...")
            
            # Last resort: try to extract whatever we can
            try:
                fallback_data = json.loads(sanitized_json_content)
                
                # Fix common format issues: strengths/weaknesses as dicts with "point" key (only for default format)
                if format_type != "CriticalNeurIPS":
                    for field in ['strengths', 'weaknesses']:
                        if field in fallback_data and fallback_data[field]:
                            # Check if it's a list of dicts with "point" key
                            if isinstance(fallback_data[field], list) and len(fallback_data[field]) > 0:
                                if isinstance(fallback_data[field][0], dict) and 'point' in fallback_data[field][0]:
                                    # Convert [{"point": "..."}, ...] to ["...", ...]
                                    fallback_data[field] = [item.get('point', str(item)) if isinstance(item, dict) else item for item in fallback_data[field]]
                
                # Add metadata
                fallback_data["paper_id"] = paper_id
                fallback_data["folder"] = folder_label
                fallback_data["run_id"] = run_id
                fallback_data["success"] = True
                fallback_data["__parsing_warning"] = "JSON parsed but Pydantic validation may have failed"
                
                # Add score mappings for compatibility with evaluation scripts
                if format_type == "CriticalNeurIPS":
                    # CriticalNeurIPS format already has soundness, presentation, contribution
                    fallback_data["rating"] = fallback_data.get("overall_score")
                else:
                    # Default format - map to standard names
                    fallback_data["soundness"] = fallback_data.get("technical_quality_score")
                    fallback_data["presentation"] = fallback_data.get("clarity_score")
                    fallback_data["contribution"] = fallback_data.get("novelty_score")
                    fallback_data["rating"] = fallback_data.get("overall_score")
                
                return fallback_data
            except:
                return {
                    "error": f"Failed to parse JSON from LLM after {MAX_RETRIES} attempts",
                    "paper_id": paper_id,
                    "folder": folder_label,
                    "run_id": run_id,
                    "raw_content": response_text[:1000] if response_text else None,
                    "last_exception": str(last_exception) if last_exception else None,
                    "success": False
                }
    
    except Exception as e:
        message = f"FATAL ERROR reviewing {paper_id} ({folder_label}): {type(e).__name__} - {e}"
        _print_method(f"Worker {worker_id}: {message}")
        import traceback
        _print_method(traceback.format_exc())
        return {
            "error": message,
            "paper_id": paper_id,
            "folder": folder_label,
            "run_id": run_id,
            "success": False
        }

def review_papers_in_folder(
    base_dir: Path,
    folder_name: str,
    output_dir: Path,
    model_name: str,
    verbose: bool,
    skip_existing: bool = False,
    num_runs: int = 1,
    max_workers: int = None,
    format_type: str = "default"
) -> List[dict]:
    """
    Review all papers in a specific folder using parallel processing with multiple API keys.
    
    Args:
        base_dir: Base directory (e.g., data/ICLR2024)
        folder_name: Folder name to review (e.g., "latest", "authors_affiliation_good")
        output_dir: Output directory for reviews
        model_name: Model name
        verbose: Enable verbose output
        skip_existing: Skip papers that already have review files
        num_runs: Number of times to review each paper
        max_workers: Maximum number of worker threads (default: number of API keys)
    """
    folder_path = base_dir / folder_name
    if not folder_path.exists():
        print(f"Warning: Folder {folder_path} does not exist, skipping...")
        return []
    
    # Get all paper directories
    paper_dirs = [d for d in folder_path.iterdir() if d.is_dir()]
    print(f"\nFound {len(paper_dirs)} papers in {folder_name}/")
    
    # Create output directory for this folder
    folder_output_dir = output_dir / folder_name
    folder_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate request delay for this model
    request_delay = get_request_delay_for_model(model_name)
    
    # Prepare tasks: (paper_id, paper_dir, run_id, api_key, key_name, request_delay)
    tasks = []
    key_names = list(GEMINI_API_KEYS.keys())
    
    for paper_dir in paper_dirs:
        paper_id = paper_dir.name.split('_')[0]
        
        for run_id in range(num_runs):
            # Check if review already exists
            review_file = folder_output_dir / paper_id / f"review_run{run_id}.json"
            if skip_existing and review_file.exists():
                if verbose:
                    print(f"Skipping {paper_id} ({folder_name}) run {run_id} - already exists")
                continue
            
            # Assign API key (round-robin)
            task_idx = len(tasks)
            key_name = key_names[task_idx % len(key_names)]
            api_key = GEMINI_API_KEYS[key_name]
            
            tasks.append((paper_id, paper_dir, run_id, api_key, key_name, review_file, request_delay))
    
    if not tasks:
        print(f"No tasks to process for {folder_name}/")
        return []
    
    print(f"Prepared {len(tasks)} review tasks for {folder_name}/")
    
    # Set max_workers based on number of API keys if not specified
    if max_workers is None:
        max_workers = len(GEMINI_API_KEYS)
    
    reviews = []
    
    # Process tasks in parallel if multiple keys, otherwise sequential
    if len(GEMINI_API_KEYS) > 1:
        print(f"Processing {len(tasks)} tasks in parallel using {len(GEMINI_API_KEYS)} API keys (max_workers={max_workers})...")
        
        def process_task(task):
            paper_id, paper_dir, run_id, api_key, key_name, review_file, task_request_delay = task
            
            # Review paper
            review_data = review_single_paper(
                paper_id=paper_id,
                paper_path=paper_dir,
                folder_label=folder_name,
                api_key=api_key,
                key_name=key_name,
                model_name=model_name,
                verbose=verbose,
                run_id=run_id,
                request_delay=task_request_delay,
                format_type=format_type
            )
            
            # Save review
            if review_data.get("success"):
                paper_output_dir = folder_output_dir / paper_id
                paper_output_dir.mkdir(parents=True, exist_ok=True)
                
                with open(review_file, 'w', encoding='utf-8') as f:
                    json.dump(review_data, f, indent=2, ensure_ascii=False)
            
            return review_data
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_task, task): task for task in tasks}
            
            for future in tqdm(as_completed(futures), total=len(tasks), desc=f"Reviewing {folder_name}"):
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
                        "folder": folder_name,
                        "success": False
                    })
    else:
        # Sequential processing
        print(f"Processing {len(tasks)} tasks sequentially...")
        for task in tqdm(tasks, desc=f"Reviewing {folder_name}"):
            paper_id, paper_dir, run_id, api_key, key_name, review_file, task_request_delay = task
            
            # Review paper
            review_data = review_single_paper(
                paper_id=paper_id,
                paper_path=paper_dir,
                folder_label=folder_name,
                api_key=api_key,
                key_name=key_name,
                model_name=model_name,
                verbose=verbose,
                run_id=run_id,
                request_delay=task_request_delay,
                format_type=format_type
            )
            
            # Save review
            if review_data.get("success"):
                paper_output_dir = folder_output_dir / paper_id
                paper_output_dir.mkdir(parents=True, exist_ok=True)
                
                with open(review_file, 'w', encoding='utf-8') as f:
                    json.dump(review_data, f, indent=2, ensure_ascii=False)
            
            reviews.append(review_data)
    
    return reviews

def main():
    parser = argparse.ArgumentParser(
        description="Review papers from specified folders using Gemini API"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing the folders (e.g., data/ICLR2024 or data/sampled_data/ICLR2024)"
    )
    parser.add_argument(
        "--folders",
        type=str,
        nargs="+",
        required=True,
        help="Folder names to review (e.g., latest authors_affiliation_good authors_affiliation_bad)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for reviews (default: base_dir/../reviews/)"
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
    parser.add_argument(
        "--format",
        type=str,
        choices=["default", "CriticalNeurIPS"],
        default="default",
        help="Review format: 'default' or 'CriticalNeurIPS' (default: default)"
    )
    
    args = parser.parse_args()
    
    # Check if Gemini is available
    if not GENAI_AVAILABLE:
        print("Error: google-generativeai not installed. Install with: pip install google-generativeai")
        exit(1)
    
    if not GEMINI_API_KEYS:
        print("Error: No Gemini API keys found in environment variables or .env file.")
        exit(1)
    
    base_dir = Path(args.base_dir)
    
    # Set default output directory
    if args.output_dir is None:
        output_dir = base_dir.parent / "reviews" / base_dir.name
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Base directory: {base_dir}")
    print(f"Folders to review: {args.folders}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model_name}")
    print(f"Format: {args.format}")
    print(f"Number of API keys: {len(GEMINI_API_KEYS)}")
    model_rpm = GEMINI_MODEL_RPM_LIMITS.get(args.model_name, 30)
    model_delay = get_request_delay_for_model(args.model_name)
    print(f"RPM limit: {model_rpm} per key")
    print(f"Request delay: {model_delay:.2f} seconds per key")
    print(f"Total throughput: {len(GEMINI_API_KEYS) * model_rpm} RPM ({len(GEMINI_API_KEYS)} keys × {model_rpm} RPM)")
    print(f"Number of runs per paper: {args.num_runs}")
    print()
    
    # Review papers in each folder
    all_reviews = []
    for folder_name in args.folders:
        reviews = review_papers_in_folder(
            base_dir=base_dir,
            folder_name=folder_name,
            output_dir=output_dir,
            model_name=args.model_name,
            verbose=args.verbose,
            skip_existing=args.skip_existing,
            num_runs=args.num_runs,
            max_workers=args.max_workers,
            format_type=args.format
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

