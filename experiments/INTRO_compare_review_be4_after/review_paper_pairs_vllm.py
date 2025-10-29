import os
import json
import argparse
import time
import re
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List
import base64
import io
import math
import urllib.parse

# Attempt to import Pillow
try:
    from PIL import Image
    if not hasattr(Image, 'Resampling'):
        Image.Resampling = Image
except ImportError:
    print("WARNING: Pillow library not found. Install it (`pip install Pillow`) to process images.")
    Image = None

# --- Environment & API Configuration ---
load_dotenv()

# --- Constants ---
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2
RETRYABLE_STATUS_CODES = [429, 500, 502, 503, 504]

# --- Image Constants ---
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024  # API limit: 20MB
TARGET_RESIZE_BYTES = 10 * 1024 * 1024  # Softer target for resizing
MAX_RESIZE_ATTEMPTS = 4
MIN_DIMENSION_AFTER_RESIZE = 50

# --- Context Length Constants ---
# Model-specific context limits (tokens)
MODEL_CONTEXT_LIMITS = {
    "SEA-E": 32768,
    "default": 128000,  # Conservative default
}

# Reserve tokens for completion
COMPLETION_TOKENS = 4096

# Approximate tokens per character (conservative estimate)
# For English text, roughly 1 token = 4 characters
CHARS_PER_TOKEN = 4.5

# --- Pydantic Models for Structured Review Output ---
class PaperReview(BaseModel):
    """Pydantic model for the structured review output (JSON format)."""
    summary: str = Field(
        description="A 2-3 sentence summary of the paper's main contribution and approach."
    )
    strengths: List[str] = Field(
        description="A list of the paper's key strengths (3-5 points)."
    )
    weaknesses: List[str] = Field(
        description="A list of the paper's key weaknesses and limitations (3-5 points)."
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

class SEAEReview(BaseModel):
    """Pydantic model for SEA-E format review output."""
    summary: str = Field(description="Summary of the paper in 100-150 words")
    strengths: List[str] = Field(description="List of paper strengths")
    weaknesses: List[str] = Field(description="List of paper weaknesses")
    questions: List[str] = Field(description="List of questions about the paper")
    soundness: str = Field(description="Soundness rating (1-4: poor/fair/good/excellent)")
    presentation: str = Field(description="Presentation rating (1-4: poor/fair/good/excellent)")
    contribution: str = Field(description="Contribution rating (1-4: poor/fair/good/excellent)")
    rating: str = Field(description="Overall rating (1-10)")
    paper_decision: str = Field(description="Decision (Accept/Reject) with reasons")

class ReviewPrompts:
    @staticmethod
    def detect_model_type(model_name: str) -> str:
        """Detect the model type based on model name."""
        model_name_lower = model_name.lower()
        if "sea-e" in model_name_lower or "seae" in model_name_lower:
            return "SEA-E"
        # Add more model types here as needed
        return "default"
    
    @staticmethod
    def get_system_prompt(model_type: str = "default") -> str:
        """Returns the system prompt for paper review based on model type."""
        if model_type == "SEA-E":
            return """You are a highly experienced, conscientious, and fair academic reviewer, please help me review this paper. The review should be organized into nine sections: 
1. Summary: A summary of the paper in 100-150 words.
2. Strengths/Weaknesses/Questions: The Strengths/Weaknesses/Questions of paper, which should be listed in bullet points, with each point supported by specific examples from the article where possible.
3. Soundness/Contribution/Presentation: Rate the paper's Soundness/Contribution/Presentation, and match this score to the corresponding description from the list below and provide the result. The possible scores and their descriptions are: 
    1 poor
    2 fair
    3 good
    4 excellent
4. Rating: Give this paper an appropriate rating, match this rating to the corresponding description from the list below and provide the result. The possible Ratings and their descriptions are: 
    1 strong reject
    2 reject, significant issues present
    3 reject, not good enough
    4 possibly reject, but has redeeming facets
    5 marginally below the acceptance threshold
    6 marginally above the acceptance threshold
    7 accept, but needs minor improvements 
    8 accept, good paper
    9 strong accept, excellent work
    10 strong accept, should be highlighted at the conference   
5. Paper Decision: It must include the Decision itself(Accept or Reject) and the reasons for this decision, based on the criteria of originality, methodological soundness, significance of results, and clarity and logic of presentation.

Here is the template for a review format, you must follow this format to output your review result:
**Summary:**
Summary content

**Strengths:**
- Strength 1
- Strength 2
- ...

**Weaknesses:**
- Weakness 1
- Weakness 2
- ...

**Questions:**
- Question 1
- Question 2
- ...

**Soundness:**
Soundness result

**Presentation:**
Presentation result

**Contribution:**
Contribution result

**Rating:**
Rating result

**Paper Decision:**
- Decision: Accept/Reject
- Reasons: reasons content


Please ensure your feedback is objective and constructive."""
        else:
            # Default JSON format
            return """You are an expert peer reviewer for a top-tier machine learning conference (NeurIPS, ICML, or ICLR). Your task is to provide a thorough, balanced, and constructive review of the submitted research paper.

Your review should assess the paper across multiple dimensions:
1. **Clarity**: How well-written and organized is the paper?
2. **Novelty**: How original and innovative is the contribution?
3. **Technical Quality**: How sound and rigorous is the technical approach?
4. **Experimental Rigor**: How comprehensive and convincing are the experiments?

You must provide your assessment in a specific JSON format with the following fields:
- summary: A 2-3 sentence overview of the paper
- strengths: A list of key strengths (3-5 points as separate strings)
- weaknesses: A list of key weaknesses (3-5 points as separate strings)
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
    def get_user_prompt(paper_content: str, paper_version: str, flaw_context: Optional[str] = None, model_type: str = "default") -> str:
        """Constructs the user-facing prompt for review."""
        flaw_info = ""
        if flaw_context:
            flaw_info = f"""

Note: This paper has been identified as having the following potential issues in peer review:
<flaw_context>
{flaw_context}
</flaw_context>

Please consider these issues in your assessment, but conduct your own independent evaluation as well.
"""
        
        if model_type == "SEA-E":
            return f"""The paper is as follows:

<paper_content>
{paper_content}
</paper_content>
{flaw_info}"""
        else:
            return f"""Please review the following research paper ({paper_version}):

<paper_content>
{paper_content}
</paper_content>
{flaw_info}

Provide a comprehensive review following the specified JSON format.
"""

def _sanitize_json_string(json_str: str) -> str:
    """Cleans common JSON errors from LLM output."""
    json_str = json_str.strip().strip("```json").strip("```")
    json_str = re.sub(r',\s*(?=[}\]])', '', json_str)
    return json_str

def _parse_seae_format(content: str) -> dict:
    """
    Parses SEA-E markdown format into structured dictionary.
    
    Expected format:
    **Summary:**
    text
    
    **Strengths:**
    - item 1
    - item 2
    
    **Weaknesses:**
    - item 1
    
    **Questions:**
    - item 1
    
    **Soundness:**
    score text
    
    **Presentation:**
    score text
    
    **Contribution:**
    score text
    
    **Rating:**
    score text
    
    **Paper Decision:**
    - Decision: Accept/Reject
    - Reasons: reasons
    """
    result = {
        "summary": "",
        "strengths": [],
        "weaknesses": [],
        "questions": [],
        "soundness": "",
        "presentation": "",
        "contribution": "",
        "rating": "",
        "paper_decision": "",
    }
    
    # Split content by section headers (handles multi-word headers like "Paper Decision")
    sections = re.split(r'\*\*([^*:]+):\*\*', content)
    
    # Process sections (odd indices are headers, even indices are content)
    current_section = None
    for i in range(1, len(sections), 2):
        if i+1 < len(sections):
            section_name = sections[i].lower().strip()
            section_content = sections[i+1].strip()
            
            if section_name == "summary":
                result["summary"] = section_content
            
            elif section_name == "strengths":
                # Extract bullet points
                strengths = re.findall(r'^\s*-\s*(.+)$', section_content, re.MULTILINE)
                result["strengths"] = [s.strip() for s in strengths if s.strip()]
            
            elif section_name == "weaknesses":
                weaknesses = re.findall(r'^\s*-\s*(.+)$', section_content, re.MULTILINE)
                result["weaknesses"] = [w.strip() for w in weaknesses if w.strip()]
            
            elif section_name == "questions":
                questions = re.findall(r'^\s*-\s*(.+)$', section_content, re.MULTILINE)
                result["questions"] = [q.strip() for q in questions if q.strip()]
            
            elif section_name == "soundness":
                result["soundness"] = section_content
            
            elif section_name == "presentation":
                result["presentation"] = section_content
            
            elif section_name == "contribution":
                result["contribution"] = section_content
            
            elif section_name == "rating":
                result["rating"] = section_content
            
            elif section_name in ["paper decision", "paperdecision", "decision"]:
                # Extract decision and reasons
                decision_match = re.search(r'-\s*Decision:\s*(\w+)', section_content, re.IGNORECASE)
                reasons_match = re.search(r'-\s*Reasons?:\s*(.+)', section_content, re.IGNORECASE | re.DOTALL)
                
                decision = decision_match.group(1) if decision_match else "Unknown"
                reasons = reasons_match.group(1).strip() if reasons_match else section_content
                
                result["paper_decision"] = f"Decision: {decision}\nReasons: {reasons}"
    
    return result

def _estimate_tokens(text: str) -> int:
    """
    Estimate token count from text.
    Uses conservative approximation: 1 token ≈ 4 characters
    """
    return len(text) // CHARS_PER_TOKEN

def _truncate_paper_content(
    paper_content: str, 
    max_tokens: int,
    preserve_ratio: float = 0.7,
    verbose: bool = False,
    worker_id: int = 0
) -> tuple[str, bool]:
    """
    Truncate paper content to fit within token limit.
    
    Args:
        paper_content: Full paper text
        max_tokens: Maximum tokens allowed
        preserve_ratio: Ratio of content to keep from beginning (0.7 = 70% from start, 30% from end)
        verbose: Print truncation info
        worker_id: Worker ID for logging
    
    Returns:
        (truncated_content, was_truncated)
    """
    _print_method = tqdm.write if not verbose else print
    
    current_tokens = _estimate_tokens(paper_content)
    
    if current_tokens <= max_tokens:
        return paper_content, False
    
    # Calculate target character count
    target_chars = max_tokens * CHARS_PER_TOKEN
    
    # Add truncation notice
    truncation_notice = "\n\n[... CONTENT TRUNCATED DUE TO LENGTH LIMITS ...]\n\n"
    truncation_notice_tokens = _estimate_tokens(truncation_notice)
    
    # Adjust target to account for notice
    available_chars = (max_tokens - truncation_notice_tokens) * CHARS_PER_TOKEN
    
    # Split content: preserve_ratio from start, rest from end
    start_chars = int(available_chars * preserve_ratio)
    end_chars = available_chars - start_chars
    
    # Truncate
    start_content = paper_content[:start_chars]
    end_content = paper_content[-end_chars:] if end_chars > 0 else ""
    
    truncated = start_content + truncation_notice + end_content
    
    if verbose:
        _print_method(
            f"Worker {worker_id}: Truncated paper from {current_tokens} to ~{_estimate_tokens(truncated)} tokens "
            f"({len(paper_content)} → {len(truncated)} chars)"
        )
    
    return truncated, True

def find_paper_markdown(paper_folder: Path) -> Optional[Path]:
    """Finds the paper.md file in the structured_paper_output directory."""
    paper_md_path = paper_folder / "structured_paper_output" / "paper.md"
    if paper_md_path.exists():
        return paper_md_path
    
    # Fallback: search for any .md file
    md_files = list(paper_folder.glob("**/*.md"))
    return md_files[0] if md_files else None

def process_images_for_api(
    markdown_content: str,
    original_paper_dir: Path,
    max_figures: int,
    verbose: bool,
    worker_id: int
) -> list:
    """Finds, resizes, and encodes images referenced in markdown for the API."""
    _print_method = tqdm.write if not verbose else print
    if Image is None or max_figures == 0:
        return []

    content_list_images = []
    added_figures_count = 0
    
    md_img_patterns = [
        r"!\[[^\]]*\]\(([^)\s]+?)(?:\s+[\"'][^\"']*[\"'])?\)", 
        r"<img\s+[^>]*?src\s*=\s*[\"']([^\"']+)[\"'][^>]*?>",
    ]
    valid_image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.webp']
    
    all_matches = []
    for pattern_str in md_img_patterns:
        for match in re.finditer(pattern_str, markdown_content, flags=re.IGNORECASE | re.DOTALL):
            all_matches.append({'match': match, 'start_pos': match.start()})
    all_matches.sort(key=lambda x: x['start_pos'])

    if verbose and all_matches:
        _print_method(f"Worker {worker_id}: Found {len(all_matches)} potential image references.")
    
    processed_image_paths = set()
    for item in all_matches:
        if added_figures_count >= max_figures:
            if verbose: _print_method(f"Worker {worker_id}: Reached max_figures ({max_figures}).")
            break
        
        raw_path = item['match'].group(1)
        decoded_path_str = urllib.parse.unquote(raw_path)

        if urllib.parse.urlparse(decoded_path_str).scheme in ['http', 'https']:
            continue # Skip web URLs

        # Resolve image path against the original paper's directory
        candidate_path = (original_paper_dir / decoded_path_str).resolve()
        
        if str(candidate_path) in processed_image_paths:
            continue

        if not candidate_path.is_file():
            continue # Skip if file doesn't exist

        try:
            initial_size = candidate_path.stat().st_size
            if verbose: _print_method(f"Worker {worker_id}: Processing image '{candidate_path}' (Size: {initial_size / 1024**2:.2f}MB).")

            image_bytes = None
            img_format = candidate_path.suffix.lstrip('.').upper()
            if img_format == 'JPG': img_format = 'JPEG'
            
            if initial_size <= MAX_IMAGE_SIZE_BYTES:
                with open(candidate_path, "rb") as f:
                    image_bytes = f.read()
            else:
                # Resize logic
                pil_img = Image.open(candidate_path)
                img_format = pil_img.format or 'JPEG' 
                
                for attempt in range(MAX_RESIZE_ATTEMPTS):
                    scale_factor = math.sqrt(TARGET_RESIZE_BYTES / initial_size) 
                    new_dims = (int(pil_img.width * scale_factor), int(pil_img.height * scale_factor))
                    
                    if new_dims[0] < MIN_DIMENSION_AFTER_RESIZE or new_dims[1] < MIN_DIMENSION_AFTER_RESIZE:
                        if verbose: _print_method(f"Worker {worker_id}: Resizing {candidate_path} stopped, would go below min dimension.")
                        break

                    resized_img = pil_img.resize(new_dims, Image.Resampling.LANCZOS)
                    buffer = io.BytesIO()
                    save_params = {'format': img_format}
                    if img_format == 'JPEG': save_params['quality'] = 85
                    
                    resized_img.save(buffer, **save_params)
                    
                    if buffer.tell() <= MAX_IMAGE_SIZE_BYTES:
                        image_bytes = buffer.getvalue()
                        if verbose: _print_method(f"Worker {worker_id}: Resized '{candidate_path}' to {len(image_bytes) / 1024**2:.2f}MB.")
                        break
                    
                    pil_img = resized_img
                    initial_size = buffer.tell()

            if image_bytes:
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                content_list_images.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
                added_figures_count += 1
                processed_image_paths.add(str(candidate_path))
            elif verbose:
                 _print_method(f"Worker {worker_id}: Could not resize '{candidate_path}' under max bytes.")

        except Exception as e:
            _print_method(f"Worker {worker_id}: Error processing image {candidate_path}: {e}")

    return content_list_images

def review_single_paper_vllm(
    paper_id: str,
    paper_path: Path,
    version_label: str,
    flaw_descriptions: list,
    vllm_endpoint: str,
    model_name: str,
    max_figures: int,
    verbose: bool,
    run_id: int = 0
) -> dict:
    """
    Reviews a single paper version using vLLM and returns structured results.
    """
    worker_id = concurrent.futures.thread.get_ident() if hasattr(concurrent.futures.thread, 'get_ident') else os.getpid()
    _print_method = tqdm.write if not verbose else print
    
    try:
        # Detect model type
        model_type = ReviewPrompts.detect_model_type(model_name)
        
        # Read paper content
        paper_md = find_paper_markdown(paper_path)
        if not paper_md:
            return {
                "error": f"Could not find paper markdown for {paper_id} at {paper_path}",
                "paper_id": paper_id,
                "version": version_label,
                "run_id": run_id,
                "model_type": model_type
            }
        
        with open(paper_md, 'r', encoding='utf-8') as f:
            paper_content = f.read()
        
        # Get context limit for this model type
        max_context_tokens = MODEL_CONTEXT_LIMITS.get(model_type, MODEL_CONTEXT_LIMITS["default"])
        available_tokens = max_context_tokens - COMPLETION_TOKENS
        
        # Prepare prompts (before truncation to estimate overhead)
        flaw_context = None
        if flaw_descriptions:
            flaw_context = "\n".join([f"- {flaw}" for flaw in flaw_descriptions])
        
        system_prompt = ReviewPrompts.get_system_prompt(model_type)
        user_prompt_template = ReviewPrompts.get_user_prompt("", version_label, flaw_context, model_type)
        
        # Estimate overhead tokens (system prompt + user prompt template)
        overhead_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt_template)
        
        # Account for images (rough estimate: ~765 tokens per image)
        image_blocks = []
        if max_figures > 0:
            image_blocks = process_images_for_api(
                paper_content, paper_path, max_figures, verbose, worker_id
            )
            # Estimate image tokens (conservative: 1000 tokens per image)
            overhead_tokens += len(image_blocks) * 1000
        
        # Calculate available tokens for paper content
        max_paper_tokens = available_tokens - overhead_tokens - 500  # 500 buffer
        
        # Truncate paper content if needed
        paper_content, was_truncated = _truncate_paper_content(
            paper_content, 
            max_paper_tokens, 
            preserve_ratio=0.7,
            verbose=verbose,
            worker_id=worker_id
        )
        
        # Now create the full user prompt with (possibly truncated) paper content
        user_prompt_text = ReviewPrompts.get_user_prompt(paper_content, version_label, flaw_context, model_type)
        
        # Build messages in OpenAI format (vLLM compatible)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_prompt_text}] + image_blocks}
        ]
        
        response_obj = None
        last_exception = None
        
        for attempt in range(MAX_RETRIES):
            try:
                if verbose:
                    _print_method(f"Worker {worker_id}: Reviewing {paper_id} ({version_label}, run {run_id}), attempt {attempt + 1}/{MAX_RETRIES}")
                
                # Call vLLM endpoint
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": 4096,
                    "temperature": 0.7,
                }
                
                response = requests.post(
                    f"{vllm_endpoint}/v1/chat/completions",
                    json=payload,
                    timeout=300.0,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    response_obj = response.json()
                    if verbose:
                        _print_method(f"Worker {worker_id}: Successfully reviewed {paper_id} ({version_label}, run {run_id})")
                    break
                elif response.status_code in RETRYABLE_STATUS_CODES:
                    wait_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                    _print_method(f"Worker {worker_id}: Retrying {paper_id} ({version_label}, run {run_id}) in {wait_time}s due to status {response.status_code}...")
                    time.sleep(wait_time)
                else:
                    _print_method(f"Worker {worker_id}: Non-retryable error for {paper_id} ({version_label}, run {run_id}): {response.status_code} - {response.text}")
                    break
                    
            except requests.exceptions.RequestException as e:
                last_exception = e
                _print_method(f"Worker {worker_id}: Request error for {paper_id} ({version_label}, run {run_id}): {e}")
                wait_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                time.sleep(wait_time)
                
            except Exception as e:
                last_exception = e
                _print_method(f"Worker {worker_id}: Unexpected error for {paper_id} ({version_label}, run {run_id}): {e}")
                wait_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                time.sleep(wait_time)
        
        if response_obj is None or 'choices' not in response_obj:
            err_msg = f"All API attempts failed for {paper_id} ({version_label}, run {run_id})."
            if last_exception:
                err_msg += f" Last error: {type(last_exception).__name__} - {last_exception}"
            return {
                "error": err_msg,
                "paper_id": paper_id,
                "version": version_label,
                "run_id": run_id
            }
        
        raw_content = response_obj['choices'][0]['message']['content']
        
        # Parse based on model type
        try:
            if model_type == "SEA-E":
                # Parse SEA-E markdown format
                if verbose:
                    _print_method(f"Worker {worker_id}: Parsing SEA-E format for {paper_id} ({version_label}, run {run_id})")
                
                parsed_data = _parse_seae_format(raw_content)
                
                # Validate with Pydantic
                try:
                    validated_review = SEAEReview.model_validate(parsed_data)
                    review_data = validated_review.model_dump()
                except Exception as val_error:
                    if verbose:
                        _print_method(f"Worker {worker_id}: SEA-E validation warning: {val_error}")
                    review_data = parsed_data
                    review_data["__validation_warning"] = str(val_error)
                
                review_data["paper_id"] = paper_id
                review_data["version"] = version_label
                review_data["run_id"] = run_id
                review_data["model_type"] = model_type
                review_data["success"] = True
                review_data["raw_content"] = raw_content
                review_data["was_truncated"] = was_truncated
                return review_data
            
            else:
                # Parse JSON format (default)
                sanitized_json_content = _sanitize_json_string(raw_content)
                
                try:
                    parsed_review = PaperReview.model_validate_json(sanitized_json_content)
                    review_data = parsed_review.model_dump()
                    review_data["paper_id"] = paper_id
                    review_data["version"] = version_label
                    review_data["run_id"] = run_id
                    review_data["model_type"] = model_type
                    review_data["success"] = True
                    review_data["was_truncated"] = was_truncated
                    return review_data
                    
                except Exception as pydantic_error:
                    _print_method(f"Worker {worker_id}: Pydantic validation failed for {paper_id} ({version_label}, run {run_id}). Error: {pydantic_error}")
                    _print_method(f"Worker {worker_id}: Raw JSON: {sanitized_json_content[:500]}...")
                    
                    try:
                        fallback_data = json.loads(sanitized_json_content)
                        fallback_data["paper_id"] = paper_id
                        fallback_data["version"] = version_label
                        fallback_data["run_id"] = run_id
                        fallback_data["model_type"] = model_type
                        fallback_data["success"] = True
                        fallback_data["was_truncated"] = was_truncated
                        fallback_data["__pydantic_validation_error"] = str(pydantic_error)
                        return fallback_data
                    except json.JSONDecodeError as json_e:
                        return {
                            "error": "Failed to parse JSON from LLM",
                            "paper_id": paper_id,
                            "version": version_label,
                            "run_id": run_id,
                            "model_type": model_type,
                            "raw_content": raw_content[:1000],
                            "pydantic_error": str(pydantic_error),
                            "json_error": str(json_e),
                            "was_truncated": was_truncated,
                            "success": False
                        }
                        
        except Exception as parse_error:
            _print_method(f"Worker {worker_id}: Parse error for {paper_id} ({version_label}, run {run_id}): {parse_error}")
            return {
                "error": f"Failed to parse {model_type} format",
                "paper_id": paper_id,
                "version": version_label,
                "run_id": run_id,
                "model_type": model_type,
                "raw_content": raw_content[:1000],
                "parse_error": str(parse_error),
                "was_truncated": was_truncated,
                "success": False
            }
    
    except Exception as e:
        model_type = ReviewPrompts.detect_model_type(model_name)
        message = f"FATAL ERROR reviewing {paper_id} ({version_label}, run {run_id}): {type(e).__name__} - {e}"
        _print_method(f"Worker {worker_id}: {message}")
        import traceback
        _print_method(traceback.format_exc())
        return {
            "error": message,
            "paper_id": paper_id,
            "version": version_label,
            "run_id": run_id,
            "model_type": model_type,
            "success": False
        }

def review_paper_pair(
    pair_row: dict,
    vllm_endpoint: str,
    model_name: str,
    output_dir: Path,
    max_figures: int,
    verbose: bool,
    version_filter: str = "both",
    skip_existing: bool = False,
    num_runs: int = 1
) -> str:
    """
    Reviews both versions of a paper pair and saves results.
    
    Args:
        version_filter: Which version(s) to review - "v1", "latest", or "both"
        skip_existing: If True, skip reviewing papers that already have review files
        num_runs: Number of times to review each paper (for analysis of variance)
    """
    paper_id = pair_row['paperid']
    v1_folder = Path(pair_row['v1_folder_path'])
    latest_folder = Path(pair_row['latest_folder_path'])
    
    # Parse flaw descriptions if available
    flaw_descriptions = []
    if 'flaw_descriptions' in pair_row and pair_row['flaw_descriptions']:
        flaw_str = pair_row['flaw_descriptions']
        if isinstance(flaw_str, str) and flaw_str.startswith('['):
            try:
                flaw_descriptions = json.loads(flaw_str)
            except:
                flaw_descriptions = []
        elif isinstance(flaw_str, list):
            flaw_descriptions = flaw_str
    
    # Create paper output directory
    paper_output_dir = output_dir / paper_id
    paper_output_dir.mkdir(parents=True, exist_ok=True)
    
    all_reviews_success = True
    
    # Review v1 (if needed) - multiple runs
    if version_filter in ["v1", "both"]:
        for run_id in range(num_runs):
            v1_review_path = paper_output_dir / f"v1_review_run{run_id}.json"
            
            if skip_existing and v1_review_path.exists():
                if verbose:
                    tqdm.write(f"Skipping v1 review for {paper_id} run {run_id} (already exists)")
            else:
                v1_review = review_single_paper_vllm(
                    paper_id, v1_folder, "v1", flaw_descriptions,
                    vllm_endpoint, model_name, max_figures, verbose, run_id
                )
                
                with open(v1_review_path, 'w', encoding='utf-8') as f:
                    json.dump(v1_review, f, ensure_ascii=False, indent=2)
                
                if not v1_review.get("success", False):
                    all_reviews_success = False
    
    # Review latest (if needed) - multiple runs
    if version_filter in ["latest", "both"]:
        for run_id in range(num_runs):
            latest_review_path = paper_output_dir / f"latest_review_run{run_id}.json"
            
            if skip_existing and latest_review_path.exists():
                if verbose:
                    tqdm.write(f"Skipping latest review for {paper_id} run {run_id} (already exists)")
            else:
                latest_review = review_single_paper_vllm(
                    paper_id, latest_folder, "latest", flaw_descriptions,
                    vllm_endpoint, model_name, max_figures, verbose, run_id
                )
                
                with open(latest_review_path, 'w', encoding='utf-8') as f:
                    json.dump(latest_review, f, ensure_ascii=False, indent=2)
                
                if not latest_review.get("success", False):
                    all_reviews_success = False
    
    return f"Successfully reviewed pair {paper_id}" if all_reviews_success else f"Partial/failed review for {paper_id}"

def main():
    parser = argparse.ArgumentParser(description="Review paper pairs (v1 vs latest) using vLLM.")
    parser.add_argument("--csv_file", type=str, required=True, 
                       help="Path to filtered_pairs.csv file.")
    parser.add_argument("--output_dir", type=str, default="./pair_reviews_vllm/",
                       help="Output directory for review results.")
    parser.add_argument("--vllm_endpoint", type=str, required=True,
                       help="vLLM server endpoint URL (e.g., http://localhost:8000)")
    parser.add_argument("--model_name", type=str, required=True,
                       help="Model name hosted on vLLM server.")
    parser.add_argument("--max_workers", type=int, default=3,
                       help="Max worker threads for concurrent processing.")
    parser.add_argument("--max_figures", type=int, default=5,
                       help="Max figures to include per paper (0 for none).")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output.")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of pairs to process (for testing).")
    parser.add_argument("--num_runs", type=int, default=1,
                       help="Number of times to review each paper (for variance analysis).")
    
    # Debug/Save API Credit Options
    parser.add_argument("--version", type=str, choices=["v1", "latest", "both"], default="both",
                       help="Which version(s) to review: 'v1', 'latest', or 'both' (default: both).")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip papers that already have review files.")
    
    args = parser.parse_args()
    
    # Test vLLM connection
    try:
        response = requests.get(f"{args.vllm_endpoint}/health", timeout=10)
        if response.status_code != 200:
            print(f"Warning: vLLM health check returned status {response.status_code}")
    except Exception as e:
        print(f"Warning: Could not connect to vLLM endpoint at {args.vllm_endpoint}: {e}")
        print("Proceeding anyway, but API calls may fail.")
    
    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {args.csv_file}")
        exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        exit(1)
    
    # Apply limit if specified
    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to first {args.limit} pairs for testing.")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect model type
    model_type = ReviewPrompts.detect_model_type(args.model_name)
    
    print(f"Preparing to review {len(df)} paper pairs...")
    print(f"vLLM Endpoint: {args.vllm_endpoint}")
    print(f"Model: {args.model_name}")
    print(f"Detected Model Type: {model_type}")
    print(f"Version filter: {args.version}")
    print(f"Number of runs per paper: {args.num_runs}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Output directory: {output_dir}")
    print(f"Max workers: {args.max_workers}")
    print(f"Max figures per paper: {args.max_figures}")
    
    # Calculate estimated API calls
    estimated_calls = len(df) * args.num_runs
    if args.version == "both":
        estimated_calls *= 2
    print(f"Estimated API calls: {estimated_calls} (max, may be less with --skip_existing)\n")
    
    # Prepare tasks
    tasks = []
    for _, row in df.iterrows():
        tasks.append((
            row.to_dict(), args.vllm_endpoint, args.model_name, output_dir, 
            args.max_figures, args.verbose, args.version, args.skip_existing, args.num_runs
        ))
    
    # Process with thread pool
    processed_count = 0
    error_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {
            executor.submit(review_paper_pair, *task): task 
            for task in tasks
        }
        
        progress = tqdm(concurrent.futures.as_completed(future_to_task), 
                       total=len(tasks), desc="Reviewing Paper Pairs")
        
        for future in progress:
            task_info = future_to_task[future]
            paper_id = task_info[0]['paperid']
            
            try:
                result_message = future.result()
                if "Successfully reviewed" in result_message:
                    processed_count += 1
                else:
                    error_count += 1
                    tqdm.write(f"Review failed for {paper_id}: {result_message}")
            except Exception as exc:
                error_count += 1
                tqdm.write(f"Exception for {paper_id}: {exc}")
                import traceback
                tqdm.write(traceback.format_exc())
    
    # Generate summary CSV (aggregate across all runs)
    summary_data = []
    for _, row in df.iterrows():
        paper_id = row['paperid']
        paper_dir = output_dir / paper_id
        
        if not paper_dir.exists():
            continue
        
        for run_id in range(args.num_runs):
            v1_review_path = paper_dir / f"v1_review_run{run_id}.json"
            latest_review_path = paper_dir / f"latest_review_run{run_id}.json"
            
            v1_data = {}
            latest_data = {}
            
            if v1_review_path.exists():
                with open(v1_review_path, 'r') as f:
                    v1_data = json.load(f)
            
            if latest_review_path.exists():
                with open(latest_review_path, 'r') as f:
                    latest_data = json.load(f)
            
            if v1_data or latest_data:
                summary_row = {
                    "paper_id": paper_id,
                    "run_id": run_id,
                    "arxiv_id": row.get('arxiv_id', ''),
                }
                
                # Add v1 scores
                for field in ["clarity_score", "novelty_score", "technical_quality_score", 
                             "experimental_rigor_score", "overall_score", "confidence"]:
                    summary_row[f"v1_{field}"] = v1_data.get(field, '')
                summary_row["v1_recommendation"] = v1_data.get("recommendation", '')
                summary_row["v1_success"] = v1_data.get("success", False)
                
                # Add latest scores
                for field in ["clarity_score", "novelty_score", "technical_quality_score", 
                             "experimental_rigor_score", "overall_score", "confidence"]:
                    summary_row[f"latest_{field}"] = latest_data.get(field, '')
                summary_row["latest_recommendation"] = latest_data.get("recommendation", '')
                summary_row["latest_success"] = latest_data.get("success", False)
                
                # Add score changes
                if v1_data.get("success") and latest_data.get("success"):
                    for field in ["clarity_score", "novelty_score", "technical_quality_score", 
                                 "experimental_rigor_score", "overall_score"]:
                        if field in v1_data and field in latest_data:
                            v1_score = v1_data[field]
                            latest_score = latest_data[field]
                            summary_row[f"{field}_change"] = latest_score - v1_score
                
                summary_data.append(summary_row)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = output_dir / "review_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nSummary CSV saved to: {summary_csv_path}")
    
    print("\n--- Processing Complete ---")
    print(f"Total pairs successfully reviewed: {processed_count}")
    print(f"Failed/Errored reviews: {error_count}")
    print(f"Results saved in: {output_dir}")
    print("---------------------------\n")

if __name__ == "__main__":
    main()

# Example usage:
"""
# Test with 1 paper, v1 version only, 1 run:
python review_paper_pairs_vllm.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_vllm_test" \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "Qwen/Qwen2-VL-7B-Instruct" \
  --version v1 \
  --limit 1 \
  --max_figures 5 \
  --verbose

# Full run with 3 runs per paper for variance analysis:
python review_paper_pairs_vllm.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_vllm_multi" \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "Qwen/Qwen2-VL-7B-Instruct" \
  --version both \
  --num_runs 3 \
  --max_figures 5 \
  --max_workers 5 \
  --verbose

# Continue from previous run - skip existing:
python review_paper_pairs_vllm.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_vllm_multi" \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "Qwen/Qwen2-VL-7B-Instruct" \
  --version both \
  --num_runs 3 \
  --skip_existing \
  --max_workers 5
"""

