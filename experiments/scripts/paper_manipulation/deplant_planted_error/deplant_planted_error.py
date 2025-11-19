#!/usr/bin/env python3
"""
Script to fix planted errors in papers by using Gemini to revise sections at known locations.

This script reads planted_error files and their modification summaries (CSV files),
then uses Gemini to fix the errors at the specific locations where modifications were made.
The fixed papers are saved in a new de-planted_error folder.

Usage:
    python deplant_planted_error.py \
        --data_dir path/to/with_appendix \
        --conference NeurIPS2024 \
        [--model_name gemini-2.0-flash] \
        [--categories 1a,1b,1c] \
        [--max_workers 4]
"""

import os
import json
import argparse
import time
import re
import html
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import threading
from collections import deque
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple, Dict
import google.generativeai as genai

# --- Environment & API Configuration ---
# Try to find .env file - search from script location up to root, or use current working directory
script_path = Path(__file__).resolve()
# Go up 4 levels: deplant_planted_error -> paper_manipulation -> scripts -> experiments (where .env is)
experiments_path = script_path.parent.parent.parent.parent
env_path = experiments_path / '.env'

# If .env not found at expected location, try current working directory
if not env_path.exists():
    env_path = Path.cwd() / '.env'
    if not env_path.exists():
        # Try going up from current directory
        current = Path.cwd()
        for _ in range(5):
            test_env = current / '.env'
            if test_env.exists():
                env_path = test_env
                break
            current = current.parent

# Load .env file
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Loaded .env from: {env_path}")
else:
    # Try loading without explicit path (searches from current directory up)
    load_dotenv()
    print(f"⚠️ .env file not found at expected locations, trying default search")

# Load multiple Gemini API keys
GEMINI_API_KEYS = {}
for i in range(1, 10):  # Support up to 9 keys
    key = os.getenv(f'GEMINI_API_KEY_{i}')
    if key:
        GEMINI_API_KEYS[str(i)] = key

if not GEMINI_API_KEYS:
    raise ValueError(
        f"No Gemini API keys found in environment variables (GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.).\n"
        f"Please ensure your .env file is in the experiments/ directory and contains GEMINI_API_KEY_1, etc.\n"
        f"Searched for .env at: {experiments_path / '.env'}, {Path.cwd() / '.env'}"
    )

print(f"✅ Loaded {len(GEMINI_API_KEYS)} Gemini API keys: {list(GEMINI_API_KEYS.keys())}")

# Default model - can be overridden via command line argument
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash-lite"

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

# Gemini model TPM limits (Tokens Per Minute)
GEMINI_MODEL_TPM_LIMITS = {
    "gemini-2.0-flash-lite": 1000000,
    "gemini-2.0-flash-exp": 1000000,
    "gemini-2.0-flash-preview-image-generation": 1000000,
    "gemini-2.0-flash": 1000000,
    "gemini-2.5-flash-lite": 1000000,
    "gemini-2.5-flash-tts": 1000000,
    "gemini-2.5-flash": 1000000,
    "gemini-2.5-pro": 1000000,
    "gemma-3-27b-it": 1000000,
}

def get_request_delay_for_model(model_name: str) -> float:
    """Calculate request delay in seconds based on model's RPM limit."""
    rpm_limit = GEMINI_MODEL_RPM_LIMITS.get(model_name, 30)
    return 60.0 / rpm_limit

# Global variable to store the model name (set in main())
GEMINI_MODEL = DEFAULT_GEMINI_MODEL

# Global variable to store the request delay (set in main())
GEMINI_REQUEST_DELAY = get_request_delay_for_model(DEFAULT_GEMINI_MODEL)

# Rate limiting tracking
key_request_times: Dict[str, deque] = {}
key_lock = threading.Lock()

# Track dynamic delays per key (increases when 429 errors occur)
key_dynamic_delays: Dict[str, float] = {}
key_delay_lock = threading.Lock()

# Track last 429 error time per key (for cooldown period)
key_last_429_time: Dict[str, float] = {}
key_429_lock = threading.Lock()

# TPM (Tokens Per Minute) tracking
key_token_usage: Dict[str, deque] = {}
tpm_lock = threading.Lock()

# --- Pydantic Models for Fixing Errors ---

class FixedSection(BaseModel):
    fixed_content: str = Field(..., description="The corrected content for the section that fixes the planted error. This should maintain the same structure and style as the original paper, but correct the flaw.")
    explanation: str = Field(..., description="A brief explanation of what was fixed and how the error was corrected.")

class FixErrorResponse(BaseModel):
    fixed_section: FixedSection = Field(..., description="The fixed section with explanation.")

class MultipleFixesResponse(BaseModel):
    fixed_sections: List[FixedSection] = Field(..., description="List of fixed sections, one for each modification in the same order as provided.")

# --- Helper Functions ---

def get_api_key_for_task(task_idx: int) -> Tuple[str, str]:
    """Get API key for a task using round-robin assignment."""
    key_names = list(GEMINI_API_KEYS.keys())
    key_name = key_names[task_idx % len(key_names)]
    return key_name, GEMINI_API_KEYS[key_name]

def estimate_tokens(text: str) -> int:
    """Estimate token count from text. Rough approximation: ~4 characters per token."""
    return len(text) // 4

def wait_for_rate_limit(key_name: str, request_delay: float = None, estimated_tokens: int = 0, tpm_limit: int = 1000000, rpm_limit: int = 30):
    """Wait if necessary to respect rate limits (both RPM and TPM). Uses sliding window for accurate tracking."""
    if request_delay is None:
        request_delay = GEMINI_REQUEST_DELAY
    
    current_time = time.time()
    
    # Check if we need to wait due to recent 429 error (cooldown period)
    with key_429_lock:
        if key_name in key_last_429_time:
            time_since_429 = current_time - key_last_429_time[key_name]
            cooldown_period = 60.0
            if time_since_429 < cooldown_period:
                wait_time = cooldown_period - time_since_429
                tqdm.write(f"⏳ Cooldown period after 429 error (key {key_name}), waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                current_time = time.time()
    
    # Get dynamic delay multiplier for this key
    with key_delay_lock:
        delay_multiplier = key_dynamic_delays.get(key_name, 1.0)
    
    # Apply dynamic delay multiplier with additional safety buffer
    effective_delay = request_delay * 1.5 * delay_multiplier
    
    # RPM limiting using sliding window (last 60 seconds)
    with key_lock:
        if key_name not in key_request_times:
            key_request_times[key_name] = deque()
        
        # Clean old requests (older than 60 seconds)
        window_start = current_time - 60.0
        while key_request_times[key_name] and key_request_times[key_name][0] < window_start:
            key_request_times[key_name].popleft()
        
        # Check if we can make a request now
        recent_requests = len(key_request_times[key_name])
        
        if recent_requests >= rpm_limit:
            if key_request_times[key_name]:
                oldest_time = key_request_times[key_name][0]
                wait_time = 60.0 - (current_time - oldest_time) + 2.0
                if wait_time > 0:
                    tqdm.write(f"⏳ RPM limit reached (key {key_name}), waiting {wait_time:.1f}s ({recent_requests}/{rpm_limit} requests in last 60s)")
                    time.sleep(wait_time)
                    current_time = time.time()
                    
                    # Clean again after waiting
                    window_start = current_time - 60.0
                    while key_request_times[key_name] and key_request_times[key_name][0] < window_start:
                        key_request_times[key_name].popleft()
        else:
            # Check minimum delay between requests
            if key_request_times[key_name]:
                last_request_time = key_request_times[key_name][-1]
                elapsed = current_time - last_request_time
                if elapsed < effective_delay:
                    sleep_time = effective_delay - elapsed
                    time.sleep(sleep_time)
                    current_time = time.time()
        
        # Record this request
        key_request_times[key_name].append(current_time)
    
    # TPM limiting - use sliding window
    if estimated_tokens > 0:
        with tpm_lock:
            if key_name not in key_token_usage:
                key_token_usage[key_name] = deque()
            
            # Clean old entries (older than 60 seconds)
            window_start = current_time - 60.0
            while key_token_usage[key_name] and key_token_usage[key_name][0][0] < window_start:
                key_token_usage[key_name].popleft()
            
            # Calculate current token usage in the window
            current_usage = sum(tokens for _, tokens in key_token_usage[key_name])
            
            # Check if adding this request would exceed TPM limit (use 90% threshold for safety)
            tpm_threshold = int(tpm_limit * 0.9)
            if current_usage + estimated_tokens > tpm_threshold:
                if key_token_usage[key_name]:
                    oldest_time = key_token_usage[key_name][0][0]
                    wait_time = 60.0 - (current_time - oldest_time) + 2.0
                    if wait_time > 0:
                        tqdm.write(f"⏳ TPM limit approaching (key {key_name}), waiting {wait_time:.1f}s (current: {current_usage}/{tpm_limit} tokens, threshold: {tpm_threshold})")
                        time.sleep(wait_time)
                        current_time = time.time()
                        
                        # Clean again after waiting
                        window_start = current_time - 60.0
                        while key_token_usage[key_name] and key_token_usage[key_name][0][0] < window_start:
                            key_token_usage[key_name].popleft()
            
            # Record this request's token usage
            key_token_usage[key_name].append((current_time, estimated_tokens))

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

def find_heading_in_lines(lines: list, target_heading: str) -> int:
    """Find heading index in lines using multiple matching strategies. Returns -1 if not found."""
    if not target_heading.strip():
        return -1
    
    # Normalize the target heading
    target_heading_clean = target_heading.strip()
    target_heading_clean = re.sub(r'^[#*]+\s*', '', target_heading_clean).strip()
    
    match_index = -1
    
    # Strategy 1: Exact match (with or without markdown prefix)
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if line_stripped == target_heading or line_stripped == target_heading_clean:
            match_index = i
            break
        line_clean = re.sub(r'^[#*]+\s*', '', line_stripped).strip()
        if line_clean == target_heading_clean or line_clean == target_heading.strip():
            match_index = i
            break
    
    # Strategy 2: Match after stripping whitespace and markdown
    if match_index == -1:
        for i, line in enumerate(lines):
            line_clean = line.strip().strip('#* \t')
            target_clean = target_heading_clean.strip('#* \t')
            if line_clean == target_clean and line_clean:
                match_index = i
                break
    
    # Strategy 3: Aggressive cleaning and partial match
    if match_index == -1:
        cleaned_target = clean_heading_text_aggressively(target_heading_clean)
        for i, line in enumerate(lines):
            cleaned_line = clean_heading_text_aggressively(line)
            if cleaned_line and cleaned_target:
                if cleaned_line.lower() == cleaned_target.lower():
                    match_index = i
                    break
                if cleaned_line.lower().startswith(cleaned_target.lower()) or \
                   cleaned_target.lower().startswith(cleaned_line.lower()):
                    match_index = i
                    break
    
    # Strategy 4: Match text content (ignore markdown formatting)
    if match_index == -1:
        target_text = re.sub(r'[#*`]', '', target_heading_clean).strip()
        for i, line in enumerate(lines):
            line_text = re.sub(r'[#*`]', '', line).strip()
            if line_text and target_text:
                if line_text.lower() == target_text.lower() or \
                   line_text.lower().startswith(target_text.lower()) or \
                   target_text.lower().startswith(line_text.lower()):
                    match_index = i
                    break
    
    return match_index

def normalize_text_for_matching(text: str) -> str:
    """Normalize text for better matching by removing HTML entities, extra whitespace, etc."""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = ' '.join(text.split())
    return text.lower().strip()

def find_marker_in_text(text: str, marker: str, context_lines: int = 3) -> Optional[int]:
    """Find a marker text in the full text, returning the line index where it starts."""
    lines = text.split('\n')
    marker_normalized = normalize_text_for_matching(marker)
    marker_words = [w for w in marker_normalized.split() if len(w) > 2]
    
    if not marker_words:
        return None
    
    # Strategy 1: Exact match (case-insensitive, normalized)
    for i, line in enumerate(lines):
        line_normalized = normalize_text_for_matching(line)
        if marker_normalized in line_normalized:
            return i
    
    # Strategy 2: Check if all significant words appear in a line
    for i, line in enumerate(lines):
        line_normalized = normalize_text_for_matching(line)
        if all(word in line_normalized for word in marker_words):
            return i
    
    # Strategy 3: Fuzzy match - look for lines containing most marker words
    best_match_idx = None
    best_match_score = 0
    min_words_needed = max(2, len(marker_words) - 1)
    
    for i, line in enumerate(lines):
        line_normalized = normalize_text_for_matching(line)
        matching_words = sum(1 for word in marker_words if word in line_normalized)
        if matching_words >= min_words_needed and matching_words > best_match_score:
            best_match_score = matching_words
            best_match_idx = i
    
    if best_match_idx is not None:
        return best_match_idx
    
    # Strategy 4: Try with context (check multiple consecutive lines)
    for i in range(len(lines) - context_lines + 1):
        context = ' '.join(lines[i:i+context_lines])
        context_normalized = normalize_text_for_matching(context)
        if marker_normalized in context_normalized:
            return i
        matching_words = sum(1 for word in marker_words if word in context_normalized)
        if matching_words >= min_words_needed:
            return i
    
    return None

def extract_heading_from_content(content: str) -> Optional[str]:
    """Extract the heading line from content (first line that looks like a heading)."""
    if not content or not content.strip():
        return None
    
    lines = content.split('\n')
    for i, line in enumerate(lines[:10]):  # Check first 10 lines only
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # Strategy 1: Markdown heading (starts with #)
        if line_stripped.startswith('#'):
            return line_stripped
        
        # Strategy 2: Bold heading (starts and ends with **)
        if line_stripped.startswith('**') and line_stripped.endswith('**') and len(line_stripped) > 4:
            return line_stripped
        
        # Strategy 3: First non-empty line that's short and looks like a heading
        if i < 3 and len(line_stripped) < 150 and not line_stripped.startswith(' ') and \
           not line_stripped.startswith('\t') and not line_stripped.startswith('*') and \
           not line_stripped.startswith('-') and not line_stripped.startswith('1.'):
            if not (line_stripped.endswith('.') and len(line_stripped) > 50):
                return line_stripped
    
    return None

def clean_json_schema_for_gemini(schema: dict) -> dict:
    """Remove unsupported fields and inline definitions to make schema compatible with Gemini."""
    import copy
    
    UNSUPPORTED_FIELDS = {'$defs', 'title', 'description', '$schema', 'definitions'}
    
    schema = copy.deepcopy(schema)
    defs = schema.pop('$defs', {})
    
    def clean_and_inline_refs(obj):
        """Recursively remove unsupported fields and inline $ref references."""
        if isinstance(obj, dict):
            cleaned = {k: v for k, v in obj.items() if k not in UNSUPPORTED_FIELDS}
            
            if '$ref' in obj:
                ref_path = obj['$ref']
                if ref_path.startswith('#/$defs/'):
                    def_name = ref_path.replace('#/$defs/', '')
                    if def_name in defs:
                        inlined = copy.deepcopy(defs[def_name])
                        return clean_and_inline_refs(inlined)
                return cleaned
            
            return {k: clean_and_inline_refs(v) for k, v in cleaned.items()}
        elif isinstance(obj, list):
            return [clean_and_inline_refs(item) for item in obj]
        return obj
    
    return clean_and_inline_refs(schema)

def call_gemini_with_retries(api_key: str, key_name: str, prompt: str, response_model: type, max_retries: int = 5, request_delay: float = None, tpm_limit: int = 1000000, rpm_limit: int = 30) -> Optional[BaseModel]:
    """Call Gemini API with retries and structured output parsing. Handles 429 errors with exponential backoff."""
    if request_delay is None:
        request_delay = GEMINI_REQUEST_DELAY
    
    # Estimate tokens for this request
    estimated_input_tokens = estimate_tokens(prompt)
    estimated_output_tokens = 2000
    estimated_total_tokens = estimated_input_tokens + estimated_output_tokens
    
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
            # Wait for rate limit before each attempt
            wait_for_rate_limit(key_name, request_delay, estimated_total_tokens, tpm_limit, rpm_limit)
            
            response = model.generate_content(prompt)
            json_text = response.text.strip()
            
            # Clean JSON if needed
            json_text = re.sub(r'```json\s*', '', json_text)
            json_text = re.sub(r'```\s*$', '', json_text)
            json_text = json_text.strip()
            
            result = response_model.model_validate_json(json_text)
            
            # Success! Reset dynamic delay multiplier gradually
            with key_delay_lock:
                if key_name in key_dynamic_delays and key_dynamic_delays[key_name] > 1.0:
                    key_dynamic_delays[key_name] = max(1.0, key_dynamic_delays[key_name] * 0.98)
            
            return result
        except Exception as e:
            error_str = str(e)
            error_type = type(e).__name__
            
            # Check if it's a 429 Resource Exhausted error
            is_rate_limit_error = (
                "429" in error_str or 
                "ResourceExhausted" in error_type or
                "Resource has been exhausted" in error_str or
                "quota" in error_str.lower()
            )
            
            if is_rate_limit_error:
                # Record 429 error time for cooldown period
                with key_429_lock:
                    key_last_429_time[key_name] = time.time()
                
                # Increase dynamic delay multiplier for this key
                with key_delay_lock:
                    if key_name not in key_dynamic_delays:
                        key_dynamic_delays[key_name] = 1.0
                    key_dynamic_delays[key_name] = min(5.0, key_dynamic_delays[key_name] + 1.0)
                    current_multiplier = key_dynamic_delays[key_name]
                
                # Exponential backoff for rate limit errors
                backoff_time = min(10 * (2 ** attempt), 120)
                if attempt < max_retries - 1:
                    tqdm.write(f"⚠️ Rate limit hit (key {key_name}), waiting {backoff_time}s before retry {attempt + 1}/{max_retries} (delay multiplier: {current_multiplier:.1f}x)")
                    time.sleep(backoff_time)
                    continue
            else:
                # For other errors, use shorter backoff
                if attempt < max_retries - 1:
                    backoff_time = 2 ** attempt
                    time.sleep(backoff_time)
                    continue
            
            # Last attempt failed
            tqdm.write(f"❌ Error calling Gemini API (key {key_name}): {error_type} - {error_str[:200]}")
            if attempt == max_retries - 1:
                import traceback
                traceback.print_exc()
            return None
    
    return None

# --- Main Processing Functions ---

def extract_markers_from_target_heading(flawed_paper: str, target_heading: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract start_marker and end_marker from flawed_paper using target_heading.
    Returns (start_marker, end_marker) or (None, None) if not found.
    """
    if not target_heading:
        return None, None
    
    lines = flawed_paper.split('\n')
    heading_index = find_heading_in_lines(lines, target_heading)
    
    if heading_index == -1:
        return None, None
    
    # Find the end of this section (next heading or end of document)
    start_line = heading_index
    end_line = len(lines)
    
    for i in range(start_line + 1, len(lines)):
        line = lines[i].strip()
        # Check if this is a new heading
        if line.startswith('#') or (line.startswith('**') and line.endswith('**') and len(line) > 4):
            end_line = i
            break
    
    # Extract the section from flawed paper
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
    
    return None, None

def find_section_by_heading(flawed_paper: str, target_heading: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Find a section by its heading. Returns (start_line, end_line) or (None, None) if not found.
    Uses multiple matching strategies similar to try_apply_modifications.
    """
    if not target_heading or not target_heading.strip():
        return None, None
    
    lines = flawed_paper.split('\n')
    target_heading_clean = target_heading.strip()
    
    match_index = -1
    
    # Strategy 1: Exact match
    for i, line in enumerate(lines):
        if line.strip() == target_heading_clean:
            match_index = i
            break
    
    # Strategy 2: Match after stripping whitespace and markdown
    if match_index == -1:
        semi_cleaned_target = target_heading_clean.strip('#* \t')
        for i, line in enumerate(lines):
            semi_cleaned_line = line.strip().strip('#* \t')
            if semi_cleaned_line == semi_cleaned_target and semi_cleaned_line:
                match_index = i
                break
    
    # Strategy 3: Aggressive cleaning
    if match_index == -1:
        aggressively_cleaned_target = clean_heading_text_aggressively(target_heading_clean)
        for i, line in enumerate(lines):
            aggressively_cleaned_line = clean_heading_text_aggressively(line)
            if aggressively_cleaned_line and aggressively_cleaned_target:
                if aggressively_cleaned_line.lower().startswith(aggressively_cleaned_target.lower()):
                    match_index = i
                    break
    
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

def fix_error_at_location(
    flawed_paper: str,
    start_marker: Optional[str],
    end_marker: Optional[str],
    target_heading: str,
    flaw_description: str,
    original_reasoning: str,
    api_key: str,
    key_name: str,
    model_name: str,
    request_delay: float = None,
    tpm_limit: int = 1000000,
    rpm_limit: int = 30
) -> Optional[FixedSection]:
    """
    Use Gemini to fix an error at a specific location in the flawed paper.
    Can work with either start_marker/end_marker OR just target_heading.
    """
    lines = flawed_paper.split('\n')
    start_line = None
    end_line = None
    
    # Strategy 1: Use markers if available
    if start_marker and end_marker:
        start_line = find_marker_in_text(flawed_paper, start_marker)
        if start_line is not None:
            lines_after_start = lines[start_line+1:]
            text_after_start = '\n'.join(lines_after_start)
            end_line_relative = find_marker_in_text(text_after_start, end_marker)
            if end_line_relative is not None:
                end_line = start_line + 1 + end_line_relative
    
    # Strategy 2: Use target_heading to find section (fallback or primary method)
    if start_line is None or end_line is None:
        if target_heading:
            found_start, found_end = find_section_by_heading(flawed_paper, target_heading)
            if found_start is not None and found_end is not None:
                start_line = found_start
                end_line = found_end
            else:
                tqdm.write(f"  ⚠️ Could not find section with heading: {target_heading[:50]}...")
                return None
        else:
            tqdm.write(f"  ⚠️ No markers or target_heading provided")
            return None
    
    # Extract the flawed section
    flawed_section = '\n'.join(lines[start_line:end_line])
    
    # Get context around the section (50 lines before and after)
    context_before = '\n'.join(lines[max(0, start_line-50):start_line])
    context_after = '\n'.join(lines[end_line:min(len(lines), end_line+51)])
    
    # Create prompt for fixing the error
    prompt = f"""You are revising a research paper section to address an identified issue.

**Issue Description:**
{flaw_description}

**Original Reasoning:**
{original_reasoning}

**Target Heading:**
{target_heading}

**Context Before the Section:**
{context_before[-1000:]}

**Section to Revise:**
{flawed_section}

**Context After the Section:**
{context_after[:1000]}

**Revision Guidelines:**

1. If the author(s) promised to provide additional experiments, provide concrete experimental results and quantitative evidence as if they have been conducted. Include specific numerical metrics (accuracy, F1 scores, loss values, p-values, confidence intervals) rather than statements about future work or planned evaluations.

2. When comparisons or evaluations are mentioned, include detailed quantitative results with specific numbers comparing against baseline methods.

3. Add experimental details such as ablation study results, hyperparameter analyses, and performance metrics to substantiate claims.

4. Maintain the paper's writing style, structure, and formatting from outside of the modified section. Preserve all existing figures and citations. The section must start with the exact heading line: {target_heading}

Return the revised section that addresses the issue while maintaining high research paper standards.
"""
    
    try:
        response = call_gemini_with_retries(
            api_key=api_key,
            key_name=key_name,
            prompt=prompt,
            response_model=FixErrorResponse,
            max_retries=3,
            request_delay=request_delay,
            tpm_limit=tpm_limit,
            rpm_limit=rpm_limit
        )
        
        if response and response.fixed_section:
            return response.fixed_section
    except Exception as e:
        tqdm.write(f"  ⚠️ Error fixing section: {e}")
    
    return None

def fix_all_errors_at_once(
    flawed_paper: str,
    modifications: List[Dict],
    flaw_description: str,
    api_key: str,
    key_name: str,
    model_name: str,
    request_delay: float = None,
    tpm_limit: int = 1000000,
    rpm_limit: int = 30
) -> Optional[List[FixedSection]]:
    """
    Fix all modifications at once in a single LLM call for better coherence.
    Returns a list of FixedSection objects in the same order as modifications.
    """
    lines = flawed_paper.split('\n')
    
    # Collect all sections that need to be fixed
    sections_to_fix = []
    for mod_idx, mod in enumerate(modifications):
        start_marker = mod.get('start_marker', '') or None
        end_marker = mod.get('end_marker', '') or None
        target_heading = mod.get('target_heading', '')
        original_reasoning = mod.get('reasoning', '')
        
        # Extract target_heading if not available
        if not target_heading:
            new_content = mod.get('new_content', '')
            if new_content:
                extracted_heading = extract_heading_from_content(new_content)
                if extracted_heading:
                    target_heading = extracted_heading
        
        if not target_heading:
            tqdm.write(f"  ⚠️ Modification {mod_idx} missing target_heading, skipping")
            continue
        
        # Find section location
        start_line = None
        end_line = None
        
        # Strategy 1: Use markers if available
        if start_marker and end_marker:
            start_line = find_marker_in_text(flawed_paper, start_marker)
            if start_line is not None:
                lines_after_start = lines[start_line+1:]
                text_after_start = '\n'.join(lines_after_start)
                end_line_relative = find_marker_in_text(text_after_start, end_marker)
                if end_line_relative is not None:
                    end_line = start_line + 1 + end_line_relative
        
        # Strategy 2: Use target_heading
        if start_line is None or end_line is None:
            found_start, found_end = find_section_by_heading(flawed_paper, target_heading)
            if found_start is not None and found_end is not None:
                start_line = found_start
                end_line = found_end
            else:
                tqdm.write(f"  ⚠️ Could not find section with heading: {target_heading[:50]}...")
                continue
        
        # Extract the flawed section
        flawed_section = '\n'.join(lines[start_line:end_line])
        
        sections_to_fix.append({
            'mod_idx': mod_idx,
            'target_heading': target_heading,
            'flawed_section': flawed_section,
            'start_line': start_line,
            'end_line': end_line,
            'original_reasoning': original_reasoning
        })
    
    if not sections_to_fix:
        tqdm.write(f"  ⚠️ No valid sections found to fix")
        return None
    
    # Build comprehensive prompt with all sections
    sections_prompt = ""
    for i, section_info in enumerate(sections_to_fix):
        # Get context around each section
        start_line = section_info['start_line']
        end_line = section_info['end_line']
        context_before = '\n'.join(lines[max(0, start_line-30):start_line])
        context_after = '\n'.join(lines[end_line:min(len(lines), end_line+31)])
        
        sections_prompt += f"""
**Modification {i+1}:**
- Target Heading: {section_info['target_heading']}
- Original Reasoning: {section_info['original_reasoning']}
- Context Before:
{context_before[-800:]}
- Flawed Section (needs to be fixed):
{section_info['flawed_section']}
- Context After:
{context_after[:800]}
---
"""
    
    # Create comprehensive prompt
    prompt = f"""You are revising a research paper to address identified issues in specific sections.

**Overall Issue Description:**
{flaw_description}

**Sections to Revise:**
{sections_prompt}

**Revision Guidelines:**

1. Provide concrete experimental results and quantitative evidence. Include specific numerical metrics (accuracy, F1 scores, loss values, p-values, confidence intervals) rather than statements about future work or planned evaluations.

2. When comparisons or evaluations are mentioned, include detailed quantitative results with specific numbers comparing against baseline methods.

3. Add experimental details such as ablation study results, hyperparameter analyses, and performance metrics to substantiate claims.

4. Maintain the paper's writing style, structure, and formatting. Preserve all existing tables, figures, and citations. Each section must start with its exact heading line.

5. Ensure revisions are coherent across sections and consistent with the overall paper context.

Return the revised sections in the same order as provided.
"""
    
    try:
        response = call_gemini_with_retries(
            api_key=api_key,
            key_name=key_name,
            prompt=prompt,
            response_model=MultipleFixesResponse,
            max_retries=3,
            request_delay=request_delay,
            tpm_limit=tpm_limit,
            rpm_limit=rpm_limit
        )
        
        if response and response.fixed_sections:
            # Verify we got the right number of fixes
            if len(response.fixed_sections) != len(sections_to_fix):
                tqdm.write(f"  ⚠️ Expected {len(sections_to_fix)} fixes, got {len(response.fixed_sections)}")
                # Pad or truncate as needed
                if len(response.fixed_sections) < len(sections_to_fix):
                    # Pad with None
                    while len(response.fixed_sections) < len(sections_to_fix):
                        response.fixed_sections.append(None)
                else:
                    # Truncate
                    response.fixed_sections = response.fixed_sections[:len(sections_to_fix)]
            
            return response.fixed_sections
    except Exception as e:
        tqdm.write(f"  ⚠️ Error fixing sections: {e}")
        import traceback
        traceback.print_exc()
    
    return None

def apply_fix_to_paper(
    flawed_paper: str,
    start_marker: Optional[str],
    end_marker: Optional[str],
    target_heading: str,
    fixed_content: str
) -> Tuple[str, bool, Optional[str]]:
    """
    Apply a fix to the flawed paper by replacing the section.
    Can work with either start_marker/end_marker OR just target_heading.
    Returns (fixed_paper, success, error_message).
    """
    lines = flawed_paper.split('\n')
    start_line = None
    end_line = None
    
    # Strategy 1: Use markers if available
    if start_marker and end_marker:
        start_line = find_marker_in_text(flawed_paper, start_marker)
        if start_line is not None:
            lines_after_start = lines[start_line+1:]
            text_after_start = '\n'.join(lines_after_start)
            end_line_relative = find_marker_in_text(text_after_start, end_marker)
            if end_line_relative is not None:
                end_line = start_line + 1 + end_line_relative
    
    # Strategy 2: Use target_heading to find section (fallback or primary method)
    if start_line is None or end_line is None:
        if target_heading:
            found_start, found_end = find_section_by_heading(flawed_paper, target_heading)
            if found_start is not None and found_end is not None:
                start_line = found_start
                end_line = found_end
            else:
                return flawed_paper, False, f"Could not find section with heading: {target_heading[:50]}..."
        else:
            return flawed_paper, False, "No markers or target_heading provided"
    
    # Apply fix
    pre_section_lines = lines[:start_line]
    post_section_lines = lines[end_line:]  # Include end_line in replacement (it's the next heading)
    fixed_content_lines = fixed_content.split('\n')
    
    # Reconstruct
    fixed_lines = pre_section_lines + fixed_content_lines + post_section_lines
    fixed_paper = '\n'.join(fixed_lines)
    
    return fixed_paper, True, None

def process_paper_flaw(
    category: str,
    paper_folder: str,
    flaw_id: str,
    flaw_description: str,
    modifications: List[Dict],
    flawed_paper_path: Path,
    output_dir: Path,
    task_idx: int,
    model_name: str,
    request_delay: float = None,
    tpm_limit: int = 1000000,
    rpm_limit: int = 30
) -> Optional[Dict]:
    """
    Process a single flaw for a paper, fixing all modifications.
    Returns None if any error occurs, so --skip_existing will retry it.
    """
    try:
        # Read flawed paper
        try:
            with open(flawed_paper_path, 'r', encoding='utf-8') as f:
                flawed_paper = f.read()
        except Exception as e:
            tqdm.write(f"  ❌ Error reading flawed paper {flawed_paper_path}: {e}")
            return None
        
        # Get API key for this task
        key_name, api_key = get_api_key_for_task(task_idx)
    
        # Extract all sections and their metadata before fixing
        sections_metadata = []
        for mod_idx, mod in enumerate(modifications):
            start_marker = mod.get('start_marker', '') or None
            end_marker = mod.get('end_marker', '') or None
            target_heading = mod.get('target_heading', '')
            original_reasoning = mod.get('reasoning', '')
            
            # Extract target_heading if not available
            if not target_heading:
                new_content = mod.get('new_content', '')
                if new_content:
                    extracted_heading = extract_heading_from_content(new_content)
                    if extracted_heading:
                        target_heading = extracted_heading
            
            # Extract planted_error_content from flawed paper
            planted_error_content = ""
            if target_heading:
                start_line, end_line = find_section_by_heading(flawed_paper, target_heading)
                if start_line is not None and end_line is not None:
                    lines = flawed_paper.split('\n')
                    section_lines = lines[start_line:end_line]
                    planted_error_content = '\n'.join(section_lines).strip()
            
            sections_metadata.append({
                'mod_idx': mod_idx,
                'start_marker': start_marker,
                'end_marker': end_marker,
                'target_heading': target_heading,
                'planted_error_content': planted_error_content
            })
        
        # Fix all errors at once
        tqdm.write(f"  🔧 Fixing all {len(modifications)} modification(s) at once for {paper_folder}/{flaw_id}...")
        fixed_sections = fix_all_errors_at_once(
            flawed_paper=flawed_paper,
            modifications=modifications,
            flaw_description=flaw_description,
            api_key=api_key,
            key_name=key_name,
            model_name=model_name,
            request_delay=request_delay,
            tpm_limit=tpm_limit,
            rpm_limit=rpm_limit
        )
        
        # Process results and apply fixes
        fixed_paper = flawed_paper
        fix_attempts = []
        
        if fixed_sections:
            # Map fixes to modifications - need to track which modifications were processed in fix_all_errors_at_once
            # The fix_all_errors_at_once function processes modifications in order and skips invalid ones
            # So we need to reconstruct the mapping
            mod_idx_to_fix_idx = {}
            fix_idx = 0
            for mod_idx, mod in enumerate(modifications):
                target_heading = mod.get('target_heading', '')
                if not target_heading:
                    new_content = mod.get('new_content', '')
                    if new_content:
                        extracted_heading = extract_heading_from_content(new_content)
                        if extracted_heading:
                            target_heading = extracted_heading
                
                if target_heading:
                    # Check if we can find this section (same check as in fix_all_errors_at_once)
                    start_line, end_line = find_section_by_heading(flawed_paper, target_heading)
                    if start_line is not None and end_line is not None:
                        mod_idx_to_fix_idx[mod_idx] = fix_idx
                        fix_idx += 1
            
            # Apply fixes in reverse order to preserve line numbers
            mod_indices_sorted = sorted(mod_idx_to_fix_idx.keys(), reverse=True)
            
            for mod_idx in mod_indices_sorted:
                fix_idx = mod_idx_to_fix_idx[mod_idx]
                metadata = sections_metadata[mod_idx]
                fixed_section = fixed_sections[fix_idx] if fix_idx < len(fixed_sections) else None
                
                if fixed_section and fixed_section.fixed_content:
                    # Apply the fix
                    fixed_paper, success, error_msg = apply_fix_to_paper(
                        flawed_paper=fixed_paper,
                        start_marker=metadata['start_marker'],
                        end_marker=metadata['end_marker'],
                        target_heading=metadata['target_heading'],
                        fixed_content=fixed_section.fixed_content
                    )
                    
                    # Extract deplanted_error_content from fixed paper
                    deplanted_error_content = ""
                    if success and metadata['target_heading']:
                        start_line, end_line = find_section_by_heading(fixed_paper, metadata['target_heading'])
                        if start_line is not None and end_line is not None:
                            lines = fixed_paper.split('\n')
                            section_lines = lines[start_line:end_line]
                            deplanted_error_content = '\n'.join(section_lines).strip()
                    
                    fix_attempts.append({
                        'modification_index': mod_idx,
                        'target_heading': metadata['target_heading'],
                        'success': success,
                        'error': error_msg or '',
                        'start_marker': metadata['start_marker'][:100] if metadata['start_marker'] else '',
                        'end_marker': metadata['end_marker'][:100] if metadata['end_marker'] else '',
                        'planted_error_content': metadata['planted_error_content'],
                        'deplanted_error_content': deplanted_error_content,
                        'fixed_content': fixed_section.fixed_content[:500] + '...' if len(fixed_section.fixed_content) > 500 else fixed_section.fixed_content,
                        'explanation': fixed_section.explanation
                    })
                    
                    if success:
                        tqdm.write(f"  ✅ Successfully fixed modification {mod_idx+1}")
                    else:
                        tqdm.write(f"  ❌ Failed to apply fix for modification {mod_idx+1}: {error_msg}")
                else:
                    fix_attempts.append({
                        'modification_index': mod_idx,
                        'target_heading': metadata['target_heading'],
                        'success': False,
                        'error': 'Failed to generate fix from LLM' if not fixed_section else 'Empty fix content',
                        'start_marker': metadata['start_marker'][:100] if metadata['start_marker'] else '',
                        'end_marker': metadata['end_marker'][:100] if metadata['end_marker'] else '',
                        'planted_error_content': metadata['planted_error_content'],
                        'deplanted_error_content': '',
                        'fixed_content': '',
                        'explanation': ''
                    })
                    tqdm.write(f"  ❌ Failed to generate fix for modification {mod_idx+1}")
            
            # Handle modifications that weren't processed (missing target_heading or section not found)
            for mod_idx, metadata in enumerate(sections_metadata):
                if mod_idx not in mod_idx_to_fix_idx:
                    fix_attempts.append({
                        'modification_index': mod_idx,
                        'target_heading': metadata['target_heading'],
                        'success': False,
                        'error': 'Missing target_heading or section not found',
                        'start_marker': metadata['start_marker'][:100] if metadata['start_marker'] else '',
                        'end_marker': metadata['end_marker'][:100] if metadata['end_marker'] else '',
                        'planted_error_content': metadata['planted_error_content'],
                        'deplanted_error_content': '',
                        'fixed_content': '',
                        'explanation': ''
                    })
        else:
            # No fixes generated - mark all as failed
            for mod_idx, metadata in enumerate(sections_metadata):
                fix_attempts.append({
                    'modification_index': mod_idx,
                    'target_heading': metadata['target_heading'],
                    'success': False,
                    'error': 'Failed to generate fixes from LLM',
                    'start_marker': metadata['start_marker'][:100] if metadata['start_marker'] else '',
                    'end_marker': metadata['end_marker'][:100] if metadata['end_marker'] else '',
                    'planted_error_content': metadata['planted_error_content'],
                    'deplanted_error_content': '',
                    'fixed_content': '',
                    'explanation': ''
                })
            tqdm.write(f"  ❌ Failed to generate fixes for all modifications")
        
        # Sort fix_attempts by modification_index for consistency
        fix_attempts.sort(key=lambda x: x['modification_index'])
        
        # Calculate success rate
        successful_fixes = sum(1 for attempt in fix_attempts if attempt['success'])
        total_modifications = len(fix_attempts)
        
        # Only write files if we have at least some successful fixes
        # If all failed, don't write anything so --skip_existing will retry it
        if successful_fixes == 0:
            tqdm.write(f"  ⚠️ All modifications failed for {paper_folder}/{flaw_id}, not writing output (will retry with --skip_existing)")
            return None
        
        # Save fixed paper only if we have successful fixes
        de_planted_error_dir = output_dir / category / 'de-planted_error' / paper_folder
        de_planted_error_dir.mkdir(parents=True, exist_ok=True)
        
        # Save in flawed_papers subdirectory to match structure
        flawed_papers_dir = de_planted_error_dir / 'flawed_papers'
        flawed_papers_dir.mkdir(parents=True, exist_ok=True)
        fixed_paper_path = flawed_papers_dir / f"{flaw_id}.md"
        
        with open(fixed_paper_path, 'w', encoding='utf-8') as f:
            f.write(fixed_paper)
        
        return {
            'category': category,
            'paper_folder': paper_folder,
            'flaw_id': flaw_id,
            'flaw_description': flaw_description,
            'total_modifications': total_modifications,
            'successful_fixes': successful_fixes,
            'fix_attempts': fix_attempts,
            'fixed_paper_path': str(fixed_paper_path),
            'success': successful_fixes == total_modifications and total_modifications > 0
        }
    except Exception as e:
        # Catch any unexpected errors during processing
        tqdm.write(f"  ❌ Unexpected error processing {paper_folder}/{flaw_id}: {e}")
        import traceback
        traceback.print_exc()
        # Don't write anything if there's an error
        return None

def is_already_fixed(
    data_dir: Path,
    conference: str,
    category: str,
    paper_folder: str,
    flaw_id: str,
    check_success: bool = False
) -> bool:
    """
    Check if a paper flaw has already been fixed.
    
    Args:
        data_dir: Base data directory
        conference: Conference name
        category: Category name
        paper_folder: Paper folder name
        flaw_id: Flaw ID
        check_success: If True, also check if the fix was successful (requires CSV summary)
    
    Returns:
        True if already fixed (and optionally successful), False otherwise
    """
    # Check if fixed paper exists
    fixed_paper_path = data_dir / conference / category / 'de-planted_error' / paper_folder / 'flawed_papers' / f"{flaw_id}.md"
    
    if not fixed_paper_path.exists():
        return False
    
    # If we don't need to check success, just return True if file exists
    if not check_success:
        return True
    
    # Check if fix was successful by looking at the summary CSV
    summary_csv_path = data_dir / conference / category / 'de-planted_error' / f'{category}_fix_summary.csv'
    
    if not summary_csv_path.exists():
        # File exists but no summary, assume it's fixed but we don't know if successful
        return True
    
    try:
        df = pd.read_csv(summary_csv_path)
        # Find the row for this flaw
        flaw_rows = df[df['flaw_id'] == flaw_id]
        
        if flaw_rows.empty:
            # File exists but not in summary, assume it's fixed
            return True
        
        # Check if all modifications were successful
        for _, row in flaw_rows.iterrows():
            modifications_json = row.get('llm_generated_modifications', '')
            if modifications_json:
                try:
                    if isinstance(modifications_json, str):
                        modifications = json.loads(modifications_json)
                    else:
                        modifications = modifications_json
                    
                    # Check if all modifications were successful
                    for mod in modifications:
                        if not mod.get('success', False):
                            return False  # At least one modification failed
                except (json.JSONDecodeError, TypeError):
                    # Can't parse, assume it's fixed but we don't know if successful
                    return True
        
        return True  # All modifications were successful
    except Exception:
        # Error reading CSV, assume it's fixed
        return True

def process_category(
    data_dir: Path,
    conference: str,
    category: str,
    model_name: str,
    request_delay: float,
    tpm_limit: int,
    rpm_limit: int,
    task_counter: List[int],
    skip_existing: bool = False,
    check_success: bool = False
) -> List[Dict]:
    """
    Process all papers in a category.
    """
    category_dir = data_dir / conference / category
    planted_error_dir = category_dir / 'planted_error'
    
    if not planted_error_dir.exists():
        tqdm.write(f"⚠️ Planted error directory not found: {planted_error_dir}")
        return []
    
    results = []
    
    # Find all CSV files
    csv_files = list(planted_error_dir.rglob("*_modifications_summary.csv"))
    
    for csv_path in csv_files:
        paper_folder = csv_path.parent.name
        
        # Read CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            tqdm.write(f"  ⚠️ Error reading CSV {csv_path}: {e}")
            continue
        
        # Process each flaw in the CSV
        for _, row in df.iterrows():
            flaw_id = row.get('flaw_id', '')
            flaw_description = row.get('flaw_description', '')
            modifications_json = row.get('llm_generated_modifications', '')
            
            if not flaw_id or not modifications_json:
                continue
            
            try:
                # Parse modifications
                if isinstance(modifications_json, str):
                    modifications = json.loads(modifications_json)
                else:
                    modifications = modifications_json
                
                # Find flawed paper
                flawed_paper_path = planted_error_dir / paper_folder / 'flawed_papers' / f"{flaw_id}.md"
                if not flawed_paper_path.exists():
                    # Try alternative location
                    flawed_paper_path = planted_error_dir / paper_folder / f"{flaw_id}.md"
                
                if not flawed_paper_path.exists():
                    tqdm.write(f"  ⚠️ Flawed paper not found: {flaw_id}.md in {paper_folder}")
                    continue
                
                # Check if already fixed (if skip_existing is enabled)
                if skip_existing:
                    if is_already_fixed(data_dir, conference, category, paper_folder, flaw_id, check_success):
                        tqdm.write(f"  ⏭️  Skipping {paper_folder}/{flaw_id} (already fixed)")
                        continue
                
                # Process this flaw
                task_idx = task_counter[0]
                task_counter[0] += 1
                
                result = process_paper_flaw(
                    category=category,
                    paper_folder=paper_folder,
                    flaw_id=flaw_id,
                    flaw_description=flaw_description,
                    modifications=modifications,
                    flawed_paper_path=flawed_paper_path,
                    output_dir=data_dir / conference,
                    task_idx=task_idx,
                    model_name=model_name,
                    request_delay=request_delay,
                    tpm_limit=tpm_limit,
                    rpm_limit=rpm_limit
                )
                
                if result:
                    results.append(result)
            
            except Exception as e:
                tqdm.write(f"  ❌ Error processing {paper_folder}/{flaw_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    return results

def main():
    global GEMINI_MODEL, GEMINI_REQUEST_DELAY
    
    parser = argparse.ArgumentParser(
        description="Fix planted errors in papers using Gemini API."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the with_appendix data structure"
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
        "--skip_existing",
        action="store_true",
        help="Skip papers that have already been fixed (checks if fixed paper exists)"
    )
    parser.add_argument(
        "--skip_only_successful",
        action="store_true",
        help="When used with --skip_existing, only skip if the fix was successful (requires checking CSV summary)"
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
    
    # Set max_workers
    if args.max_workers is not None:
        max_workers = args.max_workers
    else:
        max_workers = len(GEMINI_API_KEYS)
    
    print(f"\n✅ Using {max_workers} worker threads (one per API key)")
    print(f"✅ Total theoretical throughput: {len(GEMINI_API_KEYS) * model_rpm} RPM")
    print()
    
    # Process all categories
    all_results = []
    task_counter = [0]
    
    for category in categories:
        print(f"\n{'='*80}")
        print(f"Processing category: {category}")
        print(f"{'='*80}")
        
        category_results = process_category(
            data_dir=data_dir,
            conference=args.conference,
            category=category,
            model_name=GEMINI_MODEL,
            request_delay=GEMINI_REQUEST_DELAY,
            tpm_limit=model_tpm,
            rpm_limit=model_rpm,
            task_counter=task_counter,
            skip_existing=args.skip_existing,
            check_success=args.skip_only_successful
        )
        
        all_results.extend(category_results)
        
        # Save separate CSV file for each paper folder (matching input structure)
        if category_results:
            # Group by paper_folder first
            papers_dict = {}
            for result in category_results:
                paper_folder = result['paper_folder']
                if paper_folder not in papers_dict:
                    papers_dict[paper_folder] = []
                papers_dict[paper_folder].append(result)
            
            # Create a CSV file for each paper folder
            for paper_folder, paper_results in papers_dict.items():
                # Group by flaw_id for this paper
                flaw_groups = {}
                for result in paper_results:
                    flaw_id = result['flaw_id']
                    if flaw_id not in flaw_groups:
                        flaw_groups[flaw_id] = {
                            'flaw_id': flaw_id,
                            'flaw_description': result['flaw_description'],
                            'fix_attempts': []
                        }
                    flaw_groups[flaw_id]['fix_attempts'].extend(result['fix_attempts'])
                
                # Create summary rows matching modifications_summary.csv format
                paper_summary = []
                for flaw_id, flaw_data in flaw_groups.items():
                    # Build llm_generated_modifications JSON
                    modifications_json = []
                    for attempt in flaw_data['fix_attempts']:
                        mod_json = {
                            'target_heading': attempt['target_heading'],
                            'planted_error_content': attempt.get('planted_error_content', ''),
                            'deplanted_error_content': attempt.get('deplanted_error_content', ''),
                            'explanation': attempt.get('explanation', ''),
                            'success': attempt['success'],
                            'error': attempt.get('error', '')
                        }
                        # Add optional fields if they exist
                        if attempt.get('start_marker'):
                            mod_json['start_marker'] = attempt['start_marker']
                        if attempt.get('end_marker'):
                            mod_json['end_marker'] = attempt['end_marker']
                        modifications_json.append(mod_json)
                    
                    paper_summary.append({
                        'flaw_id': flaw_id,
                        'flaw_description': flaw_data['flaw_description'],
                        'num_modifications': len(modifications_json),
                        'llm_generated_modifications': json.dumps(modifications_json, ensure_ascii=False, indent=2)
                    })
                
                # Save CSV in the paper folder
                paper_dir = data_dir / args.conference / category / 'de-planted_error' / paper_folder
                paper_dir.mkdir(parents=True, exist_ok=True)
                
                # Extract base name from paper_folder (e.g., "3s8V8QP9XV_2303_03358" -> "3s8V8QP9XV")
                # Try to match the naming pattern of input CSV files
                base_name = paper_folder.split('_')[0] if '_' in paper_folder else paper_folder
                summary_path = paper_dir / f'{base_name}_fix_summary.csv'
                
                summary_df = pd.DataFrame(paper_summary)
                summary_df.to_csv(summary_path, index=False)
                print(f"✅ Saved fix summary for {paper_folder}: {summary_path}")
    
    # Save overall results CSV
    if all_results:
        overall_summary = []
        for result in all_results:
            overall_summary.append({
                'category': result['category'],
                'paper_folder': result['paper_folder'],
                'flaw_id': result['flaw_id'],
                'flaw_description': result['flaw_description'],
                'total_modifications': result['total_modifications'],
                'successful_fixes': result['successful_fixes'],
                'success_rate': result['successful_fixes'] / result['total_modifications'] if result['total_modifications'] > 0 else 0,
                'fixed_paper_path': result['fixed_paper_path'],
                'success': result['success']
            })
        
        overall_df = pd.DataFrame(overall_summary)
        overall_path = data_dir / args.conference / 'de_planted_error_results.csv'
        overall_df.to_csv(overall_path, index=False)
        
        # Calculate statistics
        total_flaws = len(all_results)
        successful_flaws = sum(1 for r in all_results if r['success'])
        total_modifications = sum(r['total_modifications'] for r in all_results)
        total_successful_fixes = sum(r['successful_fixes'] for r in all_results)
        
        print(f"\n{'='*80}")
        print("✅ PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"📊 Statistics:")
        print(f"   Total flaws processed: {total_flaws}")
        print(f"   Fully successful fixes: {successful_flaws} ({successful_flaws/total_flaws*100:.1f}%)")
        print(f"   Total modifications: {total_modifications}")
        print(f"   Successful individual fixes: {total_successful_fixes} ({total_successful_fixes/total_modifications*100:.1f}%)")
        print(f"\n📁 Overall results saved to: {overall_path}")
        print(f"📁 Fixed papers: {data_dir / args.conference / '*/de-planted_error/'}")
        print(f"{'='*80}")
    else:
        print("\n⚠️ No results to save - no flaws were processed")

if __name__ == "__main__":
    main()
