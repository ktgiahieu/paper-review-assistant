import os
import csv
import json
import argparse
import time
import re
import html
import pandas as pd
import ast
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import threading
from collections import deque
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple, Dict
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

# Default model - can be overridden via command line argument
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash-lite"

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
    "gemma-3-27b-it": 1,          # 60/1 = 60.0 seconds
}

# Gemini model TPM limits (Tokens Per Minute) - conservative estimates
# These are approximate and may vary by account tier
GEMINI_MODEL_TPM_LIMITS = {
    "gemini-2.0-flash-lite": 1000000,  # ~1M tokens/min (conservative)
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
    rpm_limit = GEMINI_MODEL_RPM_LIMITS.get(model_name, 30)  # Default to 30 RPM
    return 60.0 / rpm_limit

# Global variable to store the model name (set in main())
GEMINI_MODEL = DEFAULT_GEMINI_MODEL

# Global variable to store the request delay (set in main())
GEMINI_REQUEST_DELAY = get_request_delay_for_model(DEFAULT_GEMINI_MODEL)

# Rate limiting tracking - improved system
# Track request timestamps for each key to enforce RPM limits accurately
key_request_times: Dict[str, deque] = {}  # deque of timestamps for requests in the last minute
key_lock = threading.Lock()

# Track dynamic delays per key (increases when 429 errors occur)
key_dynamic_delays: Dict[str, float] = {}  # Current delay multiplier per key (starts at 1.0)
key_delay_lock = threading.Lock()

# Track last 429 error time per key (for cooldown period)
key_last_429_time: Dict[str, float] = {}  # Timestamp of last 429 error
key_429_lock = threading.Lock()

# TPM (Tokens Per Minute) tracking
# Track token usage over time windows (sliding window of 60 seconds)
key_token_usage: Dict[str, deque] = {}  # deque of (timestamp, token_count) tuples
tpm_lock = threading.Lock()

# --- Pydantic Models for Structured Responses ---

class Modification(BaseModel):
    start_marker: str = Field(..., description="A unique text marker (3-10 words) that appears at the START of the section to replace. This should be the exact beginning text of the section, including the heading line.")
    end_marker: str = Field(..., description="A unique text marker (3-10 words) that appears at the END of the section to replace, just before the next section starts. This should be the exact ending text of the section.")
    new_content: str = Field(..., description="The complete, rewritten text for the entire section, including the heading. Must start with the exact heading line and end where the original section ended.")
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
            cooldown_period = 60.0  # Wait 60 seconds after a 429 error
            if time_since_429 < cooldown_period:
                wait_time = cooldown_period - time_since_429
                tqdm.write(f"⏳ Cooldown period after 429 error (key {key_name}), waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                current_time = time.time()
    
    # Get dynamic delay multiplier for this key (increases after 429 errors)
    with key_delay_lock:
        delay_multiplier = key_dynamic_delays.get(key_name, 1.0)
    
    # Apply dynamic delay multiplier with additional safety buffer
    # Use 1.5x base multiplier for extra safety, then apply dynamic multiplier
    effective_delay = request_delay * 1.5 * delay_multiplier
    
    # RPM limiting using sliding window (last 60 seconds)
    with key_lock:
        # Initialize if needed
        if key_name not in key_request_times:
            key_request_times[key_name] = deque()
        
        # Clean old requests (older than 60 seconds)
        window_start = current_time - 60.0
        while key_request_times[key_name] and key_request_times[key_name][0] < window_start:
            key_request_times[key_name].popleft()
        
        # Check if we can make a request now
        recent_requests = len(key_request_times[key_name])
        
        if recent_requests >= rpm_limit:
            # We've hit the RPM limit, need to wait until oldest request expires
            if key_request_times[key_name]:
                oldest_time = key_request_times[key_name][0]
                # Add larger buffer (2 seconds) to be more conservative
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
            # Initialize if needed
            if key_name not in key_token_usage:
                key_token_usage[key_name] = deque()
            
            # Clean old entries (older than 60 seconds)
            window_start = current_time - 60.0
            while key_token_usage[key_name] and key_token_usage[key_name][0][0] < window_start:
                key_token_usage[key_name].popleft()
            
            # Calculate current token usage in the window
            current_usage = sum(tokens for _, tokens in key_token_usage[key_name])
            
            # Check if adding this request would exceed TPM limit (use 90% threshold for safety)
            tpm_threshold = int(tpm_limit * 0.9)  # Be more conservative
            if current_usage + estimated_tokens > tpm_threshold:
                # Calculate how long to wait
                if key_token_usage[key_name]:
                    # Wait until oldest entry expires
                    oldest_time = key_token_usage[key_name][0][0]
                    # Add larger buffer (2 seconds) to be more conservative
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
    
    # Normalize the target heading - remove markdown prefixes if present
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
    # Decode HTML entities
    text = html.unescape(text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text.lower().strip()

def find_marker_in_text(text: str, marker: str, context_lines: int = 3) -> Optional[int]:
    """Find a marker text in the full text, returning the line index where it starts.
    Uses fuzzy matching to handle minor variations, HTML entities, etc."""
    lines = text.split('\n')
    marker_normalized = normalize_text_for_matching(marker)
    marker_words = [w for w in marker_normalized.split() if len(w) > 2]  # Filter out very short words
    
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
    min_words_needed = max(2, len(marker_words) - 1)  # Allow 1 word mismatch
    
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
        # Also check if most words match in context
        matching_words = sum(1 for word in marker_words if word in context_normalized)
        if matching_words >= min_words_needed:
            return i
    
    return None

def try_apply_modifications(original_markdown: str, modifications: List[Modification]) -> Tuple[str, bool, Optional[str], List[dict]]:
    """Apply modifications using start/end markers. Returns (modified_text, success, error_msg, metadata_list)."""
    current_markdown = original_markdown
    lines = original_markdown.split('\n')
    metadata_list = []
    
    # Process modifications in reverse order to preserve line indices
    for mod_idx, mod in enumerate(reversed(modifications)):
        start_marker = mod.start_marker.strip()
        end_marker = mod.end_marker.strip()
        
        if not start_marker or not end_marker:
            metadata_list.append({
                'index': mod_idx,
                'success': False,
                'error': 'Missing start_marker or end_marker',
                'reasoning': mod.reasoning
            })
            return original_markdown, False, f"Modification {mod_idx}: Missing start_marker or end_marker", metadata_list
        
        # Find start marker
        start_line = find_marker_in_text(current_markdown, start_marker)
        if start_line is None:
            metadata_list.append({
                'index': mod_idx,
                'success': False,
                'error': f'Start marker not found: "{start_marker[:50]}..."',
                'start_marker': start_marker[:100],
                'reasoning': mod.reasoning
            })
            return original_markdown, False, f"Modification {mod_idx}: Start marker not found: '{start_marker[:50]}...'", metadata_list
        
        # Find end marker (search after start_line)
        lines_after_start = lines[start_line+1:]
        text_after_start = '\n'.join(lines_after_start)
        end_line_relative = find_marker_in_text(text_after_start, end_marker)
        
        if end_line_relative is None:
            metadata_list.append({
                'index': mod_idx,
                'success': False,
                'error': f'End marker not found: "{end_marker[:50]}..."',
                'end_marker': end_marker[:100],
                'reasoning': mod.reasoning
            })
            return original_markdown, False, f"Modification {mod_idx}: End marker not found: '{end_marker[:50]}...'", metadata_list
        
        end_line = start_line + 1 + end_line_relative
        
        # Extract the section being replaced for metadata
        original_section = '\n'.join(lines[start_line:end_line+1])
        
        # Apply modification
        pre_section_lines = lines[:start_line]
        post_section_lines = lines[end_line+1:]  # +1 to include the end marker line
        new_content_lines = mod.new_content.split('\n')
        
        # Reconstruct
        lines = pre_section_lines + new_content_lines + post_section_lines
        current_markdown = '\n'.join(lines)
        
        # Store metadata
        metadata_list.append({
            'index': mod_idx,
            'success': True,
            'start_line': start_line,
            'end_line': end_line,
            'start_marker': start_marker[:100],
            'end_marker': end_marker[:100],
            'original_section_preview': original_section[:200] + '...' if len(original_section) > 200 else original_section,
            'new_content_preview': mod.new_content[:200] + '...' if len(mod.new_content) > 200 else mod.new_content,
            'reasoning': mod.reasoning
        })
    
    return current_markdown, True, None, metadata_list

def extract_section_by_heading(markdown: str, heading: str) -> Optional[str]:
    """Extract a section from markdown by its heading."""
    lines = markdown.split('\n')
    
    # Use the shared heading finding function
    match_index = find_heading_in_lines(lines, heading)
    
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

def call_gemini_with_retries(api_key: str, key_name: str, prompt: str, response_model: type, max_retries: int = 5, request_delay: float = None, tpm_limit: int = 1000000, rpm_limit: int = 30) -> Optional[BaseModel]:
    """Call Gemini API with retries and structured output parsing. Handles 429 errors with exponential backoff."""
    if request_delay is None:
        request_delay = GEMINI_REQUEST_DELAY
    
    # Estimate tokens for this request (input + estimated output)
    estimated_input_tokens = estimate_tokens(prompt)
    estimated_output_tokens = 2000  # Conservative estimate for JSON response
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
            # Wait for rate limit before each attempt (both RPM and TPM)
            wait_for_rate_limit(key_name, request_delay, estimated_total_tokens, tpm_limit, rpm_limit)
            
            response = model.generate_content(prompt)
            json_text = response.text.strip()
            
            # Clean JSON if needed
            json_text = re.sub(r'```json\s*', '', json_text)
            json_text = re.sub(r'```\s*$', '', json_text)
            json_text = json_text.strip()
            
            result = response_model.model_validate_json(json_text)
            
            # Success! Reset dynamic delay multiplier (gradually, but more slowly)
            with key_delay_lock:
                if key_name in key_dynamic_delays and key_dynamic_delays[key_name] > 1.0:
                    # Gradually reduce multiplier on success (more slowly to maintain safety)
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
                
                # Increase dynamic delay multiplier for this key (more aggressive)
                with key_delay_lock:
                    if key_name not in key_dynamic_delays:
                        key_dynamic_delays[key_name] = 1.0
                    # Increase multiplier more aggressively: 1.0 -> 2.0 -> 3.0 -> 4.0 -> 5.0
                    key_dynamic_delays[key_name] = min(5.0, key_dynamic_delays[key_name] + 1.0)
                    current_multiplier = key_dynamic_delays[key_name]
                
                # Exponential backoff for rate limit errors (more aggressive)
                # Start with longer wait times: 10, 20, 40, 80, 120 seconds
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

def plant_error(original_paper: str, flaw_description: str, api_key: str, key_name: str, request_delay: float = None, tpm_limit: int = 1000000, rpm_limit: int = 30, max_retries: int = 2) -> Tuple[Optional[str], Optional[List[Modification]], bool, Optional[str], Optional[List[dict]]]:
    """Plant an error in the paper based on flaw description. Retries with better marker guidance if markers aren't found."""
    
    for attempt in range(max_retries):
        retry_instruction = ""
        if attempt > 0:
            retry_instruction = f"""

**RETRY ATTEMPT {attempt + 1}:**
The previous attempt failed because the start_marker or end_marker could not be found in the paper. Please be more careful:
- Copy the EXACT text from the paper, including any special characters, HTML entities, or formatting
- For start_marker: Use the exact first line of the section (the heading line) plus 2-3 words from the first sentence
- For end_marker: Use the exact last 3-5 words of the section, just before the next heading starts
- Avoid using text that might be in tables or figures - use text from the main body
- If the section ends with a table, use text from the paragraph before the table
"""
        
        prompt = f"""You are an AI model creating a benchmark of flawed scholarly papers for a research experiment. Your objective is to modify a published research paper to re-introduce a specific flaw in a way that makes the paper's claims methodologically unsound.

The key is to make the paper *embody* the flaw, not just state it as a limitation. You must adopt the persona of an author who is either unaware of the flaw or actively downplaying it to make their results seem more significant. Stay true to the authors' original material and personas as much as possible, use scientific terms, avoid writing in an obviously different manner to create an illusion that the paper is coherent, well-written and the introduced flaws are as hard to detect as possible.

**The Golden Rule:**
When rewriting, you must follow this golden rule: The flaw should **NOT** be acknowledged as a weakness in the "Limitations" or "Future Work" sections. Instead, the authors should either ignore it completely or misrepresent it as a minor point or even a strength. The conclusions and abstract must be written with unearned confidence, drawing broad claims from the now-flawed, narrow evidence.
{retry_instruction}
---
Your Task:
For each section that needs modification, you must:
1. Identify the EXACT start and end of the section in the original paper
2. Provide a start_marker: Copy the EXACT first line (the heading) plus the first 2-3 words of the first sentence. This must be verbatim from the paper.
3. Provide an end_marker: Copy the EXACT last 3-5 words of the section's last paragraph, just before the next section heading starts. This must be verbatim from the paper.
4. Generate new_content: the complete rewritten section, starting with the exact heading line and ending where the original section ended

CRITICAL REQUIREMENTS FOR MARKERS:
- start_marker: Must be the exact heading line (e.g., "## Abstract") followed by the first 2-3 words of content, verbatim from the paper
- end_marker: Must be the exact last 3-5 words from the last sentence of the section, verbatim from the paper
- Do NOT use text from tables, figures, or code blocks for markers - use only paragraph text
- Copy markers EXACTLY as they appear, including any special characters, HTML entities, or spacing
- The new_content must preserve the exact heading format (e.g., "## Abstract" not "Abstract")
- Include tables, figures, and code blocks in new_content if they are part of the section
- Maintain the same structure and formatting as the original

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
    
        response = call_gemini_with_retries(api_key, key_name, prompt, ModificationGenerationResponse, request_delay=request_delay, tpm_limit=tpm_limit, rpm_limit=rpm_limit)
        
        if not response:
            if attempt < max_retries - 1:
                tqdm.write(f"  ⚠️ API call failed, retrying... (attempt {attempt + 1}/{max_retries})")
                continue
            return None, None, False, "API call failed or returned None", None
        
        if not response.modifications:
            if attempt < max_retries - 1:
                tqdm.write(f"  ⚠️ No modifications returned, retrying... (attempt {attempt + 1}/{max_retries})")
                continue
            return None, None, False, "API returned no modifications", None
        
        # Apply modifications
        flawed_paper, success, error_msg, metadata = try_apply_modifications(original_paper, response.modifications)
        
        if not success:
            if attempt < max_retries - 1:
                tqdm.write(f"  ⚠️ Marker matching failed: {error_msg}, retrying with better guidance... (attempt {attempt + 1}/{max_retries})")
                continue
            return None, response.modifications, False, error_msg or "Failed to apply modifications", metadata
        
        # Success!
        return flawed_paper, response.modifications, True, None, metadata
    
    # All retries exhausted
    return None, None, False, f"Failed after {max_retries} attempts", None

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
        # (not indented, reasonable length, might be a title)
        if i < 3 and len(line_stripped) < 150 and not line_stripped.startswith(' ') and \
           not line_stripped.startswith('\t') and not line_stripped.startswith('*') and \
           not line_stripped.startswith('-') and not line_stripped.startswith('1.'):
            # Check if it's not a paragraph (doesn't end with period and is not too long)
            if not (line_stripped.endswith('.') and len(line_stripped) > 50):
                return line_stripped
    
    return None

def generate_placebo(original_paper: str, flawed_paper: str, modifications: List[Modification], api_key: str, key_name: str, paperid: str = "", flaw_id: str = "", request_delay: float = None, tpm_limit: int = 1000000, rpm_limit: int = 30) -> Optional[str]:
    """Generate placebo/sham surgery version by learning style and rewriting original sections."""
    
    # Extract modified sections from the modifications themselves using start/end markers
    modified_sections = []
    original_sections = []
    
    for idx, mod in enumerate(modifications):
        # Use start_marker and end_marker to find the original section in original_paper
        start_marker = mod.start_marker.strip()
        end_marker = mod.end_marker.strip()
        
        if not start_marker or not end_marker:
            tqdm.write(f"  ⚠️ Modification {idx+1} missing markers, skipping (paper: {paperid}, flaw: {flaw_id})")
            continue
        
        # Find original section using markers
        lines = original_paper.split('\n')
        start_line = find_marker_in_text(original_paper, start_marker)
        if start_line is None:
            tqdm.write(f"  ⚠️ Could not find start_marker for mod {idx+1} in original paper (paper: {paperid}, flaw: {flaw_id})")
            continue
        
        lines_after_start = lines[start_line+1:]
        text_after_start = '\n'.join(lines_after_start)
        end_line_relative = find_marker_in_text(text_after_start, end_marker)
        if end_line_relative is None:
            tqdm.write(f"  ⚠️ Could not find end_marker for mod {idx+1} in original paper (paper: {paperid}, flaw: {flaw_id})")
            continue
        
        end_line = start_line + 1 + end_line_relative
        original_section = '\n'.join(lines[start_line:end_line+1])
        original_sections.append(original_section)
        
        # Use the modified section from flawed_paper (or from mod.new_content)
        modified_sections.append(mod.new_content)
        tqdm.write(f"  ✅ Extracted original section #{idx+1} using markers (paper: {paperid}, flaw: {flaw_id})")
    
    if not modified_sections:
        tqdm.write(f"  ❌ No modified sections found for placebo generation (paper: {paperid}, flaw: {flaw_id})")
        return None
    
    tqdm.write(f"  ✅ Collected {len(modified_sections)} modified sections for style analysis (paper: {paperid}, flaw: {flaw_id})")
    
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
    
    tqdm.write(f"  🔍 Analyzing writing style... (paper: {paperid}, flaw: {flaw_id})")
    style_response = call_gemini_with_retries(api_key, key_name, style_prompt, StyleAnalysis, request_delay=request_delay, tpm_limit=tpm_limit, rpm_limit=rpm_limit)
    
    if not style_response:
        tqdm.write(f"  ❌ Style analysis failed (paper: {paperid}, flaw: {flaw_id})")
        return None
    
    tqdm.write(f"  ✅ Style analysis complete. Key characteristics: {len(style_response.key_characteristics)} (paper: {paperid}, flaw: {flaw_id})")
    
    # original_sections already extracted above using markers
    if not original_sections:
        tqdm.write(f"  ❌ No original sections found for placebo generation (paper: {paperid}, flaw: {flaw_id})")
        return None
    
    tqdm.write(f"  ✅ Extracted {len(original_sections)} original sections for rewriting (paper: {paperid}, flaw: {flaw_id})")
    
    # Step 3: Rewrite original sections using the learned style (without dropping info)
    # Collect all rewritten sections first, then apply them
    tqdm.write(f"  🔍 Rewriting {len(original_sections)} sections with learned style... (paper: {paperid}, flaw: {flaw_id})")
    rewritten_sections = []
    
    for idx, original_section in enumerate(original_sections):
        # Truncate section if too long
        section_content = original_section
        if len(section_content) > 30000:  # Rough estimate for token limit
            section_content = section_content[:30000] + "\n\n[Truncated...]"
        
        # Extract heading from original section for matching
        heading_from_section = extract_heading_from_content(original_section)
        if not heading_from_section:
            heading_from_section = modifications[idx].start_marker.split('\n')[0] if idx < len(modifications) else f"Section {idx+1}"
        
        rewrite_prompt = f"""Rewrite the following section from a research paper using the writing style described below. 

**CRITICAL REQUIREMENTS:**
1. Maintain ALL factual information from the original section - do not drop any details
2. Apply the writing style characteristics to rewrite the content
3. Keep the same heading format (exact match as in the original)
4. Preserve all technical terms, numbers, and specific claims
5. Only change the writing style, not the content meaning
6. **PRESERVE ALL TABLES, FIGURES, EQUATIONS, AND CODE BLOCKS EXACTLY AS THEY ARE - DO NOT MODIFY, REMOVE, OR REWRITE THEM**
7. Tables must remain in their exact original format with all data unchanged
8. Figures and their captions must remain exactly as they are
9. Mathematical equations must remain exactly as they are

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
        
        tqdm.write(f"  🔄 Rewriting section #{idx+1}/{len(original_sections)}: {heading_from_section[:60]}... (paper: {paperid}, flaw: {flaw_id})")
        rewrite_response = call_gemini_with_retries(api_key, key_name, rewrite_prompt, PlaceboRewritingResponse, request_delay=request_delay, tpm_limit=tpm_limit, rpm_limit=rpm_limit)
        
        if rewrite_response and rewrite_response.rewritten_section:
            rewritten_sections.append({
                'heading': rewrite_response.rewritten_section.original_heading,
                'content': rewrite_response.rewritten_section.new_content,
                'start_marker': modifications[idx].start_marker if idx < len(modifications) else None,
                'end_marker': modifications[idx].end_marker if idx < len(modifications) else None
            })
            tqdm.write(f"  ✅ Successfully rewritten section #{idx+1}: {heading_from_section[:60]}... (paper: {paperid}, flaw: {flaw_id})")
        else:
            tqdm.write(f"  ❌ Failed to rewrite section #{idx+1}: {heading_from_section[:60]}... (paper: {paperid}, flaw: {flaw_id})")
    
    if not rewritten_sections:
        tqdm.write(f"  ❌ No sections were successfully rewritten (paper: {paperid}, flaw: {flaw_id})")
        return None
    
    tqdm.write(f"  ✅ Successfully rewritten {len(rewritten_sections)}/{len(original_sections)} sections (paper: {paperid}, flaw: {flaw_id})")
    
    # Apply all rewritten sections to the paper using markers
    placebo_paper = original_paper
    lines = placebo_paper.split('\n')
    
    # Process sections in reverse order to maintain line indices
    for section in reversed(rewritten_sections):
        new_content = section['content']
        start_marker = section.get('start_marker')
        end_marker = section.get('end_marker')
        
        if not new_content or not start_marker or not end_marker:
            continue
        
        # Find section using markers
        start_line = find_marker_in_text(placebo_paper, start_marker)
        if start_line is None:
            continue
        
        lines_after_start = lines[start_line+1:]
        text_after_start = '\n'.join(lines_after_start)
        end_line_relative = find_marker_in_text(text_after_start, end_marker)
        if end_line_relative is None:
            continue
        
        end_line = start_line + 1 + end_line_relative
        
        # Reconstruct
        pre_section_lines = lines[:start_line]
        post_section_lines = lines[end_line+1:]  # +1 to include end marker line
        new_content_lines = new_content.split('\n')
        lines = pre_section_lines + new_content_lines + post_section_lines
        placebo_paper = '\n'.join(lines)
    
    return placebo_paper

def process_paper(row: pd.Series, base_dir: Path, output_dir: Path, task_idx: int, request_delay: float = None, tpm_limit: int = 1000000, rpm_limit: int = 30) -> dict:
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
        tqdm.write(f"  🌱 Planting error #{flaw_idx + 1} for {paperid}...")
        flawed_paper, modifications, success, error_msg, metadata = plant_error(original_paper, flaw_description, api_key, key_name, request_delay=request_delay, tpm_limit=tpm_limit, rpm_limit=rpm_limit)
        
        if not success or not flawed_paper:
            error_detail = f": {error_msg}" if error_msg else ""
            tqdm.write(f"  ❌ Failed to plant error for {paperid}, flaw {flaw_idx + 1}{error_detail}")
            continue
        
        tqdm.write(f"  ✅ Successfully planted error #{flaw_idx + 1} for {paperid} ({len(modifications)} modifications)")
        
        # Save planted error version
        planted_error_dir = output_dir / 'planted_error' / paper_folder.name
        planted_error_dir.mkdir(parents=True, exist_ok=True)
        planted_error_path = planted_error_dir / f"{flaw_id}.md"
        
        with open(planted_error_path, 'w', encoding='utf-8') as f:
            f.write(flawed_paper)
        
        # Save JSON metadata for planted error
        planted_error_metadata = {
            'paperid': paperid,
            'flaw_id': flaw_id,
            'flaw_description': flaw_description,
            'modifications': [
                {
                    'index': idx,
                    'start_marker': mod.start_marker,
                    'end_marker': mod.end_marker,
                    'reasoning': mod.reasoning,
                    'new_content_preview': mod.new_content[:500] + '...' if len(mod.new_content) > 500 else mod.new_content
                }
                for idx, mod in enumerate(modifications)
            ],
            'metadata': metadata if metadata else []
        }
        planted_error_json_path = planted_error_dir / f"{flaw_id}.json"
        with open(planted_error_json_path, 'w', encoding='utf-8') as f:
            json.dump(planted_error_metadata, f, indent=2, ensure_ascii=False)
        
        # Generate placebo/sham surgery version
        tqdm.write(f"  💊 Generating placebo for {paperid}, flaw {flaw_idx + 1}...")
        placebo_paper = generate_placebo(original_paper, flawed_paper, modifications, api_key, key_name, paperid, flaw_id, request_delay=request_delay, tpm_limit=tpm_limit, rpm_limit=rpm_limit)
        sham_surgery_path = None
        
        if placebo_paper:
            # Save placebo version
            sham_surgery_dir = output_dir / 'sham_surgery' / paper_folder.name
            sham_surgery_dir.mkdir(parents=True, exist_ok=True)
            sham_surgery_path = sham_surgery_dir / f"{flaw_id}.md"
            
            with open(sham_surgery_path, 'w', encoding='utf-8') as f:
                f.write(placebo_paper)
            
            # Save JSON metadata for sham surgery (same structure, but note it's a placebo)
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
                'metadata': metadata if metadata else []
            }
            sham_surgery_json_path = sham_surgery_dir / f"{flaw_id}.json"
            with open(sham_surgery_json_path, 'w', encoding='utf-8') as f:
                json.dump(sham_surgery_metadata, f, indent=2, ensure_ascii=False)
            
            tqdm.write(f"  ✅ Successfully generated placebo for {paperid}, flaw {flaw_idx + 1}")
        else:
            tqdm.write(f"  ❌ Failed to generate placebo for {paperid}, flaw {flaw_idx + 1}")
        
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
    global GEMINI_MODEL, GEMINI_REQUEST_DELAY
    
    parser = argparse.ArgumentParser(description="Plant errors and generate placebo versions of papers using Gemini API.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to filtered_pairs_with_human_scores.csv")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing paper folders (e.g., data/ICLR2024)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: same as base_dir)")
    parser.add_argument("--max_workers", type=int, default=None, help="Max worker threads (default: calculated based on model and API keys)")
    parser.add_argument("--model_name", type=str, default=DEFAULT_GEMINI_MODEL, 
                        help=f"Gemini model name to use (default: {DEFAULT_GEMINI_MODEL})")
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
    
    # Set max_workers - use number of keys by default for optimal parallelization
    # The improved rate limiting system will handle RPM limits accurately
    if args.max_workers is not None:
        max_workers = args.max_workers
    else:
        # Default: use number of API keys so each worker can use a different key
        # The sliding window rate limiting will ensure we don't exceed RPM limits
        max_workers = len(GEMINI_API_KEYS)
    
    # Calculate theoretical throughput
    total_throughput = len(GEMINI_API_KEYS) * model_rpm
    estimated_tpm_per_request = 50000  # Conservative estimate for large paper processing
    # Estimate: if all workers are active at max RPM, each making requests with ~50k tokens
    estimated_tpm_throughput = max_workers * estimated_tpm_per_request  # Per minute if all workers active
    print(f"✅ Using {max_workers} worker threads (one per API key)")
    print(f"✅ Total theoretical throughput: {total_throughput} RPM ({len(GEMINI_API_KEYS)} keys × {model_rpm} RPM)")
    print(f"✅ Rate limiting: Sliding window tracking with dynamic backoff on 429 errors")
    print(f"⚠️ Estimated TPM usage: ~{estimated_tpm_throughput:,} tokens/min per key (if all workers active)")
    print(f"⚠️ TPM limit per key: {model_tpm:,} tokens/min")
    if estimated_tpm_throughput > model_tpm * 0.8:
        print(f"⚠️ WARNING: Estimated TPM usage ({estimated_tpm_throughput:,}) may exceed limit ({model_tpm:,})")
    print()
    
    # Process papers
    all_results = []
    task_counter = [0]  # Use list to allow modification in closure
    
    def process_with_counter(row):
        idx = task_counter[0]
        task_counter[0] += 1
        return process_paper(row, base_dir, output_dir, idx, request_delay=GEMINI_REQUEST_DELAY, tpm_limit=model_tpm, rpm_limit=model_rpm)
    
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
        
        # Calculate statistics
        total_flaws = len(all_results)
        successful_planted = sum(1 for r in all_results if r.get('success', False))
        successful_placebo = sum(1 for r in all_results if r.get('sham_surgery_path') is not None)
        
        print(f"\n{'='*80}")
        print("✅ PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"📊 Statistics:")
        print(f"   Total flaws processed: {total_flaws}")
        print(f"   Successfully planted errors: {successful_planted} ({successful_planted/total_flaws*100:.1f}%)")
        print(f"   Successfully generated placebos: {successful_placebo} ({successful_placebo/total_flaws*100:.1f}%)")
        print(f"\n📁 Results saved to: {results_csv_path}")
        print(f"📁 Planted errors: {output_dir / 'planted_error'}")
        print(f"📁 Sham surgery: {output_dir / 'sham_surgery'}")
        print(f"{'='*80}")
    else:
        print("\n⚠️ No results to save - no flaws were successfully processed")

if __name__ == "__main__":
    main()


