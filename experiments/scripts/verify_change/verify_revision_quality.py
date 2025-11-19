#!/usr/bin/env python3
"""
Verify LLM capability to distinguish between real revisions and LLM-generated fixes.

This script evaluates whether LLMs can distinguish between:
- "true good" = Camera ready paper (real revisions)
- "fake good" = De-planted error paper (LLM-generated fixes without real substance)

Given a pair of papers (original flawed + revised), the LLM scores how well revisions
were made on a scale of 1-9 (9 is best).

Usage:
    python verify_revision_quality.py --data_dir ../sampled_data_verify_change/no_appendix \
                                      --model_name gemini-2.0-flash-lite \
                                      --comparison_type true_good_vs_fake_good
"""

import os
import json
import argparse
import time
import re
import threading
import signal
import sys
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Global flag for graceful shutdown
should_exit = threading.Event()
results_csv_lock = threading.Lock()

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("WARNING: google-generativeai not installed. Install with: pip install google-generativeai")

# Set style
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    plt.style.use("ggplot")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
COLOR_MATCH = "#2ecc71"
COLOR_MISMATCH = "#e74c3c"
COLOR_NEUTRAL = "#95a5a6"
COLOR_WHITE = "#ffffff"

SAMPLE_SIZE_MULTIPLIER = 1

# Category label mapping for plots (short versions)
CATEGORY_LABELS = {
    '1a': 'Baselines',
    '1b': 'Scope',
    '1c': 'Ablation',
    '1d': 'Metrics',
    '2a': 'Design',
    '2b': 'Lacks Theory',
    '2c': 'Math Error',
    '3a': 'Novelty',
    '3b': 'Overclaims',
    '4a': 'Clarity',
    '4b': 'Reproducibility',
    '5a': 'Limitations',
    '5b': 'Ethical',
}

# --- Environment & API Configuration ---
load_dotenv()

# Load paid API key if available
GEMINI_API_KEY_PAID = os.getenv('GEMINI_API_KEY_PAID')

# Load multiple Gemini API keys
GEMINI_API_KEYS: Dict[str, str] = {}
for i in range(1, 10):
    key = os.getenv(f'GEMINI_API_KEY_{i}')
    if key:
        GEMINI_API_KEYS[str(i)] = key

if not GEMINI_API_KEYS:
    single_key = os.getenv('GEMINI_API_KEY')
    if single_key:
        GEMINI_API_KEYS = {'SINGLE': single_key}
    else:
        raise ValueError("No Gemini API keys found in environment variables")

# Add paid key to the pool if available
if GEMINI_API_KEY_PAID:
    GEMINI_API_KEYS['PAID'] = GEMINI_API_KEY_PAID
    print(f"✅ Loaded {len(GEMINI_API_KEYS)} Gemini API keys (including PAID key)")
else:
    print(f"✅ Loaded {len(GEMINI_API_KEYS)} Gemini API keys")

# --- Constants ---
MAX_RETRIES = 3
# Base backoff in seconds for retrying after errors (e.g., 429).
# Increased from 2 to 10 seconds to reduce how frequently retries are attempted.
INITIAL_BACKOFF_SECONDS = 6

# Gemini model RPM limits
GEMINI_MODEL_RPM_LIMITS = {
    "gemini-2.0-flash-lite": 30,
    "gemini-2.0-flash-exp": 10,
    "gemini-2.0-flash": 15,
    "gemini-2.5-flash-lite": 15,
    "gemini-2.5-flash": 5,
    "gemini-2.5-pro": 1,
}

def get_request_delay_for_model(model_name: str) -> float:
    """Calculate request delay in seconds based on model's RPM limit."""
    rpm_limit = GEMINI_MODEL_RPM_LIMITS.get(model_name, 30)
    return 60.0 / rpm_limit

# Rate limiting tracking
key_last_used: Dict[str, float] = {}
key_lock = threading.Lock()
USE_PAID_KEY = False  # Global flag to use paid key exclusively

def wait_for_rate_limit(key_name: str, delay: float):
    """Wait if necessary to respect rate limits. Skip delay for paid key or if delay is 0."""
    # Skip rate limiting for paid key or if delay is 0 or None
    if key_name == 'PAID' or USE_PAID_KEY or delay is None or delay <= 0:
        return
    
    with key_lock:
        current_time = time.time()
        if key_name in key_last_used:
            time_since_last_use = current_time - key_last_used[key_name]
            if time_since_last_use < delay:
                sleep_time = delay - time_since_last_use
                time.sleep(sleep_time)
        key_last_used[key_name] = time.time()

def get_api_key_for_task(task_idx: int, use_paid: bool = False) -> Tuple[str, str]:
    """Get API key for a task using round-robin. Prioritize paid key if use_paid=True."""
    if use_paid and 'PAID' in GEMINI_API_KEYS:
        # Always use paid key when requested
        return 'PAID', GEMINI_API_KEYS['PAID']
    
    key_names = list(GEMINI_API_KEYS.keys())
    key_name = key_names[task_idx % len(key_names)]
    return key_name, GEMINI_API_KEYS[key_name]

# --- Pydantic Models ---
class RevisionScore(BaseModel):
    """Pydantic model for revision quality score."""
    score: int = Field(
        description="Revision quality score from 1 to 9, where 9 indicates excellent revisions that substantially address the flaw.",
        ge=1, le=9
    )
    reasoning: str = Field(
        description="Brief explanation (2-3 sentences) of the score, focusing on how well the revisions address the identified flaw."
    )

# --- Prompt Templates ---
def create_verification_prompt(
    original_paper: str,
    revised_paper: str,
    flaw_description: str,
    flaw_location: str = "",
    change_details: List[Dict] = None,
    ablation_name: str = None
) -> str:
    """Create prompt for verifying revision quality."""
    # Ablation: no_location - don't include flaw location
    include_location = ablation_name != "no_location"
    location_text = f"\n**Flaw Location:** {flaw_location}\n" if (flaw_location and include_location) else ""
    
    # Add detailed change locations if provided
    detailed_changes_text = ""
    if change_details:
        detailed_changes_text = "\n**Specific Sections That Were Changed:**\n\n"
        for i, change in enumerate(change_details, 1):
            heading = change.get('target_heading', 'Unknown Section')
            planted = change.get('planted_error_content', '')
            deplanted = change.get('deplanted_error_content', '')
            explanation = change.get('explanation', '')
            
            detailed_changes_text += f"**Change {i}: {heading}**\n"
            if planted:
                detailed_changes_text += f"Original (flawed) content:\n{planted}\n\n"
            if deplanted:
                detailed_changes_text += f"Revised content:\n{deplanted}\n\n"
            # if explanation:
            #     detailed_changes_text += f"Fix explanation: {explanation}\n\n"
            detailed_changes_text += "---\n\n"
        
        detailed_changes_text += "\n**Important:** Focus your evaluation on these specific sections that were changed. Compare the original flawed content with the revised content to assess the quality of the revisions.\n\n"
    
    prompt = f"""You are evaluating how well a revised research paper addresses a specific flaw that was previously identified by reviewers. Your task is to provide a careful, critical, and professional assessment based on the actual content of the papers, not on the authors' claims alone.

**Evaluation Approach:**
Authors may sometimes make revisions that sound plausible but do not fully resolve the underlying issue. They may describe additional experiments, analyses, or baselines without providing sufficient supporting evidence, or update wording without making substantive methodological or empirical changes. Therefore, you must verify what is actually present in the revised paper by comparing it directly with the original version.

**Step 1: Understand the Flaw and Required Fix**
First, carefully analyze why the reviewer identified this flaw. Identify the core underlying issue (not just surface phrasing) and determine what a substantive fix would require, such as:
- New or improved experiments or analyses
- Additional baselines or comparisons
- Clearer and more complete methodological reporting
- Appropriately calibrated claims and conclusions

**Identified Flaw:**
{flaw_description}
{location_text}{detailed_changes_text}**Original Paper (with flaw):**
{original_paper}

**Revised Paper:**
{revised_paper}

**Step 2: Systematic Evaluation**
Compare the original and revised papers section by section, especially at the specified change locations. For each author claim (e.g., "we added comparison X" or "we now provide analysis Y"), verify that the corresponding evidence is clearly present and sufficiently detailed in the revised text.

Check systematically for the following potential problems:

1. **Fabricated or Unrealistic Results:** Results that seem too good to be true, numbers that appear invented, claims lacking methodological justification, or results that don't align with the described experimental setup.

2. **Incomplete or Vague Experimental Details:** Missing or insufficient description of experimental settings, hyperparameters, or configurations; lack of reproducibility information; vague descriptions that prevent verification; missing details about datasets, splits, or evaluation protocols.

3. **Cherry-Picking and Selective Reporting:** Only showing favorable comparisons or results; ignoring important baselines or methods; highlighting only the best metrics while hiding weaknesses; incomplete experimental coverage.

4. **Over-Exaggeration and Unsubstantiated Claims:** Claims that go beyond what the results actually support; overly strong language without proper statistical backing; extrapolating from limited evidence; making broad claims based on narrow experimental validation.

5. **Low-Effort and Superficial Changes:** Minimal text changes that don't address the core issue; adding generic statements without concrete evidence; promising future work instead of providing actual results; reorganizing or rewording without substantive improvement.

6. **Lack of Genuine Understanding:** Revisions that miss the point of the original critique; addressing symptoms rather than root causes; adding content that doesn't logically connect to the flaw; failing to demonstrate understanding of why the flaw mattered.

7. **Insufficient Depth and Rigor:** Shallow analysis when deep investigation is needed; missing ablation studies or analysis when required; lack of discussion of limitations or failure cases; insufficient comparison with state-of-the-art methods.

8. **Poor Integration and Coherence:** New content that doesn't fit with the rest of the paper; contradictions with other parts; inconsistent notation, terminology, or style; changes that create new problems.

9. **Discrepancies between Claims and Evidence:** Text claims that new baselines or analyses were added, but they are missing or only mentioned briefly; text states concerns have been addressed, but the problematic behavior still appears; any mismatch between what the revised text claims and what is actually documented.

**Step 3: Assign a Score (1-9)**
Assign a score from 1 to 9 based on how well the revisions address the flaw. Each score represents a distinct level of quality:

**Score 1:** The flaw is not addressed. The revised paper still clearly exhibits the same problem with no meaningful changes. Multiple serious issues are present (missing experiments, missing baselines, vague methodology, unsupported claims). Revisions are purely cosmetic.

**Score 2:** The flaw is very weakly addressed. There are minor additions or edits, but they do not meaningfully reduce the original concern. Important evidence, detail, or analysis is still missing. Several of the potential problems listed above are present. The revisions show minimal effort.

**Score 3:** The flaw is addressed minimally. There are some changes that relate to the flaw, but they are shallow or incomplete. Key aspects of the concern remain unresolved. The revisions show limited effort or depth, and important gaps persist.

**Score 4:** The flaw is partially addressed, but revisions are closer to superficial than substantive. Some relevant content has been added, but important limitations persist (e.g., missing baselines, insufficient analysis, unclear methodology). The response addresses only part of what is needed.

**Score 5:** The flaw is moderately addressed. The revisions provide some meaningful improvements, but there are noticeable gaps. The response covers part of what is needed, yet either depth, coverage, or clarity is still insufficient for a fully convincing fix. The revisions show understanding but incomplete execution.

**Score 6:** The flaw is addressed in a reasonably solid way, though not perfectly. Most key expectations are met, but there may be minor missing details, limited analysis, or restricted coverage that prevent this from being a strong or comprehensive fix. The revisions are mostly adequate but could be more thorough.

**Score 7:** The flaw is addressed well. The revisions are substantive, clearly targeted at the core issue, and supported by appropriate evidence and explanation. Any remaining gaps are relatively minor and do not substantially undermine the response to the flaw. The revisions demonstrate good understanding and meaningful improvement.

**Score 8:** The flaw is addressed very well. The revisions show clear understanding, strong evidence (e.g., well-designed experiments and comparisons), and thorough explanation. The response would likely satisfy a careful reviewer, with only small possible improvements. The revisions are comprehensive and well-executed.

**Score 9:** The flaw is addressed in an exemplary way. The revisions are comprehensive, carefully argued, and fully supported by detailed evidence and analysis. The work demonstrates clear understanding, rigor, and thoughtful integration with the rest of the paper. The revisions represent a model response to the identified flaw.

**Step 4: Provide Your Assessment**
Provide:
- A numerical revision score from 1 to 9 (using the definitions above).
- A clear explanation (in a few sentences) that:
  * Identifies any of the potential problems listed above that are present.
  * Comments on the depth and rigor of the changes relative to what the flaw requires.
  * States whether the revisions demonstrate genuine understanding and meaningful improvement.
  * Summarizes whether this appears to be a substantive fix or primarily a superficial change."""
    
    return prompt

# --- API Call Functions ---
def clean_json_schema(schema: dict) -> dict:
    """Remove fields from JSON schema that Gemini doesn't support."""
    # Fields that Gemini doesn't support
    unsupported_fields = {"title", "maximum", "minimum", "exclusiveMaximum", "exclusiveMinimum"}
    
    cleaned = {}
    for key, value in schema.items():
        if key in unsupported_fields:
            continue  # Skip unsupported fields
        elif key == "$defs" or key == "definitions":
            # Recursively clean definitions
            cleaned[key] = {
                k: clean_json_schema(v) if isinstance(v, dict) else v
                for k, v in value.items()
            }
        elif isinstance(value, dict):
            cleaned[key] = clean_json_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [
                clean_json_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned

def call_gemini_with_retries(
    api_key: str,
    key_name: str,
    prompt: str,
    response_model: BaseModel,
    max_retries: int = MAX_RETRIES,
    request_delay: float = None,
    model_name: str = "gemini-2.0-flash-lite"
) -> Optional[BaseModel]:
    """Call Gemini API with retries and structured output."""
    if request_delay is None:
        request_delay = get_request_delay_for_model(model_name)
    
    # Get and clean the JSON schema
    raw_schema = response_model.model_json_schema()
    cleaned_schema = clean_json_schema(raw_schema)
    
    for attempt in range(max_retries):
        try:
            wait_for_rate_limit(key_name, request_delay)
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            
            # Use structured output with cleaned schema
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=cleaned_schema
                )
            )
            
            # Parse JSON response
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            response_dict = json.loads(response_text)
            return response_model(**response_dict)
            
        except Exception as e:
            if attempt < max_retries - 1:
                backoff = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                time.sleep(backoff)
            else:
                # Only log errors that aren't schema-related (those are expected and handled)
                if "Schema" not in str(e):
                    tqdm.write(f"  ⚠️ Error after {max_retries} attempts: {e}")
                return None
    
    return None

# --- File Reading Functions ---
def read_paper_markdown(paper_path: Path) -> Optional[str]:
    """Read paper markdown file from various possible locations."""
    # Try structured_paper_output/paper.md first (for latest/)
    structured_path = paper_path / "structured_paper_output" / "paper.md"
    if structured_path.exists():
        try:
            with open(structured_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            tqdm.write(f"  ⚠️ Error reading {structured_path}: {e}")
    
    # Try paper.md directly
    paper_md = paper_path / "paper.md"
    if paper_md.exists():
        try:
            with open(paper_md, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            tqdm.write(f"  ⚠️ Error reading {paper_md}: {e}")
    
    # If paper_path is already a .md file, read it directly
    if paper_path.is_file() and paper_path.suffix == '.md':
        try:
            with open(paper_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            tqdm.write(f"  ⚠️ Error reading {paper_path}: {e}")
    
    return None

def read_modifications_summary(csv_path: Path) -> Optional[Dict]:
    """Read modifications summary CSV and extract flaw information."""
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return None
        
        # Get first row (assuming one flaw per paper for now)
        row = df.iloc[0]
        
        # Parse modifications JSON
        modifications = json.loads(row['llm_generated_modifications'])
        
        # Extract target headings and locations
        locations = []
        for mod in modifications:
            heading = mod.get('target_heading', '')
            if heading:
                locations.append(heading)
        
        location_text = "; ".join(locations) if locations else ""
        
        return {
            'flaw_id': row['flaw_id'],
            'flaw_description': row['flaw_description'],
            'flaw_location': location_text,
            'num_modifications': row['num_modifications']
        }
    except Exception as e:
        tqdm.write(f"  ⚠️ Error reading modifications summary: {e}")
        return None

def read_fix_summary(csv_path: Path) -> Optional[Dict]:
    """Read fix_summary CSV from de-planted_error folder and extract detailed change locations."""
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return None
        
        # Get first row (assuming one flaw per paper for now)
        row = df.iloc[0]
        
        # Parse modifications JSON
        modifications = json.loads(row['llm_generated_modifications'])
        
        # Extract detailed change information
        change_details = []
        for mod in modifications:
            heading = mod.get('target_heading', '')
            planted_content = mod.get('planted_error_content', '')
            deplanted_content = mod.get('deplanted_error_content', '')
            explanation = mod.get('explanation', '')
            success = mod.get('success', False)
            
            if heading:
                change_details.append({
                    'target_heading': heading,
                    'planted_error_content': planted_content[:1000] if planted_content else '',  # Limit length
                    'deplanted_error_content': deplanted_content[:1000] if deplanted_content else '',  # Limit length
                    'explanation': explanation,
                    'success': success
                })
        
        return {
            'flaw_id': row['flaw_id'],
            'flaw_description': row['flaw_description'],
            'change_details': change_details,
            'num_modifications': row['num_modifications']
        }
    except Exception as e:
        tqdm.write(f"  ⚠️ Error reading fix summary: {e}")
        return None

# --- Scoring Functions ---
def score_revision_quality(
    original_paper: str,
    revised_paper: str,
    flaw_description: str,
    flaw_location: str,
    api_key: str,
    key_name: str,
    model_name: str,
    request_delay: float = None,
    change_details: List[Dict] = None,
    ablation_name: str = None
) -> Optional[RevisionScore]:
    """Score the quality of revisions using Gemini."""
    prompt = create_verification_prompt(
        original_paper=original_paper,
        revised_paper=revised_paper,
        flaw_description=flaw_description,
        flaw_location=flaw_location,
        change_details=change_details,
        ablation_name=ablation_name
    )
    
    response = call_gemini_with_retries(
        api_key=api_key,
        key_name=key_name,
        prompt=prompt,
        response_model=RevisionScore,
        max_retries=MAX_RETRIES,
        request_delay=request_delay,
        model_name=model_name
    )
    
    return response

def process_paper_pair(
    category: str,
    paper_folder: str,
    flaw_id: str,
    original_paper_path: Path,
    revised_paper_path: Path,
    flaw_info: Dict,
    output_dir: Path,
    task_idx: int,
    model_name: str,
    request_delay: float = None,
    change_details: List[Dict] = None,
    use_paid: bool = False,
    ablation_name: str = None
) -> Optional[Dict]:
    """Process a single paper pair and score revision quality."""
    try:
        # Read papers
        original_paper = read_paper_markdown(original_paper_path)
        revised_paper = read_paper_markdown(revised_paper_path)
        
        if not original_paper or not revised_paper:
            return None
        
        # Get API key
        key_name, api_key = get_api_key_for_task(task_idx, use_paid=use_paid)
        
        # Score revision quality
        score_result = score_revision_quality(
            original_paper=original_paper,
            revised_paper=revised_paper,
            flaw_description=flaw_info['flaw_description'],
            flaw_location=flaw_info['flaw_location'],
            api_key=api_key,
            key_name=key_name,
            model_name=model_name,
            request_delay=request_delay,
            change_details=change_details,
            ablation_name=ablation_name
        )
        
        if not score_result:
            return None
        
        return {
            'category': category,
            'paper_folder': paper_folder,
            'flaw_id': flaw_id,
            'flaw_description': flaw_info['flaw_description'],
            'score': score_result.score,
            'reasoning': score_result.reasoning
        }
        
    except Exception as e:
        tqdm.write(f"  ❌ Error processing {paper_folder}/{flaw_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Statistics and Plotting Functions ---
def compute_treatment_effect(df: pd.DataFrame) -> Dict:
    """Compute treatment effect statistics overall and per category."""
    # Group by paper to get paired comparisons
    results = []
    
    for (category, paper_folder, flaw_id), group in df.groupby(['category', 'paper_folder', 'flaw_id']):
        true_good_row = group[group['revision_type'] == 'true_good']
        fake_good_row = group[group['revision_type'] == 'fake_good']
        
        if len(true_good_row) > 0 and len(fake_good_row) > 0:
            true_good_score = true_good_row.iloc[0]['score']
            fake_good_score = fake_good_row.iloc[0]['score']
            difference = true_good_score - fake_good_score
            
            results.append({
                'category': category,
                'paper_folder': paper_folder,
                'flaw_id': flaw_id,
                'true_good_score': true_good_score,
                'fake_good_score': fake_good_score,
                'difference': difference
            })
    
    if not results:
        return {}
    
    diff_df = pd.DataFrame(results)
    differences = diff_df['difference'].values
    true_good_scores = diff_df['true_good_score'].values
    fake_good_scores = diff_df['fake_good_score'].values
    
    mean_diff = float(np.mean(differences))
    std_diff = float(np.std(differences, ddof=1))
    
    # Treatment effect = mean difference / std of differences
    treatment_effect = mean_diff / std_diff if std_diff > 0 else 0.0
    
    # Cohen's d for paired samples with correlation correction
    n = len(differences)
    se = None
    ci_low = None
    ci_high = None
    
    if std_diff == 0:
        # No variance in differences
        d_prime = float(np.sign(mean_diff)) * np.inf if mean_diff != 0 else 0.0
        cohen_d = d_prime
    else:
        # 1. Calculate d' (d_z) = mean_diff / std_diff
        d_prime = mean_diff / std_diff
        
        # 2. Calculate correlation r between true_good and fake_good scores
        if n > 1:
            correlation = float(np.corrcoef(true_good_scores, fake_good_scores)[0, 1])
        else:
            correlation = 0.0
        
        # 3. Apply correction: cohen_d = d_prime / sqrt(1 - r)
        if correlation >= 0.99999:
            cohen_d = float(np.sign(d_prime)) * np.inf if d_prime != 0 else 0.0
            correction_factor = 1.0
        else:
            correction_factor = np.sqrt(1 - correlation)
            cohen_d = float(d_prime / correction_factor)
        
        # 4. Calculate Standard Error for the corrected Cohen's d
        if np.isfinite(d_prime) and np.isfinite(cohen_d) and n > 1:
            # Approx. SE for d_prime (d_z)
            se_d_prime = np.sqrt((1.0 / n) + (d_prime ** 2) / (2.0 * (n - 1)))
            # Scale the SE for the corrected d
            se = float(se_d_prime / correction_factor)
            # Apply sample size multiplier: SE scales as 1/√n, so 4x data = 2x smaller SE
            se = se / np.sqrt(SAMPLE_SIZE_MULTIPLIER)
            ci_low = float(cohen_d - 1.96 * se)
            ci_high = float(cohen_d + 1.96 * se)
    
    # Standard error for mean difference (separate from Cohen's d)
    se_mean = std_diff / np.sqrt(n) if n > 0 else 0.0
    se_mean = se_mean / np.sqrt(SAMPLE_SIZE_MULTIPLIER)
    mean_ci_low = mean_diff - 1.96 * se_mean
    mean_ci_high = mean_diff + 1.96 * se_mean
    
    # Compute per-category statistics
    category_stats = []
    for category in diff_df['category'].unique():
        cat_df = diff_df[diff_df['category'] == category]
        cat_differences = cat_df['difference'].values
        
        if len(cat_differences) > 0:
            cat_mean = float(np.mean(cat_differences))
            cat_std = float(np.std(cat_differences, ddof=1))
            cat_n = len(cat_differences)
            cat_true_good = cat_df['true_good_score'].values
            cat_fake_good = cat_df['fake_good_score'].values
            
            # Standard error for mean difference
            cat_se = cat_std / np.sqrt(cat_n) if cat_n > 0 else 0.0
            cat_se = cat_se / np.sqrt(SAMPLE_SIZE_MULTIPLIER)  # Reduce SE for larger sample size
            cat_ci_low = cat_mean - 1.96 * cat_se
            cat_ci_high = cat_mean + 1.96 * cat_se
            
            # Cohen's d for paired samples with correlation correction
            cat_se_d = None
            cat_ci_d_low = None
            cat_ci_d_high = None
            
            if cat_std == 0:
                cat_d_prime = float(np.sign(cat_mean)) * np.inf if cat_mean != 0 else 0.0
                cat_cohen_d = cat_d_prime
            else:
                # 1. Calculate d' (d_z) = cat_mean / cat_std
                cat_d_prime = cat_mean / cat_std
                
                # 2. Calculate correlation r between true_good and fake_good scores
                if cat_n > 1:
                    cat_correlation = float(np.corrcoef(cat_true_good, cat_fake_good)[0, 1])
                else:
                    cat_correlation = 0.0
                
                # 3. Apply correction: cat_cohen_d = cat_d_prime / sqrt(1 - r)
                if cat_correlation >= 0.99999:
                    cat_cohen_d = float(np.sign(cat_d_prime)) * np.inf if cat_d_prime != 0 else 0.0
                    cat_correction_factor = 1.0
                else:
                    cat_correction_factor = np.sqrt(1 - cat_correlation)
                    cat_cohen_d = float(cat_d_prime / cat_correction_factor)
                
                # 4. Calculate Standard Error for the corrected Cohen's d
                if np.isfinite(cat_d_prime) and np.isfinite(cat_cohen_d) and cat_n > 1:
                    # Approx. SE for d_prime (d_z)
                    cat_se_d_prime = np.sqrt((1.0 / cat_n) + (cat_d_prime ** 2) / (2.0 * (cat_n - 1)))
                    # Scale the SE for the corrected d
                    cat_se_d = float(cat_se_d_prime / cat_correction_factor)
                    # Apply sample size multiplier
                    cat_se_d = cat_se_d / np.sqrt(SAMPLE_SIZE_MULTIPLIER)
                    cat_ci_d_low = float(cat_cohen_d - 1.96 * cat_se_d)
                    cat_ci_d_high = float(cat_cohen_d + 1.96 * cat_se_d)
                else:
                    cat_se_d = 0.0
                    cat_ci_d_low = cat_cohen_d if np.isfinite(cat_cohen_d) else None
                    cat_ci_d_high = cat_cohen_d if np.isfinite(cat_cohen_d) else None
            
            # Ensure cat_se_d is set
            if cat_se_d is None:
                cat_se_d = 0.0
            
            category_stats.append({
                'category': category,
                'n_pairs': cat_n,
                'mean_difference': cat_mean,
                'std_difference': cat_std,
                'standard_error': cat_se,
                'ci_low': cat_ci_low,
                'ci_high': cat_ci_high,
                'cohen_d': float(cat_cohen_d) if np.isfinite(cat_cohen_d) else None,
                'cohen_d_se': cat_se_d,
                'cohen_d_ci_low': cat_ci_d_low,
                'cohen_d_ci_high': cat_ci_d_high,
            })
    
    category_stats_df = pd.DataFrame(category_stats)
    
    # Ensure se is set for Cohen's d
    if se is None:
        se = 0.0
    
    return {
        'n_pairs': n,
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'treatment_effect': treatment_effect,
        'cohen_d': float(cohen_d) if np.isfinite(cohen_d) else None,
        'cohen_d_se': se,  # Standard error for Cohen's d
        'cohen_d_ci_low': ci_low,  # CI for Cohen's d
        'cohen_d_ci_high': ci_high,  # CI for Cohen's d
        'standard_error': se_mean,  # Standard error for mean difference
        'ci_low': mean_ci_low,  # CI for mean difference
        'ci_high': mean_ci_high,  # CI for mean difference
        'differences': differences.tolist(),
        'detailed_results': diff_df,
        'category_stats': category_stats_df
    }

def detect_lazy_authors(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Detect cases where authors might be lazy (true_good score < 5)."""
    # Group by paper to get paired comparisons
    results = []
    
    for (category, paper_folder, flaw_id), group in df.groupby(['category', 'paper_folder', 'flaw_id']):
        true_good_row = group[group['revision_type'] == 'true_good']
        fake_good_row = group[group['revision_type'] == 'fake_good']
        
        if len(true_good_row) > 0:
            true_good_score = true_good_row.iloc[0]['score']
            true_good_reasoning = true_good_row.iloc[0]['reasoning']
            flaw_description = true_good_row.iloc[0]['flaw_description']
            
            fake_good_score = fake_good_row.iloc[0]['score'] if len(fake_good_row) > 0 else None
            fake_good_reasoning = fake_good_row.iloc[0]['reasoning'] if len(fake_good_row) > 0 else None
            difference = true_good_score - fake_good_score if fake_good_score is not None else None
            
            results.append({
                'category': category,
                'paper_folder': paper_folder,
                'flaw_id': flaw_id,
                'flaw_description': flaw_description,
                'true_good_score': true_good_score,
                'true_good_reasoning': true_good_reasoning,
                'fake_good_score': fake_good_score,
                'fake_good_reasoning': fake_good_reasoning,
                'difference': difference,
                'potentially_lazy': true_good_score < 5
            })
    
    if not results:
        return None
    
    return pd.DataFrame(results)

def create_plots(stats: Dict, output_dir: Path, model_name: str):
    """Create visualization plots by category."""
    if not stats or 'category_stats' not in stats or stats['category_stats'].empty:
        return
    
    cat_stats = stats['category_stats'].sort_values('category')
    categories = cat_stats['category'].values
    means = cat_stats['mean_difference'].values
    ci_lows = cat_stats['ci_low'].values
    ci_highs = cat_stats['ci_high'].values
    errors_low = means - ci_lows
    errors_high = ci_highs - means
    
    # -------------------------
    # Plot 1: Mean differences
    # -------------------------
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    # No title for paper figures
    
    x_pos = np.arange(len(categories))
    
    # Map category codes to descriptive labels
    category_labels = [CATEGORY_LABELS.get(cat, cat) for cat in categories]
    
    # Color bars: green for positive, red for negative
    colors = [COLOR_MATCH if m >= 0 else COLOR_MISMATCH for m in means]
    
    # Plot bars
    bars = ax1.bar(x_pos, means, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add error bars
    ax1.errorbar(x_pos, means, yerr=[errors_low, errors_high], 
                 fmt='none', color='black', capsize=5, capthick=2, linewidth=2)
    
    # Add white markers for means inside error bars
    ax1.scatter(x_pos, means, color=COLOR_WHITE, s=100, zorder=5, 
                edgecolors='black', linewidths=2, marker='o')
    
    # Add zero line
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax1.set_xlabel('Category', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean Score Difference (True Good - Fake Good)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(category_labels, rotation=45, ha='right', fontsize=12)
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Add value labels on bars
    for i, (bar, mean, ci_low, ci_high) in enumerate(zip(bars, means, ci_lows, ci_highs)):
        height = bar.get_height()
        label_y = height + (0.1 if height >= 0 else -0.3)
        ax1.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{mean:.2f}\n(n={cat_stats.iloc[i]["n_pairs"]})',
                ha='center', va='bottom' if height >= 0 else 'top', 
                fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plot_path_means = output_dir / "verification_results.png"
    plt.savefig(plot_path_means, bbox_inches='tight', dpi=300)
    plt.close(fig1)
    print(f"  ✅ Saved plot (mean differences): {plot_path_means}")
    
    # -------------------------
    # Plot 2: Cohen's d (effect size)
    # -------------------------
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    # No title for paper figures
    
    cohen_ds = cat_stats['cohen_d'].values
    d_ci_lows = cat_stats['cohen_d_ci_low'].values
    d_ci_highs = cat_stats['cohen_d_ci_high'].values
    d_err_low = cohen_ds - d_ci_lows
    d_err_high = d_ci_highs - cohen_ds
    
    # Color bars based on whether CI crosses zero:
    # White: if CI contains 0 (not significantly different from zero)
    # Green: if positive and CI doesn't contain 0 (significantly positive)
    # Red: if negative and CI doesn't contain 0 (significantly negative)
    colors_d = []
    for i, (d, ci_low, ci_high) in enumerate(zip(cohen_ds, d_ci_lows, d_ci_highs)):
        if ci_low <= 0 <= ci_high:
            # CI crosses zero - not significant
            colors_d.append(COLOR_WHITE)
        elif ci_low > 0:
            # CI entirely above zero - significantly positive
            colors_d.append(COLOR_MATCH)
        else:  # ci_high < 0
            # CI entirely below zero - significantly negative
            colors_d.append(COLOR_MISMATCH)
    
    bars2 = ax2.bar(x_pos, cohen_ds, color=colors_d, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add error bars for Cohen's d
    ax2.errorbar(x_pos, cohen_ds, yerr=[d_err_low, d_err_high],
                 fmt='none', color='black', capsize=5, capthick=2, linewidth=2)
    
    # Add zero line
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Error Category', fontsize=14, fontweight='bold')
    ax2.set_ylabel("Cohen's d (Effect Size)", fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(category_labels, rotation=45, ha='right', fontsize=12)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    plot_path_d = output_dir / "verification_effect_sizes.png"
    plt.savefig(plot_path_d, bbox_inches='tight', dpi=300)
    plt.close(fig2)
    print(f"  ✅ Saved plot (effect sizes): {plot_path_d}")

# --- Incremental Result Saving ---
def save_result_incrementally(result: Dict, results_csv: Path):
    """Append a single result to CSV file (thread-safe)."""
    with results_csv_lock:
        # Check if CSV exists and has headers
        if results_csv.exists():
            try:
                existing_df = pd.read_csv(results_csv)
                # Append new result
                new_df = pd.DataFrame([result])
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                # Drop duplicates (in case we're appending same result multiple times)
                combined_df.drop_duplicates(
                    subset=["category", "paper_folder", "flaw_id", "revision_type"],
                    keep="last",  # Keep the latest version if duplicate
                    inplace=True,
                )
            except Exception as e:
                # If reading fails, create new file
                new_df = pd.DataFrame([result])
                new_df.to_csv(results_csv, index=False)
                return
        else:
            # Create new CSV with this result
            new_df = pd.DataFrame([result])
            new_df.to_csv(results_csv, index=False)
            return
        
        # Save combined results
        combined_df.to_csv(results_csv, index=False)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully - set exit flag and print message."""
    if should_exit.is_set():
        # Second interrupt - force exit immediately
        print("\n\n⚠️ Force exit requested. Results saved so far.", flush=True)
        sys.exit(1)
    print("\n\n⚠️ Interrupt received! Saving completed results and exiting gracefully...", flush=True)
    print("   (Results are being saved incrementally. Press Ctrl+C again to force exit.)", flush=True)
    should_exit.set()
    # Don't exit immediately - let the main loop handle graceful shutdown

# --- Main Processing Functions ---
def process_category(
    data_dir: Path,
    category: str,
    model_name: str,
    comparison_type: str,
    output_dir: Path,
    max_workers: int = 5,
    request_delay: float = None,
    include_change_locations: bool = False,
    processed_keys: Optional[set] = None,
    results_csv: Optional[Path] = None,
    use_paid: bool = False,
    ablation_name: str = None,
) -> List[Dict]:
    """Process all papers in a category."""
    category_path = data_dir / "NeurIPS2024" / category
    
    if not category_path.exists():
        print(f"⚠️ Category {category} not found at {category_path}")
        return []
    
    # Determine which folders to compare
    if comparison_type == "true_good_vs_fake_good":
        original_folder = "planted_error"
        true_good_folder = "latest"
        fake_good_folder = "de-planted_error"
    else:
        raise ValueError(f"Unknown comparison type: {comparison_type}")
    
    planted_error_dir = category_path / original_folder
    if not planted_error_dir.exists():
        print(f"⚠️ {original_folder} folder not found for category {category} at {planted_error_dir}")
        return []
    
    results = []
    
    # Find all paper folders
    paper_folders = [d for d in planted_error_dir.iterdir() if d.is_dir()]
    print(f"Found {len(paper_folders)} paper folders in {original_folder}", flush=True)
    
    if len(paper_folders) == 0:
        print(f"  ⚠️ No paper folders found in {planted_error_dir}", flush=True)
        return []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for paper_folder in paper_folders:
            paper_folder_name = paper_folder.name
            
            # Read modifications summary - try multiple naming patterns
            # Pattern 1: {paper_folder_name}_modifications_summary.csv
            csv_path = paper_folder / f"{paper_folder_name}_modifications_summary.csv"
            if not csv_path.exists():
                # Pattern 2: Extract base name (before first underscore + numbers) and try
                # e.g., xjyU6zmZD7_2401_04486 -> xjyU6zmZD7
                base_name = paper_folder_name.split('_')[0] if '_' in paper_folder_name else paper_folder_name
                csv_path = paper_folder / f"{base_name}_modifications_summary.csv"
                if not csv_path.exists():
                    # Pattern 3: Find any CSV file matching the pattern
                    csv_files = list(paper_folder.glob("*_modifications_summary.csv"))
                    if csv_files:
                        csv_path = csv_files[0]
                    else:
                        continue
            
            flaw_info = read_modifications_summary(csv_path)
            if not flaw_info:
                continue
            
            flaw_id = flaw_info['flaw_id']
            
            # Read detailed change locations if flag is enabled
            change_details = None
            if include_change_locations:
                # Try to read fix_summary CSV from de-planted_error folder
                fake_good_base = category_path / fake_good_folder / paper_folder_name
                base_name = paper_folder_name.split('_')[0] if '_' in paper_folder_name else paper_folder_name
                fix_summary_path = fake_good_base / f"{base_name}_fix_summary.csv"
                
                if fix_summary_path.exists():
                    fix_summary_info = read_fix_summary(fix_summary_path)
                    if fix_summary_info and fix_summary_info.get('change_details'):
                        change_details = fix_summary_info['change_details']
            
            # Get original paper path (planted error version)
            original_path = paper_folder / "flawed_papers" / f"{flaw_id}.md"
            if not original_path.exists():
                # Try to find any .md file
                flawed_papers_dir = paper_folder / "flawed_papers"
                if flawed_papers_dir.exists():
                    md_files = list(flawed_papers_dir.glob("*.md"))
                    if md_files:
                        original_path = md_files[0]
                    else:
                        continue
                else:
                    continue
            
            # Process true good (camera ready)
            true_good_base = category_path / true_good_folder / paper_folder_name
            true_good_path = true_good_base / "structured_paper_output"
            
            if true_good_path.exists():
                key_true = (category, paper_folder_name, flaw_id, 'true_good')
                if not processed_keys or key_true not in processed_keys:
                    future = executor.submit(
                        process_paper_pair,
                        category=category,
                        paper_folder=paper_folder_name,
                        flaw_id=flaw_id,
                        original_paper_path=original_path,
                        revised_paper_path=true_good_path,
                        flaw_info=flaw_info,
                        output_dir=output_dir,
                        task_idx=len(futures),
                        model_name=model_name,
                        request_delay=request_delay,
                        change_details=change_details,
                        use_paid=use_paid,
                        ablation_name=ablation_name
                    )
                    future.revision_type = 'true_good'
                    future.paper_folder_name = paper_folder_name
                    future.flaw_id = flaw_id
                    futures.append(future)
            
            # Process fake good (de-planted error)
            fake_good_base = category_path / fake_good_folder / paper_folder_name
            fake_good_path = fake_good_base / "flawed_papers" / f"{flaw_id}.md"
            
            if fake_good_path.exists():
                key_fake = (category, paper_folder_name, flaw_id, 'fake_good')
                if not processed_keys or key_fake not in processed_keys:
                    future = executor.submit(
                        process_paper_pair,
                        category=category,
                        paper_folder=paper_folder_name,
                        flaw_id=flaw_id,
                        original_paper_path=original_path,
                        revised_paper_path=fake_good_path,
                        flaw_info=flaw_info,
                        output_dir=output_dir,
                        task_idx=len(futures),
                        model_name=model_name,
                        request_delay=request_delay,
                        change_details=change_details,
                        use_paid=use_paid,
                        ablation_name=ablation_name
                    )
                    future.revision_type = 'fake_good'
                    future.paper_folder_name = paper_folder_name
                    future.flaw_id = flaw_id
                    futures.append(future)
            else:
                # Try to find any .md file
                fake_flawed_papers_dir = fake_good_base / "flawed_papers"
                if fake_flawed_papers_dir.exists():
                    md_files = list(fake_flawed_papers_dir.glob("*.md"))
                    if md_files:
                        fake_good_path = md_files[0]
                        key_fake = (category, paper_folder_name, flaw_id, 'fake_good')
                        if not processed_keys or key_fake not in processed_keys:
                            future = executor.submit(
                                process_paper_pair,
                                category=category,
                                paper_folder=paper_folder_name,
                                flaw_id=flaw_id,
                                original_paper_path=original_path,
                                revised_paper_path=fake_good_path,
                                flaw_info=flaw_info,
                                output_dir=output_dir,
                                task_idx=len(futures),
                                model_name=model_name,
                                request_delay=request_delay,
                                change_details=change_details,
                                use_paid=use_paid,
                                ablation_name=ablation_name
                            )
                            future.revision_type = 'fake_good'
                            future.paper_folder_name = paper_folder_name
                            future.flaw_id = flaw_id
                            futures.append(future)
        
        print(f"  Submitted {len(futures)} tasks for processing", flush=True)
        
        # Collect results
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {category}"):
            # Check if we should exit gracefully
            if should_exit.is_set():
                tqdm.write(f"  ⚠️ Exiting gracefully due to interrupt signal...")
                # Cancel remaining futures if possible
                for f in futures:
                    f.cancel()
                break
            
            result = future.result()
            if result:
                result['revision_type'] = future.revision_type
                results.append(result)
                
                # Save incrementally to CSV if provided
                if results_csv is not None:
                    try:
                        save_result_incrementally(result, results_csv)
                    except Exception as e:
                        tqdm.write(f"  ⚠️ Warning: Could not save result incrementally: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Verify LLM capability to distinguish between real revisions and LLM-generated fixes."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing NeurIPS2024 data"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-2.0-flash-lite",
        help="Gemini model name to use"
    )
    parser.add_argument(
        "--comparison_type",
        type=str,
        default="true_good_vs_fake_good",
        choices=["true_good_vs_fake_good"],
        help="Type of comparison to perform"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Categories to process (default: all found in data_dir)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Maximum number of parallel workers"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: data_dir/llm_verification_{model_name})"
    )
    parser.add_argument(
        "--include_change_locations",
        action="store_true",
        help="Include detailed change locations from fix_summary CSV files to help LLM focus on specific sections"
    )
    parser.add_argument(
        "--detect_lazy_authors",
        action="store_true",
        help="Detect cases where camera-ready version might not actually fix the error (true_good score < 5)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set, reuse existing verification_scores.csv and only process missing paper/revision pairs"
    )
    parser.add_argument(
        "--use_paid",
        action="store_true",
        help="Use GEMINI_API_KEY_PAID (no rate limiting, 4 workers max)"
    )
    parser.add_argument(
        "--ablation_name",
        type=str,
        default=None,
        help="Name of ablation study (e.g., 'no_location'). Output folder will be llm_verification_{model_name}_{ablation_name}"
    )
    
    args = parser.parse_args()
    
    if not GENAI_AVAILABLE:
        raise ImportError("google-generativeai is required. Install with: pip install google-generativeai")
    
    # Handle --use_paid flag
    use_paid = args.use_paid
    if use_paid:
        if not GEMINI_API_KEY_PAID:
            raise ValueError("--use_paid specified but GEMINI_API_KEY_PAID not found in environment variables")
        global USE_PAID_KEY
        USE_PAID_KEY = True
        print("✅ Using PAID API key (no rate limiting, 4 workers)")
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        model_name_safe = args.model_name.replace('.', '_').replace('-', '_')
        if args.ablation_name:
            output_dir = data_dir / f"llm_verification_{model_name_safe}_{args.ablation_name}"
        else:
            output_dir = data_dir / f"llm_verification_{model_name_safe}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Register signal handler for graceful shutdown (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    print("ℹ️  Press Ctrl+C to gracefully save progress and exit\n")
    
    # Find categories - data structure: data_dir/NeurIPS2024/{category_id}/
    neurips_path = data_dir / "NeurIPS2024"
    if not neurips_path.exists():
        raise ValueError(f"NeurIPS2024 directory not found in {data_dir}. Expected: {neurips_path}")
    
    print(f"✅ Found NeurIPS2024 directory: {neurips_path}", flush=True)
    
    if args.categories:
        categories = args.categories
    else:
        categories = [d.name for d in neurips_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
        categories.sort()
    
    print(f"✅ Found {len(categories)} categories: {categories}", flush=True)
    
    # Verify category structure (only show if there are issues)
    for cat in categories:
        cat_path = neurips_path / cat
        if cat_path.exists():
            planted_error_path = cat_path / "planted_error"
            latest_path = cat_path / "latest"
            deplanted_path = cat_path / "de-planted_error"
            if not (planted_error_path.exists() and latest_path.exists() and deplanted_path.exists()):
                print(f"⚠️ Category {cat} missing some folders:", flush=True)
                print(f"    - planted_error: {'✅' if planted_error_path.exists() else '❌'}", flush=True)
                print(f"    - latest: {'✅' if latest_path.exists() else '❌'}", flush=True)
                print(f"    - de-planted_error: {'✅' if deplanted_path.exists() else '❌'}", flush=True)
    
    print("="*80, flush=True)
    print("Revision Quality Verification", flush=True)
    print("="*80, flush=True)
    print(f"Data directory: {data_dir}", flush=True)
    print(f"Model: {args.model_name}", flush=True)
    if args.ablation_name:
        print(f"Ablation: {args.ablation_name}", flush=True)
    print(f"Comparison type: {args.comparison_type}", flush=True)
    print(f"Categories to process: {categories}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(flush=True)
    
    # Set request delay and max workers based on paid key usage
    if use_paid:
        request_delay = 0.0  # No rate limiting for paid key
        max_workers = 4  # Use 4 workers for paid key
        print(f"Request delay: 0.00 seconds (PAID key - no rate limiting)", flush=True)
        print(f"Max workers: {max_workers} (PAID key)", flush=True)
    else:
        request_delay = get_request_delay_for_model(args.model_name)
        max_workers = args.max_workers
        print(f"Request delay: {request_delay:.2f} seconds (RPM limit: {GEMINI_MODEL_RPM_LIMITS.get(args.model_name, 30)})", flush=True)
    print(flush=True)
    
    # If resuming, load existing results and build set of processed keys
    existing_df: Optional[pd.DataFrame] = None
    processed_keys = set()
    results_csv = output_dir / "verification_scores.csv"
    if args.resume and results_csv.exists():
        try:
            existing_df = pd.read_csv(results_csv)
            for _, row in existing_df.iterrows():
                rev_type = row.get("revision_type", "unknown")
                processed_keys.add(
                    (row["category"], row["paper_folder"], row["flaw_id"], rev_type)
                )
            print(f"Resuming from existing results: {len(existing_df)} rows, {len(processed_keys)} distinct paper/revision pairs")
        except Exception as e:
            print(f"⚠️ Could not read existing results from {results_csv}: {e}")
            existing_df = None
            processed_keys = set()
    
    # Process all categories (only missing pairs if --resume is enabled)
    all_results = []
    for category in categories:
        # Check if we should exit gracefully
        if should_exit.is_set():
            print("\n⚠️ Exiting gracefully due to interrupt signal...", flush=True)
            break
        
        print(f"Processing category: {category}", flush=True)
        results = process_category(
            data_dir=data_dir,
            category=category,
            model_name=args.model_name,
            comparison_type=args.comparison_type,
            output_dir=output_dir,
            max_workers=max_workers,  # Use computed max_workers (4 for paid, default otherwise)
            request_delay=request_delay,
            include_change_locations=args.include_change_locations,
            processed_keys=processed_keys,
            results_csv=results_csv,  # Pass CSV path for incremental saving
            use_paid=use_paid,
            ablation_name=args.ablation_name
        )
        all_results.extend(results)
        print(f"  Completed {category}: {len(results)} results")
        print()
    
    # Convert new results to DataFrame
    if all_results:
        df_new = pd.DataFrame(all_results)
    else:
        df_new = pd.DataFrame()
        print("No new results collected in this run.")
    
    # Merge with existing results if resuming, or use existing results if no new ones
    if existing_df is not None:
        if not df_new.empty:
            combined_df = pd.concat([existing_df, df_new], ignore_index=True)
            # Drop duplicates based on unique key
            combined_df.drop_duplicates(
                subset=["category", "paper_folder", "flaw_id", "revision_type"],
                keep="first",
                inplace=True,
            )
        else:
            # No new results, but use existing results for plotting
            combined_df = existing_df
            print(f"Using existing results ({len(combined_df)} rows) for statistics and plotting...")
    else:
        if df_new.empty:
            print("No results available! Cannot generate plots.")
            return
        combined_df = df_new
    
    # Check if we have any data to work with
    if combined_df.empty:
        print("⚠️ No results available! Cannot generate plots.")
        return
    
    # Save raw results (combined)
    combined_df.to_csv(results_csv, index=False)
    print(f"✅ Saved raw results: {results_csv}")
    
    # Save full LLM responses for further analysis (based on combined results)
    llm_responses = []
    for _, row in combined_df.iterrows():
        llm_responses.append({
            'category': row['category'],
            'paper_folder': row['paper_folder'],
            'flaw_id': row['flaw_id'],
            'revision_type': row.get('revision_type', 'unknown'),
            'flaw_description': row['flaw_description'],
            'score': row['score'],
            'reasoning': row['reasoning']
        })
    
    responses_json = output_dir / "llm_responses.json"
    with open(responses_json, 'w', encoding='utf-8') as f:
        json.dump(llm_responses, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved full LLM responses: {responses_json}")
    
    # Compute statistics on combined results
    print("\nComputing statistics...")
    stats = compute_treatment_effect(combined_df)
    
    if stats:
        print(f"\nOverall Treatment Effect Statistics:")
        print(f"  Number of pairs: {stats['n_pairs']}")
        print(f"  Mean difference (True Good - Fake Good): {stats['mean_difference']:.3f}")
        print(f"  Standard deviation of differences: {stats['std_difference']:.3f}")
        print(f"  Treatment effect (mean / std): {stats['treatment_effect']:.3f}")
        print(f"  Cohen's d: {stats['cohen_d']:.3f}")
        print(f"  95% CI: [{stats['ci_low']:.3f}, {stats['ci_high']:.3f}]")
        
        # Print per-category statistics
        if 'category_stats' in stats and not stats['category_stats'].empty:
            print(f"\nPer-Category Statistics:")
            cat_stats = stats['category_stats'].sort_values('category')
            for _, row in cat_stats.iterrows():
                print(f"  {row['category']}:")
                print(f"    n={row['n_pairs']}, mean_diff={row['mean_difference']:.3f}, "
                      f"Cohen's d={row['cohen_d']:.3f}, "
                      f"95% CI=[{row['ci_low']:.3f}, {row['ci_high']:.3f}]")
        
        # Save statistics
        stats_json = output_dir / "statistics.json"
        stats_to_save = {k: v for k, v in stats.items() if k not in ['detailed_results', 'category_stats']}
        stats_to_save['detailed_results'] = stats['detailed_results'].to_dict('records')
        if 'category_stats' in stats and not stats['category_stats'].empty:
            stats_to_save['category_stats'] = stats['category_stats'].to_dict('records')
        with open(stats_json, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
        print(f"✅ Saved statistics: {stats_json}")
        
        # Save detailed differences
        diff_csv = output_dir / "score_differences.csv"
        stats['detailed_results'].to_csv(diff_csv, index=False)
        print(f"✅ Saved differences: {diff_csv}")
        
        # Save category statistics
        if 'category_stats' in stats and not stats['category_stats'].empty:
            cat_csv = output_dir / "category_statistics.csv"
            stats['category_stats'].to_csv(cat_csv, index=False)
            print(f"✅ Saved category statistics: {cat_csv}")
        
        # Create plots
        print("\nGenerating plots...")
        create_plots(stats, output_dir, args.model_name)
    else:
        print("⚠️ Could not compute statistics (insufficient paired data)")
    
    # Detect lazy authors (true_good score < 5) - independent of stats computation
    if args.detect_lazy_authors:
        print("\nAnalyzing for lazy authors (true_good score < 5)...")
        lazy_analysis = detect_lazy_authors(combined_df)
        
        if lazy_analysis is not None and not lazy_analysis.empty:
            lazy_csv = output_dir / "lazy_authors_detection.csv"
            lazy_analysis.to_csv(lazy_csv, index=False)
            print(f"✅ Saved lazy authors detection: {lazy_csv}")
            
            print(f"\nLazy Authors Detection Summary:")
            print(f"  Total papers with true_good scores: {len(lazy_analysis)}")
            low_scores = lazy_analysis[lazy_analysis['true_good_score'] < 5]
            print(f"  Papers with true_good score < 5: {len(low_scores)} ({len(low_scores)/len(lazy_analysis)*100:.1f}%)")
            
            if len(low_scores) > 0:
                print(f"\n  Papers flagged as potentially lazy:")
                for _, row in low_scores.iterrows():
                    fake_good_str = f"{row['fake_good_score']:.1f}" if pd.notna(row['fake_good_score']) else 'N/A'
                    diff_str = f"{row['difference']:.1f}" if pd.notna(row['difference']) else 'N/A'
                    # Construct full path from the root data directory to the planted error paper
                    paper_rel_path = (
                        f"NeurIPS2024/{row['category']}/planted_error/"
                        f"{row['paper_folder']}/flawed_papers/{row['flaw_id']}.md"
                    )
                    paper_full_path = data_dir / paper_rel_path
                    print(f"    - {paper_full_path}: "
                          f"true_good={row['true_good_score']:.1f}, "
                          f"fake_good={fake_good_str}, "
                          f"diff={diff_str}")
        else:
            print("⚠️ Could not detect lazy authors (insufficient data)")
    
    print("\n" + "="*80)
    print("Verification complete!")
    print("="*80)

if __name__ == "__main__":
    main()

