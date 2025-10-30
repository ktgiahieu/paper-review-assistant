import os
import csv
import json
import argparse
import time
import re
import pandas as pd
import anthropic
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("WARNING: google-generativeai not installed. Install with: pip install google-generativeai")

# --- Environment & API Configuration ---
load_dotenv()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- Constants ---
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2
RETRYABLE_STATUS_CODES = [429, 500, 502, 503, 504]

# --- Pydantic Model for Structured Review Output ---
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
    def get_user_prompt(paper_content: str, paper_version: str, flaw_context: Optional[str] = None, format_type: str = "default") -> str:
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
        
        if format_type == "CriticalNeurIPS":
            return f"""Please review the following research paper with exceptional rigor and depth.

<paper_content>
{paper_content}
</paper_content>
{flaw_info}

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
{flaw_info}

Provide a comprehensive review following the specified JSON format.
"""

def _sanitize_json_string(json_str: str) -> str:
    """Cleans common JSON errors from LLM output."""
    json_str = json_str.strip().strip("```json").strip("```")
    json_str = re.sub(r',\s*(?=[}\]])', '', json_str)
    return json_str

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
    version_label: str,
    flaw_descriptions: list,
    api_client,  # Can be anthropic.Anthropic or genai model
    model_name: str,
    verbose: bool,
    run_id: int = 0,
    api_type: str = "anthropic",  # "anthropic" or "gemini"
    format_type: str = "default"  # "default" or "CriticalNeurIPS"
) -> dict:
    """
    Reviews a single paper version and returns structured results.
    
    Args:
        api_type: "anthropic" or "gemini"
        format_type: "default" or "CriticalNeurIPS"
    """
    worker_id = concurrent.futures.thread.get_ident() if hasattr(concurrent.futures.thread, 'get_ident') else os.getpid()
    _print_method = tqdm.write if not verbose else print
    
    try:
        # Read paper content
        paper_md = find_paper_markdown(paper_path)
        if not paper_md:
            return {
                "error": f"Could not find paper markdown for {paper_id} at {paper_path}",
                "paper_id": paper_id,
                "version": version_label,
                "run_id": run_id,
                "model_type": f"{api_type}_{format_type}",
                "success": False
            }
        
        with open(paper_md, 'r', encoding='utf-8') as f:
            paper_content = f.read()
        
        # Prepare flaw context if available
        flaw_context = None
        if flaw_descriptions:
            flaw_context = "\n".join([f"- {flaw}" for flaw in flaw_descriptions])
        
        system_prompt = ReviewPrompts.get_system_prompt(format_type)
        user_prompt_text = ReviewPrompts.get_user_prompt(paper_content, version_label, flaw_context, format_type)
        
        response_text = None
        last_exception = None
        
        for attempt in range(MAX_RETRIES):
            try:
                if verbose:
                    _print_method(f"Worker {worker_id}: Reviewing {paper_id} ({version_label}), attempt {attempt + 1}/{MAX_RETRIES}")
                
                if api_type == "anthropic":
                    response_obj = api_client.messages.create(
                        model=model_name,
                        max_tokens=4096,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt_text}],
                        timeout=300.0,
                    )
                    response_text = response_obj.content[0].text
                    
                elif api_type == "gemini":
                    # Gemini API call
                    # For Gemini, combine system and user prompts
                    full_prompt = f"{system_prompt}\n\n{user_prompt_text}"
                    response_obj = api_client.generate_content(full_prompt)
                    response_text = response_obj.text
                else:
                    raise ValueError(f"Unknown API type: {api_type}")
                
                if verbose:
                    _print_method(f"Worker {worker_id}: Successfully reviewed {paper_id} ({version_label})")
                break
                
            except anthropic.APIStatusError as e:
                last_exception = e
                if e.status_code in RETRYABLE_STATUS_CODES:
                    wait_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                    _print_method(f"Worker {worker_id}: Retrying {paper_id} ({version_label}) in {wait_time}s due to status {e.status_code}...")
                    time.sleep(wait_time)
                else:
                    _print_method(f"Worker {worker_id}: Non-retryable error for {paper_id} ({version_label}): {e}")
                    break
                    
            except Exception as e:
                last_exception = e
                _print_method(f"Worker {worker_id}: Unexpected error for {paper_id} ({version_label}): {e}")
                wait_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                time.sleep(wait_time)
        
        if response_text is None:
            err_msg = f"All API attempts failed for {paper_id} ({version_label})."
            if last_exception:
                err_msg += f" Last error: {type(last_exception).__name__} - {last_exception}"
            return {
                "error": err_msg,
                "paper_id": paper_id,
                "version": version_label,
                "run_id": run_id,
                "model_type": f"{api_type}_{format_type}",
                "success": False
            }
        
        raw_json_content = response_text
        sanitized_json_content = _sanitize_json_string(raw_json_content)
        
        model_type_label = f"{api_type}_{format_type}"
        
        try:
            # Choose the appropriate Pydantic model based on format
            if format_type == "CriticalNeurIPS":
                parsed_review = CriticalNeurIPSReview.model_validate_json(sanitized_json_content)
                review_data = parsed_review.model_dump()
                
                # Map scores for compatibility with evaluation scripts
                review_data["rating"] = review_data.get("overall_score")
                # soundness, presentation, contribution already match
                
            else:
                # Default format
                parsed_review = PaperReview.model_validate_json(sanitized_json_content)
                review_data = parsed_review.model_dump()
                
                # Add score mappings for evaluate_numerical_scores.py compatibility
                review_data["soundness"] = review_data.get("technical_quality_score")
                review_data["presentation"] = review_data.get("clarity_score")
                review_data["contribution"] = review_data.get("novelty_score")
                review_data["rating"] = review_data.get("overall_score")
            
            review_data["paper_id"] = paper_id
            review_data["version"] = version_label
            review_data["run_id"] = run_id
            review_data["model_type"] = model_type_label
            review_data["success"] = True
            
            return review_data
            
        except Exception as pydantic_error:
            _print_method(f"Worker {worker_id}: Pydantic validation failed for {paper_id} ({version_label}). Error: {pydantic_error}")
            _print_method(f"Worker {worker_id}: Raw JSON: {sanitized_json_content[:500]}...")
            
            try:
                fallback_data = json.loads(sanitized_json_content)
                fallback_data["paper_id"] = paper_id
                fallback_data["version"] = version_label
                fallback_data["run_id"] = run_id
                fallback_data["model_type"] = model_type_label
                fallback_data["success"] = True
                fallback_data["__pydantic_validation_error"] = str(pydantic_error)
                
                # Add score mappings for evaluator compatibility
                if format_type == "CriticalNeurIPS":
                    fallback_data["rating"] = fallback_data.get("overall_score")
                else:
                    fallback_data["soundness"] = fallback_data.get("technical_quality_score")
                    fallback_data["presentation"] = fallback_data.get("clarity_score")
                    fallback_data["contribution"] = fallback_data.get("novelty_score")
                    fallback_data["rating"] = fallback_data.get("overall_score")
                
                return fallback_data
            except json.JSONDecodeError as json_e:
                return {
                    "error": "Failed to parse JSON from LLM",
                    "paper_id": paper_id,
                    "version": version_label,
                    "run_id": run_id,
                    "model_type": model_type_label,
                    "raw_content": raw_json_content[:1000],
                    "pydantic_error": str(pydantic_error),
                    "json_error": str(json_e),
                    "success": False
                }
    
    except Exception as e:
        message = f"FATAL ERROR reviewing {paper_id} ({version_label}): {type(e).__name__} - {e}"
        _print_method(f"Worker {worker_id}: {message}")
        import traceback
        _print_method(traceback.format_exc())
        return {
            "error": message,
            "paper_id": paper_id,
            "version": version_label,
            "run_id": run_id,
            "model_type": f"{api_type}_{format_type}",
            "success": False
        }

def review_paper_pair(
    pair_row: dict,
    api_client,  # anthropic.Anthropic or genai model
    model_name: str,
    output_dir: Path,
    verbose: bool,
    version_filter: str = "both",
    skip_existing: bool = False,
    num_runs: int = 1,
    api_type: str = "anthropic",
    format_type: str = "default"
) -> str:
    """
    Reviews both versions of a paper pair and saves results.
    
    Args:
        version_filter: Which version(s) to review - "v1", "latest", or "both"
        skip_existing: If True, skip reviewing papers that already have review files
        num_runs: Number of times to run each review (for variance analysis)
    """
    paper_id = pair_row['paperid']
    v1_folder = Path(pair_row['v1_folder_path'])
    latest_folder = Path(pair_row['latest_folder_path'])
    
    # Create paper output directory
    paper_output_dir = output_dir / paper_id
    paper_output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    all_reviews_success = True
    
    # Review v1 (if needed) - multiple runs
    if version_filter in ["v1", "both"]:
        for run_id in range(num_runs):
            v1_review_path = paper_output_dir / f"v1_review_run{run_id}.json"
            
            if skip_existing and v1_review_path.exists():
                if verbose:
                    tqdm.write(f"Skipping v1 review for {paper_id} run {run_id} (already exists)")
            else:
                v1_review = review_single_paper(
                    paper_id, v1_folder, "v1", flaw_descriptions,
                    api_client, model_name, verbose, run_id,
                    api_type=api_type, format_type=format_type
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
                latest_review = review_single_paper(
                    paper_id, latest_folder, "latest", flaw_descriptions,
                    api_client, model_name, verbose, run_id,
                    api_type=api_type, format_type=format_type
                )
                
                with open(latest_review_path, 'w', encoding='utf-8') as f:
                    json.dump(latest_review, f, ensure_ascii=False, indent=2)
                
                if not latest_review.get("success", False):
                    all_reviews_success = False
    
    return f"Successfully reviewed pair {paper_id}" if all_reviews_success else f"Partial/failed review for {paper_id}"

def main():
    parser = argparse.ArgumentParser(description="Review paper pairs (v1 vs latest) using LLM.")
    parser.add_argument("--csv_file", type=str, required=True, 
                       help="Path to filtered_pairs.csv file.")
    parser.add_argument("--output_dir", type=str, default="./pair_reviews/",
                       help="Output directory for review results.")
    parser.add_argument("--model_name", type=str, default="claude-haiku-4-5-20251001",
                       help="Model name (Anthropic or Gemini model name).")
    parser.add_argument("--api", type=str, choices=["anthropic", "gemini"], default="anthropic",
                       help="API to use: 'anthropic' or 'gemini' (default: anthropic).")
    parser.add_argument("--format", type=str, choices=["default", "CriticalNeurIPS"], default="default",
                       help="Review format: 'default' or 'CriticalNeurIPS' (default: default).")
    parser.add_argument("--max_workers", type=int, default=3,
                       help="Max worker threads for concurrent processing.")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output.")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of pairs to process (for testing).")
    parser.add_argument("--num_runs", type=int, default=1,
                       help="Number of times to review each paper (for variance analysis). Default: 1")
    
    # Debug/Save API Credit Options
    parser.add_argument("--version", type=str, choices=["v1", "latest", "both"], default="both",
                       help="Which version(s) to review: 'v1', 'latest', or 'both' (default: both). Use 'v1' or 'latest' to save API credits.")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip papers that already have review files (saves API credits on reruns).")
    
    args = parser.parse_args()
    
    # Initialize API client based on selected API
    if args.api == "anthropic":
        if not ANTHROPIC_API_KEY:
            print("Error: ANTHROPIC_API_KEY not found in environment variables or .env file.")
            exit(1)
        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        except Exception as e:
            print(f"Error initializing Anthropic client: {e}")
            exit(1)
            
    elif args.api == "gemini":
        if not GENAI_AVAILABLE:
            print("Error: google-generativeai library not installed.")
            print("Install with: pip install google-generativeai")
            exit(1)
        if not GEMINI_API_KEY:
            print("Error: GEMINI_API_KEY not found in environment variables or .env file.")
            exit(1)
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            client = genai.GenerativeModel(args.model_name)
        except Exception as e:
            print(f"Error initializing Gemini client: {e}")
            exit(1)
    else:
        print(f"Error: Unknown API type: {args.api}")
        exit(1)
    
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
    
    print(f"Preparing to review {len(df)} paper pairs...")
    print(f"API: {args.api}")
    print(f"Model: {args.model_name}")
    print(f"Format: {args.format}")
    print(f"Version filter: {args.version}")
    print(f"Number of runs per paper: {args.num_runs}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Output directory: {output_dir}")
    print(f"Max workers: {args.max_workers}")
    
    # Calculate estimated API calls
    estimated_calls = len(df) * args.num_runs
    if args.version == "both":
        estimated_calls *= 2
    print(f"Estimated API calls: {estimated_calls} (max, may be less with --skip_existing)\n")
    
    # Prepare tasks
    tasks = []
    for _, row in df.iterrows():
        tasks.append((row.to_dict(), client, args.model_name, output_dir, args.verbose, args.version, args.skip_existing, args.num_runs, args.api, args.format))
    
    # Process with thread pool
    processed_count = 0
    error_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {
            executor.submit(review_paper_pair, task[0], task[1], task[2], task[3], task[4], task[5], task[6], task[7], task[8], task[9]): task 
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
    
    print("\n--- Processing Complete ---")
    print(f"Total pairs successfully reviewed: {processed_count}")
    print(f"Failed/Errored reviews: {error_count}")
    print(f"Results saved in: {output_dir}")
    print("---------------------------\n")

if __name__ == "__main__":
    main()

# Example usage:
"""
# Full production run (both versions, 3 runs each for variance analysis):
python review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./data/ICLR2024_pairs/reviews_anthropic" \
  --model_name "claude-haiku-4-5-20251001" \
  --num_runs 3 \
  --max_workers 3 \
  --verbose

# Single run (fastest, no variance analysis):
python review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./data/ICLR2024_pairs/reviews_anthropic" \
  --model_name "claude-haiku-4-5-20251001" \
  --max_workers 3

# DEBUG MODE - Test with 1 paper, only v1 version (saves API credits):
python review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_test" \
  --model_name "claude-haiku-4-5-20251001" \
  --version v1 \
  --limit 1 \
  --verbose

# Continue from previous run - only review what's missing (saves API credits):
python review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./data/ICLR2024_pairs/reviews_anthropic" \
  --model_name "claude-haiku-4-5-20251001" \
  --version latest \
  --num_runs 3 \
  --skip_existing \
  --max_workers 3

# Complete testing workflow:
# Step 1: Test with 1 paper, v1 only
python review_paper_pairs.py --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" --output_dir "./reviews_test" --version v1 --limit 1

# Step 2: If successful, add latest version to the same paper
python review_paper_pairs.py --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" --output_dir "./reviews_test" --version latest --limit 1 --skip_existing

# Step 3: Expand to more papers with multiple runs
python review_paper_pairs.py --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" --output_dir "./reviews_test" --version both --limit 10 --num_runs 3 --skip_existing

# Step 4: Full run
python review_paper_pairs.py --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" --output_dir "./data/ICLR2024_pairs/reviews_anthropic" --version both --num_runs 3 --skip_existing

# Step 5: Analyze results with evaluator
python evaluate_numerical_scores.py --reviews_dir ./data/ICLR2024_pairs/reviews_anthropic --output_dir ./evaluation_anthropic
"""

