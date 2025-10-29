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

# --- Environment & API Configuration ---
load_dotenv()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

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

class ReviewPrompts:
    @staticmethod
    def get_system_prompt() -> str:
        """Returns the system prompt for paper review."""
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
    def get_user_prompt(paper_content: str, paper_version: str, flaw_context: Optional[str] = None) -> str:
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
    anthropic_client: anthropic.Anthropic,
    model_name: str,
    verbose: bool
) -> dict:
    """
    Reviews a single paper version and returns structured results.
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
                "version": version_label
            }
        
        with open(paper_md, 'r', encoding='utf-8') as f:
            paper_content = f.read()
        
        # Prepare flaw context if available
        flaw_context = None
        if flaw_descriptions:
            flaw_context = "\n".join([f"- {flaw}" for flaw in flaw_descriptions])
        
        system_prompt = ReviewPrompts.get_system_prompt()
        user_prompt_text = ReviewPrompts.get_user_prompt(paper_content, version_label, flaw_context)
        
        response_obj = None
        last_exception = None
        
        for attempt in range(MAX_RETRIES):
            try:
                if verbose:
                    _print_method(f"Worker {worker_id}: Reviewing {paper_id} ({version_label}), attempt {attempt + 1}/{MAX_RETRIES}")
                
                response_obj = anthropic_client.messages.create(
                    model=model_name,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt_text}],
                    timeout=300.0,
                )
                
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
        
        if response_obj is None or not response_obj.content:
            err_msg = f"All API attempts failed for {paper_id} ({version_label})."
            if last_exception:
                err_msg += f" Last error: {type(last_exception).__name__} - {last_exception}"
            return {
                "error": err_msg,
                "paper_id": paper_id,
                "version": version_label
            }
        
        raw_json_content = response_obj.content[0].text
        sanitized_json_content = _sanitize_json_string(raw_json_content)
        
        try:
            parsed_review = PaperReview.model_validate_json(sanitized_json_content)
            review_data = parsed_review.model_dump()
            review_data["paper_id"] = paper_id
            review_data["version"] = version_label
            review_data["success"] = True
            return review_data
            
        except Exception as pydantic_error:
            _print_method(f"Worker {worker_id}: Pydantic validation failed for {paper_id} ({version_label}). Error: {pydantic_error}")
            _print_method(f"Worker {worker_id}: Raw JSON: {sanitized_json_content[:500]}...")
            
            try:
                fallback_data = json.loads(sanitized_json_content)
                fallback_data["paper_id"] = paper_id
                fallback_data["version"] = version_label
                fallback_data["success"] = True
                fallback_data["__pydantic_validation_error"] = str(pydantic_error)
                return fallback_data
            except json.JSONDecodeError as json_e:
                return {
                    "error": "Failed to parse JSON from LLM",
                    "paper_id": paper_id,
                    "version": version_label,
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
            "success": False
        }

def review_paper_pair(
    pair_row: dict,
    anthropic_client: anthropic.Anthropic,
    model_name: str,
    output_dir: Path,
    verbose: bool,
    version_filter: str = "both",
    skip_existing: bool = False
) -> str:
    """
    Reviews both versions of a paper pair and saves results.
    
    Args:
        version_filter: Which version(s) to review - "v1", "latest", or "both"
        skip_existing: If True, skip reviewing papers that already have review files
    """
    paper_id = pair_row['paperid']
    v1_folder = Path(pair_row['v1_folder_path'])
    latest_folder = Path(pair_row['latest_folder_path'])
    
    # Check if reviews already exist
    paper_output_dir = output_dir / paper_id
    v1_review_path = paper_output_dir / "v1_review.json"
    latest_review_path = paper_output_dir / "latest_review.json"
    
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
    
    # Review v1 (if needed)
    v1_review = None
    if version_filter in ["v1", "both"]:
        if skip_existing and v1_review_path.exists():
            if verbose:
                tqdm.write(f"Skipping v1 review for {paper_id} (already exists)")
            with open(v1_review_path, 'r') as f:
                v1_review = json.load(f)
        else:
            v1_review = review_single_paper(
                paper_id, v1_folder, "v1", flaw_descriptions,
                anthropic_client, model_name, verbose
            )
    
    # Review latest (if needed)
    latest_review = None
    if version_filter in ["latest", "both"]:
        if skip_existing and latest_review_path.exists():
            if verbose:
                tqdm.write(f"Skipping latest review for {paper_id} (already exists)")
            with open(latest_review_path, 'r') as f:
                latest_review = json.load(f)
        else:
            latest_review = review_single_paper(
                paper_id, latest_folder, "latest", flaw_descriptions,
                anthropic_client, model_name, verbose
            )
    
    # Save individual reviews (only if they were generated/loaded)
    paper_output_dir.mkdir(parents=True, exist_ok=True)
    
    if v1_review is not None:
        with open(paper_output_dir / "v1_review.json", 'w', encoding='utf-8') as f:
            json.dump(v1_review, f, ensure_ascii=False, indent=2)
    
    if latest_review is not None:
        with open(paper_output_dir / "latest_review.json", 'w', encoding='utf-8') as f:
            json.dump(latest_review, f, ensure_ascii=False, indent=2)
    
    # Create comparison summary (only if both reviews are available)
    if v1_review is not None and latest_review is not None:
        comparison = {
            "paper_id": paper_id,
            "v1": v1_review,
            "latest": latest_review,
            "score_changes": {}
        }
        
        # Calculate score changes if both reviews succeeded
        if v1_review.get("success") and latest_review.get("success"):
            for score_field in ["clarity_score", "novelty_score", "technical_quality_score", 
                               "experimental_rigor_score", "overall_score"]:
                if score_field in v1_review and score_field in latest_review:
                    v1_score = v1_review[score_field]
                    latest_score = latest_review[score_field]
                    comparison["score_changes"][score_field] = {
                        "v1": v1_score,
                        "latest": latest_score,
                        "change": latest_score - v1_score
                    }
        
        with open(paper_output_dir / "comparison.json", 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    # Determine success
    v1_success = v1_review.get("success", False) if v1_review else True  # True if not requested
    latest_success = latest_review.get("success", False) if latest_review else True
    success = v1_success and latest_success
    
    return f"Successfully reviewed pair {paper_id}" if success else f"Partial/failed review for {paper_id}"

def main():
    parser = argparse.ArgumentParser(description="Review paper pairs (v1 vs latest) using LLM.")
    parser.add_argument("--csv_file", type=str, required=True, 
                       help="Path to filtered_pairs.csv file.")
    parser.add_argument("--output_dir", type=str, default="./pair_reviews/",
                       help="Output directory for review results.")
    parser.add_argument("--model_name", type=str, default="claude-haiku-4-5-20251001",
                       help="Anthropic model name to use.")
    parser.add_argument("--max_workers", type=int, default=3,
                       help="Max worker threads for concurrent processing.")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output.")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of pairs to process (for testing).")
    
    # Debug/Save API Credit Options
    parser.add_argument("--version", type=str, choices=["v1", "latest", "both"], default="both",
                       help="Which version(s) to review: 'v1', 'latest', or 'both' (default: both). Use 'v1' or 'latest' to save API credits.")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip papers that already have review files (saves API credits on reruns).")
    
    args = parser.parse_args()
    
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not found in environment variables or .env file.")
        exit(1)
    
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except Exception as e:
        print(f"Error initializing Anthropic client: {e}")
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
    print(f"Model: {args.model_name}")
    print(f"Version filter: {args.version}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Output directory: {output_dir}")
    print(f"Max workers: {args.max_workers}")
    
    # Calculate estimated API calls
    estimated_calls = len(df)
    if args.version == "both":
        estimated_calls *= 2
    print(f"Estimated API calls: {estimated_calls} (max, may be less with --skip_existing)\n")
    
    # Prepare tasks
    tasks = []
    for _, row in df.iterrows():
        tasks.append((row.to_dict(), client, args.model_name, output_dir, args.verbose, args.version, args.skip_existing))
    
    # Process with thread pool
    processed_count = 0
    error_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {
            executor.submit(review_paper_pair, task[0], task[1], task[2], task[3], task[4], task[5], task[6]): task 
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
    
    # Generate summary CSV
    summary_data = []
    for _, row in df.iterrows():
        paper_id = row['paperid']
        paper_dir = output_dir / paper_id
        
        if (paper_dir / "comparison.json").exists():
            with open(paper_dir / "comparison.json", 'r') as f:
                comp = json.load(f)
                
                summary_row = {
                    "paper_id": paper_id,
                    "arxiv_id": row.get('arxiv_id', ''),
                }
                
                # Add v1 scores
                v1_data = comp.get('v1', {})
                for field in ["clarity_score", "novelty_score", "technical_quality_score", 
                             "experimental_rigor_score", "overall_score", "confidence"]:
                    summary_row[f"v1_{field}"] = v1_data.get(field, '')
                summary_row["v1_recommendation"] = v1_data.get("recommendation", '')
                summary_row["v1_success"] = v1_data.get("success", False)
                
                # Add latest scores
                latest_data = comp.get('latest', {})
                for field in ["clarity_score", "novelty_score", "technical_quality_score", 
                             "experimental_rigor_score", "overall_score", "confidence"]:
                    summary_row[f"latest_{field}"] = latest_data.get(field, '')
                summary_row["latest_recommendation"] = latest_data.get("recommendation", '')
                summary_row["latest_success"] = latest_data.get("success", False)
                
                # Add score changes
                score_changes = comp.get('score_changes', {})
                for field in ["clarity_score", "novelty_score", "technical_quality_score", 
                             "experimental_rigor_score", "overall_score"]:
                    if field in score_changes:
                        summary_row[f"{field}_change"] = score_changes[field].get("change", '')
                
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
# Full production run (both versions):
python review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./data/ICLR2024_pairs/reviews" \
  --model_name "claude-3-5-sonnet-20241022" \
  --max_workers 3 \
  --verbose

# DEBUG MODE - Test with 1 paper, only v1 version (saves API credits):
python review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./data/ICLR2024_pairs/reviews_debug" \
  --model_name "claude-3-5-haiku-20241022" \
  --version v1 \
  --limit 1 \
  --verbose

# DEBUG MODE - Review only v1 versions for all papers (half the API calls):
python review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./data/ICLR2024_pairs/reviews" \
  --model_name "claude-3-5-sonnet-20241022" \
  --version v1 \
  --max_workers 3

# Continue from previous run - only review what's missing (saves API credits):
python review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./data/ICLR2024_pairs/reviews" \
  --model_name "claude-3-5-sonnet-20241022" \
  --version latest \
  --skip_existing \
  --max_workers 3

# Complete testing workflow:
# Step 1: Test with 1 paper, v1 only
python review_paper_pairs.py --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" --output_dir "./reviews_test" --version v1 --limit 1

# Step 2: If successful, add latest version to the same paper
python review_paper_pairs.py --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" --output_dir "./reviews_test" --version latest --limit 1 --skip_existing

# Step 3: Expand to more papers
python review_paper_pairs.py --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" --output_dir "./reviews_test" --version both --limit 10 --skip_existing

# Step 4: Full run
python review_paper_pairs.py --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" --output_dir "./data/ICLR2024_pairs/reviews" --version both --skip_existing
"""

