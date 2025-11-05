import os
import argparse
import json
import pandas as pd
from pathlib import Path
import concurrent.futures
import time
import re

from tqdm import tqdm
import anthropic
from dotenv import load_dotenv
import pydantic

# Make sure evaluation_prompts.py is in the same directory or accessible
from evaluation_prompts import FlawEvaluation, EvaluationPrompts

# --- Environment & API Configuration ---
# Load environment variables from a .env file.
# Your .env file should contain: ANTHROPIC_API_KEY="your_api_key_here"
load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("Error: ANTHROPIC_API_KEY not found in environment variables or .env file.")

# --- Exception Handling for Retries ---
_RETRYABLE_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
    pydantic.ValidationError,
    json.JSONDecodeError,
)

def _sanitize_json_string(json_str):
    """A simple function to clean common JSON errors from LLM output."""
    json_str = re.sub(r',\s*(?=[}\]])', '', json_str)
    json_str = json_str.strip().strip("```json").strip("```")
    return json_str

def evaluate_single_flaw(
    anthropic_client,
    model_name,
    review_content,
    flaw_id,
    flaw_description,
    verbose=False
):
    """
    Calls the Anthropic API to evaluate if a flaw is present in a review.

    Args:
        anthropic_client: The configured Anthropic client instance.
        model_name (str): The name of the Anthropic model to use.
        review_content (str): The full text of the human review.
        flaw_id (str): The ID of the flaw being evaluated.
        flaw_description (str): The detailed description of the flaw.
        verbose (bool): If True, prints detailed logs.

    Returns:
        A dictionary containing the validated evaluation data or an error message.
    """
    MAX_RETRIES = 3
    INITIAL_BACKOFF_SECONDS = 2
    _print_method = tqdm.write if not verbose else print
    
    evaluation_schema = FlawEvaluation.model_json_schema()
    
    user_prompt = EvaluationPrompts.get_evaluation_prompt(
        review_text=review_content,
        flaw_id=flaw_id,
        flaw_description=flaw_description,
        json_schema=evaluation_schema
    )

    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            response_obj = anthropic_client.messages.create(
                model=model_name,
                max_tokens=1024,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            raw_json_content = response_obj.content[0].text
            cleaned_json = _sanitize_json_string(raw_json_content)
            validated_evaluation = FlawEvaluation.model_validate_json(cleaned_json)
            return validated_evaluation.model_dump()

        except Exception as e:
            last_exception = e
            if isinstance(e, _RETRYABLE_EXCEPTIONS):
                wait_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                if verbose: _print_method(f"Retryable API error ({type(e).__name__}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                if verbose: _print_method(f"Non-retryable error during evaluation for {flaw_id}: {type(e).__name__} - {e}")
                break
    
    return {
        "flaw_id": flaw_id,
        "error": f"Failed after {MAX_RETRIES} attempts.",
        "last_exception": str(last_exception)
    }

def process_row_wrapper(args_dict):
    """Helper function to unpack arguments for the executor map."""
    return evaluate_single_flaw(**args_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate human reviews for specific flaws using the Anthropic API.")
    parser.add_argument("--reviews_file", type=str, required=True, help="Input CSV file with columns [openreview_id, flaw_id, reviewer_id, human_review].")
    parser.add_argument("--flaws_file", type=str, required=True, help="CSV file with columns [openreview_id, flaw_id, flaw_description] to provide context to the LLM.")
    parser.add_argument("--output_file", type=str, default="./review_evaluations.json", help="Path to save the final aggregated evaluation JSON file.")
    parser.add_argument("--model_name", type=str, default="claude-3-sonnet-20240229", help="Anthropic model name for evaluation.")
    parser.add_argument("--max_workers", type=int, default=10, help="Max worker threads for concurrent processing.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output for debugging.")
    args = parser.parse_args()

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        print("Successfully initialized Anthropic client.")
    except Exception as e:
        print(f"Error initializing Anthropic client: {e}"); exit(1)

    # --- 1. Load and Prepare Data ---
    print(f"Loading reviews from: {args.reviews_file}")
    reviews_df = pd.read_csv(args.reviews_file)
    print(f"Loading flaw descriptions from: {args.flaws_file}")
    flaws_df = pd.read_csv(args.flaws_file)

    # Merge the two dataframes to get the flaw_description for each review
    data_df = pd.merge(
        reviews_df,
        flaws_df,
        on=['openreview_id', 'flaw_id'],
        how='left'
    )
    
    # Drop rows where a flaw description couldn't be found
    original_rows = len(data_df)
    data_df.dropna(subset=['flaw_description', 'human_review'], inplace=True)
    if len(data_df) < original_rows:
        print(f"Warning: Dropped {original_rows - len(data_df)} rows due to missing flaw descriptions or reviews.")

    if data_df.empty:
        print("No valid data to process after merging and cleaning. Exiting.")
        exit(0)

    print(f"Found {len(data_df)} valid reviews to evaluate with model: {args.model_name}")

    # --- 2. Create Tasks for Concurrent Processing ---
    tasks = []
    for _, row in data_df.iterrows():
        tasks.append({
            "anthropic_client": client,
            "model_name": args.model_name,
            "review_content": row['human_review'],
            "flaw_id": row['flaw_id'],
            "flaw_description": row['flaw_description'],
            "verbose": args.verbose
        })

    # --- 3. Run Evaluation ---
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        progress = tqdm(executor.map(process_row_wrapper, tasks), total=len(tasks), desc="Evaluating Reviews")
        for i, result in enumerate(progress):
            # Combine original data with the evaluation result
            combined_result = {
                "openreview_id": data_df.iloc[i]['openreview_id'],
                "reviewer_id": data_df.iloc[i]['reviewer_id'],
                "evaluation": result
            }
            all_results.append(combined_result)

    # --- 4. Save Results ---
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n--- Evaluation Complete ---")
    print(f"Results for {len(all_results)} evaluations saved to {args.output_file}")
    print("---------------------------\n")
