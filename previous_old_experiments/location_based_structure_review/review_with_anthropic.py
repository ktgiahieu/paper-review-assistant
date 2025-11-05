import os
import glob
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
import base64
import io
import math
import urllib.parse
from pydantic import BaseModel, Field
from typing import Literal

# Attempt to import Pillow
try:
    from PIL import Image
    if not hasattr(Image, 'Resampling'):
        Image.Resampling = Image
except ImportError:
    print("WARNING: Pillow library not found. Install it (`pip install Pillow`) to process images.")
    Image = None

# --- Merged Content from flaw_prompts.py ---

FLAW_TAXONOMY = """
Category 1: Empirical Evaluation Flaws
(Concerns the experiments and evidence used to support the claims.)
1a: Insufficient Baselines/Comparisons
1b: Weak or Limited Scope of Experiments
1c: Lack of Necessary Ablation or Analysis
1d: Flawed Evaluation Metrics or Setup

Category 2: Methodological & Theoretical Flaws
(Concerns the core technical contribution—the algorithm, model, or theory itself.)
2a: Fundamental Technical Limitation
2b: Missing or Incomplete Theoretical Foundation
2c: Technical or Mathematical Error

Category 3: Positioning & Contribution Flaws
(Concerns how the work is framed and what it claims to contribute.)
3a: Insufficient Novelty / Unacknowledged Prior Work
3b: Overstated Claims or Mismatch Between Claim and Evidence

Category 4: Presentation & Reproducibility Flaws
(Concerns the quality of the writing and the ability for others to understand and use the work.)
4a: Lack of Clarity / Ambiguity
4b: Missing Implementation or Methodological Details

Category 5: Failure to Address Limitations or Ethical Concerns
(Concerns the omission of a proper discussion of the work's boundaries and potential consequences.)
5a: Unacknowledged Technical Limitations
5b: Unaddressed Ethical or Societal Impact
"""

# This literal is no longer strictly enforced by the pydantic model for the *input*
# but is kept here for reference.
FlawCategory = Literal[
    "1a", "1b", "1c", "1d",
    "2a", "2b", "2c",
    "3a", "3b",
    "4a", "4b",
    "5a", "5b"
]

# --- MODIFICATION: Updated Pydantic Model ---
class IdentifiedFlaw(BaseModel):
    """Pydantic model for the structured flaw analysis output."""
    flaw_types: str = Field(
        description="A comma-separated string of all applicable flaw codes from the provided taxonomy (e.g., '1a', '3b, 4a', '2c')."
    )
    root_cause_reasoning: str = Field(
        description="A single sentence identifying the single root cause that causes these flaw types symptoms."
    )

class FlawPrompts:
    @staticmethod
    # --- MODIFICATION: Updated System Prompt ---
    def get_system_prompt() -> str:
        """
        Returns the system prompt that instructs the model on its task, taxonomy, and output format.
        """
        return f"""
You are an expert academic reviewer. Your task is to identify all applicable flaws in a specific section of a research paper, based on a provided flaw taxonomy.

You will be given the full text of the paper for context, and a specific "LOCATION TO INVESTIGATE". Your analysis must focus exclusively on the text within that location. The location may contain multiple distinct sections, separated by markers; you should analyze all of them as a whole.

First, carefully review the flaw taxonomy provided below.
<flaw_taxonomy>
{FLAW_TAXONOMY}
</flaw_taxonomy>

Your goal is to identify all flaw types from the taxonomy that are present in the "LOCATION TO INVESTIGATE".

After identifying the flaws, you must respond in a specific JSON format. The JSON object must contain two keys: 'flaw_types' and 'root_cause_reasoning'.
- 'flaw_types': A comma-separated string of all applicable flaw codes (e.g., '1a', '3b, 4a', '2c').
- 'root_cause_reasoning': A single sentence identifying the single root cause that causes these flaw types symptoms.

Do not add any other text, comments, or explanations outside of the JSON structure.
Your response MUST be a single, valid JSON object.
"""

    @staticmethod
    def get_user_prompt(paper_content: str, location_to_investigate: str) -> str:
        """
        Constructs the user-facing prompt containing the paper and the specific location.
        """
        return f"""
Here is the full paper for context:
<full_paper_content>
{paper_content}
</full_paper_content>

Now, please perform your analysis on the following section. Your entire evaluation should be based on identifying all applicable flaws within this specific text block. (Note: This block may contain multiple modified sections concatenated together).

<location_to_investigate>
{location_to_investigate}
</location_to_investigate>
"""
# --- End of Merged Content ---


# --- Environment & API Configuration ---
load_dotenv()
# Ensure you have ANTHROPIC_API_KEY in your .env file or environment variables
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# --- Constants ---
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2
RETRYABLE_STATUS_CODES = [429, 500, 502, 503, 504]

# --- Image Constants ---
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024  # API limit: 20MB
TARGET_RESIZE_BYTES = 10 * 1024 * 1024 # Softer target for resizing
MAX_RESIZE_ATTEMPTS = 4
MIN_DIMENSION_AFTER_RESIZE = 50

def _sanitize_json_string(json_str: str) -> str:
    """A simple function to clean common JSON errors from LLM output."""
    # Remove code block markers
    json_str = json_str.strip().strip("```json").strip("```")
    # Remove trailing commas before object/array close
    json_str = re.sub(r',\s*(?=[}\]])', '', json_str)
    return json_str

def find_markdown_file(base_dir: Path, paper_id: str) -> Path | None:
    """Finds the first markdown file within a directory that matches the paper_id pattern."""
    search_path = base_dir / paper_id
    # print(search_path) # Kept commented out as it was in the original
    if not search_path.exists():
        return None
    
    md_files = list(search_path.glob("*.md"))
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
                # Ensure img_format is correct even if suffix was weird
                img_format = pil_img.format or 'JPEG' 
                
                for attempt in range(MAX_RESIZE_ATTEMPTS):
                    # Use sqrt for area-based resizing
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
                    
                    # Prepare for next iteration
                    pil_img = resized_img
                    initial_size = buffer.tell() # Update size for next scale calc

            if image_bytes:
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                mime_type = Image.MIME.get(img_format, 'image/jpeg')
                
                content_list_images.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": base64_image,
                    },
                })
                added_figures_count += 1
                processed_image_paths.add(str(candidate_path))
            elif verbose:
                 _print_method(f"Worker {worker_id}: Could not resize '{candidate_path}' under max bytes.")

        except Exception as e:
            _print_method(f"Worker {worker_id}: Error processing image {candidate_path}: {e}")

    return content_list_images


def analyze_flaw_location(
    paper_id: str,
    paper_content: str,
    location_to_investigate: str,
    location_index: int, # This will now always be 0 for a given paper
    image_content_blocks: list,
    output_dir: Path,
    anthropic_client: anthropic.Anthropic,
    model_name: str,
    verbose: bool
):
    """
    Analyzes a single flaw location (or concatenated locations) within a paper using the Anthropic API.
    """
    worker_id = concurrent.futures.thread.get_ident() if hasattr(concurrent.futures.thread, 'get_ident') else os.getpid()
    _print_method = tqdm.write if not verbose else print
    
    try:
        system_prompt = FlawPrompts.get_system_prompt()
        user_prompt_text = FlawPrompts.get_user_prompt(paper_content, location_to_investigate)

        # Combine images and text for multimodal input
        user_content = image_content_blocks + [{"type": "text", "text": user_prompt_text}]

        response_obj = None
        last_exception = None

        for attempt in range(MAX_RETRIES):
            try:
                if verbose:
                    _print_method(f"Worker {worker_id}: Paper {paper_id}, Loc {location_index}: API call attempt {attempt + 1}/{MAX_RETRIES}")
                
                response_obj = anthropic_client.messages.create(
                    model=model_name,
                    max_tokens=1024,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_content}],
                    timeout=300.0,
                )
                if verbose:
                     _print_method(f"Worker {worker_id}: Paper {paper_id}, Loc {location_index}: API call successful.")
                break
            except anthropic.APIStatusError as e:
                last_exception = e
                if e.status_code in RETRYABLE_STATUS_CODES:
                    wait_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                    _print_method(f"Worker {worker_id}: Retrying API call for {paper_id} (Loc {location_index}) in {wait_time}s due to status {e.status_code}...")
                    time.sleep(wait_time)
                else:
                    _print_method(f"Worker {worker_id}: Non-retryable API error for {paper_id} (Loc {location_index}): {e}")
                    break
            except Exception as e:
                last_exception = e
                _print_method(f"Worker {worker_id}: An unexpected error occurred during API call for {paper_id} (Loc {location_index}): {e}")
                # Adding a small backoff even for unexpected errors
                wait_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                time.sleep(wait_time)
                # break # This was in the original, but for a general exception, retrying might be good. Let's stick to original logic.

        if response_obj is None or not response_obj.content:
            err_msg = f"All API call attempts failed for {paper_id} at location {location_index}."
            if last_exception:
                err_msg += f" Last error: {type(last_exception).__name__} - {last_exception}"
            _print_method(f"Worker {worker_id}: {err_msg}")
            return err_msg

        raw_json_content = response_obj.content[0].text
        sanitized_json_content = _sanitize_json_string(raw_json_content)

        try:
            # The model_validate_json will now use the *updated* IdentifiedFlaw model
            parsed_flaw = IdentifiedFlaw.model_validate_json(sanitized_json_content)
            final_json_output = parsed_flaw.model_dump()
        except Exception as pydantic_error:
            _print_method(f"Worker {worker_id}: CRITICAL - Pydantic validation failed for {paper_id} (Loc {location_index}). Error: {pydantic_error}")
            _print_method(f"Worker {worker_id}: Raw sanitized JSON: {sanitized_json_content}")
            _print_method(f"Worker {worker_id}: Attempting to save raw, sanitized JSON as fallback.")
            try:
                final_json_output = json.loads(sanitized_json_content)
                final_json_output["__pydantic_validation_error"] = str(pydantic_error)
            except json.JSONDecodeError as json_e:
                _print_method(f"Worker {worker_id}: CRITICAL - Fallback JSON parsing also failed for {paper_id} (Loc {location_index}). Error: {json_e}")
                final_json_output = {
                    "error": "Failed to parse or validate JSON from LLM.",
                    "raw_content": raw_json_content,
                    "sanitized_content": sanitized_json_content
                }

        output_path = output_dir / paper_id
        output_path.mkdir(parents=True, exist_ok=True)
        # File name will now be 'flaw_analysis_0.json' for all papers, as location_index is 0
        output_file = output_path / f"flaw_analysis_{location_index}.json" 

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_json_output, f, ensure_ascii=False, indent=2)

        return f"Successfully analyzed flaw {location_index} for paper {paper_id}"

    except Exception as e:
        message = f"Worker {worker_id}: FATAL ERROR processing flaw {location_index} for paper {paper_id}: {type(e).__name__} - {e}"
        _print_method(message)
        import traceback
        _print_method(traceback.format_exc()) # More detailed traceback
        return message

def main():
    parser = argparse.ArgumentParser(description="Generate structured flaw reviews for papers using the Anthropic API.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the sampled_flaws.csv file.")
    parser.add_argument("--papers_dir", type=str, required=True, help="Base directory where the 'flawed_papers' folder is located.")
    parser.add_argument("--original_data_dir", type=str, required=True, help="Base directory of the original, non-flawed papers containing figures.")
    parser.add_argument("--output_dir", type=str, default="./flaw_reviews/", help="Output directory for flaw analysis JSON files.")
    parser.add_argument("--model_name", type=str, default="claude-3-sonnet-20240229", help="Anthropic model name to use.")
    parser.add_argument("--max_workers", type=int, default=5, help="Max worker threads for concurrent processing.")
    parser.add_argument("--max_figures", type=int, default=0, help="Max figures to process per paper (0 for none).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
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
        print(f"Error reading CSV file {args.csv_file}: {e}")
        exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    tasks = []
    print(f"Preparing to analyze {len(df)} papers from CSV...")
    
    required_column = 'llm_generated_modifications'
    if required_column not in df.columns:
        print(f"Error: Required column '{required_column}' not found in {args.csv_file}")
        print(f"Available columns: {df.columns.to_list()}")
        exit(1)
        
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Reading CSV & Finding Papers"):
        paper_id_col = row.iloc[0] 
        # Handle potential variance in how paper ID is stored
        paper_id = str(paper_id_col)
        if '/' in paper_id or '\\' in paper_id:
             paper_id = Path(paper_id).name
        paper_id = paper_id.split(',')[0].strip() # Clean up

        if not paper_id:
            tqdm.write(f"Warning: Empty paper ID at row {index}. Skipping.")
            continue
            
        markdown_file = find_markdown_file(Path(args.papers_dir), paper_id)
        if not markdown_file:
            tqdm.write(f"Warning: Could not find markdown file for paper ID '{paper_id}'. Skipping.")
            continue
            
        try:
            with open(markdown_file, 'r', encoding='utf-8') as f:
                paper_content = f.read()
        except Exception as e:
            tqdm.write(f"Warning: Could not read file {markdown_file} for paper ID '{paper_id}'. Error: {e}. Skipping.")
            continue
        
        # Process images for this paper
        original_paper_dir = Path(args.original_data_dir) / paper_id
        image_blocks = []
        if original_paper_dir.exists():
            if args.verbose: tqdm.write(f"Processing images for {paper_id}...")
            image_blocks = process_images_for_api(
                paper_content, original_paper_dir, args.max_figures, args.verbose, index
            )
        elif args.verbose:
            tqdm.write(f"Note: Original data directory not found for {paper_id} at {original_paper_dir}")


        try:
            modifications_str = row[required_column]
            if not isinstance(modifications_str, str) or not modifications_str.strip().startswith('['):
                 tqdm.write(f"Warning: Skipping paper ID '{paper_id}'. '{required_column}' is not valid JSON array string: {modifications_str[:100]}...")
                 continue

            modifications = json.loads(modifications_str)
            # --- MODIFICATION: Concatenate all locations ---
            # First, extract all valid locations
            locations = [mod['new_content'] for mod in modifications if 'new_content' in mod and mod['new_content']]
            
            if not locations:
                tqdm.write(f"Warning: No 'new_content' found for paper ID '{paper_id}'. Skipping.")
                continue

            # Concatenate all locations into one string with a clear separator
            separator = "\n\n" + ("=" * 40) + "\n[NEXT MODIFIED SECTION]\n" + ("=" * 40) + "\n\n"
            all_locations_text = separator.join(locations)
            
            # Add a single task for the *entire* set of modifications for this paper
            # We use '0' for the location_index as there's only one combined location now.
            tasks.append((paper_id, paper_content, all_locations_text, 0, image_blocks))
            # --- END MODIFICATION ---

        except (json.JSONDecodeError, TypeError, KeyError) as e:
            tqdm.write(f"Warning: Could not parse JSON or find locations for paper ID '{paper_id}'. Error: {e}. Skipping.")
            continue
        except Exception as e:
            tqdm.write(f"Warning: Unexpected error processing row for paper ID '{paper_id}'. Error: {e}. Skipping.")
            continue
    
    print(f"Found a total of {len(tasks)} locations to analyze across all papers.")
    if not tasks:
        print("No tasks to run. Exiting.")
        return

    processed_count, error_count = 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {
            executor.submit(
                analyze_flaw_location, task[0], task[1], task[2], task[3], task[4],
                Path(args.output_dir), client, args.model_name, args.verbose
            ): task for task in tasks
        }
        
        progress = tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks), desc="Analyzing Flaws")
        for future in progress:
            task_info = future_to_task[future] # (paper_id, ..., location_index, ...)
            paper_id_info = task_info[0]
            loc_index_info = task_info[3]
            try:
                result_message = future.result()
                if "Successfully analyzed" in result_message:
                    processed_count += 1
                else:
                    error_count += 1
                    tqdm.write(f"Analysis failed for {paper_id_info} (Loc {loc_index_info}): {result_message}")
            except Exception as exc:
                error_count += 1
                tqdm.write(f"A flaw analysis process ({paper_id_info}, Loc {loc_index_info}) generated an unhandled exception: {exc}")
                import traceback
                tqdm.write(traceback.format_exc())

    print("\n--- Processing Complete ---")
    print(f"Total locations analyzed (by paper): {processed_count}")
    print(f"Failed/Errored analyses: {error_count}")
    print(f"Results saved in: {args.output_dir}")
    print("---------------------------\n")


if __name__ == "__main__":
    main()

# Example usage
"""
python review_with_anthropic_modified.py \
  --csv_file "./exp_data_2_per_group_NeurIPS2024/sampled_flaws.csv" \
  --papers_dir "./exp_data_2_per_group_NeurIPS2024/flawed_papers" \
  --original_data_dir "./exp_data_2_per_group_NeurIPS2024/original_papers" \
  --output_dir "./exp_data_2_per_group_NeurIPS2024/structured_reviews" \
  --max_workers 10 \
  --model_name "claude-3-haiku-20240307" \
  --verbose
"""

