import openreview
import os
import json
import csv
import re
import argparse
import concurrent.futures
import time
import threading
from pathlib import Path
from openai import AzureOpenAI, APIError, Timeout, RateLimitError, APIConnectionError
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from typing import List, Tuple, Optional
from tqdm import tqdm
import tiktoken

from generation_prompt import ConsensusExtractionPrompt, FlawInjectionPrompt

# --- Pydantic Schemas for Structured LLM Responses ---

class Flaw(BaseModel):
    flaw_id: str = Field(..., description="A short, snake_case identifier for the flaw (e.g., 'limited_scope', 'missing_baseline').")
    description: str = Field(..., description="A detailed description of the flaw, explaining its significance based on the review-rebuttal consensus.")

class FlawExtractionResponse(BaseModel):
    critical_flaws: List[Flaw] = Field(..., description="A list of critical flaws identified from the review process.")

class Modification(BaseModel):
    target_heading: str = Field(..., description="The exact markdown heading of the section to replace.")
    new_content: str = Field(..., description="The complete, rewritten text for the entire section, including the heading.")
    reasoning: str = Field(..., description="A brief explanation for the modification.")

class ModificationGenerationResponse(BaseModel):
    modifications: List[Modification] = Field(..., description="A list of sections to modify to re-introduce the flaw.")

# --- Configuration ---
load_dotenv()

# OpenReview Credentials
OPENREVIEW_USERNAME = os.environ.get('OPENREVIEW_USERNAME')
OPENREVIEW_PASSWORD = os.environ.get('OPENREVIEW_PASSWORD')

# Azure OpenAI Credentials
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")

CONTEXT_LENGTH_LIMIT = 150000  # The context limit of the model being used
TOKEN_BUFFER = 10000 # Buffer for system prompts, response format, etc.
# --- Helper Functions & Classes ---

def get_tokenizer():
    """Initializes and returns a tiktoken tokenizer."""
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return tiktoken.encoding_for_model("gpt-4")

tokenizer = get_tokenizer()

def truncate_text(text: str, max_tokens: int) -> str:
    """Truncates text to a maximum number of tokens."""
    tokens = tokenizer.encode(text, allowed_special="all")
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens)
    return text

def get_openreview_client(venue):
    """Initializes and returns a connected OpenReview client."""
    try:
        client = openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net',
            username=OPENREVIEW_USERNAME,
            password=OPENREVIEW_PASSWORD
        )
        all_notes = None
        # Get correct client version for this venue
        client_version = "v2"
        api_venue_group = client.get_group(venue)
        api_venue_domain = api_venue_group.domain
        if api_venue_domain:
            print("This venue is available for OpenReview Client V2. Proceeding...")
        else:
            print("This venue is not available for OpenReview Client V2. Switching to Client V1...")
            client = openreview.Client(
                baseurl='https://api.openreview.net', 
                username=OPENREVIEW_USERNAME, 
                password=OPENREVIEW_PASSWORD
            )
            client_version = "v1"
            
            all_notes = client.get_all_notes(invitation=venue + "/-/Blind_Submission", details = 'replies')
            all_notes = {note.id: note for note in all_notes}
        return client, client_version, all_notes
    except Exception as e:
        print(f"Failed to connect to OpenReview: {e}")
        return None
    
    

def format_reviews_for_llm(note_details: dict) -> str:
    """Formats OpenReview note details into a clean text block for LLM analysis."""
    output_text = []
    if not note_details or 'replies' not in note_details:
        return "No review or comment data found."

    for reply in sorted(note_details['replies'], key=lambda x: x.get('number', 0)):
        signatures = ", ".join(reply.get('signatures', ['Unknown']))
        content = reply.get('content', {})

        if any(key in content for key in ['strengths', 'weaknesses', 'rating']):
            output_text.append(f"--- Review by {signatures} ---\n")
            for key, value in content.items():
                if isinstance(value, dict) and 'value' in value:
                    value = value['value']
                if isinstance(value, str) and value.strip():
                    output_text.append(f"## {key.replace('_', ' ').title()}\n{value.strip()}\n")
        elif 'comment' in content:
            output_text.append(f"--- Comment by {signatures} ---\n")
            comment_value = content['comment']
            if not isinstance(comment_value, str):
                comment_value = comment_value.get('value', content['comment'])
            if isinstance(comment_value, str) and comment_value.strip():
                output_text.append(f"{comment_value.strip()}\n")

    return "\n".join(output_text)

def clean_heading_text_aggressively(text: str) -> str:
    """
    This is the most aggressive cleaning function, used as a last resort.
    It strips almost all non-alphanumeric characters.
    """
    # Remove HTML tags, e.g., <span...>
    text = re.sub(r'<.*?>', '', text)
    # Remove markdown reference-style links, e.g., [sec:comparison]
    text = re.sub(r'\[[^\]]*?\]', '', text)
    # Remove markdown inline links, keeping the link text, e.g., [text](url) -> text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # Remove all LaTeX commands and math delimiters
    text = re.sub(r'\\[a-zA-Z@]+({.*?})?|[\{\}\$\(\)\\]', '', text)
    # Remove leading/trailing markdown markup and whitespace
    text = text.strip().strip('#*').strip()
    # Remove trailing punctuation that might differ
    text = text.rstrip('.,;:')
    # Normalize internal whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def try_apply_modifications(original_markdown: str, modifications: List[Modification]) -> Tuple[str, bool, Optional[str]]:
    """
    Tries to apply a list of modifications using a tiered matching strategy,
    from most to least strict, to ensure accuracy.
    """
    current_markdown = original_markdown
    lines = original_markdown.split('\n')
    
    for mod in modifications:
        target_heading = mod.target_heading.strip()
        if not target_heading:
            continue

        match_index = -1

        # Tier 1: Exact, verbatim match (highest precision)
        for i, line in enumerate(lines):
            if line.strip() == target_heading:
                match_index = i
                break
        
        # Tier 2: Match after stripping leading/trailing whitespace and markdown characters
        if match_index == -1:
            semi_cleaned_target = target_heading.strip('#* \t')
            for i, line in enumerate(lines):
                semi_cleaned_line = line.strip().strip('#* \t')
                if semi_cleaned_line == semi_cleaned_target and semi_cleaned_line:
                    match_index = i
                    break

        # Tier 3: Aggressive cleaning (last resort)
        # This handles complex cases with embedded HTML, LaTeX, etc.
        if match_index == -1:
            aggressively_cleaned_target = clean_heading_text_aggressively(target_heading)
            for i, line in enumerate(lines):
                aggressively_cleaned_line = clean_heading_text_aggressively(line)
                
                if not aggressively_cleaned_line or not aggressively_cleaned_target:
                    continue

                # Use startswith for leniency, as cleaned lines might have extra text
                if aggressively_cleaned_line.lower().startswith(aggressively_cleaned_target.lower()):
                    match_index = i
                    break

        if match_index == -1:
            tqdm.write(f"Failure: Could not find target heading '{target_heading}'.")
            return original_markdown, False, target_heading

        # --- If a match was found, apply the modification ---
        start_line = match_index
        
        # Find the end of the section by looking for the next heading of any type.
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

        # Reconstruct the markdown with the modification
        pre_section_lines = lines[:start_line]
        post_section_lines = lines[end_line:]
        
        new_content_lines = mod.new_content.split('\n')
        
        # Join sections back together. We must operate on the list of lines to preserve structure.
        lines = pre_section_lines + new_content_lines + post_section_lines
        current_markdown = '\n'.join(lines)
            
    return current_markdown, True, None



def call_llm_with_retries(client, prompt, response_model, purpose="LLM Call"):
    """Generic function to call the Azure OpenAI API with retries and Pydantic parsing."""
    max_retries = 3
    backoff_factor = 2
    for attempt in range(max_retries):
        try:
            response = client.beta.chat.completions.parse(
                model=AZURE_DEPLOYMENT_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format=response_model,
                timeout=600.0,
            )
            json_response = response.choices[0].message.content
            return response_model.model_validate_json(json_response)
        
        except (json.JSONDecodeError, ValidationError) as e:
            return None
            
        except Exception as e:
            tqdm.write(f"Worker {threading.get_ident()}: [{purpose}] Retryable API error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(backoff_factor ** attempt)
            else:
                tqdm.write(f"Worker {threading.get_ident()}: [{purpose}] All API retries failed.")
                raise

    return None

def process_paper(paper_md_path: Path, input_base_dir: Path, output_base_dir: Path, or_client, azure_client, or_client_version, all_or_notes):
    """Main worker function to process a single paper."""
    worker_id = threading.get_ident()
    MAX_MODIFICATION_ATTEMPTS = 3
    try:
        paper_folder = paper_md_path.parent.parent
        openreview_id = '_'.join(paper_folder.name.split('_')[:-2])
        
        with open(paper_md_path, 'r', encoding='utf-8') as f:
            original_paper_text = f.read()

        if or_client_version == "v2":
            note = or_client.get_note(openreview_id, details='replies')
        elif or_client_version == "v1":
            note = all_or_notes[openreview_id]
        else:
            raise ValueError(f"Unknown OpenReview client version: {or_client_version}")

        review_text = format_reviews_for_llm(note.details)
        
        # --- Truncate inputs to avoid context length errors ---
        review_tokens = len(tokenizer.encode(review_text, allowed_special="all"))
        paper_max_tokens = CONTEXT_LENGTH_LIMIT - review_tokens - TOKEN_BUFFER
        truncated_paper_text = truncate_text(original_paper_text, paper_max_tokens)


        flaw_extraction_prompt = ConsensusExtractionPrompt.generate_prompt_for_review_process(
            review_process=review_text, paper_text=truncated_paper_text
        )
        flaw_response = call_llm_with_retries(azure_client, flaw_extraction_prompt, FlawExtractionResponse, "Flaw Extraction")

        if not flaw_response or not flaw_response.critical_flaws:
            # tqdm.write(f"Worker {worker_id}: No critical flaws found or extracted for {openreview_id}.")
            return []

        results_for_global_csv = []
        all_modifications_for_local_csv = []

        for flaw in flaw_response.critical_flaws:
            was_successful = False
            last_error_feedback = None # <<< Initialize error feedback for the loop

            for attempt in range(MAX_MODIFICATION_ATTEMPTS):

                # --- Generate prompt with feedback from previous failed attempts ---
                modification_prompt = FlawInjectionPrompt.generate_prompt_for_flaw_injection(
                    flaw_description=flaw.description, 
                    original_paper_text=original_paper_text,
                    last_error=last_error_feedback # <<< Pass the error to the prompt generator
                )
                
                mod_response = call_llm_with_retries(azure_client, modification_prompt, ModificationGenerationResponse, "Modification Generation")

                if not mod_response or not mod_response.modifications:
                    # tqdm.write(f"Worker {worker_id}: LLM returned no modifications for flaw '{flaw.flaw_id}'. Not retrying for this flaw.")
                    break 

                # --- Attempt to apply the modifications ---
                flawed_paper_text, success, failed_heading = try_apply_modifications(original_paper_text, mod_response.modifications)

                if success:                    
                    relative_paper_folder = paper_folder.relative_to(input_base_dir)
                    # Define output folder inside the loop to ensure it's created
                    output_paper_folder = output_base_dir / relative_paper_folder / "flawed_papers"
                    output_paper_folder.mkdir(parents=True, exist_ok=True)
                    
                    output_filename = f"{flaw.flaw_id}.md" # Simplified filename
                    output_filepath = output_paper_folder / output_filename

                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        f.write(flawed_paper_text)
                    
                    modifications_json_string = json.dumps([mod.model_dump() for mod in mod_response.modifications], indent=2)

                    # Append to list for the global summary CSV
                    results_for_global_csv.append({
                        'openreview_id': openreview_id,
                        'flaw_id': flaw.flaw_id,
                        'flaw_description': flaw.description,
                        'output_path': str(output_filepath),
                        'num_modifications': len(mod_response.modifications),
                        'llm_generated_modifications': modifications_json_string
                    })
                    
                    # Append to list for the paper-specific modifications CSV
                    all_modifications_for_local_csv.append({
                        'flaw_id': flaw.flaw_id,
                        'flaw_description': flaw.description,
                        'num_modifications': len(mod_response.modifications),
                        'llm_generated_modifications': modifications_json_string
                    })
                    was_successful = True
                    break # Success, so break the attempt loop for this flaw
                else:
                    # --- If application fails, prepare feedback for the next attempt ---
                    tqdm.write(f"Worker {worker_id}: Modification application failed on attempt {attempt + 1}. Retrying LLM call with feedback.")
                    last_error_feedback = failed_heading # <<< Set the feedback for the next loop iteration
            
            if not was_successful:
                tqdm.write(f"Worker {worker_id}: All {MAX_MODIFICATION_ATTEMPTS} attempts failed for flaw '{flaw.flaw_id}'. Skipping.")
        
        # --- Write the local modifications CSV for this paper ---
        if all_modifications_for_local_csv:
            # This path is now well-defined even if only one flaw succeeded
            local_csv_path = output_base_dir / paper_folder.relative_to(input_base_dir) / f"{openreview_id}_modifications_summary.csv"
            local_csv_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            with open(local_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=all_modifications_for_local_csv[0].keys())
                writer.writeheader()
                writer.writerows(all_modifications_for_local_csv)

        return results_for_global_csv
    
    except Exception as e:
        tqdm.write(f"Worker {worker_id}: FATAL ERROR processing {paper_md_path.parent.parent.name}: {e}")
        import traceback
        traceback.print_exc()
        return []

# --- Main Workflow Execution ---
def main():
    parser = argparse.ArgumentParser(description="Generate flawed paper versions based on OpenReview discussions.")
    parser.add_argument("--input_dir", type=str, required=True, help="Base input directory (e.g., 'ICLR2024_latest').")
    parser.add_argument("--output_dir", type=str, default="flawed_papers_v2", help="Directory to save the flawed papers and results CSV.")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of parallel threads to run.")
    parser.add_argument("--venue", default="ICLR.cc/2025/Conference", help="OpenReview venue ID (e.g., ICLR.cc/2024/Conference)")
    args = parser.parse_args()

    # --- Environment Variable Check ---
    required_env_vars = ['OPENREVIEW_USERNAME', 'OPENREVIEW_PASSWORD', 'AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_KEY', 'AZURE_OPENAI_DEPLOYMENT_NAME']
    if not all(os.environ.get(var) for var in required_env_vars):
        print("Error: One or more required environment variables are not set. Please check your .env file or environment.")
        print("Required:", ", ".join(required_env_vars))
        return

    input_base_dir = Path(args.input_dir)
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input directory: {input_base_dir.resolve()}")
    print(f"Output directory: {output_base_dir.resolve()}")

    paper_paths = list(input_base_dir.glob("**/paper.md"))
    if not paper_paths:
        print(f"No 'paper.md' files found in {input_base_dir}. Please check the path and folder structure.")
        return
    
    print(f"Found {len(paper_paths)} papers to process.")

    or_client, or_client_version, all_or_notes = get_openreview_client(args.venue)
    
    if not or_client:
        print("Failed to initialize OpenReview client. Check credentials or network. Exiting.")
        return

    try:
        azure_client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT, api_key=AZURE_API_KEY, api_version=AZURE_API_VERSION
        )
        # Simple test call to check credentials
        azure_client.models.list()
        print("Successfully connected to Azure OpenAI.")
    except Exception as e:
        print(f"Failed to initialize or connect to Azure OpenAI: {e}")
        return


    all_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_path = {
            executor.submit(process_paper, path, input_base_dir, output_base_dir, or_client, azure_client, or_client_version, all_or_notes): path
            for path in paper_paths
        }
        
        progress_bar = tqdm(concurrent.futures.as_completed(future_to_path), total=len(paper_paths), desc="Processing Papers")
        for future in progress_bar:
            try:
                result = future.result()
                if result:
                    all_results.extend(result)
            except Exception as e:
                path = future_to_path[future]
                tqdm.write(f"Main Thread: An unexpected error occurred for paper at {path}: {e}")

    if not all_results:
        print("\nWorkflow finished, but no flawed papers were successfully generated.")
        return

    results_csv_path = output_base_dir / "flawed_papers_global_summary.csv"
    print(f"\nWriting global summary of {len(all_results)} generated files to {results_csv_path}...")
    
    with open(results_csv_path, 'w', newline='', encoding='utf-8') as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    print("Workflow complete.")

if __name__ == "__main__":
    main()