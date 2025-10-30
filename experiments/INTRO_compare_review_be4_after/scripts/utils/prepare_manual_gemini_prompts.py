#!/usr/bin/env python3
"""
Prepare Manual Gemini Prompts for UI-Based Reviews

This script prepares folders for manual Gemini Pro reviews via web UI:
- Creates structured folders for each paper
- Generates prompts ready to paste into Gemini UI
- Extracts and compresses figures (max 10 per paper)
- Creates placeholder JSON outputs

Usage:
    python prepare_manual_gemini_prompts.py \
        --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
        --output_dir ./manual_gemini_reviews/ \
        --format CriticalNeurIPS \
        --max_figures 10
"""

import os
import json
import argparse
import pandas as pd
import re
import shutil
from pathlib import Path
from PIL import Image
from typing import List, Optional, Tuple

# --- Constants ---
MAX_IMAGE_SIZE = 4 * 1024 * 1024  # 4MB per image (Gemini UI limit)
SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']

def find_paper_markdown(paper_folder: Path) -> Optional[Path]:
    """Finds the paper.md file."""
    paper_md_path = paper_folder / "structured_paper_output" / "paper.md"
    if paper_md_path.exists():
        return paper_md_path
    
    # Fallback
    md_files = list(paper_folder.glob("**/*.md"))
    return md_files[0] if md_files else None

def extract_figure_paths(paper_md: Path, max_figures: int = 10) -> List[Path]:
    """
    Extract figure paths from paper.md in reading order.
    
    Returns up to max_figures figure paths.
    """
    with open(paper_md, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find image references:
    # 1) Markdown: ![alt](path)
    md_pattern = r'!\[[^\]]*\]\(([^)\s]+)'
    md_matches = re.findall(md_pattern, content)
    # 2) HTML: <img src="path" ...>
    html_pattern = r'<img\s+[^>]*src=["\']([^"\'>\s]+)'
    html_matches = re.findall(html_pattern, content)
    matches = md_matches + html_matches
    
    figures = []
    # Try resolving paths relative to the markdown folder and its parent
    candidates = [paper_md.parent, paper_md.parent.parent]
    
    for match in matches:
        # Skip external URLs
        if match.startswith('http'):
            continue  # Skip external URLs
        # Clean surrounding quotes
        clean = match.strip().strip('"').strip("'")
        # Resolve against candidates
        resolved: Optional[Path] = None
        for base_dir in candidates:
            candidate = (base_dir / clean).resolve()
            if candidate.exists():
                resolved = candidate
                break
        if not resolved:
            # Also try URL-decoded path variants for spaces (%20)
            try:
                from urllib.parse import unquote
                decoded = unquote(clean)
                for base_dir in candidates:
                    candidate = (base_dir / decoded).resolve()
                    if candidate.exists():
                        resolved = candidate
                        break
            except Exception:
                pass
        if not resolved:
            continue
        # Accept only known raster formats here; others handled later during copy
        if resolved.suffix.lower() in SUPPORTED_IMAGE_FORMATS:
            figures.append(resolved)
        else:
            # Still include non-raster (e.g., .svg, .pdf) so we can attempt copy later
            figures.append(resolved)
        if len(figures) >= max_figures:
            break
    
    return figures

def compress_image(input_path: Path, output_path: Path, max_size: int = MAX_IMAGE_SIZE) -> bool:
    """
    Compress image to fit within max_size.
    
    Returns True if successful, False otherwise.
    """
    try:
        # Try opening with PIL; if unsupported (e.g., SVG/PDF), fallback to raw copy
        try:
            img = Image.open(input_path)
        except Exception:
            # Fallback: copy as-is if under size limit
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(input_path, output_path)
                if output_path.stat().st_size <= max_size:
                    return True
                # Too large and can't compress without renderer; skip
                return False
            except Exception as copy_err:
                print(f"Error copying non-raster image {input_path}: {copy_err}")
                return False
        
        # Convert RGBA to RGB if necessary
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        
        # Start with original quality
        quality = 95
        
        while quality > 20:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, format='JPEG', quality=quality, optimize=True)
            
            if output_path.stat().st_size <= max_size:
                return True
            
            quality -= 10
        
        # If still too large, resize
        scale = 0.8
        original_size = img.size
        
        while scale > 0.3:
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            resized = img.resize(new_size, Image.Resampling.LANCZOS)
            resized.save(output_path, format='JPEG', quality=85, optimize=True)
            
            if output_path.stat().st_size <= max_size:
                return True
            
            scale -= 0.1
        
        return False
        
    except Exception as e:
        print(f"Error compressing {input_path}: {e}")
        return False

def generate_prompt(paper_content: str, paper_version: str, format_type: str = "CriticalNeurIPS", 
                    flaw_context: Optional[str] = None) -> str:
    """Generate the complete prompt for manual Gemini UI input."""
    
    # System prompt
    if format_type == "CriticalNeurIPS":
        system_prompt = """You are a top-tier academic reviewer for NeurIPS, known for writing exceptionally thorough, incisive, and constructive critiques. Your goal is to synthesize multiple expert perspectives into a single, coherent review that elevates the entire research field.

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
        system_prompt = """You are an expert peer reviewer for a top-tier machine learning conference (NeurIPS, ICML, or ICLR). Your task is to provide a thorough, balanced, and constructive review of the submitted research paper.

Your review should assess the paper across multiple dimensions:
1. **Clarity**: How well-written and organized is the paper?
2. **Novelty**: How original and innovative is the contribution?
3. **Technical Quality**: How sound and rigorous is the technical approach?
4. **Experimental Rigor**: How comprehensive and convincing are the experiments?

Be critical but fair. Provide constructive feedback."""
    
    # Flaw context
    flaw_info = ""
    if flaw_context:
        flaw_info = f"""

Note: This paper has been identified as having the following potential issues in peer review:
<flaw_context>
{flaw_context}
</flaw_context>

Please consider these issues in your assessment, but conduct your own independent evaluation as well.
"""
    
    # User prompt with JSON schema
    if format_type == "CriticalNeurIPS":
        user_prompt = f"""Please review the following research paper with exceptional rigor and depth.

<paper_content>
{paper_content}
</paper_content>
{flaw_info}

**Instructions:**
1. Thoroughly read the entire paper.
2. Adopt the comprehensive, critical persona described in your system instructions above.
3. Generate one complete review that fills all the fields in the required JSON format.
4. Your response MUST be a single, valid JSON object. Do not include any text, markdown, or code formatting before or after the JSON object.

**Required JSON Schema (STRICT):**
Generate a single JSON object with these keys (NO extra keys, NO markdown):
* "summary": (string) A brief, neutral summary of the paper and its contributions. Not a critique or copy of the abstract.
* "strengths": (array of strings) Each item a single concise point; no nested lists, no numbering, no markdown.
* "weaknesses": (array of strings) Each item a single concise point; no nested lists, no numbering, no markdown.
* "questions": (array of strings) 3-5 actionable questions/suggestions for authors; no numbering, no markdown.
* "limitations_and_societal_impact": (string) Whether limitations and societal impacts are addressed; include constructive suggestions if not.
* "soundness": (integer) Must be 4, 3, 2, or 1. (4=excellent, 3=good, 2=fair, 1=poor)
* "presentation": (integer) Must be 4, 3, 2, or 1. (4=excellent, 3=good, 2=fair, 1=poor)
* "contribution": (integer) Must be 4, 3, 2, or 1. (4=excellent, 3=good, 2=fair, 1=poor)
* "overall_score": (integer) Must be 10, 9, 8, 7, 6, 5, 4, 3, 2, or 1. (10=Award quality, 8=Strong Accept, 6=Weak Accept, 5=Borderline, 4=Borderline reject, 2=Strong Reject)
* "confidence": (integer) Must be 5, 4, 3, 2, or 1. (5=Certain, 4=Confident, 3=Fairly confident, 2=Willing to defend, 1=Educated guess)

Your response MUST be a single, valid JSON object with no markdown fences or extra commentary."""
    else:
        user_prompt = f"""Please review the following research paper ({paper_version}):

<paper_content>
{paper_content}
</paper_content>
{flaw_info}

Provide a comprehensive review in JSON format with the following fields:
- summary: A 2-3 sentence overview of the paper
- strengths: Array of key strengths (3-5 points)
- weaknesses: Array of key weaknesses (3-5 points)
- clarity_score: Score from 1-10
- novelty_score: Score from 1-10
- technical_quality_score: Score from 1-10
- experimental_rigor_score: Score from 1-10
- overall_score: Score from 1-10 (1=strong reject, 10=strong accept)
- confidence: Your confidence level from 1-5
- recommendation: One of 'Strong Accept', 'Accept', 'Weak Accept', 'Borderline', 'Weak Reject', 'Reject', or 'Strong Reject'
- detailed_comments: 3-5 sentences explaining your assessment

Your response MUST be a single, valid JSON object with no additional text."""
    
    # Combine everything
    full_prompt = f"""{system_prompt}

===== YOUR TASK =====

{user_prompt}
"""
    
    return full_prompt

def create_placeholder_json(format_type: str, paper_id: str, version: str, run_id: int) -> dict:
    """Create a placeholder JSON with instructions."""
    
    placeholder = {
        "_instructions": "PASTE GEMINI'S JSON OUTPUT HERE (replace this entire object with Gemini's response)",
        "_paper_id": paper_id,
        "_version": version,
        "_run_id": run_id,
        "_format": format_type,
        "_status": "PENDING - Waiting for manual input",
        "_steps": [
            "1. Open the input/prompt.txt file",
            "2. Copy the entire prompt",
            "3. Paste into Gemini Pro UI (https://aistudio.google.com/)",
            "4. If there are figures in input/figures/, upload them to Gemini",
            "5. Wait for Gemini to generate the review",
            "6. Copy ONLY the JSON output from Gemini",
            "7. Replace this entire file content with the JSON from Gemini",
            "8. Save this file",
            "9. Run process_manual_gemini_outputs.py to validate and format"
        ]
    }
    
    if format_type == "CriticalNeurIPS":
        placeholder["_expected_fields"] = {
            "summary": "string",
            "strengths": "array of strings",
            "weaknesses": "array of strings",
            "questions": "array of strings",
            "limitations_and_societal_impact": "string",
            "soundness": "integer (1-4)",
            "presentation": "integer (1-4)",
            "contribution": "integer (1-4)",
            "overall_score": "integer (1-10)",
            "confidence": "integer (1-5)"
        }
    else:
        placeholder["_expected_fields"] = {
            "summary": "string",
            "strengths": "array of strings",
            "weaknesses": "array of strings",
            "clarity_score": "integer (1-10)",
            "novelty_score": "integer (1-10)",
            "technical_quality_score": "integer (1-10)",
            "experimental_rigor_score": "integer (1-10)",
            "overall_score": "integer (1-10)",
            "confidence": "integer (1-5)",
            "recommendation": "string",
            "detailed_comments": "string"
        }
    
    return placeholder

def prepare_paper_review(
    paper_id: str,
    paper_path: Path,
    version_label: str,
    flaw_descriptions: list,
    output_base_dir: Path,
    format_type: str,
    max_figures: int,
    num_runs: int
) -> Tuple[int, int]:
    """
    Prepare all materials for manual review of one paper version.
    
    Returns: (success_count, error_count)
    """
    success_count = 0
    error_count = 0
    
    try:
        # Find paper markdown
        paper_md = find_paper_markdown(paper_path)
        if not paper_md:
            print(f"❌ Could not find paper.md for {paper_id} ({version_label})")
            return 0, 1
        
        # Read paper content
        with open(paper_md, 'r', encoding='utf-8') as f:
            paper_content = f.read()
        
        # Prepare flaw context
        flaw_context = None
        if flaw_descriptions:
            flaw_context = "\n".join([f"- {flaw}" for flaw in flaw_descriptions])
        
        # Extract figures
        figures = extract_figure_paths(paper_md, max_figures)
        
        # Create folders and files for each run
        for run_id in range(num_runs):
            run_folder = output_base_dir / paper_id / f"{version_label}_run{run_id}"
            input_folder = run_folder / "input"
            output_folder = run_folder / "output"
            figures_folder = input_folder / "figures"
            
            input_folder.mkdir(parents=True, exist_ok=True)
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # Generate and save prompt
            prompt = generate_prompt(paper_content, version_label, format_type, flaw_context)
            prompt_file = input_folder / "prompt.txt"
            
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            # Copy and compress figures
            if figures:
                figures_folder.mkdir(parents=True, exist_ok=True)
                
                figure_list = []
                for idx, fig_path in enumerate(figures, 1):
                    # Use .jpg for rasterized outputs; preserve original extension if copy-as-is
                    target_name_jpg = f"figure_{idx:02d}.jpg"
                    output_fig_jpg = figures_folder / target_name_jpg
                    
                    if compress_image(fig_path, output_fig_jpg):
                        # If we ended up copying a non-raster (same extension), adjust name
                        if output_fig_jpg.suffix.lower() not in ['.jpg', '.jpeg']:
                            # compress_image only writes JPG or copies original; detect copied original
                            # If copied original (non-JPG), rename target to preserve original suffix
                            orig_ext = fig_path.suffix.lower()
                            preserved = figures_folder / f"figure_{idx:02d}{orig_ext}"
                            try:
                                output_fig_jpg.rename(preserved)
                                figure_list.append(preserved.name)
                            except Exception:
                                # Fallback: list original filename
                                figure_list.append(output_fig_jpg.name)
                        else:
                            figure_list.append(target_name_jpg)
                    else:
                        print(f"⚠️  Failed to include {fig_path.name}")
                
                # Create figure list file
                if figure_list:
                    with open(input_folder / "figure_list.txt", 'w') as f:
                        f.write("Figures to upload to Gemini (in order):\n\n")
                        for fig in figure_list:
                            f.write(f"- {fig}\n")
                        f.write("\nUpload these in the Gemini UI along with the prompt.\n")
                else:
                    # If we detected figures but none could be included, note it in README
                    pass
            
            # Create placeholder output JSON
            placeholder = create_placeholder_json(format_type, paper_id, version_label, run_id)
            placeholder_file = output_folder / "review.json"
            
            with open(placeholder_file, 'w', encoding='utf-8') as f:
                json.dump(placeholder, f, indent=2)
            
            # Create README
            readme_content = f"""# Manual Review for {paper_id} ({version_label}, Run {run_id})

## Steps to Complete This Review

1. **Read the prompt**:
   - Open `input/prompt.txt`
   - This contains the complete review instructions

2. **Go to Gemini Pro**:
   - Visit: https://aistudio.google.com/
   - Make sure you're using Gemini 1.5 Pro (supports images and long context)

3. **Upload figures** (if any):
   - Check `input/figures/` folder
   - Upload all figures to Gemini in the order listed in `input/figure_list.txt`

4. **Paste the prompt**:
   - Copy the entire content of `input/prompt.txt`
   - Paste into Gemini's chat interface

5. **Wait for response**:
   - Gemini will generate a JSON review
   - This may take 1-2 minutes

6. **Copy the output**:
   - Copy ONLY the JSON object from Gemini's response
   - Do NOT include any markdown formatting or explanatory text
   - Just the raw JSON: `{{"summary": "...", ...}}`

7. **Paste into output file**:
   - Open `output/review.json`
   - Replace the ENTIRE content with the JSON from Gemini
   - Save the file

8. **Verify**:
   - Make sure the JSON is valid (use a JSON validator if unsure)
   - Check that all required fields are present

9. **Process the outputs**:
   - After completing all reviews, run:
   - `python process_manual_gemini_outputs.py --input_dir {output_base_dir} --output_dir ./reviews_gemini_manual/`

## Format: {format_type}

## Number of figures: {len(figures)}

## Status: ⏳ PENDING
"""
            
            with open(run_folder / "README.md", 'w') as f:
                f.write(readme_content)
            
            success_count += 1
            
    except Exception as e:
        print(f"❌ Error preparing {paper_id} ({version_label}): {e}")
        error_count += 1
    
    return success_count, error_count

def main():
    parser = argparse.ArgumentParser(
        description="Prepare manual Gemini Pro reviews via web UI"
    )
    parser.add_argument(
        "--csv_file", type=str, required=True,
        help="Path to filtered_pairs.csv"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./manual_gemini_reviews/",
        help="Output directory for prepared materials"
    )
    parser.add_argument(
        "--format", type=str, choices=["default", "CriticalNeurIPS"], 
        default="CriticalNeurIPS",
        help="Review format (default: CriticalNeurIPS)"
    )
    parser.add_argument(
        "--max_figures", type=int, default=10,
        help="Maximum figures per paper (default: 10)"
    )
    parser.add_argument(
        "--num_runs", type=int, default=1,
        help="Number of review runs per paper (default: 1)"
    )
    parser.add_argument(
        "--version", type=str, choices=["v1", "latest", "both"], 
        default="both",
        help="Which version(s) to prepare (default: both)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of papers (for testing)"
    )
    
    args = parser.parse_args()
    
    # Load CSV
    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"❌ Error: CSV file not found at {args.csv_file}")
        exit(1)
    
    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to first {args.limit} papers for testing.\n")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Manual Gemini Pro Review Preparation")
    print("="*80)
    print(f"Papers: {len(df)}")
    print(f"Format: {args.format}")
    print(f"Max figures: {args.max_figures}")
    print(f"Runs per paper: {args.num_runs}")
    print(f"Version filter: {args.version}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Calculate total reviews needed
    total_reviews = len(df) * args.num_runs
    if args.version == "both":
        total_reviews *= 2
    print(f"Total manual reviews needed: {total_reviews}")
    print(f"Estimated time (at 2 min/review): {total_reviews * 2} minutes\n")
    
    total_success = 0
    total_errors = 0
    
    for idx, row in df.iterrows():
        paper_id = row['paperid']
        v1_folder = Path(row['v1_folder_path'])
        latest_folder = Path(row['latest_folder_path'])
        
        # Parse flaw descriptions
        flaw_descriptions = []
        if pd.notna(row.get('flaw_descriptions')) and row.get('flaw_descriptions'):
            import ast
            try:
                flaws = ast.literal_eval(row['flaw_descriptions'])
                if isinstance(flaws, list):
                    flaw_descriptions = [f.strip() for f in flaws if f.strip()]
            except:
                pass
        
        print(f"[{idx+1}/{len(df)}] Preparing {paper_id}...")
        
        # Prepare v1
        if args.version in ["v1", "both"]:
            success, errors = prepare_paper_review(
                paper_id, v1_folder, "v1", flaw_descriptions,
                output_dir, args.format, args.max_figures, args.num_runs
            )
            total_success += success
            total_errors += errors
        
        # Prepare latest
        if args.version in ["latest", "both"]:
            success, errors = prepare_paper_review(
                paper_id, latest_folder, "latest", flaw_descriptions,
                output_dir, args.format, args.max_figures, args.num_runs
            )
            total_success += success
            total_errors += errors
    
    print("\n" + "="*80)
    print("PREPARATION COMPLETE")
    print("="*80)
    print(f"✅ Successfully prepared: {total_success} reviews")
    print(f"❌ Errors: {total_errors}")
    print()
    print("Next steps:")
    print("1. Navigate to each paper folder in:", output_dir)
    print("2. Follow the README.md instructions in each run folder")
    print("3. Complete reviews manually using Gemini Pro UI")
    print("4. After all reviews are done, run:")
    print(f"   python scripts/utils/process_manual_gemini_outputs.py \\")
    print(f"     --input_dir {output_dir} \\")
    print(f"     --output_dir ./reviews_gemini_manual/")
    print("="*80)

if __name__ == "__main__":
    main()

