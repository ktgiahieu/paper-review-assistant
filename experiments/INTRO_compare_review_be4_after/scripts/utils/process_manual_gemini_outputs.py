#!/usr/bin/env python3
"""
Process Manual Gemini Outputs

This script processes manually-pasted Gemini outputs and formats them
for compatibility with the evaluation pipeline.

Usage:
    python process_manual_gemini_outputs.py \
        --input_dir ./manual_gemini_reviews/ \
        --output_dir ./reviews_gemini_manual/ \
        --format CriticalNeurIPS
"""

import os
import json
import argparse
import re
from pathlib import Path
from typing import Dict, Optional
from pydantic import BaseModel, Field, ValidationError
from typing import List

# --- Pydantic Models ---
class PaperReview(BaseModel):
    """Default review format."""
    summary: str
    strengths: List[str]
    weaknesses: List[str]
    clarity_score: int = Field(ge=1, le=10)
    novelty_score: int = Field(ge=1, le=10)
    technical_quality_score: int = Field(ge=1, le=10)
    experimental_rigor_score: int = Field(ge=1, le=10)
    overall_score: int = Field(ge=1, le=10)
    confidence: int = Field(ge=1, le=5)
    recommendation: str
    detailed_comments: str

class CriticalNeurIPSStructured(BaseModel):
    """CriticalNeurIPS format with lists for strengths/weaknesses/questions."""
    summary: str
    strengths: List[str]
    weaknesses: List[str]
    questions: List[str]
    limitations_and_societal_impact: str
    soundness: int = Field(ge=1, le=4)
    presentation: int = Field(ge=1, le=4)
    contribution: int = Field(ge=1, le=4)
    overall_score: int = Field(ge=1, le=10)
    confidence: int = Field(ge=1, le=5)

class CriticalNeurIPSLegacy(BaseModel):
    """Legacy CriticalNeurIPS format with combined text fields."""
    summary: str
    strengths_and_weaknesses: str
    questions: str
    limitations_and_societal_impact: str
    soundness: int = Field(ge=1, le=4)
    presentation: int = Field(ge=1, le=4)
    contribution: int = Field(ge=1, le=4)
    overall_score: int = Field(ge=1, le=10)
    confidence: int = Field(ge=1, le=5)

def _split_bullets_or_numbered(text: str) -> List[str]:
    """Split a block of text into list items using bullets or numbering."""
    if not text:
        return []
    lines = text.splitlines()
    items: List[str] = []
    buf: List[str] = []
    def flush():
        if buf:
            joined = ' '.join([s.strip() for s in buf]).strip(' -•\t')
            if joined:
                items.append(joined)
            buf.clear()
    for ln in lines:
        stripped = ln.strip()
        if re.match(r"^([\-*•]|\d+[\.)])\s+", stripped):
            flush()
            buf.append(re.sub(r"^([\-*•]|\d+[\.)])\s+", "", stripped))
        else:
            if stripped:
                buf.append(stripped)
            else:
                flush()
    flush()
    # Filter empties and limit overly long items
    return [it for it in (s.strip() for s in items) if it]

def _parse_strengths_weaknesses(combined: str) -> (List[str], List[str]):
    """Parse a combined strengths_and_weaknesses string to two lists."""
    if not combined:
        return [], []
    # Try to split by headings
    parts = re.split(r"(?i)\*\*?\s*strengths\s*\*\*?|##\s*strengths|\bstrengths\b\s*:|\n\s*strengths\s*\n", combined)
    strengths_text = ''
    weaknesses_text = ''
    if len(parts) > 1:
        after_strengths = parts[1]
        wk_parts = re.split(r"(?i)\*\*?\s*weaknesses\s*\*\*?|##\s*weaknesses|\bweaknesses\b\s*:|\n\s*weaknesses\s*\n", after_strengths)
        if len(wk_parts) > 1:
            strengths_text = wk_parts[0]
            weaknesses_text = wk_parts[1]
        else:
            strengths_text = after_strengths
    else:
        # Fallback: try to split bullets roughly in half
        bullets = _split_bullets_or_numbered(combined)
        if bullets:
            mid = max(1, len(bullets)//2)
            return bullets[:mid], bullets[mid:]
        return [], []
    strengths = _split_bullets_or_numbered(strengths_text)
    weaknesses = _split_bullets_or_numbered(weaknesses_text)
    return strengths, weaknesses

def sanitize_json_string(json_str: str) -> str:
    """Clean common JSON errors from manual input."""
    # Remove markdown code blocks
    json_str = json_str.strip()
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    if json_str.startswith("```"):
        json_str = json_str[3:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    
    json_str = json_str.strip()
    
    # Remove trailing commas
    json_str = re.sub(r',\s*(?=[}\]])', '', json_str)
    
    # Fix common escape issues
    def fix_escapes(match):
        escaped_char = match.group(1)
        if escaped_char in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']:
            return match.group(0)
        return '\\\\' + escaped_char
    
    json_str = re.sub(r'\\(.)', fix_escapes, json_str)
    
    return json_str

def process_review_file(
    review_file: Path,
    format_type: str,
    paper_id: str,
    version: str,
    run_id: int
) -> Optional[Dict]:
    """
    Process a single manual review JSON file.
    
    Returns formatted review dict or None if invalid.
    """
    try:
        with open(review_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if it's still a placeholder
        if '_instructions' in content or '_status' in content:
            return None  # Skipping placeholder
        
        # Sanitize and parse
        sanitized = sanitize_json_string(content)
        
        try:
            raw_data = json.loads(sanitized)
        except json.JSONDecodeError as e:
            print(f"  ❌ JSON decode error in {review_file.name}: {e}")
            print(f"     First 200 chars: {sanitized[:200]}")
            return None
        
        # Validate with Pydantic
        try:
            if format_type == "CriticalNeurIPS":
                # Prefer the structured list format
                try:
                    validated_review = CriticalNeurIPSStructured.model_validate(raw_data)
                    review_data = validated_review.model_dump()
                except ValidationError:
                    # Attempt legacy format and convert
                    legacy = CriticalNeurIPSLegacy.model_validate(raw_data)
                    strengths, weaknesses = _parse_strengths_weaknesses(legacy.strengths_and_weaknesses)
                    questions_list = _split_bullets_or_numbered(legacy.questions)
                    review_data = {
                        "summary": legacy.summary,
                        "strengths": strengths,
                        "weaknesses": weaknesses,
                        "questions": questions_list,
                        "limitations_and_societal_impact": legacy.limitations_and_societal_impact,
                        "soundness": legacy.soundness,
                        "presentation": legacy.presentation,
                        "contribution": legacy.contribution,
                        "overall_score": legacy.overall_score,
                        "confidence": legacy.confidence,
                    }
                # Add score mappings for evaluation compatibility
                review_data["rating"] = review_data["overall_score"]
                
            else:
                # Default format
                validated_review = PaperReview.model_validate(raw_data)
                review_data = validated_review.model_dump()
                
                # Add score mappings for evaluation compatibility
                review_data["soundness"] = review_data["technical_quality_score"]
                review_data["presentation"] = review_data["clarity_score"]
                review_data["contribution"] = review_data["novelty_score"]
                review_data["rating"] = review_data["overall_score"]
            
            # Add metadata
            review_data["paper_id"] = paper_id
            review_data["version"] = version
            review_data["run_id"] = run_id
            review_data["model_type"] = f"gemini_manual_{format_type}"
            review_data["success"] = True
            review_data["source"] = "manual_gemini_ui"
            
            return review_data
            
        except ValidationError as e:
            print(f"  ❌ Validation error in {review_file.name}:")
            print(f"     {e}")
            
            # Try fallback with partial data
            try:
                review_data = raw_data.copy()
                review_data["paper_id"] = paper_id
                review_data["version"] = version
                review_data["run_id"] = run_id
                review_data["model_type"] = f"gemini_manual_{format_type}"
                review_data["success"] = False
                review_data["validation_error"] = str(e)
                review_data["source"] = "manual_gemini_ui"
                
                # Try to add score mappings if possible
                if format_type == "CriticalNeurIPS":
                    if "overall_score" in review_data:
                        review_data["rating"] = review_data["overall_score"]
                else:
                    if "technical_quality_score" in review_data:
                        review_data["soundness"] = review_data["technical_quality_score"]
                    if "clarity_score" in review_data:
                        review_data["presentation"] = review_data["clarity_score"]
                    if "novelty_score" in review_data:
                        review_data["contribution"] = review_data["novelty_score"]
                    if "overall_score" in review_data:
                        review_data["rating"] = review_data["overall_score"]
                
                print(f"     Using partial data with validation errors noted")
                return review_data
                
            except:
                return None
    
    except Exception as e:
        print(f"  ❌ Error processing {review_file}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Process manual Gemini outputs and format for evaluation"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Input directory with manual_gemini_reviews structure"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./reviews_gemini_manual/",
        help="Output directory for formatted reviews"
    )
    parser.add_argument(
        "--format", type=str, choices=["default", "CriticalNeurIPS"],
        default=None,
        help="Expected format (auto-detect if not specified)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"❌ Error: Input directory not found: {input_dir}")
        exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Processing Manual Gemini Outputs")
    print("="*80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Scan for review files
    review_files = list(input_dir.glob("*/*/output/review.json"))
    
    if not review_files:
        print("❌ No review files found!")
        print(f"   Expected structure: {input_dir}/<paper_id>/<version>_run<N>/output/review.json")
        exit(1)
    
    print(f"Found {len(review_files)} review files\n")
    
    processed_count = 0
    skipped_placeholders = 0
    error_count = 0
    
    for review_file in review_files:
        # Extract metadata from path
        # Structure: input_dir/paper_id/version_runN/output/review.json
        parts = review_file.parts
        paper_id = parts[-4]
        version_run = parts[-3]
        
        # Parse version_runN
        match = re.match(r'(.+)_run(\d+)', version_run)
        if not match:
            print(f"⚠️  Skipping {review_file} (invalid folder name format)")
            error_count += 1
            continue
        
        version = match.group(1)
        run_id = int(match.group(2))
        
        # Auto-detect format if not specified
        format_type = args.format
        if not format_type:
            # Try to detect from README or placeholder
            try:
                placeholder_file = review_file
                with open(placeholder_file, 'r') as f:
                    content = f.read()
                    if '"_format"' in content:
                        format_match = re.search(r'"_format":\s*"([^"]+)"', content)
                        if format_match:
                            format_type = format_match.group(1)
            except:
                pass
            
            if not format_type:
                format_type = "CriticalNeurIPS"  # Default assumption
        
        print(f"Processing {paper_id} ({version}, run {run_id})...")
        
        # Process the review
        review_data = process_review_file(review_file, format_type, paper_id, version, run_id)
        
        if review_data is None:
            if '_instructions' in open(review_file).read():
                print(f"  ⏭️  Skipped (still placeholder)")
                skipped_placeholders += 1
            else:
                print(f"  ❌ Failed to process")
                error_count += 1
            continue
        
        # Save to output directory
        paper_output_dir = output_dir / paper_id
        paper_output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = paper_output_dir / f"{version}_review_run{run_id}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(review_data, f, indent=2, ensure_ascii=False)
        
        status = "✅" if review_data.get("success") else "⚠️ "
        print(f"  {status} Saved to {output_file}")
        processed_count += 1
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"✅ Successfully processed: {processed_count}")
    print(f"⏭️  Skipped (placeholders): {skipped_placeholders}")
    print(f"❌ Errors: {error_count}")
    print()
    
    if processed_count > 0:
        print("Reviews are now ready for evaluation!")
        print()
        print("Run evaluation scripts:")
        print(f"  python scripts/evaluation/evaluate_numerical_scores.py \\")
        print(f"    --reviews_dir {output_dir}")
        print()
        print(f"  python scripts/evaluation/evaluate_flaw_detection.py \\")
        print(f"    --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \\")
        print(f"    --reviews_dir {output_dir}")
    
    if skipped_placeholders > 0:
        print()
        print(f"⚠️  Note: {skipped_placeholders} reviews are still placeholders")
        print("   Complete those reviews and run this script again")
    
    print("="*80)

if __name__ == "__main__":
    main()

