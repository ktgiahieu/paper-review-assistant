#!/usr/bin/env python3
"""
Script to review papers from specified folders using vLLM.

This script can review papers from multiple folders (e.g., latest/, authors_affiliation_good/, authors_affiliation_bad/)
and save reviews to an output directory with proper structure, using the same input/output format as review_papers_gemini.py
but using vLLM instead of Gemini API.
"""

import os
import json
import argparse
import time
import re
import requests
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple, Dict
import concurrent.futures

# --- Environment & API Configuration ---
load_dotenv()

# --- Constants ---
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2
RETRYABLE_STATUS_CODES = [429, 500, 502, 503, 504]

# --- Timeout Constants ---
MODEL_TIMEOUTS = {
    "default": 300,  # 5 minutes default
    "ReviewForm": 600,  # 10 minutes for critical analysis
    "CycleReviewer": 900,  # 15 minutes for 4 reviewers + meta review
}

# --- Pydantic Models for Structured Review Output ---
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

class ReviewFormReview(BaseModel):
    """Pydantic model for a comprehensive, critical NeurIPS-style review."""
    summary: str = Field(
        description="A brief, neutral summary of the paper and its contributions. This should not be a critique or a copy of the abstract."
    )
    strengths: List[str] = Field(
        description="An array of strings, each representing a single concise strength point. No nested lists, no numbering, no markdown."
    )
    weaknesses: List[str] = Field(
        description="An array of strings, each representing a single concise weakness point. No nested lists, no numbering, no markdown."
    )
    questions: List[str] = Field(
        description="An array of strings, each representing an actionable question or suggestion for the authors (ideally 3-5 key points). No numbering, no markdown."
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

class CycleReviewerIndividual(BaseModel):
    """Pydantic model for individual reviewer in CycleReviewer format."""
    summary: str = Field(description="Summary of the paper")
    soundness: str = Field(description="Soundness rating and description")
    presentation: str = Field(description="Presentation rating and description")
    contribution: str = Field(description="Contribution rating and description")
    strengths: List[str] = Field(description="List of paper strengths")
    weaknesses: List[str] = Field(description="List of paper weaknesses")
    questions: List[str] = Field(description="List of questions")
    flag_for_ethics_review: str = Field(description="Ethics review flag")
    rating: str = Field(description="Rating with justification")
    confidence: str = Field(description="Confidence level")

class CycleReviewerReview(BaseModel):
    """Pydantic model for CycleReviewer format review output."""
    reviewers: List[CycleReviewerIndividual] = Field(description="List of 4 individual reviews")
    meta_review: str = Field(description="Meta review summary")
    justification_for_why_not_higher_score: str = Field(description="Why not higher score")
    justification_for_why_not_lower_score: str = Field(description="Why not lower score")
    paper_decision: str = Field(description="Final decision (Accept/Reject)")

class ReviewPrompts:
    @staticmethod
    def get_system_prompt(format_type: str = "default") -> str:
        """Returns the system prompt for paper review based on format type."""
        if format_type == "ReviewForm":
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
        elif format_type == "CycleReviewer":
            return """You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers. For each paper submitted, conduct a comprehensive review addressing the following aspects:

1. Summary: Briefly outline main points and objectives.
2. Soundness: Assess methodology and logical consistency.
3. Presentation: Evaluate clarity, organization, and visual aids.
4. Contribution: Analyze significance and novelty in the field.
5. Strengths: Identify the paper's strongest aspects.
6. Weaknesses: Point out areas for improvement.
7. Questions: Pose questions for the authors.
8. Rating: Score 1-10, justify your rating.
9. Meta Review: Provide overall assessment and recommendation (Accept/Reject).

Maintain objectivity and provide specific examples from the paper to support your evaluation.

You need to fill out **4** review opinions."""
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
    def get_user_prompt(paper_content: str, paper_version: str, format_type: str = "default") -> str:
        """Constructs the user-facing prompt for review."""
        if format_type == "ReviewForm":
            return f"""Please review the following research paper with exceptional rigor and depth.

<paper_content>
{paper_content}
</paper_content>

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
        elif format_type == "CycleReviewer":
            return f"""<paper_content>
{paper_content}
</paper_content>

Please provide 4 comprehensive review opinions following the CycleReviewer format with markdown sections:
- ## Reviewer (for each of 4 reviewers)
  - ### Summary
  - ### Soundness
  - ### Presentation
  - ### Contribution
  - ### Strengths
  - ### Weaknesses
  - ### Questions
  - ### Flag for Ethics Review
  - ### Rating
  - ### Confidence
- ## Meta Review
  - ### Justification for Why Not Higher Score
  - ### Justification for Why Not Lower Score
- ## Paper Decision"""
        else:
            return f"""
Please review the following research paper ({paper_version}):

<paper_content>
{paper_content}
</paper_content>

Provide a comprehensive review following the specified JSON format.
"""

def _parse_cyclereviewer_format(content: str) -> dict:
    """
    Parses CycleReviewer markdown format into structured dictionary.
    
    Expected format: Multiple "## Reviewer" sections followed by "## Meta Review" and "## Paper Decision"
    """
    result = {
        "reviewers": [],
        "meta_review": "",
        "justification_for_why_not_higher_score": "",
        "justification_for_why_not_lower_score": "",
        "paper_decision": ""
    }
    
    # Split by ## headings
    sections = re.split(r'\n## (.+?)\n', content)
    
    i = 1  # Start from first section name
    while i < len(sections):
        section_name = sections[i].strip()
        section_content = sections[i+1] if i+1 < len(sections) else ""
        
        if section_name.lower() == "reviewer":
            # Parse individual reviewer
            reviewer = {
                "summary": "",
                "soundness": "",
                "presentation": "",
                "contribution": "",
                "strengths": [],
                "weaknesses": [],
                "questions": [],
                "flag_for_ethics_review": "",
                "rating": "",
                "confidence": ""
            }
            
            # Split by ### subsections
            subsections = re.split(r'\n### (.+?)\n', section_content)
            for j in range(1, len(subsections), 2):
                if j+1 < len(subsections):
                    subsection_name = subsections[j].lower().strip()
                    subsection_content = subsections[j+1].strip()
                    
                    if subsection_name == "summary":
                        reviewer["summary"] = subsection_content
                    elif subsection_name == "soundness":
                        reviewer["soundness"] = subsection_content
                    elif subsection_name == "presentation":
                        reviewer["presentation"] = subsection_content
                    elif subsection_name == "contribution":
                        reviewer["contribution"] = subsection_content
                    elif subsection_name == "strengths":
                        # Extract numbered or bulleted list (-, *, •, or 1., 2., etc.)
                        strengths = re.findall(r'^\s*(?:\d+\.|\-|\*|•)\s*(.+)$', subsection_content, re.MULTILINE)
                        reviewer["strengths"] = [s.strip() for s in strengths if s.strip()]
                    elif subsection_name == "weaknesses":
                        weaknesses = re.findall(r'^\s*(?:\d+\.|\-|\*|•)\s*(.+)$', subsection_content, re.MULTILINE)
                        reviewer["weaknesses"] = [w.strip() for w in weaknesses if w.strip()]
                    elif subsection_name == "questions":
                        questions = re.findall(r'^\s*(?:\d+\.|\-|\*|•)\s*(.+)$', subsection_content, re.MULTILINE)
                        reviewer["questions"] = [q.strip() for q in questions if q.strip()]
                    elif subsection_name == "flag for ethics review":
                        reviewer["flag_for_ethics_review"] = subsection_content
                    elif subsection_name == "rating":
                        reviewer["rating"] = subsection_content
                    elif subsection_name == "confidence":
                        reviewer["confidence"] = subsection_content
            
            result["reviewers"].append(reviewer)
        
        elif section_name.lower() == "meta review":
            # Parse meta review section
            # The content includes justifications, so need to extract them
            subsections = re.split(r'\n### (.+?)\n', section_content)
            
            # First part before any ### is the main meta review
            if subsections:
                result["meta_review"] = subsections[0].strip()
            
            # Extract justifications
            for j in range(1, len(subsections), 2):
                if j+1 < len(subsections):
                    subsection_name = subsections[j].lower().strip()
                    subsection_content = subsections[j+1].strip()
                    
                    if "higher" in subsection_name:
                        result["justification_for_why_not_higher_score"] = subsection_content
                    elif "lower" in subsection_name:
                        result["justification_for_why_not_lower_score"] = subsection_content
        
        elif section_name.lower() == "paper decision":
            result["paper_decision"] = section_content.strip()
        
        i += 2  # Move to next section
    
    return result

def _sanitize_json_string(json_str: str) -> str:
    """Cleans common JSON errors from LLM output."""
    # Remove markdown code blocks
    json_str = json_str.strip()
    json_str = re.sub(r'^```json\s*', '', json_str, flags=re.MULTILINE)
    json_str = re.sub(r'^```\s*$', '', json_str, flags=re.MULTILINE)
    json_str = json_str.strip()
    
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',\s*(?=[}\]])', '', json_str)
    
    # Try to parse and re-serialize to fix escape issues
    try:
        temp = json.loads(json_str)
        json_str = json.dumps(temp, ensure_ascii=False)
    except json.JSONDecodeError:
        pass
    
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
    folder_label: str,
    vllm_endpoint: str,
    model_name: str,
    verbose: bool,
    run_id: int = 0,
    format_type: str = "default",
    timeout: Optional[int] = None
) -> dict:
    """
    Reviews a single paper and returns structured results using vLLM.
    
    Args:
        paper_id: Paper identifier
        paper_path: Path to paper directory
        folder_label: Label for the folder (e.g., "latest", "authors_affiliation_good")
        vllm_endpoint: vLLM server endpoint URL
        model_name: Model name
        verbose: Enable verbose output
        run_id: Run ID for multiple runs
        format_type: Review format type ("default" or "ReviewForm")
        timeout: Request timeout in seconds (default: model-specific)
    """
    worker_id = concurrent.futures.thread.get_ident() if hasattr(concurrent.futures.thread, 'get_ident') else os.getpid()
    _print_method = tqdm.write if not verbose else print
    
    # Get timeout for this format type
    if timeout is None:
        timeout = MODEL_TIMEOUTS.get(format_type, MODEL_TIMEOUTS["default"])
    
    try:
        # Read paper content
        paper_md = find_paper_markdown(paper_path)
        if not paper_md:
            return {
                "error": f"Could not find paper markdown for {paper_id} at {paper_path}",
                "paper_id": paper_id,
                "folder": folder_label,
                "run_id": run_id,
                "success": False
            }
        
        with open(paper_md, 'r', encoding='utf-8') as f:
            paper_content = f.read()
        
        system_prompt = ReviewPrompts.get_system_prompt(format_type)
        user_prompt_text = ReviewPrompts.get_user_prompt(paper_content, folder_label, format_type)
        
        response_obj = None
        response_text = None
        last_exception = None
        parsed_review = None
        
        # Retry loop: includes both API calls and JSON parsing
        for attempt in range(MAX_RETRIES):
            try:
                if verbose:
                    _print_method(f"Worker {worker_id}: Reviewing {paper_id} ({folder_label}), attempt {attempt + 1}/{MAX_RETRIES}")
                
                # Build messages in OpenAI format (vLLM compatible)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_text}
                ]
                
                # Call vLLM endpoint
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": 4096,
                    "temperature": 0.7,
                }
                
                response = requests.post(
                    f"{vllm_endpoint}/v1/chat/completions",
                    json=payload,
                    timeout=timeout,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    response_obj = response.json()
                    if 'choices' in response_obj and len(response_obj['choices']) > 0:
                        response_text = response_obj['choices'][0]['message']['content']
                    else:
                        raise ValueError("Invalid response format from vLLM")
                    
                    # For CycleReviewer, parse markdown format directly
                    if format_type == "CycleReviewer":
                        try:
                            parsed_data = _parse_cyclereviewer_format(response_text)
                            parsed_review = CycleReviewerReview.model_validate(parsed_data)
                            # Success! Break out of retry loop
                            break
                        except Exception as cycle_error:
                            if attempt < MAX_RETRIES - 1:
                                _print_method(f"Worker {worker_id}: CycleReviewer parsing failed for {paper_id} ({folder_label}), attempt {attempt + 1}/{MAX_RETRIES}. Retrying...")
                                _print_method(f"Worker {worker_id}: Error: {cycle_error}")
                                wait_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                                time.sleep(wait_time)
                                continue
                            else:
                                # Last attempt failed, will be handled below
                                last_exception = cycle_error
                                break
                    
                    # For JSON formats, continue with JSON parsing
                elif response.status_code in RETRYABLE_STATUS_CODES:
                    wait_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                    _print_method(f"Worker {worker_id}: Retrying {paper_id} ({folder_label}) in {wait_time}s due to status {response.status_code}...")
                    time.sleep(wait_time)
                    continue
                else:
                    error_text = response.text
                    _print_method(f"Worker {worker_id}: Non-retryable error for {paper_id} ({folder_label}): {response.status_code} - {error_text}")
                    break
                
                # Try to parse JSON (for default and ReviewForm formats)
                if format_type != "CycleReviewer":
                    raw_json_content = response_text
                    sanitized_json_content = _sanitize_json_string(raw_json_content)
                    
                    # Try Pydantic validation first
                    try:
                        # Choose the appropriate Pydantic model based on format
                        if format_type == "ReviewForm":
                            parsed_review = ReviewFormReview.model_validate_json(sanitized_json_content)
                        else:
                            parsed_review = PaperReview.model_validate_json(sanitized_json_content)
                        # Success! Break out of retry loop
                        break
                    except Exception as pydantic_error:
                        # If Pydantic fails, try regular JSON parsing as fallback
                        try:
                            parsed_dict = json.loads(sanitized_json_content)
                            
                            # Fix common format issues: strengths/weaknesses/questions as dicts with "point" key
                            # Also handle cases where questions might be a string instead of array
                            for field in ['strengths', 'weaknesses', 'questions']:
                                if field in parsed_dict and parsed_dict[field]:
                                    # If it's a string, try to split it (for questions field)
                                    if isinstance(parsed_dict[field], str):
                                        # Try to split by newlines or common separators
                                        if format_type == "ReviewForm" and field == "questions":
                                            # Split questions string into array
                                            questions_list = [q.strip() for q in parsed_dict[field].split('\n') if q.strip()]
                                            if questions_list:
                                                parsed_dict[field] = questions_list
                                            else:
                                                # Single question as string
                                                parsed_dict[field] = [parsed_dict[field]]
                                        continue
                                    
                                    # Check if it's a list of dicts with "point" key
                                    if isinstance(parsed_dict[field], list) and len(parsed_dict[field]) > 0:
                                        if isinstance(parsed_dict[field][0], dict) and 'point' in parsed_dict[field][0]:
                                            # Convert [{"point": "..."}, ...] to ["...", ...]
                                            parsed_dict[field] = [item.get('point', str(item)) if isinstance(item, dict) else item for item in parsed_dict[field]]
                                        
                                        # Also handle if items are strings but need cleaning
                                        parsed_dict[field] = [str(item).strip() for item in parsed_dict[field] if str(item).strip()]
                            
                            # Create a valid review object from the dict based on format type
                            if format_type == "ReviewForm":
                                parsed_review = ReviewFormReview(**parsed_dict)
                            else:
                                parsed_review = PaperReview(**parsed_dict)
                            break
                        except (json.JSONDecodeError, ValueError) as json_e:
                            # JSON parsing failed - this is a retryable error
                            if attempt < MAX_RETRIES - 1:
                                _print_method(f"Worker {worker_id}: JSON parsing failed for {paper_id} ({folder_label}), attempt {attempt + 1}/{MAX_RETRIES}. Retrying...")
                                _print_method(f"Worker {worker_id}: Error: {json_e}")
                                response_text = None  # Reset to trigger retry
                                wait_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                                time.sleep(wait_time)
                                continue
                            else:
                                # Last attempt failed, raise to be caught below
                                raise json_e
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                _print_method(f"Worker {worker_id}: Request error for {paper_id} ({folder_label}), attempt {attempt + 1}/{MAX_RETRIES}: {e}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    break
            except Exception as e:
                last_exception = e
                _print_method(f"Worker {worker_id}: Error for {paper_id} ({folder_label}), attempt {attempt + 1}/{MAX_RETRIES}: {e}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    break
        
        # Check if we successfully parsed the review
        if parsed_review is not None:
            # Successfully parsed
            review_data = parsed_review.model_dump()
            
            # Add metadata
            review_data["paper_id"] = paper_id
            review_data["folder"] = folder_label
            review_data["run_id"] = run_id
            review_data["success"] = True
            
            # Add score mappings for compatibility with evaluation scripts
            if format_type == "ReviewForm":
                # ReviewForm format already has soundness, presentation, contribution
                review_data["rating"] = review_data.get("overall_score")
            elif format_type == "CycleReviewer":
                # CycleReviewer format: extract scores from reviewers if needed
                # For evaluation, we might need to aggregate scores from the 4 reviewers
                # For now, we'll keep the structure as-is since it's different from other formats
                # The evaluation scripts may need to handle CycleReviewer format specially
                pass
            else:
                # Default format - map to standard names
                review_data["soundness"] = review_data.get("technical_quality_score")
                review_data["presentation"] = review_data.get("clarity_score")
                review_data["contribution"] = review_data.get("novelty_score")
                review_data["rating"] = review_data.get("overall_score")
            
            if verbose:
                _print_method(f"Worker {worker_id}: Successfully reviewed {paper_id} ({folder_label})")
            
            return review_data
        
        # If we get here, all attempts failed
        if response_obj is None:
            err_msg = f"All API attempts failed for {paper_id} ({folder_label})."
            if last_exception:
                err_msg += f" Last error: {type(last_exception).__name__} - {last_exception}"
            return {
                "error": err_msg,
                "paper_id": paper_id,
                "folder": folder_label,
                "run_id": run_id,
                "success": False
            }
        else:
            # API call succeeded but JSON parsing failed after all retries
            if response_text is None:
                return {
                    "error": f"API call failed for {paper_id} ({folder_label})",
                    "paper_id": paper_id,
                    "folder": folder_label,
                    "run_id": run_id,
                    "success": False
                }
            
            sanitized_json_content = _sanitize_json_string(response_text)
            _print_method(f"Worker {worker_id}: Failed to parse JSON after {MAX_RETRIES} attempts for {paper_id} ({folder_label})")
            _print_method(f"Worker {worker_id}: Raw JSON (first 500 chars): {sanitized_json_content[:500]}...")
            
            # Last resort: try to extract whatever we can
            try:
                fallback_data = json.loads(sanitized_json_content)
                
                # Fix common format issues
                for field in ['strengths', 'weaknesses', 'questions']:
                    if field in fallback_data and fallback_data[field]:
                        if isinstance(fallback_data[field], str):
                            if format_type == "ReviewForm" and field == "questions":
                                questions_list = [q.strip() for q in fallback_data[field].split('\n') if q.strip()]
                                if questions_list:
                                    fallback_data[field] = questions_list
                                else:
                                    fallback_data[field] = [fallback_data[field]]
                            continue
                        
                        if isinstance(fallback_data[field], list) and len(fallback_data[field]) > 0:
                            if isinstance(fallback_data[field][0], dict) and 'point' in fallback_data[field][0]:
                                fallback_data[field] = [item.get('point', str(item)) if isinstance(item, dict) else item for item in fallback_data[field]]
                            fallback_data[field] = [str(item).strip() for item in fallback_data[field] if str(item).strip()]
                
                # Add metadata
                fallback_data["paper_id"] = paper_id
                fallback_data["folder"] = folder_label
                fallback_data["run_id"] = run_id
                fallback_data["success"] = True
                fallback_data["__parsing_warning"] = "JSON parsed but Pydantic validation may have failed"
                
                # Add score mappings for compatibility with evaluation scripts
                if format_type == "ReviewForm":
                    fallback_data["rating"] = fallback_data.get("overall_score")
                elif format_type == "CycleReviewer":
                    # CycleReviewer format: keep structure as-is
                    pass
                else:
                    fallback_data["soundness"] = fallback_data.get("technical_quality_score")
                    fallback_data["presentation"] = fallback_data.get("clarity_score")
                    fallback_data["contribution"] = fallback_data.get("novelty_score")
                    fallback_data["rating"] = fallback_data.get("overall_score")
                
                return fallback_data
            except:
                return {
                    "error": f"Failed to parse JSON from LLM after {MAX_RETRIES} attempts",
                    "paper_id": paper_id,
                    "folder": folder_label,
                    "run_id": run_id,
                    "raw_content": response_text[:1000] if response_text else None,
                    "last_exception": str(last_exception) if last_exception else None,
                    "success": False
                }
    
    except Exception as e:
        message = f"FATAL ERROR reviewing {paper_id} ({folder_label}): {type(e).__name__} - {e}"
        _print_method(f"Worker {worker_id}: {message}")
        import traceback
        _print_method(traceback.format_exc())
        return {
            "error": message,
            "paper_id": paper_id,
            "folder": folder_label,
            "run_id": run_id,
            "success": False
        }

def review_papers_in_folder(
    base_dir: Path,
    folder_name: str,
    output_dir: Path,
    vllm_endpoint: str,
    model_name: str,
    verbose: bool,
    skip_existing: bool = False,
    num_runs: int = 1,
    max_workers: int = 3,
    format_type: str = "default",
    timeout: Optional[int] = None
) -> List[dict]:
    """
    Review all papers in a specific folder using parallel processing with vLLM.
    
    Args:
        base_dir: Base directory (e.g., data/ICLR2024)
        folder_name: Folder name to review (e.g., "latest", "authors_affiliation_good")
        output_dir: Output directory for reviews
        vllm_endpoint: vLLM server endpoint URL
        model_name: Model name
        verbose: Enable verbose output
        skip_existing: Skip papers that already have review files
        num_runs: Number of times to review each paper
        max_workers: Maximum number of worker threads
        format_type: Review format type ("default" or "ReviewForm")
        timeout: Request timeout in seconds (default: model-specific)
    """
    folder_path = base_dir / folder_name
    if not folder_path.exists():
        print(f"Warning: Folder {folder_path} does not exist, skipping...")
        return []
    
    # Get all paper directories
    paper_dirs = [d for d in folder_path.iterdir() if d.is_dir()]
    print(f"\nFound {len(paper_dirs)} papers in {folder_name}/")
    
    # Create output directory for this folder
    folder_output_dir = output_dir / folder_name
    folder_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare tasks: (paper_id, paper_dir, run_id, review_file)
    tasks = []
    
    for paper_dir in paper_dirs:
        paper_id = paper_dir.name.split('_')[0]
        
        for run_id in range(num_runs):
            # Check if review already exists
            review_file = folder_output_dir / paper_id / f"review_run{run_id}.json"
            if skip_existing and review_file.exists():
                if verbose:
                    print(f"Skipping {paper_id} ({folder_name}) run {run_id} - already exists")
                continue
            
            tasks.append((paper_id, paper_dir, run_id, review_file))
    
    if not tasks:
        print(f"No tasks to process for {folder_name}/")
        return []
    
    print(f"Prepared {len(tasks)} review tasks for {folder_name}/")
    
    reviews = []
    
    # Process tasks in parallel
    print(f"Processing {len(tasks)} tasks in parallel (max_workers={max_workers})...")
    
    def process_task(task):
        paper_id, paper_dir, run_id, review_file = task
        
        # Review paper
        review_data = review_single_paper(
            paper_id=paper_id,
            paper_path=paper_dir,
            folder_label=folder_name,
            vllm_endpoint=vllm_endpoint,
            model_name=model_name,
            verbose=verbose,
            run_id=run_id,
            format_type=format_type,
            timeout=timeout
        )
        
        # Save review
        if review_data.get("success"):
            paper_output_dir = folder_output_dir / paper_id
            paper_output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(review_file, 'w', encoding='utf-8') as f:
                json.dump(review_data, f, indent=2, ensure_ascii=False)
        
        return review_data
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_task, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc=f"Reviewing {folder_name}"):
            try:
                review_data = future.result()
                reviews.append(review_data)
            except Exception as e:
                task = futures[future]
                paper_id = task[0]
                print(f"Error processing {paper_id}: {e}")
                reviews.append({
                    "error": str(e),
                    "paper_id": paper_id,
                    "folder": folder_name,
                    "success": False
                })
    
    return reviews

def main():
    parser = argparse.ArgumentParser(
        description="Review papers from specified folders using vLLM"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing the folders (e.g., data/ICLR2024 or sampled_data/ICLR2024)"
    )
    parser.add_argument(
        "--folders",
        type=str,
        nargs="+",
        required=True,
        help="Folder names to review (e.g., latest authors_affiliation_good authors_affiliation_bad)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for reviews (default: base_dir/../reviews/)"
    )
    parser.add_argument(
        "--vllm_endpoint",
        type=str,
        required=True,
        help="vLLM server endpoint URL (e.g., http://localhost:8000)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name hosted on vLLM server"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=3,
        help="Max worker threads for concurrent processing (default: 3)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of times to review each paper (default: 1)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip papers that already have review files"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["default", "ReviewForm", "CycleReviewer"],
        default="default",
        help="Review format: 'default', 'ReviewForm', or 'CycleReviewer' (default: default)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Request timeout in seconds (default: model-specific)"
    )
    
    args = parser.parse_args()
    
    # Test vLLM connection
    try:
        response = requests.get(f"{args.vllm_endpoint}/health", timeout=10)
        if response.status_code != 200:
            print(f"Warning: vLLM health check returned status {response.status_code}")
    except Exception as e:
        print(f"Warning: Could not connect to vLLM endpoint at {args.vllm_endpoint}: {e}")
        print("Proceeding anyway, but API calls may fail.")
    
    base_dir = Path(args.base_dir)
    
    # Set default output directory
    if args.output_dir is None:
        output_dir = base_dir.parent / "reviews" / base_dir.name
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Base directory: {base_dir}")
    print(f"Folders to review: {args.folders}")
    print(f"Output directory: {output_dir}")
    print(f"vLLM Endpoint: {args.vllm_endpoint}")
    print(f"Model: {args.model_name}")
    print(f"Format: {args.format}")
    print(f"Max workers: {args.max_workers}")
    print(f"Number of runs per paper: {args.num_runs}")
    
    # Show timeout info
    if args.timeout:
        print(f"Request timeout: {args.timeout}s (custom)")
    else:
        default_timeout = MODEL_TIMEOUTS.get(args.format, MODEL_TIMEOUTS["default"])
        print(f"Request timeout: {default_timeout}s (format-specific default)")
    print()
    
    # Review papers in each folder
    all_reviews = []
    for folder_name in args.folders:
        reviews = review_papers_in_folder(
            base_dir=base_dir,
            folder_name=folder_name,
            output_dir=output_dir,
            vllm_endpoint=args.vllm_endpoint,
            model_name=args.model_name,
            verbose=args.verbose,
            skip_existing=args.skip_existing,
            num_runs=args.num_runs,
            max_workers=args.max_workers,
            format_type=args.format,
            timeout=args.timeout
        )
        all_reviews.extend(reviews)
    
    # Print summary
    successful = sum(1 for r in all_reviews if r.get("success", False))
    failed = len(all_reviews) - successful
    
    print("\n" + "="*60)
    print("Review Summary:")
    print(f"  Total reviews: {len(all_reviews)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Results saved in: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

