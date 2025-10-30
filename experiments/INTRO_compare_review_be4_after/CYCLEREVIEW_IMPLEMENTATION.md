# CycleReviewer Implementation Summary

## Overview

Added support for **CycleReviewer-Llama-3.1-70B** model with multi-reviewer format output.

## Model Characteristics

**Format:** Markdown with multiple reviewers and meta review  
**Structure:** 4 individual reviewers + meta review + paper decision  
**Detection:** Model names containing "cyclereviewer", "cycle-reviewer", or "cycle_reviewer"

## Implementation

### 1. Pydantic Models

```python
class CycleReviewerIndividual(BaseModel):
    """Individual reviewer's opinion."""
    summary: str
    soundness: str
    presentation: str
    contribution: str
    strengths: List[str]
    weaknesses: List[str]
    questions: List[str]
    flag_for_ethics_review: str
    rating: str
    confidence: str

class CycleReviewerReview(BaseModel):
    """Complete review with multiple reviewers."""
    reviewers: List[CycleReviewerIndividual]  # 4 reviewers
    meta_review: str
    justification_for_why_not_higher_score: str
    justification_for_why_not_lower_score: str
    paper_decision: str
```

### 2. System Prompt

```python
"""You are an expert academic reviewer tasked with providing a thorough and balanced 
evaluation of research papers. For each paper submitted, conduct a comprehensive review 
addressing the following aspects:

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
```

### 3. Parser Function

`_parse_cyclereviewer_format(content: str) -> dict`

**Parsing Strategy:**
1. Split by `## Heading` to find major sections (Reviewer, Meta Review, Paper Decision)
2. For each Reviewer section:
   - Split by `### Subheading` to find fields
   - Extract bullet/numbered lists for strengths, weaknesses, questions
   - Store as dictionary
3. For Meta Review:
   - Extract main meta review text
   - Find justification subsections
4. Extract final Paper Decision

**Key Regex Patterns:**
- Section headers: `\n## (.+?)\n`
- Subsection headers: `\n### (.+?)\n`
- List items: `^\s*(?:\d+\.|\-)\s*(.+)$` (handles both numbered and bulleted)

### 4. Integration

Added to `review_single_paper_vllm()` parsing logic:

```python
elif model_type == "CycleReviewer":
    parsed_data = _parse_cyclereviewer_format(raw_content)
    
    try:
        validated_review = CycleReviewerReview.model_validate(parsed_data)
        review_data = validated_review.model_dump()
    except Exception as val_error:
        # Fallback to unvalidated data with warning
        review_data = parsed_data
        review_data["__validation_warning"] = str(val_error)
    
    # Add metadata
    review_data["paper_id"] = paper_id
    review_data["version"] = version_label
    # ... etc
```

## Output Structure

### JSON Output

```json
{
  "reviewers": [
    {
      "summary": "This paper proposes...",
      "soundness": "2 fair",
      "presentation": "2 fair",
      "contribution": "2 fair",
      "strengths": [
        "Novel framework introduction",
        "Extensive experiments",
        "Applicable across model scales"
      ],
      "weaknesses": [
        "Lacks clear explanation of CoNN design",
        "No discussion of limitations",
        "Missing computational complexity analysis"
      ],
      "questions": [
        "Can you provide more details on architecture?",
        "What are the limitations?",
        "Computational complexity analysis?"
      ],
      "flag_for_ethics_review": "No ethics review needed.",
      "rating": "3: reject, not good enough",
      "confidence": "3: You are fairly confident..."
    },
    // ... 3 more reviewers
  ],
  "meta_review": "This paper proposes a method to integrate compiled neural networks...",
  "justification_for_why_not_higher_score": "The main weakness is the lack of clarity...",
  "justification_for_why_not_lower_score": "The main strength is the introduction of a new method...",
  "paper_decision": "Reject",
  "paper_id": "abc123",
  "version": "v1",
  "run_id": 0,
  "model_type": "CycleReviewer",
  "success": true,
  "was_truncated": false,
  "chars_per_token_used": 3.0
}
```

## Usage

```bash
python review_paper_pairs_vllm.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_cycle" \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "CycleReviewer-Llama-3.1-70B" \
  --num_runs 3 \
  --max_figures 5 \
  --verbose
```

## Key Features

### 1. Multi-Reviewer Simulation
- Generates 4 independent reviewer opinions
- Each with unique strengths, weaknesses, questions
- Different confidence levels and ratings

### 2. Comprehensive Meta Review
- Synthesizes all reviewer opinions
- Provides balanced assessment
- Justifies score decisions (why not higher/lower)

### 3. Structured Decision Process
- Clear Accept/Reject decision
- Based on aggregated reviewer feedback
- Follows academic conference review process

## Parsing Complexity

**Complexity:** High - Multiple nested structures

**Challenges:**
1. Multiple `## Reviewer` sections to detect and separate
2. Each reviewer has 10+ fields to extract
3. Lists can be numbered or bulleted
4. Meta review has subsections
5. Need to maintain order and structure

**Robustness:**
- Handles variations in heading formats
- Tolerates missing sections (returns empty strings/lists)
- Validates with Pydantic but falls back gracefully
- Includes `__validation_warning` if structure doesn't match exactly

## Context Management

**Typical Output Size:** ~4,000-8,000 tokens (4 reviewers + meta)

**Fits Well Within Limits:**
- Much larger than SEA-E (~1,500 tokens)
- But still manageable for most models
- Benefits from adaptive truncation if paper is very long

## Testing

### Test Case Structure

```python
test_output = """
## Reviewer

### Summary
Test summary...

### Strengths
1. Strength 1
2. Strength 2

### Weaknesses
- Weakness 1
- Weakness 2

### Questions
1. Question 1

### Rating
3: reject, not good enough

### Confidence
3: You are fairly confident...

[... 3 more reviewers ...]

## Meta Review
Overall assessment...

### justification_for_why_not_higher_score
Explanation...

### justification_for_why_not_lower_score
Explanation...

## Paper Decision
Reject
"""

parsed = _parse_cyclereviewer_format(test_output)
assert len(parsed["reviewers"]) == 4
assert parsed["paper_decision"] == "Reject"
```

## Comparison with SEA-E

| Feature | SEA-E | CycleReviewer |
|---------|-------|---------------|
| **Reviewers** | 1 | 4 |
| **Sections** | 9 | 11 per reviewer + meta |
| **Output Size** | ~1.5K tokens | ~4-8K tokens |
| **Complexity** | Medium | High |
| **Use Case** | Single review | Multi-reviewer consensus |
| **Decision Info** | Simple Accept/Reject | Detailed justifications |

## Benefits

### For Research
- **Variance Analysis:** 4 different reviewer perspectives
- **Consistency Check:** See if reviewers agree/disagree
- **Decision Validation:** Meta review synthesizes opinions

### For Evaluation
- **Richer Data:** Multiple opinions per paper
- **Inter-Reviewer Agreement:** Can compute agreement metrics
- **Confidence Tracking:** Each reviewer reports confidence level

## Future Enhancements

Potential improvements:
1. **Reviewer Disagreement Analysis:** Flag papers where reviewers strongly disagree
2. **Confidence Weighting:** Weight meta review by reviewer confidence
3. **Aspect Extraction:** Compute average soundness/presentation/contribution across reviewers
4. **Agreement Metrics:** Calculate inter-reviewer agreement scores

## Documentation

Updated files:
- ✅ `review_paper_pairs_vllm.py` - Core implementation
- ✅ `MODEL_FORMATS.md` - Format documentation
- ✅ `CYCLEREVIEWER_IMPLEMENTATION.md` - This file

## Status

**Implementation:** ✅ Complete  
**Testing:** ✅ Parser tested  
**Documentation:** ✅ Complete  
**Ready for use:** ✅ Yes

---

**Model Added:** October 30, 2025  
**Complexity:** High (multi-reviewer structure)  
**Recommended For:** Conference-style review simulation with multiple perspectives

