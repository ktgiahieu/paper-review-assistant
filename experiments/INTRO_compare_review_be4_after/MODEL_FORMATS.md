# Model-Specific Formats Guide

## Overview

The `review_paper_pairs_vllm.py` script now supports multiple model formats with different prompts and output parsers. The script automatically detects the model type based on the model name and applies the appropriate prompt and parser.

## Supported Models

Currently supported: **3 specialized formats** + 1 default JSON format

1. **SEA-E** - Single academic review with structured markdown sections
2. **CycleReviewer** - Multi-reviewer format with meta review (4 reviewers + meta)
3. **GenericStructured** - JSON format with explicit instructions (for non-finetuned models)
4. **Default** - Generic JSON format for other models

## Format Override

You can override automatic model detection using the `--format` option:

```bash
# Force GenericStructured format for any model
python review_paper_pairs_vllm.py \
  --model_name "meta-llama/Llama-3.1-70B-Instruct" \
  --format GenericStructured \
  --vllm_endpoint "http://localhost:8000" \
  ...

# Force SEA-E format even if model name doesn't match
python review_paper_pairs_vllm.py \
  --model_name "my-custom-model" \
  --format SEA-E \
  --vllm_endpoint "http://localhost:8000" \
  ...

# Force CycleReviewer format
python review_paper_pairs_vllm.py \
  --model_name "my-custom-model" \
  --format CycleReviewer \
  --vllm_endpoint "http://localhost:8000" \
  ...
```

**Available Options:**
- `--format SEA-E`: Use SEA-E markdown format
- `--format CycleReviewer`: Use CycleReviewer multi-reviewer format
- `--format GenericStructured`: Use GenericStructured JSON format (recommended for base models)
- `--format default`: Use default JSON format
- No `--format` flag: Auto-detect based on model name

### 1. SEA-E Format

**Model Detection:** Model names containing `"sea-e"` or `"seae"` (case-insensitive)

**Output Format:** Markdown with specific sections

**Example Usage:**
```bash
python review_paper_pairs_vllm.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_seae" \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "SEA-E" \
  --limit 1 \
  --verbose
```

**Prompt Structure:**

The SEA-E model uses a structured academic review prompt with 9 sections:
1. Summary (100-150 words)
2. Strengths/Weaknesses/Questions (bullet points)
3. Soundness/Contribution/Presentation ratings (1-4: poor/fair/good/excellent)
4. Rating (1-10 scale)
5. Paper Decision (Accept/Reject with reasons)

**Output Format:**

```markdown
**Summary:**
[100-150 word summary]

**Strengths:**
- Strength point 1
- Strength point 2
- ...

**Weaknesses:**
- Weakness point 1
- Weakness point 2
- ...

**Questions:**
- Question 1
- Question 2
- ...

**Soundness:**
[1-4 score with description]

**Presentation:**
[1-4 score with description]

**Contribution:**
[1-4 score with description]

**Rating:**
[1-10 score with description]

**Paper Decision:**
- Decision: Accept/Reject
- Reasons: [detailed reasons]
```

**Parsed JSON Structure:**

```json
{
  "summary": "string",
  "strengths": ["string", "string", ...],
  "weaknesses": ["string", "string", ...],
  "questions": ["string", "string", ...],
  "soundness": "string",
  "presentation": "string",
  "contribution": "string",
  "rating": "string",
  "paper_decision": "string",
  "paper_id": "string",
  "version": "string",
  "run_id": 0,
  "model_type": "SEA-E",
  "success": true,
  "raw_content": "string"
}
```

### 2. CycleReviewer Format

**Model Detection:** Model names containing `"cyclereviewer"`, `"cycle-reviewer"`, or `"cycle_reviewer"` (case-insensitive)

**Output Format:** Markdown with multiple reviewers and meta review

**Example Usage:**
```bash
python review_paper_pairs_vllm.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_cycle" \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "CycleReviewer-Llama-3.1-70B" \
  --limit 1 \
  --verbose
```

**Prompt Structure:**

The CycleReviewer model simulates a multi-reviewer conference review process with:
- 4 individual reviewer opinions
- Each reviewer provides: Summary, Soundness, Presentation, Contribution, Strengths, Weaknesses, Questions, Ethics Flag, Rating, Confidence
- Meta Review section with overall assessment
- Justifications for score decisions
- Final paper decision

**Output Format:**

```markdown
## Reviewer

### Summary
[Reviewer 1 summary]

### Soundness
[Rating with description]

### Presentation
[Rating with description]

### Contribution
[Rating with description]

### Strengths
1. Strength point 1
2. Strength point 2

### Weaknesses
1. Weakness point 1
2. Weakness point 2

### Questions
1. Question 1
2. Question 2

### Flag For Ethics Review
[Ethics review status]

### Rating
[1-10 rating with justification]

### Confidence
[Confidence level description]

**********

## Reviewer
[Reviewer 2... same structure]

**********

## Reviewer
[Reviewer 3... same structure]

**********

## Reviewer
[Reviewer 4... same structure]

**********

## Meta Review
[Overall assessment paragraph]

### justification_for_why_not_higher_score
[Explanation]

### justification_for_why_not_lower_score
[Explanation]

**********

## Paper Decision
Accept/Reject
```

**Parsed JSON Structure:**

```json
{
  "reviewers": [
    {
      "summary": "string",
      "soundness": "string",
      "presentation": "string",
      "contribution": "string",
      "strengths": ["string", ...],
      "weaknesses": ["string", ...],
      "questions": ["string", ...],
      "flag_for_ethics_review": "string",
      "rating": "string",
      "confidence": "string"
    },
    // ... 3 more reviewers
  ],
  "meta_review": "string",
  "justification_for_why_not_higher_score": "string",
  "justification_for_why_not_lower_score": "string",
  "paper_decision": "string",
  "paper_id": "string",
  "version": "string",
  "run_id": 0,
  "model_type": "CycleReviewer",
  "success": true,
  "raw_content": "string",
  "was_truncated": false,
  "chars_per_token_used": 3.0
}
```

### 3. GenericStructured Format

**Model Detection:** Manual override with `--format GenericStructured`

**Purpose:** For non-finetuned models that need explicit JSON formatting instructions

**Output Format:** JSON with detailed structure

**Example Usage:**
```bash
python review_paper_pairs_vllm.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_generic" \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "meta-llama/Llama-3.1-70B-Instruct" \
  --format GenericStructured \
  --num_runs 3 \
  --max_figures 5 \
  --verbose
```

**Prompt Structure:**

The GenericStructured prompt is designed for base/non-finetuned models and includes:
- Very explicit instructions about what to include
- Detailed JSON schema with examples
- Clear formatting rules (no markdown, no code blocks)
- Example of correct output format
- Field-by-field format specifications

**Key Features:**
- **Extremely Detailed Instructions**: Every field explained with format requirements
- **JSON Schema Example**: Shows exact structure expected
- **Format Enforcement**: Multiple reminders to output ONLY JSON
- **Compatible Fields**: Uses same fields as SEA-E for easy comparison

**Output Format:**

```json
{
  "summary": "string",
  "soundness": "string (1-4: poor/fair/good/excellent)",
  "presentation": "string (1-4: poor/fair/good/excellent)",
  "contribution": "string (1-4: poor/fair/good/excellent)",
  "strengths": ["string", "string", "string"],
  "weaknesses": ["string", "string", "string"],
  "questions": ["string", "string", "string"],
  "rating": "string (1-10 with description)",
  "recommendation": "string (Accept/Reject)",
  "meta_review": "string (comprehensive assessment)"
}
```

**Parsed JSON Structure:**

```json
{
  "summary": "This paper proposes...",
  "soundness": "3 good",
  "presentation": "2 fair",
  "contribution": "3 good",
  "strengths": [
    "Novel approach to problem X",
    "Comprehensive experiments",
    "Clear technical exposition"
  ],
  "weaknesses": [
    "Limited comparison with baseline Y",
    "No ablation study for component Z",
    "Computational cost not discussed"
  ],
  "questions": [
    "How does this scale to larger datasets?",
    "What is the inference time?",
    "Can this be applied to domain X?"
  ],
  "rating": "6: marginally above the acceptance threshold",
  "recommendation": "Accept",
  "meta_review": "This paper presents a solid contribution with good experimental results. The main strengths are the novel approach and comprehensive evaluation. However, some comparisons with recent work are missing. Overall, the contributions outweigh the limitations.",
  "paper_id": "string",
  "version": "string",
  "run_id": 0,
  "model_type": "GenericStructured",
  "success": true,
  "was_truncated": false,
  "chars_per_token_used": 3.0
}
```

**When to Use:**
- Base models (Llama, Mistral, etc.) without fine-tuning
- Models that don't follow instructions well
- When you need maximum format compliance
- When comparing with SEA-E results (compatible fields)

**Comparison with SEA-E:**
- Same core fields: summary, soundness, presentation, contribution, strengths, weaknesses, questions, rating
- GenericStructured adds: `recommendation` (explicit Accept/Reject) and `meta_review` (detailed assessment)
- GenericStructured uses JSON instead of Markdown (better for parsing)
- More explicit instructions to compensate for lack of fine-tuning

### 4. Default (JSON) Format

**Model Detection:** All models not matching specific patterns

**Output Format:** JSON

**Example Usage:**
```bash
python review_paper_pairs_vllm.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_default" \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "Qwen/Qwen2-VL-7B-Instruct" \
  --limit 1 \
  --verbose
```

**Prompt Structure:**

Standard peer review prompt requesting JSON output with specific fields.

**Expected JSON Output:**

```json
{
  "summary": "string",
  "strengths": ["string", "string", ...],
  "weaknesses": ["string", "string", ...],
  "clarity_score": 1-10,
  "novelty_score": 1-10,
  "technical_quality_score": 1-10,
  "experimental_rigor_score": 1-10,
  "overall_score": 1-10,
  "confidence": 1-5,
  "recommendation": "Accept/Reject variant",
  "detailed_comments": "string"
}
```

## How Model Detection Works

The script uses `ReviewPrompts.detect_model_type(model_name)` which:

1. Converts model name to lowercase
2. Checks for specific keywords:
   - Contains "sea-e" or "seae" → Returns "SEA-E"
   - No matches → Returns "default"

## Adding New Model Formats

To add support for a new model format:

### Step 1: Add Detection Logic

Edit `review_paper_pairs_vllm.py`, function `ReviewPrompts.detect_model_type()`:

```python
@staticmethod
def detect_model_type(model_name: str) -> str:
    model_name_lower = model_name.lower()
    
    # Existing checks...
    if "sea-e" in model_name_lower or "seae" in model_name_lower:
        return "SEA-E"
    elif "cyclereviewer" in model_name_lower or "cycle-reviewer" in model_name_lower:
        return "CycleReviewer"
    
    # Add your new model check
    if "your-model-keyword" in model_name_lower:
        return "YOUR-MODEL-TYPE"
    
    return "default"
```

### Step 2: Add Pydantic Model (Optional but Recommended)

```python
class YourModelReview(BaseModel):
    """Pydantic model for YOUR-MODEL format."""
    field1: str = Field(description="...")
    field2: List[str] = Field(description="...")
    # ... add all expected fields
```

### Step 3: Add Prompt

Edit `ReviewPrompts.get_system_prompt()`:

```python
if model_type == "YOUR-MODEL-TYPE":
    return """Your custom system prompt here"""
```

Edit `ReviewPrompts.get_user_prompt()`:

```python
if model_type == "YOUR-MODEL-TYPE":
    return f"""Your custom user prompt with {paper_content} here"""
```

### Step 4: Add Parser

Create a parser function:

```python
def _parse_your_model_format(content: str) -> dict:
    """Parse your model's output format."""
    result = {}
    # Your parsing logic
    return result
```

### Step 5: Integrate Parser

In `review_single_paper_vllm()`, add your case:

```python
if model_type == "YOUR-MODEL-TYPE":
    parsed_data = _parse_your_model_format(raw_content)
    # Validate and return
```

## Testing Your Parser

1. Create a test file with example output
2. Run the standalone parser test:

```python
from review_paper_pairs_vllm import _parse_your_model_format

example_output = """..."""
result = _parse_your_model_format(example_output)
print(json.dumps(result, indent=2))
```

## Model Type Information in Output

All review JSON files include a `model_type` field indicating which format was used:

```json
{
  "model_type": "SEA-E",
  ...
}
```

This allows you to:
- Track which model/format generated each review
- Handle different formats in downstream analysis
- Debug parsing issues

## Tips and Best Practices

1. **Model Name Conventions:** Use consistent naming for your models to make detection reliable.

2. **Flexible Parsing:** Make parsers robust to minor format variations (extra whitespace, missing sections, etc.).

3. **Validation:** Use Pydantic models to validate parsed output when possible.

4. **Fallback Handling:** Always include error handling and fallback to save raw content if parsing fails.

5. **Testing:** Test your parser with multiple real examples before running on full dataset.

6. **Documentation:** Document your model's expected format with examples.

## Troubleshooting

### Issue: Wrong model type detected

**Solution:** Check your model name and update `detect_model_type()` logic.

### Issue: Parsing fails

**Solution:** 
1. Check `raw_content` in output JSON
2. Test parser with standalone script
3. Update regex patterns or parsing logic

### Issue: Missing fields in output

**Solution:**
1. Check if model is following the prompt
2. Adjust prompt to be more explicit
3. Make parser more flexible to handle variations

## Example: Full Workflow with SEA-E

```bash
# 1. Start vLLM with SEA-E model
python -m vllm.entrypoints.openai.api_server \
  --model SEA-E \
  --port 8000

# 2. Run review script (automatically detects SEA-E)
python review_paper_pairs_vllm.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_seae" \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "SEA-E" \
  --num_runs 3 \
  --max_figures 5 \
  --verbose

# 3. Check output
cat reviews_seae/{paper_id}/v1_review_run0.json

# 4. Verify model_type field
jq '.model_type' reviews_seae/{paper_id}/v1_review_run0.json
# Output: "SEA-E"
```

## Summary CSV Compatibility

The summary CSV generation works with all model formats, extracting common fields when available. Format-specific fields are preserved in individual JSON files.

