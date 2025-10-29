# Model-Specific Formats Guide

## Overview

The `review_paper_pairs_vllm.py` script now supports multiple model formats with different prompts and output parsers. The script automatically detects the model type based on the model name and applies the appropriate prompt and parser.

## Supported Models

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

### 2. Default (JSON) Format

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

