# Retry Failed Reviews Guide

## Overview

**Note:** The `review_paper_pairs_vllm.py` script now includes **automatic retry** that handles most failures immediately (up to 3 attempts per review). The `retry_failed_reviews.py` script is now only needed for **persistent failures** that survive automatic retries.

See [`AUTO_RETRY_SUMMARY.md`](./AUTO_RETRY_SUMMARY.md) for details on automatic retry.

---

The `retry_failed_reviews.py` script manually identifies reviews that failed during generation and re-runs only those reviews. This is useful for recovering from persistent issues:

- **JSON parsing errors** (invalid escapes, truncated output)
- **API errors** (timeouts, rate limits, server errors)
- **Validation errors** (Pydantic validation failures)
- **Missing reviews** (incomplete runs)

## How It Works

1. **Scans review directory** for all JSON files
2. **Identifies failures** (reviews with `success: false`)
3. **Finds missing reviews** (expected but don't exist)
4. **Creates retry CSV** with only papers needing retry
5. **Runs review script** on retry CSV
6. **Verifies completion** after retry

## Common Errors Handled

### 1. JSON Parsing Errors

**Example:**
```
Worker 2487277: GenericStructured validation failed for VdkGRV1vcf (v1, run 0). 
Error: Invalid JSON: invalid escape at line 9 column 95
```

**Cause**: LLM generated JSON with invalid escape sequences (e.g., `\o` instead of `\\o`)

**Solution**: Improved `_sanitize_json_string()` function now handles:
- Invalid escape sequences (`\'`, `\o`, etc.)
- Truncated JSON
- Unescaped backslashes
- Trailing commas

### 2. Validation Errors

**Example:**
```
Pydantic validation failed: 1 validation error for GenericStructuredReview
  Field required [type=missing, input_value={...}]
```

**Cause**: LLM didn't include all required fields

**Solution**: Retry with improved prompt or check if format override needed

### 3. API Errors

**Example:**
```
Error: Connection timeout after 120 seconds
```

**Cause**: API server issues, network problems

**Solution**: Retry automatically recovers from temporary issues

## Usage

### Check for Failures (No Retry)

```bash
python retry_failed_reviews.py \
  --reviews_dir ./reviews_output \
  --csv_file ./data/filtered_pairs.csv \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "GenericStructured" \
  --check_only
```

**Output:**
```
==================================================================================
Failed Review Detector
==================================================================================
Reviews directory: ./reviews_output
CSV file: ./data/filtered_pairs.csv

Step 1: Checking for failed reviews...

❌ Found 5 failed reviews:
  - VdkGRV1vcf (v1, run 0): Failed to parse JSON from LLM
  - abc123 (latest, run 1): API timeout
  - def456 (v1, run 0): Pydantic validation failed
  ...

Step 2: Checking for missing reviews...

❌ Found 3 missing reviews:
  - xyz789 (v1, run 0): Review file does not exist
  - mno234 (latest, run 2): Review file does not exist
  ...

==================================================================================
Summary
==================================================================================
Failed reviews: 5
Missing reviews: 3
Total issues: 8

Unique papers needing retry: 6

--check_only flag set. Not retrying.
```

### Retry Failed Reviews

```bash
python retry_failed_reviews.py \
  --reviews_dir ./reviews_output \
  --csv_file ./data/filtered_pairs.csv \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "meta-llama/Llama-3.1-70B-Instruct" \
  --format GenericStructured \
  --num_runs 3 \
  --max_figures 5 \
  --verbose
```

**What Happens:**
1. Identifies 8 failed/missing reviews from 6 papers
2. Creates `./reviews_output/retry_papers.csv` with 6 papers
3. Runs `review_paper_pairs_vllm.py` on retry CSV
4. Verifies all reviews completed successfully

### Full Example Workflow

```bash
# Step 1: Initial review run
python review_paper_pairs_vllm.py \
  --csv_file ./data/filtered_pairs.csv \
  --output_dir ./reviews_llama \
  --model_name "meta-llama/Llama-3.1-70B-Instruct" \
  --format GenericStructured \
  --version both \
  --num_runs 3 \
  --max_workers 5

# Step 2: Check for failures
python retry_failed_reviews.py \
  --reviews_dir ./reviews_llama \
  --csv_file ./data/filtered_pairs.csv \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "meta-llama/Llama-3.1-70B-Instruct" \
  --format GenericStructured \
  --num_runs 3 \
  --check_only

# Step 3: Retry failures
python retry_failed_reviews.py \
  --reviews_dir ./reviews_llama \
  --csv_file ./data/filtered_pairs.csv \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "meta-llama/Llama-3.1-70B-Instruct" \
  --format GenericStructured \
  --num_runs 3 \
  --max_workers 3 \
  --verbose

# Step 4: Verify completion
python retry_failed_reviews.py \
  --reviews_dir ./reviews_llama \
  --csv_file ./data/filtered_pairs.csv \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "meta-llama/Llama-3.1-70B-Instruct" \
  --format GenericStructured \
  --num_runs 3 \
  --check_only
```

## Arguments

### Required

| Argument | Description |
|----------|-------------|
| `--reviews_dir` | Directory containing review outputs |
| `--csv_file` | Original CSV file with paper pairs |
| `--vllm_endpoint` | vLLM server endpoint URL |
| `--model_name` | Model name hosted on vLLM |

### Optional (Review Parameters)

| Argument | Default | Description |
|----------|---------|-------------|
| `--format` | None | Format override (SEA-E, CycleReviewer, GenericStructured, default) |
| `--max_figures` | 5 | Max figures to include |
| `--num_runs` | 1 | Number of runs per paper |
| `--max_workers` | 3 | Max worker threads |
| `--version` | both | Which versions to check (v1, latest, both) |
| `--verbose` | False | Verbose output |

### Retry-Specific

| Argument | Default | Description |
|----------|---------|-------------|
| `--check_only` | False | Only check for failures, don't retry |
| `--retry_output` | reviews_dir/retry_papers.csv | Where to save retry CSV |

## Output

### Retry CSV

The script creates a CSV with only papers that need retry:

```csv
paperid,v1_folder_path,latest_folder_path,flaw_descriptions
VdkGRV1vcf,/path/to/v1/VdkGRV1vcf,...,...
abc123,/path/to/v1/abc123,...,...
...
```

This CSV is automatically fed to `review_paper_pairs_vllm.py` for retry.

### Console Output

```
==================================================================================
Failed Review Detector
==================================================================================
...
Step 1: Checking for failed reviews...
✅ No failed reviews found

Step 2: Checking for missing reviews...
✅ No missing reviews

==================================================================================
Summary
==================================================================================
Failed reviews: 0
Missing reviews: 0
Total issues: 0

✅ All reviews completed successfully!
```

## JSON Sanitization Improvements

The improved `_sanitize_json_string()` function now handles:

### 1. Invalid Escape Sequences

**Before:**
```json
{
  "summary": "This uses \regularization to improve..."
}
```
❌ Invalid escape `\r` (only valid as `\r` for carriage return)

**After:**
```json
{
  "summary": "This uses \\regularization to improve..."
}
```
✅ Backslash properly escaped

### 2. Truncated JSON

**Before:**
```json
{
  "summary": "This paper proposes",
  "strengths": [
    "Novel approach",
    "Good exper
```
❌ Incomplete JSON

**After:**
```json
{
  "summary": "This paper proposes",
  "strengths": [
    "Novel approach",
    "Good exper"
  ]
}
```
✅ Automatically closed

### 3. Trailing Commas

**Before:**
```json
{
  "summary": "...",
  "strengths": ["..."],
}
```
❌ Trailing comma not allowed in JSON

**After:**
```json
{
  "summary": "...",
  "strengths": ["..."]
}
```
✅ Trailing comma removed

## Troubleshooting

### Issue: Same papers keep failing

**Cause**: Persistent issue with specific paper content (e.g., special characters, long content)

**Solutions:**
1. Check the paper content for unusual characters
2. Try different format (`--format` override)
3. Increase context length limit if using custom model
4. Manually review and fix the specific paper

### Issue: Many JSON parsing errors

**Cause**: Model not following JSON format well

**Solutions:**
1. Use `GenericStructured` format (more explicit JSON instructions)
2. Try different model
3. Adjust temperature (if controlling vLLM parameters)
4. Check vLLM server logs for issues

### Issue: API timeouts

**Cause**: Server overloaded or slow model

**Solutions:**
1. Reduce `--max_workers` (less concurrent load)
2. Check vLLM server resources (GPU memory, CPU)
3. Retry during off-peak hours
4. Increase timeout in review script (would need code change)

### Issue: Retries keep failing

**Cause**: Fundamental issue with model or setup

**Solutions:**
1. Check vLLM server is running: `curl http://localhost:8000/health`
2. Test with single paper: `--limit 1`
3. Check model format compatibility
4. Review vLLM server logs for errors

## Best Practices

### 1. Run Initial Check After Large Batch

```bash
# After running 100+ papers
python retry_failed_reviews.py \
  --reviews_dir ./reviews_output \
  --csv_file ./data/filtered_pairs.csv \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "..." \
  --check_only
```

### 2. Use Lower max_workers for Retry

```bash
# Original run: --max_workers 10
# Retry: --max_workers 3  (more stable)
python retry_failed_reviews.py \
  --reviews_dir ./reviews_output \
  --max_workers 3 \
  ...
```

### 3. Verify After Retry

Always check completion:
```bash
python retry_failed_reviews.py --check_only ...
```

### 4. Keep Retry CSV

Save retry CSV for later analysis:
```bash
python retry_failed_reviews.py \
  --retry_output ./logs/retry_$(date +%Y%m%d_%H%M%S).csv \
  ...
```

## Integration with Evaluation

After retrying, run evaluation to ensure completeness:

```bash
# Retry failures
python retry_failed_reviews.py \
  --reviews_dir ./reviews_llama \
  ...

# Evaluate (will skip failed reviews)
python evaluate_numerical_scores.py \
  --reviews_dir ./reviews_llama \
  --output_dir ./eval_llama

# Check how many papers were evaluated
# Should match total expected papers
```

## Advanced Usage

### Retry Specific Versions Only

```bash
# Only retry v1 reviews
python retry_failed_reviews.py \
  --reviews_dir ./reviews_output \
  --version v1 \
  ...

# Only retry latest reviews
python retry_failed_reviews.py \
  --reviews_dir ./reviews_output \
  --version latest \
  ...
```

### Programmatic Access

```python
from pathlib import Path
import json

def check_review_status(reviews_dir):
    """Check status of all reviews."""
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'missing': 0
    }
    
    for paper_dir in Path(reviews_dir).iterdir():
        if not paper_dir.is_dir():
            continue
        
        for review_file in paper_dir.glob("*_review_run*.json"):
            stats['total'] += 1
            
            try:
                with open(review_file) as f:
                    data = json.load(f)
                
                if data.get('success', False):
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
            except:
                stats['failed'] += 1
    
    return stats

# Usage
stats = check_review_status('./reviews_output')
print(f"Success rate: {stats['success'] / stats['total']:.1%}")
```

## Summary

The retry mechanism provides automatic recovery from common failures:

✅ **Automatic Detection**: Finds all failed and missing reviews  
✅ **Smart Retry**: Only re-runs what's needed  
✅ **Improved Sanitization**: Better JSON cleaning  
✅ **Verification**: Checks completion after retry  
✅ **Easy to Use**: Single command to retry everything  

For most cases, simply run:
```bash
python retry_failed_reviews.py \
  --reviews_dir ./reviews_output \
  --csv_file ./data/filtered_pairs.csv \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "your-model" \
  --num_runs 3
```

---

**Questions?** Check the main README.md or EVALUATION_GUIDE.md for related documentation.

