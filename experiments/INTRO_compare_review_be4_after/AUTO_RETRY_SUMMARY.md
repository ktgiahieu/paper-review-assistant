# Automatic Retry Implementation Summary

## Overview

The `review_paper_pairs_vllm.py` script now includes **automatic retry logic** that immediately re-attempts failed reviews without requiring manual intervention via `retry_failed_reviews.py`.

## The Problem

LLMs occasionally produce malformed output:
- Invalid JSON escape sequences (e.g., `\'` instead of `'`)
- Truncated JSON responses
- Pydantic validation failures
- Transient API errors

Previously, these failures required:
1. Waiting for entire batch to complete
2. Running `retry_failed_reviews.py` manually
3. Re-processing only failed papers

## The Solution

**Automatic retry at the individual review level:**

```python
def review_with_retry(paper_id, ..., max_retries=MAX_REVIEW_RETRIES):
    """Review a paper with automatic retry on failures."""
    for attempt in range(max_retries + 1):
        review = review_single_paper_vllm(...)
        
        if review.get("success", False):
            return review  # Success!
        
        if attempt < max_retries:
            print(f"⚠️  Retrying {paper_id} - Attempt {attempt + 2}/{max_retries + 1}")
            time.sleep(REVIEW_RETRY_DELAY)
    
    return review  # All attempts failed
```

## Implementation Details

### Constants Added

```python
# Review-level retries (for failed reviews with parsing/validation errors)
MAX_REVIEW_RETRIES = 2  # Retry failed reviews this many times before giving up
REVIEW_RETRY_DELAY = 5  # Seconds to wait between review retries
```

### Two-Level Retry System

1. **API-Level Retries** (existing):
   - Handles connection errors, timeouts, rate limits, 5xx errors
   - 3 retries with exponential backoff
   - Happens *within* each review attempt

2. **Review-Level Retries** (new):
   - Handles JSON parsing, validation failures
   - 2 automatic retries (3 total attempts)
   - 5-second delay between attempts
   - Happens *across* review attempts

### Example Flow

```
Attempt 1: review_single_paper_vllm()
  ├─ API retry 1: timeout → wait 2s → retry
  ├─ API retry 2: success
  └─ Parse: Invalid JSON ❌

⚠️  Retrying paper123 (v1, run 0) - Attempt 2/3
Attempt 2: review_single_paper_vllm()
  ├─ API retry 1: success
  └─ Parse: Validation error ❌

⚠️  Retrying paper123 (v1, run 0) - Attempt 3/3
Attempt 3: review_single_paper_vllm()
  ├─ API retry 1: success
  └─ Parse: Success ✅
  
✅ Retry successful for paper123 (v1, run 0)
```

## User-Visible Changes

### Console Output

Successful retry:
```
⚠️  Retrying paper_abc (v1, run 0) - Attempt 2/3
✅ Retry successful for paper_abc (v1, run 0)
```

Failed after all attempts:
```
⚠️  Retrying paper_xyz (latest, run 1) - Attempt 2/3
⚠️  Retrying paper_xyz (latest, run 1) - Attempt 3/3
❌ All 3 attempts failed for paper_xyz (latest, run 1)
   Error: Invalid JSON: invalid escape at line 9 column 95
```

### JSON Output

No changes to the output format. Failed reviews still have:
```json
{
  "success": false,
  "error": "Error message...",
  "model_type": "GenericStructured"
}
```

But this should happen **much less frequently** now!

## Benefits

1. **Reduced Manual Work**: Most transient errors are fixed automatically
2. **Faster Recovery**: Retries happen immediately (5s delay vs hours of waiting)
3. **Better Success Rate**: 3 attempts significantly reduce failure rate
4. **Transparent**: Clear console output shows retry attempts
5. **No Breaking Changes**: Same JSON format, same arguments, same workflow

## Configuration

### Default Settings

```python
MAX_REVIEW_RETRIES = 2  # 3 total attempts
REVIEW_RETRY_DELAY = 5  # seconds
```

### Customization

To change retry behavior, edit the constants in `review_paper_pairs_vllm.py`:

```python
# More aggressive retries
MAX_REVIEW_RETRIES = 4  # 5 total attempts
REVIEW_RETRY_DELAY = 10  # 10-second delays

# No automatic retries
MAX_REVIEW_RETRIES = 0  # Only 1 attempt
```

Or modify the `review_with_retry` call:
```python
v1_review = review_with_retry(
    ...,
    max_retries=5  # Override default
)
```

## When Automatic Retry Isn't Enough

Some failures are persistent (e.g., model consistently produces invalid JSON for a specific paper). For these cases:

1. The review is saved with `success: false`
2. All 3 attempts and their errors are logged
3. Use `retry_failed_reviews.py` to:
   - Identify patterns in failures
   - Manually retry with different parameters
   - Debug specific problematic papers

See [`RETRY_GUIDE.md`](./RETRY_GUIDE.md) for troubleshooting persistent failures.

## Impact on Performance

- **Successful reviews**: No impact (0 retries)
- **1st retry succeeds**: +5 seconds (minimal)
- **2nd retry succeeds**: +10 seconds (still fast)
- **All retries fail**: +10 seconds (but saves hours of manual work later)

**Trade-off:** Slightly longer runtime for individual failures, but **much faster** overall workflow and reduced manual intervention.

## Testing

To test the automatic retry logic:

1. **Simulate failures** by temporarily modifying `_sanitize_json_string` to always fail:
   ```python
   def _sanitize_json_string(json_str: str) -> str:
       return "{ invalid json"  # Force failure
   ```

2. **Run a single review**:
   ```bash
   python review_paper_pairs_vllm.py \
     --csv_file ./test_papers.csv \
     --vllm_endpoint "http://localhost:8000" \
     --model_name "GenericStructured" \
     --limit 1
   ```

3. **Expected output**:
   ```
   ⚠️  Retrying paper_id (v1, run 0) - Attempt 2/3
   ⚠️  Retrying paper_id (v1, run 0) - Attempt 3/3
   ❌ All 3 attempts failed for paper_id (v1, run 0)
   ```

4. **Restore** the original `_sanitize_json_string` function

## Related Files

- **Implementation**: `review_paper_pairs_vllm.py` (lines 36-38, 1345-1397, 1454, 1474)
- **Documentation**: `README.md` (section "Automatic Retry on Failures")
- **Manual retry**: `retry_failed_reviews.py` (for persistent failures)
- **Troubleshooting**: `RETRY_GUIDE.md`

## Summary

The automatic retry feature:
- ✅ Handles most transient JSON/validation errors automatically
- ✅ Reduces need for manual `retry_failed_reviews.py` runs
- ✅ Provides clear feedback on retry attempts
- ✅ No breaking changes to existing workflow
- ✅ Configurable retry behavior
- ✅ Two-level retry system (API + review)

**Result:** More reliable review generation with less manual intervention! 🎉

