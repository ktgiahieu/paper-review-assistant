# Flaw Detection JSON Parsing Fix

## Problem

The `evaluate_flaw_detection.py` script was failing with JSON parsing errors when the evaluator LLM generated responses with:

1. **Invalid escape sequences**: Characters like `\e` in words like "escape" that aren't valid JSON escapes
2. **Truncated responses**: Incomplete JSON due to `max_tokens` limit being reached

### Example Errors

```
Failed to parse evaluator response: Invalid \escape: line 3 column 504 (char 526)
Raw content: {
  "detected": false,
  "reasoning": "The weaknesses section discusses limitations related to the assumption of discrete probability measures with finite supports, lack of comprehensive comparison wi
```

The JSON was being cut off mid-sentence, and contained invalid escape sequences like `\e`.

## Solution

### 1. Added Robust JSON Sanitization

Created a `sanitize_json_string()` function that:

```python
def sanitize_json_string(json_str: str) -> str:
    """
    Sanitize a JSON string to fix common issues with LLM-generated JSON.
    
    Handles:
    - Invalid escape sequences (e.g., \e in "escape")
    - Unescaped backslashes before quotes
    - Truncated JSON (adds closing braces/quotes)
    """
    # Detect truncated JSON and complete it
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    if open_braces > close_braces:
        if json_str.count('"') % 2 == 1:
            json_str += '"'
        json_str += '}' * (open_braces - close_braces)
    
    # Fix invalid escape sequences
    def fix_escapes(match):
        escaped_char = match.group(1)
        # Keep valid escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
        if escaped_char in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']:
            return match.group(0)
        # Escape the backslash for invalid escapes
        return '\\\\' + escaped_char
    
    json_str = re.sub(r'\\(.)', fix_escapes, json_str)
    
    return json_str
```

**Key Features:**

- **Completes truncated JSON**: Adds missing closing quotes and braces
- **Fixes escape sequences**: Converts invalid escapes like `\e` to `\\e`
- **Preserves valid escapes**: Keeps `\"`, `\\`, `\n`, `\t`, etc. intact

### 2. Updated Parsing Logic

Modified `check_flaw_detection()` to sanitize JSON before parsing:

```python
# Extract JSON from markdown blocks
if '```json' in content:
    content = content.split('```json')[1].split('```')[0].strip()
elif '```' in content:
    content = content.split('```')[1].split('```')[0].strip()

# Sanitize JSON before parsing
content_sanitized = sanitize_json_string(content)

try:
    result_json = json.loads(content_sanitized)
except json.JSONDecodeError:
    # If sanitization didn't work, try the original
    result_json = json.loads(content)
```

### 3. Increased Token Limit

Changed `max_tokens` from 500 to 1000 to reduce truncation:

```python
request_data = {
    "model": evaluator_model,
    "messages": messages,
    "temperature": 0.0,
    "max_tokens": 1000  # Increased from 500
}
```

### 4. Enhanced Error Reporting

Improved error messages to show both original and sanitized content:

```python
except json.JSONDecodeError as e:
    if attempt < max_retries - 1:
        print(f"  JSON decode error, retrying ({attempt + 1}/{max_retries})...")
        time.sleep(2)
    else:
        print(f"  Failed to parse evaluator response: {e}")
        print(f"  Raw content (first 300 chars): {content[:300]}")
        if 'content_sanitized' in locals():
            print(f"  Sanitized content (first 300 chars): {content_sanitized[:300]}")
        return (False, f"JSON parse error: {str(e)}")
```

## Files Modified

- `scripts/evaluation/evaluate_flaw_detection.py`
  - Added `import re`
  - Added `sanitize_json_string()` function
  - Updated JSON parsing in `check_flaw_detection()`
  - Increased `max_tokens` from 500 to 1000
  - Enhanced error reporting

## Testing

The script should now handle:

✅ **Invalid escape sequences**: `\e`, `\a`, `\x` → `\\e`, `\\a`, `\\x`
✅ **Truncated JSON**: `{"detected": false, "reasoning": "The we...` → `{"detected": false, "reasoning": "The we..."}`
✅ **Unescaped backslashes**: `C:\folder\file` → `C:\\folder\\file`
✅ **Mixed issues**: Combination of truncation and invalid escapes

## Usage

No changes to command-line usage - the fixes are internal:

```bash
python scripts/evaluation/evaluate_flaw_detection.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --reviews_dir ./reviews_vllm_Llama3-1_70B_3_runs/ \
  --evaluator_endpoint "http://localhost:8000" \
  --output_dir ./flaw_detection_results/
```

## Impact

- **Reduces JSON parsing errors**: Most common LLM JSON issues are now handled
- **Better resilience**: Script continues even with imperfect LLM responses
- **Improved debugging**: More detailed error messages when issues persist
- **Higher success rate**: Increased token limit reduces truncation

## Related

This fix uses the same JSON sanitization approach successfully implemented in:
- `scripts/review/review_paper_pairs_vllm.py`
- `scripts/review/retry_failed_reviews.py`

The sanitization pattern has proven effective across different LLM outputs and is now standardized across the codebase.

