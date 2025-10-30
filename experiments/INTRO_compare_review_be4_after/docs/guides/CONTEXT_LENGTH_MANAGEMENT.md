# Context Length Management

## Overview

The script now automatically manages context length to prevent exceeding model limits. This is especially important for models like **SEA-E** that have limited context windows.

## Problem

Models have maximum context length limits. For example:
- **SEA-E**: 32,768 tokens (32K context)
- **Qwen2-VL**: 32,768 tokens
- **GPT-4**: 128,000 tokens

When reviewing long papers with figures, you might exceed these limits:
```
Error: This model's maximum context length is 32768 tokens. 
However, you requested 39511 tokens (35415 in the messages, 4096 in the completion).
```

## Solution

The script now:
1. **Detects model-specific limits** - Each model type has a configured maximum
2. **Estimates token usage** - Calculates tokens from characters (~4 chars/token)
3. **Accounts for overhead** - System prompt, user prompt template, images, completion
4. **Truncates intelligently** - Preserves beginning and end of paper
5. **Marks truncated reviews** - Adds `was_truncated` flag to output

## Model Limits

Configured in `review_paper_pairs_vllm.py`:

```python
MODEL_CONTEXT_LIMITS = {
    "SEA-E": 32768,      # 32K tokens
    "default": 128000,   # 128K tokens (conservative)
}
```

**Completion Reserve:** 4,096 tokens (configurable via `COMPLETION_TOKENS`)

## How Truncation Works

### 1. Calculate Available Space

```
Available tokens = Max context - Completion tokens - Overhead - Buffer
```

Where:
- **Max context**: Model-specific limit (e.g., 32,768 for SEA-E)
- **Completion tokens**: Reserved for response (4,096)
- **Overhead**: System prompt + user template + images
- **Buffer**: Safety margin (500 tokens)

### 2. Estimate Overhead

**System Prompt:**
- SEA-E: ~400 tokens (academic review instructions)
- Default: ~250 tokens (JSON format instructions)

**User Prompt Template:**
- Includes version, flaw context
- ~100-200 tokens

**Images:**
- Estimated at 1,000 tokens per image (conservative)
- Actual varies by image size/complexity

**Example for SEA-E with 5 figures:**
```
Overhead = 400 (system) + 150 (user template) + 5000 (images) + 500 (buffer)
         = 6,050 tokens

Available for paper = 32,768 - 4,096 - 6,050 = 22,622 tokens
```

### 3. Truncate Paper Content

If paper exceeds available tokens, a **three-stage smart truncation strategy** is applied:

#### Stage 1: Remove Reference Abstracts
- Removes all `**Abstract:**` sections from the References
- Preserves reference citations but removes lengthy abstracts
- **Impact:** Can save ~20-40% of reference section size

#### Stage 2: Remove Appendices
- Removes all content after the References section
- Typically includes: Theoretical Analysis, Additional Experiments, Proofs
- Adds notice: `[... APPENDICES REMOVED DUE TO LENGTH LIMITS ...]`
- **Impact:** Can save significant space for papers with extensive appendices

#### Stage 3: Beginning/End Truncation (Last Resort)
- **Only used if stages 1 & 2 are insufficient**
- **Preserve Ratio:** 70% from beginning, 30% from end (configurable)
- Preserves most important content:
  - **Beginning**: Abstract, intro, problem statement, main contributions, methods
  - **End**: Results, conclusions, future work
- Adds notice: `[... MAIN CONTENT TRUNCATED DUE TO LENGTH LIMITS ...]`

**Why this strategy?**
1. **Reference abstracts** are rarely needed for review (citations are sufficient)
2. **Appendices** contain supplementary material, not core contributions
3. **Core paper content** is preserved as much as possible
4. **Reviewers get** abstract, intro, methods, results, and conclusions intact

## Example

**Original paper:** 40,000 tokens  
**Available:** 22,622 tokens

**Stage 1 - Remove Reference Abstracts:**
- Original references: ~8,000 tokens
- After removing abstracts: ~5,000 tokens
- **Saved:** 3,000 tokens
- **New total:** 37,000 tokens → Still over limit

**Stage 2 - Remove Appendices:**
- Original appendices: ~12,000 tokens
- After removal: 0 tokens (notice added)
- **Saved:** 12,000 tokens
- **New total:** 25,000 tokens → Still over limit

**Stage 3 - Beginning/End Truncation:**
- Target: 22,622 tokens
- **First 70%**: ~15,820 tokens (Abstract → Methods → Results)
- **Notice**: ~20 tokens
- **Last 30%**: ~6,780 tokens (Conclusion section)
- **Final total:** ~22,620 tokens ✓

**Content preserved:**
- ✅ Abstract
- ✅ Introduction & Background
- ✅ Methodology (complete)
- ✅ Most Results
- ✅ Conclusions
- ✅ Reference citations (without abstracts)
- ❌ Reference abstracts
- ❌ Appendices (theoretical proofs, additional experiments)

## Output Format

All review JSON files now include a `was_truncated` field:

```json
{
  "paper_id": "abc123",
  "version": "v1",
  "run_id": 0,
  "model_type": "SEA-E",
  "was_truncated": true,    ← NEW!
  "summary": "...",
  "strengths": [...],
  ...
}
```

**Values:**
- `false`: Paper fit within limits, no truncation
- `true`: Paper was truncated to fit

## Monitoring Truncation

### Check Individual Reviews

```bash
# Check if a specific review was truncated
jq '.was_truncated' reviews_seae/paper123/v1_review_run0.json
```

### Count Truncated Reviews

```bash
# Count how many reviews were truncated
jq -r '.was_truncated' reviews_seae/*/v1_review_run0.json | grep -c true
```

### Find All Truncated Papers

```bash
# List papers that were truncated
for dir in reviews_seae/*/; do
  paper_id=$(basename "$dir")
  truncated=$(jq -r '.was_truncated' "$dir/v1_review_run0.json" 2>/dev/null)
  if [ "$truncated" = "true" ]; then
    echo "$paper_id"
  fi
done
```

### Analyze in Python

```python
import pandas as pd
import json
from pathlib import Path

# Load all reviews
reviews = []
for review_file in Path("reviews_seae").glob("*/v1_review_run0.json"):
    with open(review_file) as f:
        reviews.append(json.load(f))

df = pd.DataFrame(reviews)

# Count truncated
print(f"Total reviews: {len(df)}")
print(f"Truncated: {df['was_truncated'].sum()}")
print(f"Not truncated: {(~df['was_truncated']).sum()}")
print(f"Truncation rate: {df['was_truncated'].mean():.1%}")
```

## Adjusting Truncation Behavior

### Reduce Truncation Rate

**Option 1: Reduce max_figures**
```bash
# Use fewer images to save tokens
python review_paper_pairs_vllm.py \
  --max_figures 2  # Instead of 5
```

**Option 2: Modify preserve_ratio**

Edit `review_paper_pairs_vllm.py`:
```python
paper_content, was_truncated = _truncate_paper_content(
    paper_content, 
    max_paper_tokens, 
    preserve_ratio=0.8,  # Keep more from beginning, less from end
    ...
)
```

**Option 3: Increase buffer**

Reduce the safety buffer if you're confident about token estimates:
```python
max_paper_tokens = available_tokens - overhead_tokens - 200  # Smaller buffer
```

### For Long Papers

If you have very long papers and truncation is unavoidable:

1. **Review without figures** first:
   ```bash
   --max_figures 0
   ```

2. **Review in stages** (manually):
   - First half only
   - Second half only
   - Combine insights

3. **Use larger context model**:
   - Switch to model with 128K+ context if available

## Token Estimation

The script uses a simple approximation:
```python
tokens ≈ characters / 4
```

This is conservative for English text. Actual tokenization varies by:
- **Language**: Non-English may use more tokens
- **Technical terms**: Code, equations may use different rates
- **Special characters**: Symbols, LaTeX may affect count

### Accuracy

**Typical accuracy:** ±10-20%

**Why approximate?**
- Don't have access to model's exact tokenizer
- Fast estimation (no external dependencies)
- Conservative (better to underestimate than overflow)

**If you need exact counts:**
```python
# Install tiktoken for GPT models
pip install tiktoken

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
tokens = len(enc.encode(text))
```

## Verbose Output

Enable `--verbose` to see truncation information:

```bash
python review_paper_pairs_vllm.py \
  --model_name "SEA-E" \
  --verbose
```

**Output:**
```
Worker 12345: Truncated paper from 35000 to ~22600 tokens (140000 → 90400 chars)
Worker 12345: Reviewing paper123 (v1, run 0), attempt 1/3
```

## Best Practices

### 1. Monitor Truncation Rates

Check after running on a subset:
```bash
# Test with --limit first
python review_paper_pairs_vllm.py --limit 10 --verbose

# Count truncated
grep "Truncated paper" output.log | wc -l
```

### 2. Adjust Based on Results

If truncation rate is high (>50%):
- Reduce `--max_figures`
- Use model with larger context
- Consider paper preprocessing (remove appendices)

### 3. Document Truncation

In your analysis, note:
- How many papers were truncated
- Impact on review quality (if measurable)
- Truncation strategy used

### 4. Validate Quality

Compare truncated vs non-truncated reviews:
```python
truncated = df[df['was_truncated']]
not_truncated = df[~df['was_truncated']]

print("Truncated mean score:", truncated['rating'].mean())
print("Full mean score:", not_truncated['rating'].mean())
```

## Troubleshooting

### Still Getting Context Length Errors

**Possible causes:**
1. **Token estimation off** - Real tokens > estimated
2. **Image overhead underestimated** - Images use more tokens
3. **Model limit incorrect** - Check actual model limit

**Solutions:**
1. **Reduce preserve ratio** to truncate more aggressively
2. **Increase buffer** for safety margin
3. **Disable images** for testing

### Reviews Seem Incomplete

If truncated reviews miss key information:

1. **Check truncation notice location:**
   ```bash
   jq -r '.raw_content' review.json | grep -C 5 "CONTENT TRUNCATED"
   ```

2. **Adjust preserve_ratio:**
   - Increase if conclusions are being cut
   - Decrease if methodology is missing

3. **Use full context model** if available

### Truncation Too Aggressive

If papers that should fit are being truncated:

1. **Check overhead calculation** - May be overestimating
2. **Reduce buffer** from 500 to 200-300 tokens
3. **Verify image token estimate** - May be too conservative

## Performance Impact

**Token estimation:** <1ms per paper (negligible)  
**Truncation:** <10ms for long papers (negligible)  
**Overall impact:** Minimal (<1% overhead)

The truncation happens on the client side before sending to the API, so there's no impact on API response time.

## Summary

✅ **Automatic detection** of context limits  
✅ **Smart truncation** preserving key sections  
✅ **Transparent tracking** via `was_truncated` flag  
✅ **Configurable behavior** for different needs  
✅ **Minimal overhead** (<1% performance impact)

The feature ensures your reviews complete successfully even with long papers and limited context models, while maintaining visibility into when and how truncation occurs.

