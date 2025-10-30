# Anthropic Multiple Runs & Evaluator Compatibility

## Overview

Updated `review_paper_pairs.py` to support:
1. **Multiple runs per paper** (like vLLM version) for variance analysis
2. **Compatibility with `evaluate_numerical_scores.py`** for statistical analysis

## Changes Made

### 1. Multiple Runs Support

Added `--num_runs` argument to run reviews multiple times per paper:

```bash
python review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_anthropic" \
  --model_name "claude-haiku-4-5-20251001" \
  --num_runs 3  # Run each review 3 times
```

**Output structure:**
```
reviews_anthropic/
  paper_id_1/
    v1_review_run0.json
    v1_review_run1.json
    v1_review_run2.json
    latest_review_run0.json
    latest_review_run1.json
    latest_review_run2.json
  paper_id_2/
    ...
```

### 2. Evaluator Compatibility

Added required fields to JSON output:
- `run_id`: Run number (0, 1, 2, ...)
- `model_type`: Set to "Anthropic"
- `success`: Boolean indicating if review succeeded

Added score field mappings for evaluator:
- `soundness`: Mapped from `technical_quality_score`
- `presentation`: Mapped from `clarity_score`
- `contribution`: Mapped from `novelty_score`
- `rating`: Mapped from `overall_score`

**Original fields are preserved**, so JSON contains both:
```json
{
  "paper_id": "abc123",
  "version": "v1",
  "run_id": 0,
  "model_type": "Anthropic",
  "success": true,
  
  // Original Anthropic fields
  "clarity_score": 8,
  "novelty_score": 7,
  "technical_quality_score": 9,
  "experimental_rigor_score": 8,
  "overall_score": 8,
  "confidence": 4,
  "recommendation": "Accept",
  
  // Mapped fields for evaluator
  "soundness": 9,
  "presentation": 8,
  "contribution": 7,
  "rating": 8,
  
  // ... other fields
}
```

### 3. Updated Evaluator

Modified `evaluate_numerical_scores.py` to handle Anthropic format:

```python
def extract_scores_default(review_data: dict) -> Dict[str, Optional[float]]:
    """Extract numerical scores from default/Anthropic format review."""
    # Check if review has already-mapped fields (from review_paper_pairs.py)
    if 'soundness' in review_data and review_data['soundness'] is not None:
        scores = {
            'soundness': review_data.get('soundness'),
            'presentation': review_data.get('presentation'),
            'contribution': review_data.get('contribution'),
            'rating': review_data.get('rating')
        }
    else:
        # Fallback: try to map from original field names
        scores = {
            'soundness': review_data.get('technical_quality_score'),
            'presentation': review_data.get('clarity_score'),
            'contribution': review_data.get('novelty_score'),
            'rating': review_data.get('overall_score')
        }
    return scores
```

## Usage Examples

### Single Run (Fastest)

```bash
python review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_anthropic" \
  --model_name "claude-haiku-4-5-20251001" \
  --max_workers 3
```

### Multiple Runs (Variance Analysis)

```bash
python review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_anthropic" \
  --model_name "claude-haiku-4-5-20251001" \
  --num_runs 3 \
  --max_workers 3
```

### Evaluate Results

```bash
python evaluate_numerical_scores.py \
  --reviews_dir ./reviews_anthropic \
  --output_dir ./evaluation_anthropic
```

## Score Mappings

| Anthropic Field | Evaluator Field | Scale |
|----------------|----------------|-------|
| `technical_quality_score` | `soundness` | 1-10 |
| `clarity_score` | `presentation` | 1-10 |
| `novelty_score` | `contribution` | 1-10 |
| `overall_score` | `rating` | 1-10 |

**Note:** Anthropic uses 1-10 scale for all scores, while SEA-E uses 1-4 for soundness/presentation/contribution and 1-10 for rating.

## Comparison with vLLM Version

Both scripts now have feature parity:

| Feature | `review_paper_pairs.py` | `review_paper_pairs_vllm.py` |
|---------|------------------------|------------------------------|
| Multiple runs | ✅ `--num_runs` | ✅ `--num_runs` |
| Run ID in filenames | ✅ `*_run{id}.json` | ✅ `*_run{id}.json` |
| Evaluator compatible | ✅ Auto-mapping | ✅ Native fields |
| Model type field | ✅ "Anthropic" | ✅ "SEA-E", "CycleReviewer", etc. |
| Success field | ✅ Boolean | ✅ Boolean |

## Testing Workflow

```bash
# Step 1: Test with 1 paper
python review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_test" \
  --version v1 \
  --limit 1

# Step 2: Add latest version
python review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_test" \
  --version latest \
  --limit 1 \
  --skip_existing

# Step 3: Expand with multiple runs
python review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_test" \
  --version both \
  --limit 10 \
  --num_runs 3 \
  --skip_existing

# Step 4: Evaluate
python evaluate_numerical_scores.py \
  --reviews_dir ./reviews_test \
  --output_dir ./evaluation_test
```

## API Cost Estimation

**Formula:** `API calls = papers × versions × runs`

Examples:
- 100 papers, both versions, 1 run: **200 calls**
- 100 papers, both versions, 3 runs: **600 calls**
- 100 papers, v1 only, 3 runs: **300 calls**

**With `--skip_existing`:** Only processes missing reviews, significantly reducing costs on reruns.

## Benefits

1. **Variance Analysis:** Multiple runs allow measuring LLM consistency
2. **Statistical Power:** More data points improve statistical significance
3. **Direct Comparison:** Same evaluator works for Anthropic and vLLM reviews
4. **Cost Efficient:** `--skip_existing` and `--version` flags save API credits
5. **Idempotent:** Can safely rerun to complete failed reviews

## Files Modified

1. **`review_paper_pairs.py`:**
   - Added `run_id` parameter to `review_single_paper()`
   - Added `num_runs` parameter to `review_paper_pair()`
   - Added `--num_runs` command-line argument
   - Added `run_id`, `model_type`, `success` to all return dicts
   - Added score field mappings (soundness, presentation, contribution, rating)
   - Updated filename format to `*_review_run{id}.json`
   - Removed old comparison.json generation (use evaluator instead)

2. **`evaluate_numerical_scores.py`:**
   - Enhanced `extract_scores_default()` to handle mapped fields
   - Added fallback to original Anthropic field names
   - Works with both old and new JSON formats

## Summary

The Anthropic script now has:
- ✅ Multiple runs support (variance analysis)
- ✅ Evaluator compatibility (statistical analysis)
- ✅ Same output structure as vLLM version
- ✅ Backward compatible (original fields preserved)
- ✅ Feature parity with vLLM script

You can now run `claude-haiku-4-5-20251001` on all paper pairs with multiple runs and use the same evaluation script to compare with vLLM models!

