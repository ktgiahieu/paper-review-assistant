# Shared Scripts

This directory contains scripts shared across all experiments for review generation and evaluation.

## Directory Structure

```
scripts/
├── review/          # Review generation scripts
├── evaluation/      # Analysis and evaluation scripts
└── utils/          # Utility scripts
```

## Review Generation (`review/`)

### `review_papers_gemini.py`
Review papers from specified folders using Gemini API with:
- Multiple API key support (rate limiting)
- Multithreading for parallel processing
- Dynamic rate limits based on model (gemini-2.0-flash-lite, gemini-2.5-pro, etc.)
- Automatic retry on JSON parsing errors

**Usage:**
```bash
python3 review_papers_gemini.py \
    --base_dir ../../sampled_data/ICLR2024 \
    --folders latest authors_affiliation_good authors_affiliation_bad \
    --model_name gemini-2.0-flash-lite \
    --output_dir ../../sampled_data/reviews/ICLR2024
```

### `review_paper_pairs.py`
Review paper pairs (v1 vs latest) using Anthropic Claude API.

**Usage:**
```bash
python3 review_paper_pairs.py \
    --csv_file ../../data/ICLR2024_pairs/filtered_pairs.csv \
    --output_dir ../../sampled_data/reviews/ICLR2024
```

### `review_paper_pairs_vllm.py`
Review papers using vLLM (supports multiple model formats).

**Usage:**
```bash
python3 review_paper_pairs_vllm.py \
    --csv_file ../../data/ICLR2024_pairs/filtered_pairs.csv \
    --vllm_endpoint "http://localhost:8000" \
    --model_name "Meta-Llama-3.1-70B-Instruct" \
    --output_dir ../../sampled_data/reviews/ICLR2024
```

## Evaluation (`evaluation/`)

### `evaluate_good_bad_plant_effect.py`
Analyze the effect of good/bad manipulations on review scores (triplet analysis: baseline, good, bad).
Supports any manipulation pattern (author/affiliation, abstract, etc.) via the `--pattern` argument.

**Usage:**
```bash
python3 evaluate_good_bad_plant_effect.py \
    --reviews_dir ../../sampled_data/reviews/ICLR2024 \
    --output_dir ../../sampled_data/evaluation_results
```

### `evaluate_numerical_scores.py`
Extract and analyze numerical scores with paired t-tests (v1 vs latest comparison).

**Usage:**
```bash
python3 evaluate_numerical_scores.py \
    --reviews_dir ../../sampled_data/reviews/ICLR2024 \
    --output_dir ../../sampled_data/evaluation_results
```

### `calculate_mse_mae.py`
Calculate MSE and MAE comparing AI-generated vs human review scores.

**Usage:**
```bash
python3 calculate_mse_mae.py \
    --csv_file ../../data/ICLR2024_pairs/filtered_pairs_with_human_scores.csv \
    --reviews_dir ../../sampled_data/reviews/ICLR2024
```

### Other Evaluation Scripts
- `analyze_flaw_detection.py` - Analyze flaw detection with paired t-tests
- `evaluate_flaw_detection.py` - Evaluate whether AI reviews detect consensus flaws

## Utilities (`utils/`)

- `fetch_human_scores.py` - Fetch human review scores from OpenReview API
- `get_paper_pairs.py` - Extract paper pairs (v1 vs latest) from ICLR data
- `prepare_manual_gemini_prompts.py` - Prepare prompts for manual Gemini review
- `process_manual_gemini_outputs.py` - Process manual Gemini review outputs

## Path Conventions

All scripts use relative paths from the `experiments/` directory:
- `../../data/` - Original paper data
- `../../sampled_data/` - Sampled subsets
- `../../sampled_data/reviews/` - Review outputs
- `../../sampled_data/evaluation_results/` - Evaluation results

