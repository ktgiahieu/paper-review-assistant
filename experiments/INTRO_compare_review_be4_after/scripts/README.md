# Scripts Directory

This directory contains all executable Python scripts for the paper review comparison workflow.

## Directory Structure

### 📝 `review/` - Review Generation
Scripts for generating AI reviews of papers:

- **`review_paper_pairs.py`** - Generate reviews using Anthropic Claude API
- **`review_paper_pairs_vllm.py`** - Generate reviews using vLLM (supports multiple models and formats)
- **`retry_failed_reviews.py`** - Automatically retry failed review generations

### 📊 `evaluation/` - Analysis & Evaluation
Scripts for evaluating and analyzing review results:

- **`evaluate_numerical_scores.py`** - Extract and analyze numerical scores (paired t-tests, effect sizes)
- **`analyze_flaw_detection.py`** - Analyze flaw detection with paired t-tests
- **`calculate_mse_mae.py`** - Calculate MSE/MAE comparing AI vs human scores
- **`evaluate_flaw_detection.py`** - Evaluate whether AI reviews detect consensus flaws

### 🔧 `utils/` - Utility Scripts
Helper scripts for data preparation and retrieval:

- **`get_paper_pairs.py`** - Extract paper pairs (v1 vs latest) from ICLR data
- **`fetch_human_scores.py`** - Fetch human review scores from OpenReview API

## Quick Start

### 1. Generate Reviews

```bash
# Using Anthropic Claude
python scripts/review/review_paper_pairs.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ./reviews_claude/ \
  --num_runs 3

# Using vLLM (multiple formats)
python scripts/review/review_paper_pairs_vllm.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ./reviews_llama3/ \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "Meta-Llama-3.1-70B-Instruct" \
  --num_runs 3 \
  --format default
```

### 2. Evaluate Results

```bash
# Evaluate numerical scores
python scripts/evaluation/evaluate_numerical_scores.py \
  --reviews_dir ./reviews_llama3/ \
  --output_dir ./evaluation_results/

# Evaluate flaw detection
python scripts/evaluation/evaluate_flaw_detection.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --reviews_dir ./reviews_llama3/ \
  --evaluator_endpoint "http://localhost:8000" \
  --evaluator_model "Qwen3-30B"

# Compare with human scores
python scripts/utils/fetch_human_scores.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv

python scripts/evaluation/calculate_mse_mae.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs_with_human_scores.csv \
  --reviews_dir ./reviews_llama3/
```

## Dependencies

All scripts require:
```bash
pip install -r ../requirements.txt
```

## Documentation

See `../docs/guides/` for detailed user guides for each script.

