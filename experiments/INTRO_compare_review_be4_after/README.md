# Paper Review Comparison Workflow

This directory contains scripts to compare reviews between v1 and latest versions of papers.

## Quick Comparison: Anthropic vs vLLM

| Feature | `review_paper_pairs.py`<br/>(Anthropic) | `review_paper_pairs_vllm.py`<br/>(vLLM) |
|---------|----------------------------------------|----------------------------------------|
| **API** | Anthropic Claude API | Local/hosted vLLM (OpenAI-compatible) |
| **Figures/Images** | ❌ No | ✅ Yes (automatic extraction & encoding) |
| **Multiple Runs** | ❌ No | ✅ Yes (for variance analysis) |
| **Cost** | Pay-per-use ($) | Free (if self-hosted) |
| **Models** | Claude Sonnet/Haiku | Any vLLM-supported model (Qwen2-VL, etc.) |
| **Multimodal** | ❌ Text only | ✅ Text + Images |
| **Setup Complexity** | Easy (just API key) | Moderate (need vLLM server) |

**Use Anthropic** if you want quick setup and don't need figures.  
**Use vLLM** if you need figures, want multiple runs for variance analysis, or prefer local models.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

### For Anthropic API (review_paper_pairs.py)
Make sure you have `ANTHROPIC_API_KEY` in your `.env` file or environment variables.

### For vLLM (review_paper_pairs_vllm.py)
You need a running vLLM server. The script assumes an OpenAI-compatible API endpoint.

## Workflow

### Step 1: Extract Paper Pairs

```bash
python get_paper_pairs.py
```

This will:

- Read `ICLR2024.csv`
- Find papers with v1 before 2024-01-15 and latest version after that date
- Copy paper folders to `data/ICLR2024_pairs/`
- Include flaw descriptions from the flawed papers CSV
- Generate `filtered_pairs.csv`

### Step 2: Review Paper Pairs with LLM

The script supports debug modes to save API credits!

#### 🐛 DEBUG MODE - Start Small!

**Test with 1 paper, v1 only (1 API call):**

```bash
python review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_test" \
  --version v1 \
  --limit 1 \
  --verbose
```

**Add latest version to same paper (1 more API call):**

```bash
python review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_test" \
  --version latest \
  --limit 1 \
  --skip_existing
```

**Expand to 10 papers:**

```bash
python review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_test" \
  --version both \
  --limit 10 \
  --skip_existing
```

#### 🚀 PRODUCTION MODE

**Review all v1 versions first:**

```bash
python review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./data/ICLR2024_pairs/reviews" \
  --version v1 \
  --max_workers 3
```

**Then add latest versions:**

```bash
python review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./data/ICLR2024_pairs/reviews" \
  --version latest \
  --skip_existing \
  --max_workers 3
```

**Or do both at once:**

```bash
python review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./data/ICLR2024_pairs/reviews" \
  --version both \
  --skip_existing \
  --max_workers 3
```

### Step 2b: Review Paper Pairs with vLLM (Alternative)

Use `review_paper_pairs_vllm.py` for local/hosted vLLM models with **multimodal support (figures)** and **multiple runs for variance analysis**.

#### Key Differences from Anthropic Version:

1. **Figures/Images Support**: Automatically extracts and includes figures from papers
2. **Multiple Runs**: Review each paper multiple times to analyze LLM variance
3. **Local/Hosted Models**: Use your own vLLM deployment
4. **OpenAI-Compatible API**: Works with any vLLM server
5. **Smart Context Management**: Automatically truncates long papers to fit model limits

#### Prerequisites

Start a vLLM server with a multimodal model:

```bash
# Example: Starting vLLM with Qwen2-VL
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --port 8000 \
  --tensor-parallel-size 1
```

#### 🐛 DEBUG MODE - Test vLLM Setup

**Test with 1 paper, v1 only, 1 run (1 API call):**

```bash
python review_paper_pairs_vllm.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_vllm_test" \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "Qwen/Qwen2-VL-7B-Instruct" \
  --version v1 \
  --limit 1 \
  --max_figures 5 \
  --verbose
```

**Test multiple runs (3x same paper for variance analysis):**

```bash
python review_paper_pairs_vllm.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_vllm_test" \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "Qwen/Qwen2-VL-7B-Instruct" \
  --version v1 \
  --limit 1 \
  --num_runs 3 \
  --max_figures 5 \
  --verbose
```

#### 🚀 Production Mode

**Full run with figures and multiple runs:**

```bash
python review_paper_pairs_vllm.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./data/ICLR2024_pairs/reviews_vllm" \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "Qwen/Qwen2-VL-7B-Instruct" \
  --version both \
  --num_runs 3 \
  --max_figures 5 \
  --max_workers 5 \
  --verbose
```

**Continue from previous run (skip existing):**

```bash
python review_paper_pairs_vllm.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./data/ICLR2024_pairs/reviews_vllm" \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "Qwen/Qwen2-VL-7B-Instruct" \
  --version both \
  --num_runs 3 \
  --skip_existing \
  --max_workers 5
```

#### vLLM-Specific Arguments

- `--vllm_endpoint`: URL of vLLM server (e.g., `http://localhost:8000`)
- `--model_name`: Model name hosted on vLLM (e.g., `Qwen/Qwen2-VL-7B-Instruct`)
- `--max_figures`: Number of figures to include (default: 5, set 0 to disable)
- `--num_runs`: Number of times to review each paper (default: 1)
  - Use `--num_runs 3` or `--num_runs 5` for variance analysis
  - Each run is saved separately as `v1_review_run0.json`, `v1_review_run1.json`, etc.

#### vLLM Output Structure

```
reviews_vllm/
├── {paper_id}/
│   ├── v1_review_run0.json         # First run of v1
│   ├── v1_review_run1.json         # Second run of v1
│   ├── v1_review_run2.json         # Third run of v1
│   ├── latest_review_run0.json     # First run of latest
│   ├── latest_review_run1.json     # Second run of latest
│   └── latest_review_run2.json     # Third run of latest
└── review_summary.csv              # All runs aggregated (includes run_id column)
```

**Note**: With `--num_runs 3`, you get 3x the reviews, allowing you to:
- Analyze LLM response variance/consistency
- Compute confidence intervals for scores
- Study how stochastic the model's judgments are

### Context Length Management

The script automatically handles context length limits:

- **SEA-E**: 32K tokens → Papers auto-truncated if needed
- **Default models**: 128K tokens → Rare truncation
- **Truncation tracking**: All reviews include `was_truncated: true/false` flag

See `CONTEXT_LENGTH_MANAGEMENT.md` for details on:
- How truncation works (preserves beginning + end)
- Monitoring truncated reviews
- Adjusting truncation behavior
- Token estimation accuracy

## Output Structure

### Anthropic Version (review_paper_pairs.py)

```
pair_reviews/
├── {paper_id}/
│   ├── v1_review.json          # Full review of v1
│   ├── latest_review.json      # Full review of latest
│   └── comparison.json         # Side-by-side comparison
└── review_summary.csv          # All results aggregated
```

### vLLM Version (review_paper_pairs_vllm.py)

```
reviews_vllm/
├── {paper_id}/
│   ├── v1_review_run0.json         # Run 0 of v1
│   ├── v1_review_run1.json         # Run 1 of v1 (if num_runs > 1)
│   ├── latest_review_run0.json     # Run 0 of latest
│   └── latest_review_run1.json     # Run 1 of latest (if num_runs > 1)
└── review_summary.csv              # All runs with run_id column
```

## Key Arguments

### Debug/Credit-Saving Options

- `--version {v1,latest,both}`: Choose which version(s) to review

  - `v1`: Only review v1 versions (saves 50% API calls)
  - `latest`: Only review latest versions (saves 50% API calls)
  - `both`: Review both versions (default)
- `--skip_existing`: Skip papers that already have reviews (idempotent reruns)
- `--limit N`: Only process first N papers (for testing)

### Other Options

- `--model_name`: Choose model (default: claude-3-5-sonnet-20241022)
  - Use `claude-3-5-haiku-20241022` for cheaper testing
- `--max_workers`: Concurrent threads (default: 3)
- `--verbose`: Detailed logging

## Review Output Fields

Each review includes:

- **Scores (1-10)**: clarity, novelty, technical_quality, experimental_rigor, overall
- **Confidence (1-5)**: Reviewer confidence
- **Recommendation**: Strong Accept/Accept/Weak Accept/Borderline/Weak Reject/Reject/Strong Reject
- **Strengths/Weaknesses**: Bulleted lists
- **Summary**: 2-3 sentence overview
- **Detailed Comments**: Explanatory text

The `review_summary.csv` includes all scores plus score changes (e.g., `overall_score_change`).

## Cost Estimation

- **Haiku** (~$0.25 per 1M input tokens): ~$0.01-0.05 per paper review
- **Sonnet** (~$3 per 1M input tokens): ~$0.10-0.50 per paper review

Use `--limit` and `--version v1` for cheap testing!
