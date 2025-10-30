# Paper Review Comparison Workflow

This directory contains scripts to compare reviews between v1 and latest versions of papers.

## 📁 Directory Structure

```
INTRO_compare_review_be4_after/
├── scripts/              # All executable Python scripts
│   ├── review/          # Review generation (Anthropic, vLLM)
│   ├── evaluation/      # Analysis and statistical evaluation
│   └── utils/           # Data preparation utilities
├── docs/                # Complete documentation
│   ├── guides/          # User guides and tutorials
│   ├── implementation/  # Technical documentation
│   └── fixes/           # Bug fix documentation
├── data/                # Paper data and CSVs
│   └── ICLR2024_pairs/  # Paper pairs (v1 vs latest)
├── tests/               # Test scripts and outputs
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

**📌 Quick Navigation:**
- **Getting Started** → See sections below
- **Detailed Guides** → `docs/guides/`
- **Script Documentation** → `scripts/README.md`
- **API Reference** → `docs/implementation/`

## Quick Comparison: Review Methods

| Feature | `review_paper_pairs.py`<br/>(Anthropic/Gemini API) | **Manual Gemini UI**<br/>(No API) | `review_paper_pairs_vllm.py`<br/>(vLLM) |
|---------|----------------------------------------|--------------------------------------|----------------------------------------|
| **API Key Required** | ✅ Yes | ❌ No (FREE!) | N/A (self-hosted) |
| **Cost** | Pay-per-use | **FREE** | Free (self-hosted) |
| **Setup** | Easy (API key) | **Easiest (none!)** | Complex (vLLM server) |
| **Speed** | Fast (automated) | Slow (manual) | Fast (automated) |
| **Scalability** | Excellent | Limited (manual work) | Excellent |
| **Figures/Images** | Anthropic: ❌<br/>Gemini: ✅ | ✅ Yes | ✅ Yes |
| **Multiple Runs** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Formats** | Default + CriticalNeurIPS | Default + CriticalNeurIPS | All 5 formats |
| **Best For** | Large datasets, automation | **Small datasets, no budget** | Research, control |

### 🆓 NEW: Manual Gemini Workflow (No API Needed!)

Perfect for users without API access! Uses free Gemini Pro via web UI.

```bash
# 1. Prepare prompts and figures
python scripts/utils/prepare_manual_gemini_prompts.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ./manual_gemini_reviews/ \
  --format CriticalNeurIPS

# 2. Complete reviews manually at https://aistudio.google.com/
# Follow README.md in each paper folder

# 3. Process outputs
python scripts/utils/process_manual_gemini_outputs.py \
  --input_dir ./manual_gemini_reviews/ \
  --output_dir ./reviews_gemini_manual/
```

**📖 Complete Guide**: See [`docs/guides/MANUAL_GEMINI_WORKFLOW.md`](docs/guides/MANUAL_GEMINI_WORKFLOW.md)

### Choosing the Right Method

**Use Anthropic/Gemini API** if:
- You have API access
- Need automation
- Large dataset (>100 papers)
- Want speed

**Use Manual Gemini UI** if:
- **No API access** 🎯
- **Small dataset** (<100 papers)
- **No budget** 💰
- Want to learn the process

**Use vLLM** if:
- Need custom models
- Want complete control
- Self-hosting capability
- Need all format options

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

### For Anthropic/Gemini API (review_paper_pairs.py)

**Supports both Anthropic Claude and Google Gemini!**

Add to your `.env` file:
```bash
# For Anthropic (Claude)
ANTHROPIC_API_KEY=your_anthropic_key_here

# For Gemini (optional)
GEMINI_API_KEY=your_gemini_key_here
```

**New Features**:
- ✅ Gemini API support (`--api gemini`)
- ✅ CriticalNeurIPS format (`--format CriticalNeurIPS`)
- ✅ Multiple runs (`--num_runs 3`)

See [`docs/guides/ANTHROPIC_SCRIPT_UPDATES.md`](docs/guides/ANTHROPIC_SCRIPT_UPDATES.md) for details.

### For Manual Gemini (No API)

**No setup needed!** Just use the free Gemini UI at https://aistudio.google.com/

See [`docs/guides/MANUAL_GEMINI_WORKFLOW.md`](docs/guides/MANUAL_GEMINI_WORKFLOW.md) for complete guide.

### For vLLM (review_paper_pairs_vllm.py)
You need a running vLLM server. The script assumes an OpenAI-compatible API endpoint.

## Workflow

### Step 1: Extract Paper Pairs

```bash
python scripts/utils/get_paper_pairs.py
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
python scripts/review/review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_test" \
  --version v1 \
  --limit 1 \
  --verbose
```

**Add latest version to same paper (1 more API call):**

```bash
python scripts/review/review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_test" \
  --version latest \
  --limit 1 \
  --skip_existing
```

**Expand to 10 papers:**

```bash
python scripts/review/review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_test" \
  --version both \
  --limit 10 \
  --skip_existing
```

#### 🚀 PRODUCTION MODE

**Review all v1 versions first:**

```bash
python scripts/review/review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./data/ICLR2024_pairs/reviews" \
  --version v1 \
  --max_workers 3
```

**Then add latest versions:**

```bash
python scripts/review/review_paper_pairs.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./data/ICLR2024_pairs/reviews" \
  --version latest \
  --skip_existing \
  --max_workers 3
```

**Or do both at once:**

```bash
python scripts/review/review_paper_pairs.py \
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
python scripts/review/review_paper_pairs_vllm.py \
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
python scripts/review/review_paper_pairs_vllm.py \
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
python scripts/review/review_paper_pairs_vllm.py \
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
python scripts/review/review_paper_pairs_vllm.py \
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
- `--timeout`: Request timeout in seconds (default: model-specific)
  - **Default: 300s** (5 minutes) for SEA-E, GenericStructured, default
  - **Default: 600s** (10 minutes) for CriticalNeurIPS (deep analysis)
  - **Default: 900s** (15 minutes) for CycleReviewer (generates 4 reviewers)
  - Use `--timeout 1200` for slower models or complex papers

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

The script automatically handles context length limits with a **3-stage smart truncation strategy**:

1. **Remove reference abstracts** (preserves citations)
2. **Remove appendices** (preserves core paper)
3. **Beginning/end truncation** (only if needed)

**Results on sample 52K token paper:**
- After Stage 1: 24K tokens (removed abstracts, 54% savings)
- After Stage 2: 16K tokens (removed appendices, 69% total savings)
- **Fits within SEA-E 23K limit!** ✅ Stage 3 not needed
- **Core content fully preserved:** Abstract, Intro, Methods, Experiments, Results, Conclusions

**Key features:**
- **SEA-E**: 32K tokens → Smart truncation applied automatically
- **Default models**: 128K tokens → Rare truncation needed
- **Transparent**: All reviews include `was_truncated` and `chars_per_token_used` flags
- **Intelligent**: Removes non-essential content first (abstracts, appendices)
- **Adaptive**: Auto-adjusts if initial token estimation is inaccurate

### Model-Specific Formats

The vLLM script supports **multiple output formats** based on the model name:

#### SEA-E Format
```bash
python scripts/review/review_paper_pairs_vllm.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_seae" \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "SEA-E" \
  --limit 5 \
  --verbose
```

**Output:** Single structured review with Summary, Strengths, Weaknesses, Questions, Soundness/Presentation/Contribution ratings, Overall Rating, and Paper Decision.

**Robust Parsing:** The SEA-E parser automatically handles format variations:
- All bullet styles: `-`, `*`, `•` and numbered lists (`1.`, `2.`, etc.)
- Verbose text in score fields (extracts complete sentences, not truncated)
- Flexible decision formats (with/without dashes)

See [`SEAE_PARSER_IMPROVEMENTS.md`](./SEAE_PARSER_IMPROVEMENTS.md) for details.

#### CycleReviewer Format
```bash
python scripts/review/review_paper_pairs_vllm.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_cycle" \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "CycleReviewer-Llama-3.1-70B" \
  --num_runs 3 \
  --max_figures 5 \
  --verbose
```

**Output:** Multi-reviewer format with 4 independent reviewer opinions + meta review + justifications + paper decision.

**Benefits:**
- 4 different perspectives per paper
- Inter-reviewer agreement analysis
- Confidence levels per reviewer
- Comprehensive meta review synthesis

#### Default (JSON) Format
```bash
python scripts/review/review_paper_pairs_vllm.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_qwen" \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "Qwen/Qwen2-VL-7B-Instruct" \
  --num_runs 3 \
  --max_figures 5
```

**Output:** Generic JSON format with summary, strengths, weaknesses, multiple scores (clarity, novelty, technical quality, experimental rigor), overall score, confidence, recommendation.

#### GenericStructured Format (for Non-Finetuned Models)
```bash
python scripts/review/review_paper_pairs_vllm.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_llama" \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "meta-llama/Llama-3.1-70B-Instruct" \
  --format GenericStructured \
  --num_runs 3 \
  --max_figures 5
```

**Output:** JSON with extremely detailed format instructions (designed for base models)

**Features:**
- Explicit JSON schema with examples
- Field-by-field format specifications
- Compatible with SEA-E fields for comparison
- Multiple reminders to output only JSON
- Best for models without instruction fine-tuning

#### CriticalNeurIPS Format (for Deep, Scholarly Critiques)
```bash
python scripts/review/review_paper_pairs_vllm.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_critical" \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "meta-llama/Llama-3.1-70B-Instruct" \
  --format CriticalNeurIPS \
  --num_runs 3 \
  --max_figures 5
```

**Output:** Multi-faceted critical review with evidence-based feedback

**Features:**
- **Multi-perspective analysis**: Conceptual critique + Methodological scrutiny
- **Evidence-based**: Cites literature, counter-examples, alternative hypotheses
- **Scholarly rigor**: Questions assumptions, challenges from first principles
- **Longer timeout**: 10 minutes (vs 5 min for other formats) for thorough analysis
- **Markdown formatting**: Combined strengths/weaknesses section
- **Ethical considerations**: Dedicated limitations and societal impact field

**Best for:** High-quality, publication-ready reviews that deeply engage with papers

**For complete details, see:** [`CRITICAL_NEURIPS_FORMAT.md`](./docs/implementation/CRITICAL_NEURIPS_FORMAT.md) | [`Quick Start Guide`](./docs/guides/CRITICAL_NEURIPS_QUICK_START.md)

---

#### Format Override Option

You can use `--format` to override automatic model detection:

```bash
# Force GenericStructured format for any model
python scripts/review/review_paper_pairs_vllm.py \
  --model_name "any-model" \
  --format GenericStructured \
  ...

# Options: SEA-E, CycleReviewer, GenericStructured, CriticalNeurIPS, default
```

**For more details, see:** [`MODEL_FORMATS.md`](./MODEL_FORMATS.md)

**Adaptive truncation:** If context errors still occur (token estimation off), the system automatically:
1. Detects context length error from API
2. Reduces `chars_per_token` (more conservative estimate)
3. Re-truncates more aggressively
4. Retries (up to 6 attempts with different estimations)
5. Tracks which estimation worked (`chars_per_token_used`)

See documentation:
- `CONTEXT_LENGTH_MANAGEMENT.md` - 3-stage truncation strategy
- `ADAPTIVE_TRUNCATION.md` - Dynamic estimation adjustment
- `SMART_TRUNCATION_SUMMARY.md` - Quick overview

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

## Step 3: Evaluate Numerical Scores

After generating reviews, use the evaluation script to extract numerical scores and perform statistical analysis:

```bash
python scripts/evaluation/evaluate_numerical_scores.py \
  --reviews_dir ./reviews_vllm \
  --output_dir ./evaluation_results
```

### What It Does

1. **Extracts Numerical Scores**: Automatically parses soundness, presentation, contribution, and rating from all review formats
2. **Handles CycleReviewer**: Aggregates scores from 4 reviewers per paper
3. **Paired t-Tests**: Compares v1 vs latest versions to test if AI can differentiate
4. **Effect Sizes**: Computes Cohen's d to quantify magnitude of differences
5. **Inter-Reviewer Agreement**: Analyzes consistency across CycleReviewer's multiple reviewers

### Output Files

```
evaluation_results/
├── evaluation_scores.csv                    # Raw scores (all papers, versions, runs, reviewers)
├── evaluation_summary.csv                   # Summary table with t-test results
├── evaluation_detailed_results.json         # Complete statistics (t, p, Cohen's d, CI)
└── evaluation_cyclereviewer_agreement.csv   # Inter-reviewer agreement (if applicable)
```

### Example Summary Output

```
Model         Metric        N    v1_mean  latest_mean  Difference  t_statistic  p_value  Cohen's_d  Significant
SEA-E         soundness     50   2.800    2.950        0.150       2.345        0.0234   0.332      **
SEA-E         presentation  50   2.600    2.850        0.250       3.456        0.0012   0.489      ***
SEA-E         contribution  50   2.700    2.900        0.200       2.789        0.0078   0.395      **
SEA-E         rating        50   5.800    6.400        0.600       4.123        0.0001   0.584      ***
CycleReviewer soundness     50   2.750    2.900        0.150       2.123        0.0389   0.301      **
...
```

**Significance codes**: `***` p<0.01, `**` p<0.05, `ns` not significant

### Interpretation

- **Positive Difference**: Latest version scored higher (improved)
- **Negative Difference**: V1 version scored higher (regressed)
- **p < 0.05**: Statistically significant difference
- **Cohen's d**: Effect size (0.2=small, 0.5=medium, 0.8=large)

**For complete documentation, see:** [`EVALUATION_GUIDE.md`](./docs/guides/EVALUATION_GUIDE.md)

**For visualization plots:** [`EVALUATION_PLOTS_GUIDE.md`](./docs/guides/EVALUATION_PLOTS_GUIDE.md)

## Step 4: Retry Failed Reviews (If Needed)

**Note:** The review script now **automatically retries** failed reviews up to 2 additional times (3 total attempts) before giving up. This handles most transient errors like JSON parsing failures.

If some reviews still fail after automatic retries, you can manually retry only the failed ones:

```bash
python scripts/review/retry_failed_reviews.py \
  --reviews_dir ./reviews_vllm \
  --csv_file ./data/filtered_pairs.csv \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "YourModel" \
  --format GenericStructured \
  --num_runs 3
```

### What It Does

1. **Scans** all review JSON files for failures (`success: false`)
2. **Identifies** missing reviews (expected but don't exist)
3. **Creates** retry CSV with only papers needing retry
4. **Runs** review script automatically on failed papers
5. **Verifies** completion after retry

### Common Errors Fixed

- ✅ **JSON parsing errors** (invalid escapes, truncated output)
- ✅ **API errors** (timeouts, connection issues)
- ✅ **Validation errors** (Pydantic failures)
- ✅ **Missing reviews** (incomplete runs)

### Check Only (No Retry)

```bash
# Just check status without retrying
python scripts/review/retry_failed_reviews.py \
  --reviews_dir ./reviews_vllm \
  --csv_file ./data/filtered_pairs.csv \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "YourModel" \
  --check_only
```

**Output:**
```
Failed reviews: 5
Missing reviews: 3
Total issues: 8
Unique papers needing retry: 6
```

**For complete documentation, see:** [`RETRY_GUIDE.md`](./docs/guides/RETRY_GUIDE.md)

## Step 5: Compare AI vs Human Review Scores

**NEW!** You can now compare AI-generated review scores against official human reviews from OpenReview.

### Step 5a: Fetch Human Review Scores

**Test first** (recommended):
```bash
python tests/test_fetch_human_scores.py
```

This quick test verifies the OpenReview API is working correctly.

**Fetch scores:**
```bash
python scripts/utils/fetch_human_scores.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv
```

This will:
- Query OpenReview API v2 for each paper's submission with replies
- Filter replies for official reviews (per [OpenReview API guide](https://docs.openreview.net/how-to-guides/data-retrieval-and-modification/how-to-get-all-notes-for-submissions-reviews-rebuttals-etc))
- Extract numerical scores (soundness, presentation, contribution, rating)
- Aggregate across multiple human reviewers (mean ± std)
- Save to `filtered_pairs_with_human_scores.csv`

**Troubleshooting:**
If no reviews are found, use debug mode:
```bash
python scripts/utils/fetch_human_scores.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --limit 1 \
  --debug
```

**Output example:**
```
Papers with human reviews: 122/125
Average number of reviews per paper: 3.41

Human score statistics (mean ± std):
  Soundness: 2.87 ± 0.45
  Presentation: 2.91 ± 0.52
  Contribution: 2.78 ± 0.49
  Rating: 6.43 ± 1.12
```

### Step 5b: Calculate MSE and MAE

```bash
python scripts/evaluation/calculate_mse_mae.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs_with_human_scores.csv \
  --reviews_dir ./reviews_vllm_Llama3-1_70B_3_runs/ \
  --output_dir ./ai_vs_human_evaluation/
```

This will:
- Match AI and human scores by paper ID
- Calculate MSE, MAE, RMSE, and correlation
- Generate comparison plots
- Save detailed results

**Output files:**
```
ai_vs_human_evaluation/
├── ai_vs_human_results.csv          # Summary: MAE, RMSE per metric
├── ai_vs_human_detailed.csv         # All AI-Human score pairs
├── ai_vs_human_results.json         # Complete statistics
└── Plots:
    ├── {model}_ai_vs_human_scatter.png       # Scatter plots with regression
    ├── {model}_error_distributions.png       # Prediction error distributions
    └── {model}_mae_rmse_comparison.png       # Bar chart across metrics
```

**Example results:**
```
SOUNDNESS:
  Pairs: 366
  Human: 2.87 ± 0.45
  AI:    2.92 ± 0.38
  MAE:   0.42    ← Average absolute difference
  RMSE:  0.54    ← Root mean squared error
  Correlation: 0.68  ← Agreement strength

RATING:
  Pairs: 366
  Human: 6.43 ± 1.12
  AI:    6.58 ± 0.89
  MAE:   0.82
  RMSE:  1.05
  Correlation: 0.71
```

**Interpretation:**
- **MAE < 0.5** → Good agreement for 1-4 scales
- **MAE < 1.0** → Good agreement for 1-10 scales
- **Correlation > 0.6** → Strong relationship
- **Mean difference ≈ 0** → No systematic bias

### Research Questions Answered

1. **How accurate are AI reviewers?** → MAE/RMSE metrics
2. **Does AI agree with human consensus?** → Correlation coefficient
3. **Is AI systematically biased?** → Compare AI mean vs Human mean
4. **Which metrics are hardest to predict?** → Compare MAE across metrics

**For complete documentation, see:** [`AI_VS_HUMAN_GUIDE.md`](./docs/guides/AI_VS_HUMAN_GUIDE.md)

## Step 6: Evaluate Consensus Flaw Detection

**NEW!** Evaluate whether AI reviewers detect the consensus flaws (ground truth weaknesses from paper revisions).

### Step 6a: Evaluate Flaw Detection

```bash
python scripts/evaluation/evaluate_flaw_detection.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --reviews_dir ./reviews_vllm_Llama3-1_70B_3_runs/ \
  --evaluator_endpoint "http://localhost:8000" \
  --evaluator_model "Qwen3-30B-A3B-Instruct-2507-FP8"
```

This will:
- Load ground truth flaws from `flaw_descriptions` in CSV
- Extract weaknesses from AI reviews
- Use an evaluator LLM to check if each flaw is mentioned
- Calculate **recall** = (flaws detected) / (total flaws)
- Save detailed results including which specific flaws were detected

**Output example:**
```
Papers with flaw descriptions: 122
Evaluating papers: 100%|████████████| 122/122 [15:32<00:00,  7.64s/it]

Overall Recall Statistics:
  Mean recall: 0.652  ← AI detected 65% of consensus flaws
  Std recall: 0.218

By Version:
  v1: Mean recall: 0.631
  latest: Mean recall: 0.673  ← Slight improvement
```

**Output files:**
```
flaw_detection_results/
├── flaw_detection_detailed.json    # Full results with reasoning per flaw
├── flaw_detection_summary.csv      # Paper-level recall values
└── flaw_detection_per_flaw.csv     # Individual flaw detection records
```

### Step 6b: Analyze with Paired t-Test

```bash
python scripts/evaluation/analyze_flaw_detection.py \
  --results_file ./flaw_detection_results/flaw_detection_detailed.json \
  --output_dir ./flaw_detection_analysis/
```

This will:
- Compare v1 vs latest flaw detection recall
- Perform paired t-test
- Calculate effect size (Cohen's d)
- Generate comparison plots

**Example results:**
```
v1 Recall:     0.631 ± 0.224
Latest Recall: 0.673 ± 0.211
Difference:    0.042 ± 0.158
95% CI: [0.014, 0.070]

t-statistic: 2.941
p-value: 0.0039 **  ← Highly significant!
Cohen's d: 0.266 (small effect)

✅ CONCLUSION: Paper revisions significantly improved the AI's ability
   to detect consensus flaws (p=0.0039).
```

**Output files:**
```
flaw_detection_analysis/
├── flaw_detection_ttest_results.json
├── flaw_detection_comparison_summary.csv
├── flaw_detection_paired_data.csv
└── Plots:
    ├── flaw_detection_bar_comparison.png       # Bar chart with significance
    ├── flaw_detection_scatter.png              # v1 vs latest scatter
    ├── flaw_detection_difference_distribution.png  ← Key plot!
    └── flaw_detection_violin.png               # Distribution comparison
```

### How It Works

The evaluator LLM receives:
1. **Ground truth flaw** (from `flaw_descriptions`)
2. **Weaknesses section** (from AI review)
3. **Question:** Is the flaw mentioned in the weaknesses?

It responds with:
```json
{
  "detected": true,
  "reasoning": "The weaknesses section mentions lack of multiple runs, 
                which matches the ground truth flaw about single-run results."
}
```

### Research Questions Answered

1. **Do AI reviewers detect consensus flaws?** → Overall recall metric
2. **Do revisions make flaws more detectable?** → Paired t-test (v1 vs latest)
3. **Which flaws are hardest to detect?** → Per-flaw analysis
4. **Are results consistent across runs?** → Standard deviation of recall

**For complete documentation, see:** [`FLAW_DETECTION_GUIDE.md`](./docs/guides/FLAW_DETECTION_GUIDE.md)

**Note:** The evaluator includes robust JSON sanitization to handle LLM output issues. See [`FLAW_DETECTION_JSON_FIX.md`](./docs/fixes/FLAW_DETECTION_JSON_FIX.md) for details.

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

## Automatic Retry on Failures

The vLLM script includes **built-in automatic retry** for failed reviews:

- **What triggers a retry?**
  - JSON parsing errors (invalid escape sequences, truncated output)
  - Pydantic validation failures
  - Other non-API errors

- **How many retries?**
  - **2 automatic retries** (3 total attempts per review)
  - 5-second delay between retries
  - Configurable via `MAX_REVIEW_RETRIES` constant in the script

- **What about API errors?**
  - API errors (timeouts, rate limits, 5xx) have **separate retry logic**
  - Up to 3 retries with exponential backoff per attempt
  - These happen *within* each review attempt

- **Example output:**
  ```
  ⚠️  Retrying paper123 (v1, run 0) - Attempt 2/3
  ✅ Retry successful for paper123 (v1, run 0)
  ```

- **Still have failures?**
  - If a review fails after 3 attempts, it's saved with `success: false`
  - Use `retry_failed_reviews.py` to manually retry persistent failures
  - Check [`RETRY_GUIDE.md`](./docs/guides/RETRY_GUIDE.md) for troubleshooting

**This feature significantly reduces the need for manual intervention** on transient JSON/parsing errors that are common with LLMs!

**For complete documentation, see:** [`AUTO_RETRY_SUMMARY.md`](./AUTO_RETRY_SUMMARY.md)

## Cost Estimation

- **Haiku** (~$0.25 per 1M input tokens): ~$0.01-0.05 per paper review
- **Sonnet** (~$3 per 1M input tokens): ~$0.10-0.50 per paper review

Use `--limit` and `--version v1` for cheap testing!
