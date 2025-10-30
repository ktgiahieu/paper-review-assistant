# AI vs Human Review Score Comparison Guide

This guide explains how to fetch human review scores from OpenReview and calculate MSE/MAE to compare AI-generated reviews with human reviewers.

## Overview

The workflow consists of two steps:

1. **Fetch Human Scores** - Use `fetch_human_scores.py` to retrieve official review scores from OpenReview API
2. **Calculate Metrics** - Use `calculate_mse_mae.py` to compute MSE, MAE, and correlation between AI and human scores

## Prerequisites

```bash
pip install requests pandas numpy matplotlib seaborn scipy tqdm
```

All dependencies are already in `requirements.txt`.

## Step 1: Fetch Human Review Scores

### Basic Usage

```bash
python fetch_human_scores.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv
```

This will:
- Read paper IDs from the CSV
- Query OpenReview API v2 for each paper's submission with replies
- Filter replies for official reviews (invitations ending with "Official_Review")
- Extract numerical scores (soundness, presentation, contribution, rating)
- Aggregate scores across multiple reviewers (mean and std)
- Save results to `filtered_pairs_with_human_scores.csv`

**Note:** This script follows the [OpenReview API v2 guide](https://docs.openreview.net/how-to-guides/data-retrieval-and-modification/how-to-get-all-notes-for-submissions-reviews-rebuttals-etc) for retrieving reviews as replies to submission notes.

### Options

```bash
python fetch_human_scores.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_file ./data/ICLR2024_pairs/papers_with_human_scores.csv \
  --limit 10 \
  --verbose \
  --debug
```

**Arguments:**
- `--csv_file`: Input CSV with paper IDs (required)
- `--output_file`: Custom output path (optional, default: `{input}_with_human_scores.csv`)
- `--limit`: Process only first N papers for testing (optional)
- `--verbose`: Print detailed progress (optional)
- `--debug`: Print debug information showing API responses and score extraction (optional, useful for troubleshooting)

### Output Format

The script adds these columns to your CSV:

| Column | Description |
|--------|-------------|
| `num_reviews` | Number of official reviews found |
| `human_soundness_mean` | Mean soundness score (1-4 scale) |
| `human_soundness_std` | Std deviation of soundness |
| `human_presentation_mean` | Mean presentation score (1-4 scale) |
| `human_presentation_std` | Std deviation of presentation |
| `human_contribution_mean` | Mean contribution score (1-4 scale) |
| `human_contribution_std` | Std deviation of contribution |
| `human_rating_mean` | Mean overall rating (1-10 scale) |
| `human_rating_std` | Std deviation of rating |
| `human_confidence_mean` | Mean reviewer confidence (1-5 scale) |
| `human_confidence_std` | Std deviation of confidence |

### Example Output

```
Reading data/ICLR2024_pairs/filtered_pairs.csv...
Found 125 papers

Fetching human review scores from OpenReview API...
(This may take several minutes depending on API rate limits)
Fetching reviews: 100%|████████████| 125/125 [01:04<00:00,  1.94it/s]

✅ Saved results to: data/ICLR2024_pairs/filtered_pairs_with_human_scores.csv

================================================================================
Summary Statistics:
================================================================================

Total papers: 125
Papers with reviews: 122
Papers without reviews: 3

Average number of reviews per paper: 3.41

Human score availability:
  Soundness: 122/125 papers (97.6%)
  Presentation: 122/125 papers (97.6%)
  Contribution: 122/125 papers (97.6%)
  Rating: 122/125 papers (97.6%)

Human score statistics (mean ± std):
  Soundness: 2.87 ± 0.45
  Presentation: 2.91 ± 0.52
  Contribution: 2.78 ± 0.49
  Rating: 6.43 ± 1.12
```

### Rate Limiting

The script includes automatic rate limiting:
- 0.5 second delay between API requests
- Exponential backoff on errors
- Up to 3 retry attempts per paper

**Be respectful to OpenReview's API!** If processing many papers, consider running overnight.

## Step 2: Calculate MSE and MAE

### Basic Usage

```bash
python calculate_mse_mae.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs_with_human_scores.csv \
  --reviews_dir ./reviews_vllm_Llama3-1_70B_3_runs/
```

This will:
- Load human scores from the CSV
- Load AI-generated reviews from the directory
- Match scores by paper ID and version
- Calculate MSE, MAE, RMSE, and correlation
- Generate visualization plots
- Save detailed results

### Options

```bash
python calculate_mse_mae.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs_with_human_scores.csv \
  --reviews_dir ./reviews_vllm_Llama3-1_70B_3_runs/ \
  --output_dir ./ai_vs_human_evaluation/ \
  --version latest
```

**Arguments:**
- `--csv_file`: CSV with human scores (from Step 1)
- `--reviews_dir`: Directory with AI reviews
- `--output_dir`: Where to save results (default: `./ai_vs_human_evaluation/`)
- `--version`: Which paper version to evaluate: `v1`, `latest`, or `both` (default: `latest`)

### Output Files

The script generates:

1. **`ai_vs_human_results.csv`** - Summary statistics per metric
2. **`ai_vs_human_detailed.csv`** - Every AI-Human score pair
3. **`ai_vs_human_results.json`** - Complete results in JSON format
4. **Plots:**
   - `{model}_ai_vs_human_scatter.png` - Scatter plots showing AI vs Human agreement
   - `{model}_error_distributions.png` - Distribution of prediction errors
   - `{model}_mae_rmse_comparison.png` - Bar chart comparing MAE/RMSE across metrics

### Metrics Explained

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **MAE** | Mean Absolute Error | Average absolute difference between AI and human scores. Lower is better. |
| **RMSE** | Root Mean Squared Error | Square root of average squared errors. Penalizes large errors more than MAE. |
| **MSE** | Mean Squared Error | Average of squared errors. RMSE is more interpretable. |
| **Correlation** | Pearson correlation coefficient | How well AI and human scores move together. Range: -1 to 1. |

**Example interpretation:**
- MAE = 0.5 → On average, AI scores differ from human scores by 0.5 points
- Correlation = 0.8 → Strong positive relationship (AI tends to agree with humans)
- RMSE = 0.7 → Typical prediction error is about 0.7 points

### Example Output

```
================================================================================
AI vs Human Review Score Evaluation
================================================================================

Reading human scores from: data/ICLR2024_pairs/filtered_pairs_with_human_scores.csv
Papers with human reviews: 122/125
Reading AI reviews from: reviews_vllm_Llama3-1_70B_3_runs

================================================================================
Collecting scores...
================================================================================
Processing papers: 100%|████████████| 122/122 [00:01<00:00, 95.12it/s]

Collected 1464 human scores
Collected 1464 AI scores

================================================================================
Calculating MSE and MAE...
================================================================================

================================================================================
RESULTS
================================================================================

SOUNDNESS:
  Pairs: 366
  Human: 2.87 ± 0.45
  AI:    2.92 ± 0.38
  MAE:   0.42
  RMSE:  0.54
  MSE:   0.29
  Correlation: 0.68

PRESENTATION:
  Pairs: 366
  Human: 2.91 ± 0.52
  AI:    2.88 ± 0.41
  MAE:   0.48
  RMSE:  0.61
  MSE:   0.37
  Correlation: 0.62

CONTRIBUTION:
  Pairs: 366
  Human: 2.78 ± 0.49
  AI:    2.85 ± 0.44
  MAE:   0.45
  RMSE:  0.58
  MSE:   0.33
  Correlation: 0.65

RATING:
  Pairs: 366
  Human: 6.43 ± 1.12
  AI:    6.58 ± 0.89
  MAE:   0.82
  RMSE:  1.05
  MSE:   1.10
  Correlation: 0.71

✅ Saved summary results to: ai_vs_human_evaluation/ai_vs_human_results.csv
✅ Saved detailed comparison to: ai_vs_human_evaluation/ai_vs_human_detailed.csv
✅ Saved full results to: ai_vs_human_evaluation/ai_vs_human_results.json

================================================================================
Generating visualization plots...
================================================================================
  Saved: ai_vs_human_evaluation/Reviews_Vllm_Llama3-1_70B_3_Runs_ai_vs_human_scatter.png
  Saved: ai_vs_human_evaluation/Reviews_Vllm_Llama3-1_70B_3_Runs_error_distributions.png
  Saved: ai_vs_human_evaluation/Reviews_Vllm_Llama3-1_70B_3_Runs_mae_rmse_comparison.png

✅ Plots generated successfully!

================================================================================
Evaluation complete!
================================================================================
```

## Understanding the Plots

### 1. Scatter Plot: AI vs Human Scores

**What it shows:** Each point represents one paper. X-axis = human score, Y-axis = AI score.

**Key elements:**
- **Red dashed line** - Perfect agreement (AI = Human)
- **Green line** - Regression line (actual trend)
- **Statistics box** - Correlation (r), MAE, RMSE

**Interpretation:**
- Points near red line → Good agreement
- Points above line → AI scored higher than humans
- Points below line → AI scored lower than humans
- Tight clustering → Consistent predictions
- Wide scatter → Inconsistent predictions

### 2. Error Distribution Plot

**What it shows:** Distribution of prediction errors (AI - Human).

**Key elements:**
- **Black dashed line** - Zero error (perfect prediction)
- **Red line** - Mean error
- **Histogram + KDE** - Error distribution shape

**Interpretation:**
- Centered at zero → Unbiased predictions
- Right of zero → AI tends to score higher
- Left of zero → AI tends to score lower
- Narrow distribution → Consistent errors
- Wide distribution → Variable accuracy

### 3. MAE/RMSE Comparison Bar Chart

**What it shows:** MAE and RMSE for each metric side-by-side.

**Interpretation:**
- Lower bars = Better predictions
- RMSE > MAE = Some large errors present
- RMSE ≈ MAE = Errors are consistent

## Troubleshooting

### Issue: "No matching scores found"

**Cause:** Paper IDs in CSV don't match directory names in `reviews_dir`

**Solution:** 
- Check that paper IDs in CSV match folder names in reviews directory
- Ensure AI reviews were successfully generated (check for `success: true` in JSON files)

### Issue: "Papers without reviews: 125" or All reviews showing 0

**Cause:** Reviews not being found or extracted correctly

**Solution:**

1. **Use debug mode** to see what's happening:
```bash
python fetch_human_scores.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --limit 1 \
  --debug
```

This will show:
- API responses
- Number of replies found
- Which replies are official reviews
- Content keys available in reviews
- What scores are being extracted

2. **Verify paper IDs:**
- Check that paper IDs in CSV are correct OpenReview forum IDs
- Papers must be from ICLR 2024 (venue: "ICLR.cc/2024/Conference")
- Papers must be published (not under review)

3. **Check OpenReview directly:**
- Visit https://openreview.net/forum?id={paper_id}
- Verify official reviews exist
- Check if score format matches expectations

4. **Common fixes:**
- Ensure using API v2 endpoints (fixed in current version)
- Reviews must have invitation ending with "Official_Review"
- Scores must be in `content` field of review notes

### Issue: Very high MAE/RMSE

**Cause:** Could indicate:
1. Different score scales between AI and human reviews
2. AI model needs better calibration
3. Papers are genuinely difficult to review consistently

**Solution:**
- Check raw scores in `ai_vs_human_detailed.csv`
- Verify score ranges match (e.g., both use 1-4 for soundness)
- Consider if AI prompt needs adjustment

## Advanced: Comparing Multiple AI Models

To compare different AI models against humans:

```bash
# Fetch human scores once
python fetch_human_scores.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv

# Evaluate Model 1
python calculate_mse_mae.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs_with_human_scores.csv \
  --reviews_dir ./reviews_vllm_Llama3-1_70B/ \
  --output_dir ./evaluation_llama3_vs_human/

# Evaluate Model 2
python calculate_mse_mae.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs_with_human_scores.csv \
  --reviews_dir ./reviews_anthropic_claude/ \
  --output_dir ./evaluation_claude_vs_human/

# Compare results
compare results from both output directories
```

## Research Questions

These tools help answer:

1. **How accurate are AI reviewers?** → Look at MAE/RMSE
2. **Does AI agree with human consensus?** → Look at correlation
3. **Is AI systematically biased?** → Look at mean error (AI mean vs Human mean)
4. **Which metrics are hardest to predict?** → Compare MAE across metrics
5. **Does AI improve with paper revisions?** → Compare v1 vs latest MAE

## Citation

If you use these tools in your research, please cite appropriately and acknowledge OpenReview for providing the human review data.

## API Usage Notes

- **OpenReview API v2** is used: `https://api2.openreview.net`
- **Rate limits:** Script includes 0.5s delay between requests
- **Ethical use:** Only use for research purposes, respect OpenReview's terms of service
- **Data privacy:** Human reviews are publicly available on OpenReview

## Summary

| Step | Command | Input | Output |
|------|---------|-------|--------|
| 1 | `fetch_human_scores.py` | `filtered_pairs.csv` | `filtered_pairs_with_human_scores.csv` |
| 2 | `calculate_mse_mae.py` | CSV + AI reviews | MSE/MAE metrics + plots |

Happy evaluating! 📊✨

