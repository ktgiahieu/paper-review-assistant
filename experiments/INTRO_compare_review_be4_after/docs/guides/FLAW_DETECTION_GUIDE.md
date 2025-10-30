## Flaw Detection Evaluation Guide

This guide explains how to evaluate whether AI-generated reviews detect the consensus flaws identified by human reviewers during the paper revision process.

## Overview

**Research Question:** Do AI reviewers identify the same weaknesses/flaws that human reviewers found during the review process?

The workflow:
1. **Ground Truth**: Use `flaw_descriptions` from CSV (consensus flaws from paper revisions)
2. **AI Reviews**: Extract weaknesses sections from AI-generated reviews
3. **Evaluator LLM**: Ask an evaluator model if each flaw is mentioned in the weaknesses
4. **Recall Metric**: Calculate what fraction of ground truth flaws were detected
5. **Statistical Analysis**: Compare v1 vs latest to see if revisions made flaws more detectable

## Prerequisites

```bash
pip install requests pandas numpy matplotlib seaborn scipy tqdm
```

You'll also need:
- A running vLLM instance with the evaluator model (e.g., Qwen3-30B-A3B-Instruct-2507-FP8)
- AI-generated reviews (from `review_paper_pairs_vllm.py` or `review_paper_pairs.py`)
- `filtered_pairs.csv` with `flaw_descriptions` column

## Step 1: Evaluate Flaw Detection

### Basic Usage

```bash
python evaluate_flaw_detection.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --reviews_dir ./reviews_vllm_Llama3-1_70B_3_runs/ \
  --evaluator_endpoint "http://localhost:8000" \
  --evaluator_model "Qwen3-30B-A3B-Instruct-2507-FP8"
```

### What It Does

For each paper and version:
1. **Loads ground truth flaws** from `flaw_descriptions` column
2. **Extracts weaknesses** from AI review
3. **Asks evaluator LLM** if each flaw is mentioned (yes/no + reasoning)
4. **Calculates recall** = (flaws detected) / (total flaws)
5. **Saves detailed results** including which specific flaws were detected

### Options

```bash
python evaluate_flaw_detection.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --reviews_dir ./reviews_vllm_Llama3-1_70B_3_runs/ \
  --evaluator_endpoint "http://localhost:8000" \
  --evaluator_model "Qwen3-30B-A3B-Instruct-2507-FP8" \
  --output_dir ./flaw_detection_results/ \
  --version both \
  --limit 10 \
  --verbose
```

**Arguments:**
- `--csv_file`: Path to CSV with flaw_descriptions (required)
- `--reviews_dir`: Directory with AI reviews (required)
- `--evaluator_endpoint`: vLLM endpoint (default: `http://localhost:8000`)
- `--evaluator_model`: Evaluator model name (default: `Qwen3-30B-A3B-Instruct-2507-FP8`)
- `--output_dir`: Where to save results (default: `./flaw_detection_results/`)
- `--version`: Evaluate `v1`, `latest`, or `both` (default: `both`)
- `--limit`: Process only first N papers for testing
- `--verbose`: Print detailed progress

### Output Files

```
flaw_detection_results/
├── flaw_detection_detailed.json      # Full results with reasoning for each flaw
├── flaw_detection_summary.csv        # Paper-level summary (recall per paper)
└── flaw_detection_per_flaw.csv       # Flaw-level details (one row per flaw)
```

#### 1. `flaw_detection_detailed.json`

Complete results with nested structure:

```json
[
  {
    "paper_id": "ViNe1fjGME",
    "title": "Deep Temporal Graph Clustering",
    "version": "v1",
    "run_id": 0,
    "model_type": "GenericStructured",
    "num_flaws": 4,
    "num_detected": 2,
    "recall": 0.5,
    "flaws_detailed": [
      {
        "flaw": "Initial results were presented from only a single run...",
        "detected": true,
        "reasoning": "The weakness section mentions lack of statistical robustness..."
      },
      {
        "flaw": "GPU-memory and runtime analyses compared TGC solely against...",
        "detected": false,
        "reasoning": "No mention of comparison methodology issues..."
      },
      ...
    ]
  },
  ...
]
```

#### 2. `flaw_detection_summary.csv`

Paper-level aggregation:

| paper_id | title | version | run_id | model_type | num_flaws | num_detected | recall |
|----------|-------|---------|--------|------------|-----------|--------------|--------|
| ViNe1fjGME | Deep Temporal... | v1 | 0 | GenericStructured | 4 | 2 | 0.50 |
| ViNe1fjGME | Deep Temporal... | latest | 0 | GenericStructured | 4 | 3 | 0.75 |

#### 3. `flaw_detection_per_flaw.csv`

Individual flaw detection records:

| paper_id | version | run_id | flaw | detected | reasoning |
|----------|---------|--------|------|----------|-----------|
| ViNe1fjGME | v1 | 0 | Initial results... | True | Weakness mentions... |
| ViNe1fjGME | v1 | 0 | GPU-memory... | False | No mention of... |

### Example Output

```
================================================================================
Flaw Detection Evaluation
================================================================================

CSV file: ./data/ICLR2024_pairs/filtered_pairs.csv
Reviews directory: ./reviews_vllm_Llama3-1_70B_3_runs/
Evaluator endpoint: http://localhost:8000
Evaluator model: Qwen3-30B-A3B-Instruct-2507-FP8
Output directory: ./flaw_detection_results/

Loading paper data...
Found 125 papers
Papers with flaw descriptions: 122

================================================================================
Evaluating flaw detection in reviews...
================================================================================
(This may take a while depending on the number of papers and flaws)

Evaluating papers: 100%|████████████| 122/122 [15:32<00:00,  7.64s/it]

================================================================================
Saving results...
================================================================================

✅ Saved detailed results to: flaw_detection_results/flaw_detection_detailed.json
✅ Saved summary to: flaw_detection_results/flaw_detection_summary.csv
✅ Saved per-flaw details to: flaw_detection_results/flaw_detection_per_flaw.csv

================================================================================
SUMMARY STATISTICS
================================================================================

Total evaluations: 732
Unique papers: 122
Versions: ['v1' 'latest']
Model types: ['GenericStructured']

Overall Recall Statistics:
  Mean recall: 0.652
  Std recall: 0.218
  Min recall: 0.000
  Max recall: 1.000

By Version:
  v1:
    Mean recall: 0.631
    Std recall: 0.224
    N: 366
  latest:
    Mean recall: 0.673
    Std recall: 0.211
    N: 366

Flaw Detection Rate:
  Total flaws: 488
  Total detected: 318
  Detection rate: 65.2%

================================================================================
Next steps:
  Run: python analyze_flaw_detection.py \
         --results_file flaw_detection_results/flaw_detection_detailed.json
================================================================================
```

### How the Evaluator Works

The evaluator LLM receives:

**System Prompt:**
```
You are an expert evaluator tasked with determining whether a specific flaw
mentioned in a paper review matches a ground truth flaw description.

Consider a match if:
- The weakness directly mentions the same issue
- The weakness describes the same problem with different wording
- The weakness implies or relates to the ground truth flaw

Respond in JSON: {"detected": true/false, "reasoning": "explanation"}
```

**User Prompt:**
```
GROUND TRUTH FLAW:
Initial results were presented from only a single run, even though clustering
performance is known to fluctuate with random seeds.

WEAKNESSES SECTION FROM REVIEW:
- The paper lacks statistical robustness as experiments were run only once
- No error bars or confidence intervals reported
- Results may not be reproducible

Is this ground truth flaw detected/mentioned in the weaknesses section?
```

**Response:**
```json
{
  "detected": true,
  "reasoning": "The weaknesses section clearly identifies the lack of multiple runs
               and statistical robustness, which directly matches the ground truth
               flaw about single-run results."
}
```

## Step 2: Analyze Flaw Detection with Paired t-Test

### Basic Usage

```bash
python analyze_flaw_detection.py \
  --results_file ./flaw_detection_results/flaw_detection_detailed.json \
  --output_dir ./flaw_detection_analysis/
```

### What It Does

1. **Loads evaluation results** from Step 1
2. **Pairs v1 and latest** recalls for each paper
3. **Performs paired t-test** to compare v1 vs latest
4. **Calculates effect size** (Cohen's d)
5. **Generates plots** showing differences
6. **Provides interpretation** of statistical significance

### Output Files

```
flaw_detection_analysis/
├── flaw_detection_ttest_results.json           # Statistical test results
├── flaw_detection_comparison_summary.csv       # Summary table
├── flaw_detection_paired_data.csv              # Paired recall values
└── Plots:
    ├── flaw_detection_bar_comparison.png       # Bar plot with significance
    ├── flaw_detection_scatter.png              # v1 vs latest scatter
    ├── flaw_detection_difference_distribution.png  # Paired t-test visualization
    └── flaw_detection_violin.png               # Distribution comparison
```

### Example Output

```
================================================================================
Flaw Detection Analysis - Paired t-Test
================================================================================

Results file: ./flaw_detection_results/flaw_detection_detailed.json
Output directory: ./flaw_detection_analysis/

Loading flaw detection results...
Loaded 732 evaluation results

Unique papers: 122
Versions: ['v1', 'latest']
Model types: ['GenericStructured']

Paired samples: 122

================================================================================
Performing Paired t-Test...
================================================================================

================================================================================
RESULTS
================================================================================

Paired samples: 122
Unique papers: 122

v1 Recall:
  Mean: 0.631
  Std:  0.224

Latest Recall:
  Mean: 0.673
  Std:  0.211

Difference (Latest - v1):
  Mean: 0.042
  Std:  0.158
  95% CI: [0.014, 0.070]

Statistical Test:
  t-statistic: 2.941
  p-value: 0.0039
  Cohen's d: 0.266

  Significant at α=0.05: Yes**
  Significant at α=0.01: Yes**

Interpretation:
  Latest version had 0.042 higher recall than v1 (highly significant, p=0.0039).
  Effect size: small (Cohen's d=0.266).

================================================================================

✅ Saved t-test results to: flaw_detection_analysis/flaw_detection_ttest_results.json
✅ Saved summary to: flaw_detection_analysis/flaw_detection_comparison_summary.csv
✅ Saved paired data to: flaw_detection_analysis/flaw_detection_paired_data.csv

================================================================================
Generating plots...
================================================================================
  Saved: flaw_detection_analysis/flaw_detection_bar_comparison.png
  Saved: flaw_detection_analysis/flaw_detection_scatter.png
  Saved: flaw_detection_analysis/flaw_detection_difference_distribution.png
  Saved: flaw_detection_analysis/flaw_detection_violin.png

✅ Plots generated successfully!

================================================================================
Analysis complete!
================================================================================

✅ CONCLUSION: Paper revisions significantly improved the AI's ability
   to detect consensus flaws (p=0.0039).
```

## Understanding the Plots

### 1. Bar Comparison Plot

**What it shows:** Mean recall for v1 vs latest with error bars

**Key elements:**
- Blue bar = v1 mean recall
- Green bar = latest mean recall
- Error bars = standard deviation
- Significance stars above (**, *, or ns)

**Interpretation:**
- Higher bar = Better flaw detection
- Stars = Statistically significant difference

### 2. Scatter Plot

**What it shows:** Each point = one paper, X=v1 recall, Y=latest recall

**Key elements:**
- Red dashed line = No change (v1 = latest)
- Green line = Regression trend
- Points above red line = Improved in latest
- Points below red line = Worse in latest

**Interpretation:**
- Most points above diagonal → Overall improvement
- Tight clustering around diagonal → Consistent across versions
- Wide scatter → Variable improvement

### 3. Difference Distribution Plot ⭐

**What it shows:** Distribution of recall differences (Latest - v1)

**Key elements:**
- Histogram + KDE curve
- Black dashed line = Zero (no change)
- Red line = Mean difference
- Green shaded = 95% confidence interval

**Interpretation:**
- Mean away from zero → Systematic improvement/decline
- CI excludes zero → Significant difference
- Narrow distribution → Consistent changes
- Right-skewed → Most improved, few declined

**This is the KEY PLOT for understanding the paired t-test!**

### 4. Violin Plot

**What it shows:** Full distribution comparison

**Interpretation:**
- Wider = More variability
- Peak position = Most common recall value
- Compare shapes to see distribution changes

## Metrics Explained

### Recall

**Formula:** Recall = (Number of flaws detected) / (Total number of flaws)

**Range:** 0.0 to 1.0

**Interpretation:**
- **1.0** = Perfect (all flaws detected)
- **0.5** = Half of flaws detected
- **0.0** = No flaws detected

**Example:**
- Ground truth: 4 flaws
- AI detected: 3 flaws
- Recall = 3/4 = 0.75

### Statistical Significance

**p-value < 0.05:** Significant improvement/decline (95% confidence)  
**p-value < 0.01:** Highly significant (99% confidence)  
**p-value ≥ 0.05:** No significant difference

### Effect Size (Cohen's d)

**Interpretation:**
- **d < 0.2:** Negligible
- **0.2 ≤ d < 0.5:** Small
- **0.5 ≤ d < 0.8:** Medium
- **d ≥ 0.8:** Large

**Example:** d = 0.266 (small effect) means the improvement is statistically significant but modest in magnitude.

## Research Questions Answered

### 1. Do AI reviewers detect consensus flaws?

**Answer:** Check overall recall in Step 1 output

- Recall > 0.7 → Good detection
- Recall 0.5-0.7 → Moderate detection
- Recall < 0.5 → Poor detection

### 2. Do revisions make flaws more/less detectable?

**Answer:** Check paired t-test in Step 2

- **p < 0.05 & positive difference** → Revisions made flaws more obvious
- **p < 0.05 & negative difference** → Revisions obscured flaws
- **p ≥ 0.05** → No change in detectability

### 3. Which flaws are hardest to detect?

**Answer:** Check `flaw_detection_per_flaw.csv`

- Sort by `detected` column
- Look for patterns in flaws that are frequently missed

### 4. Are results consistent across runs?

**Answer:** Compare standard deviations

- Low std (< 0.1) → Consistent
- High std (> 0.3) → Variable

## Troubleshooting

### Issue: "No papers with flaw descriptions found"

**Cause:** CSV doesn't have `flaw_descriptions` column or it's empty

**Solution:**
- Check that CSV has `flaw_descriptions` column
- Verify column is not all NaN or empty strings
- Ensure flaws are in list format: `['flaw 1', 'flaw 2']`

### Issue: Evaluator returns "JSON parse error"

**Cause:** LLM didn't return valid JSON

**Solution:**
- Check evaluator model is working correctly
- Try with `--verbose` to see raw responses
- The script already handles markdown code blocks, but some models may need prompt adjustments

### Issue: Very low recall scores

**Possible causes:**
1. **AI reviews are not detailed** → Check weaknesses sections are substantial
2. **Evaluator is too strict** → Review some false negatives manually
3. **Flaw descriptions are too vague** → Ground truth flaws may need refinement

**Solution:**
- Manually inspect some cases in `flaw_detection_per_flaw.csv`
- Check `reasoning` column to understand evaluator decisions

### Issue: No significant difference (p > 0.05)

**Interpretations:**
1. **Revisions didn't change detectability** (legitimately no effect)
2. **Sample size too small** (need more papers)
3. **High variance** (some improved, some didn't)

**Solution:**
- Check Cohen's d (may have small effect that's not significant)
- Look at difference distribution plot for patterns
- Consider if result is scientifically meaningful

## Advanced Usage

### Compare Multiple AI Models

Evaluate different AI models and compare their flaw detection:

```bash
# Evaluate Model 1
python evaluate_flaw_detection.py \
  --csv_file ./data/filtered_pairs.csv \
  --reviews_dir ./reviews_llama3/ \
  --output_dir ./flaw_llama3/

# Evaluate Model 2
python evaluate_flaw_detection.py \
  --csv_file ./data/filtered_pairs.csv \
  --reviews_dir ./reviews_claude/ \
  --output_dir ./flaw_claude/

# Compare summaries
diff flaw_llama3/flaw_detection_summary.csv flaw_claude/flaw_detection_summary.csv
```

### Evaluate Specific Flaw Types

Filter CSV to papers with specific types of flaws:

```python
import pandas as pd

df = pd.read_csv('filtered_pairs.csv')

# Filter papers mentioning "statistical" in flaws
df_stats = df[df['flaw_descriptions'].str.contains('statistical', na=False)]
df_stats.to_csv('papers_with_stats_flaws.csv', index=False)
```

Then run evaluation on the filtered CSV.

## Performance Notes

### Time Estimates

For 122 papers with ~4 flaws each:
- **Step 1 (Evaluation):** ~15-30 minutes
  - Depends on evaluator model speed
  - ~2-3 seconds per flaw
- **Step 2 (Analysis):** < 10 seconds

### API Usage

- **Evaluator calls:** `N_papers × N_versions × N_runs × N_flaws_per_paper`
- **Example:** 122 papers × 2 versions × 1 run × 4 flaws = ~976 API calls
- **Rate limiting:** 0.2s delay between calls (built-in)

### Cost Considerations

If using paid API for evaluator:
- Estimate tokens per call: ~300-500 tokens
- Total tokens: ~976 calls × 400 tokens = ~390K tokens
- Cost depends on model pricing

## Best Practices

1. **Test first:** Use `--limit 5` to test on a small subset
2. **Use verbose mode:** `--verbose` to monitor progress
3. **Check evaluator:** Manually verify some evaluator decisions are reasonable
4. **Multiple runs:** If using `--num_runs` in review generation, all runs will be evaluated
5. **Save raw results:** Keep `flaw_detection_detailed.json` for reanalysis

## Summary

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `evaluate_flaw_detection.py` | CSV + Reviews | Recall per paper-version |
| 2 | `analyze_flaw_detection.py` | Recall data | Paired t-test + plots |

**Key Insight:** This workflow tells you whether paper revisions made the consensus flaws more or less obvious to AI reviewers, which is a proxy for measuring the effectiveness of revisions in addressing reviewer concerns.

Happy evaluating! 🎯📊

