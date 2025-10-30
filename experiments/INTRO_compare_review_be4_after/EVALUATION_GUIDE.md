# Numerical Score Evaluation Guide

## Overview

The `evaluate_numerical_scores.py` script automatically extracts numerical scores from review JSON files and performs paired t-tests to determine if AI reviewers can differentiate between v1 and latest paper versions.

## Supported Metrics

The script extracts and analyzes four numerical scores:

| Metric | Range | Description |
|--------|-------|-------------|
| **Soundness** | 1-4 | Methodology and logical consistency (1=poor, 2=fair, 3=good, 4=excellent) |
| **Presentation** | 1-4 | Clarity, organization, visual aids (1=poor, 2=fair, 3=good, 4=excellent) |
| **Contribution** | 1-4 | Significance and novelty (1=poor, 2=fair, 3=good, 4=excellent) |
| **Rating** | 1-10 | Overall quality (1=strong reject, 10=strong accept) |

## Format Support

### SEA-E Format
- Extracts: soundness, presentation, contribution, rating
- Single reviewer per paper

### CycleReviewer Format
- Extracts: soundness, presentation, contribution, rating from each reviewer
- **4 reviewers per paper** → 4 sets of scores
- Aggregated by averaging across reviewers for t-tests
- Special inter-reviewer agreement analysis

### GenericStructured Format
- Extracts: soundness, presentation, contribution, rating
- Single reviewer per paper

### Default Format
- Extracts: rating only (from `overall_score`)
- Other metrics not available

## Usage

### Basic Usage

```bash
python evaluate_numerical_scores.py \
  --reviews_dir ./reviews_output \
  --output_dir ./evaluation_results
```

### With Custom Prefix

```bash
python evaluate_numerical_scores.py \
  --reviews_dir ./reviews_seae \
  --output_dir ./eval_seae \
  --output_prefix seae_evaluation
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--reviews_dir` | Yes | - | Directory with review outputs (contains paper subdirectories) |
| `--output_dir` | No | `./evaluation_results` | Where to save evaluation results |
| `--output_prefix` | No | `evaluation` | Prefix for output filenames |

## Output Files

The script generates several output files:

### 1. Raw Scores (`*_scores.csv`)

Complete dataset with all extracted scores:

```csv
paper_id,version,run_id,model_type,reviewer_id,soundness,presentation,contribution,rating
abc123,v1,0,SEA-E,0,3.0,2.0,3.0,6.0
abc123,latest,0,SEA-E,0,3.0,3.0,3.0,7.0
def456,v1,0,CycleReviewer,0,2.0,2.0,2.0,5.0
def456,v1,0,CycleReviewer,1,3.0,2.0,2.0,6.0
def456,v1,0,CycleReviewer,2,2.0,3.0,3.0,5.0
def456,v1,0,CycleReviewer,3,3.0,3.0,3.0,6.0
...
```

**Columns:**
- `paper_id`: Unique paper identifier
- `version`: "v1" or "latest"
- `run_id`: Run number (0, 1, 2, ... if using `--num_runs`)
- `model_type`: Format used (SEA-E, CycleReviewer, GenericStructured, default)
- `reviewer_id`: Reviewer number (0 for single-reviewer formats, 0-3 for CycleReviewer)
- `soundness`, `presentation`, `contribution`, `rating`: Numerical scores

### 2. Summary Table (`*_summary.csv`)

Concise overview of all t-test results:

```csv
Model,Metric,N,v1_mean,latest_mean,Difference,t_statistic,p_value,Cohen's_d,Significant
SEA-E,soundness,50,2.800,2.950,0.150,2.345,0.0234,0.332,**
SEA-E,presentation,50,2.600,2.850,0.250,3.456,0.0012,0.489,***
SEA-E,contribution,50,2.700,2.900,0.200,2.789,0.0078,0.395,**
SEA-E,rating,50,5.800,6.400,0.600,4.123,0.0001,0.584,***
...
```

**Significance Codes:**
- `***`: p < 0.01 (highly significant)
- `**`: p < 0.05 (significant)
- `ns`: not significant

### 3. Detailed Results (`*_detailed_results.json`)

Complete statistical analysis in JSON format:

```json
{
  "SEA-E": {
    "soundness": {
      "metric": "soundness",
      "n_pairs": 50,
      "n_papers": 50,
      "v1_mean": 2.8,
      "v1_std": 0.52,
      "latest_mean": 2.95,
      "latest_std": 0.48,
      "mean_difference": 0.15,
      "std_difference": 0.41,
      "t_statistic": 2.345,
      "p_value": 0.0234,
      "cohens_d": 0.332,
      "ci_95_lower": 0.021,
      "ci_95_upper": 0.279,
      "significant_at_0.05": true,
      "significant_at_0.01": false,
      "interpretation": "Latest version scored 0.150 points higher than v1 (significant, p=0.0234). Effect size: small (Cohen's d=0.332)."
    },
    ...
  },
  ...
}
```

### 4. CycleReviewer Agreement (`*_cyclereviewer_agreement.csv`)

Inter-reviewer agreement analysis (only generated if CycleReviewer reviews exist):

```csv
paper_id,version,run_id,metric,n_reviewers,mean,std,min,max,range
abc123,v1,0,soundness,4,2.5,0.577,2.0,3.0,1.0
abc123,v1,0,presentation,4,2.5,0.577,2.0,3.0,1.0
abc123,v1,0,contribution,4,2.5,0.577,2.0,3.0,1.0
abc123,v1,0,rating,4,5.5,0.577,5.0,6.0,1.0
...
```

## Statistical Analysis

### Paired t-Test

The script performs **paired t-tests** comparing v1 vs latest versions:

**Hypothesis:**
- **H₀ (Null)**: No difference between v1 and latest scores (μ_diff = 0)
- **H₁ (Alternative)**: Latest scores differ from v1 scores (μ_diff ≠ 0)

**Methodology:**
1. For each paper pair, compute score difference: `Δ = latest - v1`
2. Test if mean difference significantly differs from zero
3. Report t-statistic, p-value, and effect size (Cohen's d)

### Effect Size (Cohen's d)

Interpreting effect sizes:

| Cohen's d | Interpretation |
|-----------|----------------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| > 0.8 | Large |

### Confidence Intervals

95% confidence intervals for mean difference:
- If CI includes 0 → not significant
- If CI excludes 0 → significant difference

## CycleReviewer Special Handling

### Multiple Reviewers

CycleReviewer provides **4 independent reviews** per paper. The script handles this by:

1. **Extraction**: Extracts scores from all 4 reviewers separately
2. **Storage**: Stores each reviewer's scores individually (`reviewer_id` 0-3)
3. **Aggregation**: For t-tests, averages scores across 4 reviewers per paper
4. **Agreement Analysis**: Computes inter-reviewer statistics (std, range)

### Example: CycleReviewer Data Flow

**Input (1 paper, 1 version, 1 run):**
- 4 reviewers → 4 score records

**For t-test:**
- Average across 4 reviewers → 1 aggregated score per paper-version

**Agreement Analysis:**
- Compute std, range across 4 reviewers
- Low std/range = high agreement
- High std/range = reviewer disagreement

### Inter-Reviewer Agreement Metrics

| Metric | Meaning |
|--------|---------|
| **std** (standard deviation) | Spread of scores across reviewers. Lower = more agreement. |
| **range** | max - min. Range of 0 = perfect agreement, higher = more disagreement. |

**Example:**
```
Paper ABC, Rating:
  Reviewer 0: 6
  Reviewer 1: 5
  Reviewer 2: 7
  Reviewer 3: 6
  
  mean = 6.0
  std = 0.816
  range = 2 (max 7 - min 5)
```

## Interpretation Guide

### Example Output

```
Analyzing model type: SEA-E

SOUNDNESS:
  Pairs: 50 (from 50 papers)
  v1:     2.800 ± 0.520
  Latest: 2.950 ± 0.480
  Diff:   0.150 ± 0.410
  t(49) = 2.345, p = 0.0234
  Cohen's d = 0.332
  95% CI: [0.021, 0.279]
  Latest version scored 0.150 points higher than v1 (significant, p=0.0234). 
  Effect size: small (Cohen's d=0.332).
```

### What This Means

1. **Pairs: 50**: We have 50 paired comparisons (50 papers with both v1 and latest versions)

2. **v1: 2.800 ± 0.520**: 
   - Average soundness score for v1 papers: 2.8 (out of 4)
   - Standard deviation: 0.52

3. **Latest: 2.950 ± 0.480**:
   - Average soundness score for latest papers: 2.95
   - On average, latest papers scored 0.15 points higher

4. **Diff: 0.150 ± 0.410**:
   - Mean difference between paired papers
   - Standard deviation of differences

5. **t(49) = 2.345, p = 0.0234**:
   - t-statistic with 49 degrees of freedom
   - p-value = 0.0234 < 0.05 → **statistically significant**
   - We can reject the null hypothesis (no difference)

6. **Cohen's d = 0.332**:
   - Small effect size
   - Difference is statistically significant but effect is modest

7. **95% CI: [0.021, 0.279]**:
   - We're 95% confident the true mean difference is between 0.021 and 0.279
   - Interval excludes 0 → confirms significance

8. **Conclusion**: AI reviewers **can differentiate** v1 from latest versions in terms of soundness, with latest versions receiving slightly higher scores on average.

## Research Questions Answered

### 1. Can AI reviewers differentiate v1 vs latest?

**Answer**: Check p-values for each metric:
- p < 0.05: Yes, significant difference detected
- p ≥ 0.05: No significant difference

### 2. Which metrics show the biggest differences?

**Answer**: Compare Cohen's d values:
- Higher |d| = larger effect
- Check which metrics have largest effect sizes

### 3. Are latest papers consistently better?

**Answer**: Check sign of mean_difference:
- Positive: Latest scored higher
- Negative: V1 scored higher
- Near zero: No consistent direction

### 4. How reliable are CycleReviewer's multiple reviewers?

**Answer**: Check inter-reviewer agreement:
- Low std/range: Reviewers agree (reliable)
- High std/range: Reviewers disagree (less reliable)

## Advanced Usage

### Filter by Model Type

To analyze only one model type, filter the CSV:

```python
import pandas as pd

# Load raw scores
df = pd.read_csv('evaluation_scores.csv')

# Filter to SEA-E only
df_seae = df[df['model_type'] == 'SEA-E']
df_seae.to_csv('seae_scores.csv', index=False)

# Re-run analysis on filtered data
```

### Analyze Specific Papers

```python
# Papers with highest score improvements
df = pd.read_csv('evaluation_scores.csv')

# Compute improvements per paper
improvements = df.pivot_table(
    index='paper_id',
    columns='version',
    values='rating',
    aggfunc='mean'
)
improvements['improvement'] = improvements['latest'] - improvements['v1']
top_improved = improvements.nlargest(10, 'improvement')
print(top_improved)
```

### Compare Multiple Runs

If using `--num_runs`, you can analyze consistency:

```python
# Compute variance across runs
df = pd.read_csv('evaluation_scores.csv')
variance = df.groupby(['paper_id', 'version', 'model_type'])['rating'].std()
print(f"Average variance across runs: {variance.mean():.3f}")
```

## Troubleshooting

### "No valid scores found"

**Cause**: No reviews with the specified metric  
**Solution**: 
- Check that review JSON files have `success: true`
- Verify scores are in expected format (e.g., "3 good", "6: accept")
- Check model_type matches expected format

### "Insufficient paired data"

**Cause**: Not enough papers with both v1 and latest reviews  
**Solution**:
- Ensure reviews were run for both versions (`--version both`)
- Check `reviews_dir` structure matches expected format
- Verify paper_ids match between v1 and latest

### CycleReviewer shows 0 reviewers

**Cause**: Reviews file doesn't have `reviewers` array  
**Solution**:
- Verify CycleReviewer format was used
- Check raw JSON structure
- May need to re-run reviews with correct format

## Example Workflow

### Complete Analysis Pipeline

```bash
# 1. Run reviews for both versions (multiple runs for variance)
python review_paper_pairs_vllm.py \
  --csv_file ./data/filtered_pairs.csv \
  --output_dir ./reviews_seae \
  --model_name "SEA-E" \
  --version both \
  --num_runs 3 \
  --max_workers 5

# 2. Evaluate numerical scores
python evaluate_numerical_scores.py \
  --reviews_dir ./reviews_seae \
  --output_dir ./eval_seae \
  --output_prefix seae

# 3. View summary
cat ./eval_seae/seae_summary.csv

# 4. Open detailed results
python -m json.tool ./eval_seae/seae_detailed_results.json
```

### Compare Multiple Models

```bash
# Run reviews with different models
python review_paper_pairs_vllm.py --model_name "SEA-E" --output_dir ./reviews_seae ...
python review_paper_pairs_vllm.py --model_name "CycleReviewer-Llama-3.1-70B" --output_dir ./reviews_cycle ...
python review_paper_pairs_vllm.py --model_name "meta-llama/Llama-3.1-70B-Instruct" --format GenericStructured --output_dir ./reviews_llama ...

# Evaluate each
python evaluate_numerical_scores.py --reviews_dir ./reviews_seae --output_dir ./eval_seae
python evaluate_numerical_scores.py --reviews_dir ./reviews_cycle --output_dir ./eval_cycle
python evaluate_numerical_scores.py --reviews_dir ./reviews_llama --output_dir ./eval_llama

# Compare results
python compare_evaluations.py ./eval_seae ./eval_cycle ./eval_llama  # (you'd need to write this script)
```

## Tips for Best Results

### 1. Use Multiple Runs

```bash
--num_runs 3  # or 5 for more robust statistics
```

**Benefits:**
- Accounts for LLM stochasticity
- More reliable mean estimates
- Can assess consistency

### 2. Ensure Balanced Dataset

- Same papers for both v1 and latest
- Complete reviews (check `success: true`)
- Sufficient sample size (aim for N ≥ 30)

### 3. Check Data Quality

Before running evaluation:
```bash
# Count successful reviews
grep -r '"success": true' ./reviews_dir | wc -l

# Check for errors
grep -r '"error"' ./reviews_dir
```

### 4. Interpret Effect Sizes, Not Just p-values

- Small p-value doesn't mean large practical difference
- Always check Cohen's d and mean difference
- Consider domain context (e.g., 0.5 point difference on 1-10 scale)

## Citation

If using this evaluation framework in research, consider documenting:
- Statistical test used (paired t-test)
- Number of paper pairs
- Effect sizes (Cohen's d)
- Confidence intervals
- Multiple testing corrections (if analyzing many metrics)

---

**Questions or Issues?** Check the main README.md or open an issue on GitHub.

