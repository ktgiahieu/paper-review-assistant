# Numerical Score Evaluator - Implementation Summary

**Date:** October 30, 2025  
**Status:** ✅ Complete  
**File:** `evaluate_numerical_scores.py`

## Overview

Created a comprehensive evaluation script that extracts numerical scores from review JSON files across all formats and performs paired t-tests to determine if AI reviewers can differentiate between v1 and latest paper versions.

## Features Implemented

### ✅ Score Extraction

Extracts 4 numerical metrics from all formats:

| Metric | Range | Formats |
|--------|-------|---------|
| **Soundness** | 1-4 | SEA-E, CycleReviewer, GenericStructured |
| **Presentation** | 1-4 | SEA-E, CycleReviewer, GenericStructured |
| **Contribution** | 1-4 | SEA-E, CycleReviewer, GenericStructured |
| **Rating** | 1-10 | All formats |

### ✅ Format Support

- **SEA-E**: Single reviewer → 1 set of scores
- **CycleReviewer**: 4 reviewers → 4 sets of scores, aggregated for t-tests
- **GenericStructured**: Single reviewer → 1 set of scores
- **Default**: Rating only (from `overall_score`)

### ✅ Statistical Analysis

1. **Paired t-Tests**: Compare v1 vs latest for each metric
2. **Effect Sizes**: Cohen's d to quantify magnitude
3. **Confidence Intervals**: 95% CI for mean difference
4. **Interpretation**: Automatic human-readable summaries

### ✅ CycleReviewer Special Handling

- Extracts scores from all 4 reviewers separately
- Stores with `reviewer_id` (0-3)
- Aggregates by averaging for t-tests
- Computes inter-reviewer agreement (std, range)

### ✅ Multiple Runs Support

- Handles `--num_runs` from review script
- Each run analyzed independently
- Can aggregate or compare consistency

## Implementation Details

### Key Functions

```python
extract_numerical_value(score_str)
  # Extracts number from "3 good", "6: accept", etc.

extract_scores_seae(review_data)
  # SEA-E format extraction

extract_scores_cyclereviewer(review_data)
  # CycleReviewer format (4 reviewers)
  # Returns list of 4 score dicts

extract_scores_generic_structured(review_data)
  # GenericStructured format extraction

extract_scores_default(review_data)
  # Default format (rating only)

collect_all_scores(reviews_dir)
  # Recursively finds all review JSONs
  # Returns DataFrame with all scores

compute_paired_statistics(df, metric)
  # Performs paired t-test
  # Computes effect size, CI
  # Returns complete statistical results

analyze_by_model_type(df)
  # Runs analysis separately per model type
  # Returns nested dict of results

analyze_cyclereviewer_agreement(df)
  # Special analysis for multiple reviewers
  # Computes inter-reviewer statistics
```

### Data Flow

```
1. Input: reviews_dir/
   └── paper_id/
       ├── v1_review_run0.json
       ├── v1_review_run1.json
       ├── latest_review_run0.json
       └── latest_review_run1.json

2. Extract scores → DataFrame
   Columns: paper_id, version, run_id, model_type, reviewer_id, 
            soundness, presentation, contribution, rating

3. For CycleReviewer: 
   - 4 rows per (paper, version, run)
   - Aggregate by mean for t-test

4. Paired t-test:
   - Group by (paper_id, run_id)
   - Pivot: v1 vs latest
   - Test if mean(latest - v1) ≠ 0

5. Output Results:
   ├── *_scores.csv (raw data)
   ├── *_summary.csv (table)
   ├── *_detailed_results.json (full stats)
   └── *_cyclereviewer_agreement.csv (if applicable)
```

## Output Formats

### 1. Raw Scores CSV

```csv
paper_id,version,run_id,model_type,reviewer_id,soundness,presentation,contribution,rating
paper001,v1,0,SEA-E,0,3.0,2.0,3.0,6.0
paper001,latest,0,SEA-E,0,3.0,3.0,3.0,7.0
paper002,v1,0,CycleReviewer,0,2.0,2.0,2.0,5.0
paper002,v1,0,CycleReviewer,1,3.0,2.0,2.0,6.0
paper002,v1,0,CycleReviewer,2,2.0,3.0,3.0,5.0
paper002,v1,0,CycleReviewer,3,3.0,3.0,3.0,6.0
```

**Usage**: Can be loaded into pandas, R, or Excel for further analysis

### 2. Summary CSV

```csv
Model,Metric,N,v1_mean,latest_mean,Difference,t_statistic,p_value,Cohen's_d,Significant
SEA-E,soundness,50,2.800,2.950,0.150,2.345,0.0234,0.332,**
SEA-E,presentation,50,2.600,2.850,0.250,3.456,0.0012,0.489,***
```

**Usage**: Quick overview for publications, presentations

### 3. Detailed JSON

```json
{
  "SEA-E": {
    "soundness": {
      "n_pairs": 50,
      "v1_mean": 2.8,
      "latest_mean": 2.95,
      "mean_difference": 0.15,
      "t_statistic": 2.345,
      "p_value": 0.0234,
      "cohens_d": 0.332,
      "ci_95_lower": 0.021,
      "ci_95_upper": 0.279,
      "significant_at_0.05": true,
      "interpretation": "..."
    }
  }
}
```

**Usage**: Programmatic access, full statistics

### 4. CycleReviewer Agreement CSV

```csv
paper_id,version,run_id,metric,n_reviewers,mean,std,min,max,range
paper001,v1,0,soundness,4,2.5,0.577,2.0,3.0,1.0
```

**Usage**: Analyze reviewer consistency

## Statistical Methods

### Paired t-Test

**Null Hypothesis (H₀)**: μ_diff = 0 (no difference)  
**Alternative (H₁)**: μ_diff ≠ 0 (significant difference)

**Test Statistic**:
```
t = mean(differences) / SE(differences)
df = n_pairs - 1
```

**Decision**:
- p < 0.05 → Reject H₀ (significant)
- p ≥ 0.05 → Fail to reject H₀ (not significant)

### Effect Size (Cohen's d)

```
d = mean_difference / std_difference
```

**Interpretation**:
- |d| < 0.2: Negligible
- 0.2 ≤ |d| < 0.5: Small
- 0.5 ≤ |d| < 0.8: Medium
- |d| ≥ 0.8: Large

### Confidence Interval

95% CI for mean difference using t-distribution:
```
CI = mean_diff ± t_critical * SE_diff
```

## Usage Examples

### Basic Analysis

```bash
python evaluate_numerical_scores.py \
  --reviews_dir ./reviews_seae \
  --output_dir ./eval_seae
```

### Multiple Models

```bash
# SEA-E
python evaluate_numerical_scores.py \
  --reviews_dir ./reviews_seae \
  --output_dir ./eval_seae \
  --output_prefix seae

# CycleReviewer
python evaluate_numerical_scores.py \
  --reviews_dir ./reviews_cycle \
  --output_dir ./eval_cycle \
  --output_prefix cycle

# GenericStructured
python evaluate_numerical_scores.py \
  --reviews_dir ./reviews_llama \
  --output_dir ./eval_llama \
  --output_prefix llama
```

### Load Results in Python

```python
import pandas as pd
import json

# Load raw scores
scores = pd.read_csv('eval_seae/seae_scores.csv')

# Load summary
summary = pd.read_csv('eval_seae/seae_summary.csv')

# Load detailed results
with open('eval_seae/seae_detailed_results.json') as f:
    results = json.load(f)

# Access specific metric
soundness_results = results['SEA-E']['soundness']
print(f"p-value: {soundness_results['p_value']}")
print(f"Cohen's d: {soundness_results['cohens_d']}")
```

## Research Applications

### Question 1: Can AI differentiate v1 vs latest?

**Analysis**: Check p-values
```python
summary[summary['p_value'] < 0.05]  # Significant differences
```

**Interpretation**:
- p < 0.05 for soundness → Yes, AI detects soundness differences
- p ≥ 0.05 for rating → No, AI cannot detect rating differences

### Question 2: Which metric shows largest effect?

**Analysis**: Compare Cohen's d values
```python
summary.sort_values('Cohen\'s_d', ascending=False)
```

**Interpretation**: Higher |d| = larger practical effect

### Question 3: Are CycleReviewer's 4 reviewers consistent?

**Analysis**: Check inter-reviewer agreement
```python
agreement = pd.read_csv('eval_cycle/cycle_cyclereviewer_agreement.csv')
agreement.groupby('metric')['std'].mean()
```

**Interpretation**: Lower std = more consistent

### Question 4: Do results generalize across runs?

**Analysis**: Compare results across different `run_id`
```python
scores.groupby(['metric', 'run_id']).apply(
    lambda g: compute_paired_statistics(g, g.name[0])
)
```

## Integration with Review Workflow

### Complete Pipeline

```bash
# Step 1: Generate reviews
python review_paper_pairs_vllm.py \
  --csv_file ./data/filtered_pairs.csv \
  --output_dir ./reviews_seae \
  --model_name "SEA-E" \
  --version both \
  --num_runs 3 \
  --max_workers 5

# Step 2: Evaluate scores
python evaluate_numerical_scores.py \
  --reviews_dir ./reviews_seae \
  --output_dir ./eval_seae

# Step 3: Analyze results
python analyze_results.py ./eval_seae  # Custom analysis script
```

## Advantages

### 1. Format Agnostic
- Works with all 4 formats (SEA-E, CycleReviewer, GenericStructured, default)
- Automatically detects format from `model_type` field

### 2. CycleReviewer Aware
- Properly handles 4 reviewers
- Aggregates for t-tests
- Separate agreement analysis

### 3. Comprehensive Statistics
- Not just p-values
- Effect sizes, CIs, interpretations
- Publication-ready results

### 4. Multiple Output Formats
- CSV for Excel/pandas
- JSON for programmatic access
- Human-readable interpretations

### 5. Robust Extraction
- Handles various score formats: "3 good", "6: accept", "3.5"
- Graceful error handling
- Missing value support

## Limitations & Future Work

### Current Limitations

1. **Assumes Paired Data**: Requires both v1 and latest for each paper
2. **No Multiple Testing Correction**: Bonferroni/FDR could be added
3. **No Visualization**: Could add plots (boxplots, violin plots)
4. **Single Test**: Only paired t-test (could add Wilcoxon signed-rank)

### Potential Enhancements

1. **Visualization Module**:
   ```python
   python plot_results.py --eval_dir ./eval_seae --output plots/
   ```

2. **Multiple Testing Correction**:
   ```python
   from statsmodels.stats.multitest import multipletests
   _, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')
   ```

3. **Non-Parametric Alternative**:
   ```python
   from scipy.stats import wilcoxon
   statistic, p_value = wilcoxon(differences)
   ```

4. **Subgroup Analysis**:
   ```python
   # By paper topic, length, etc.
   analyze_by_subgroup(df, 'topic')
   ```

5. **Power Analysis**:
   ```python
   from statsmodels.stats.power import TTestPower
   power = TTestPower().solve_power(effect_size=0.5, nobs=50, alpha=0.05)
   ```

## Testing

### Validation Checks

```bash
# Check score extraction
python -c "
from evaluate_numerical_scores import extract_numerical_value
assert extract_numerical_value('3 good') == 3.0
assert extract_numerical_value('6: accept') == 6.0
print('✅ Score extraction works')
"

# Test on sample data
mkdir -p test_reviews/paper001
echo '{"success": true, "model_type": "SEA-E", "paper_id": "paper001", "version": "v1", "run_id": 0, "soundness": "3 good", "presentation": "2 fair", "contribution": "3 good", "rating": "6: marginally above"}' > test_reviews/paper001/v1_review_run0.json
echo '{"success": true, "model_type": "SEA-E", "paper_id": "paper001", "version": "latest", "run_id": 0, "soundness": "3 good", "presentation": "3 good", "contribution": "3 good", "rating": "7: accept"}' > test_reviews/paper001/latest_review_run0.json

python evaluate_numerical_scores.py --reviews_dir test_reviews --output_dir test_eval

# Should produce 4 files
ls test_eval/
```

## Dependencies

Added to `requirements.txt`:
```
scipy>=1.10.0      # For t-tests
numpy>=1.24.0      # For numerical operations
```

Existing dependencies used:
```
pandas>=2.0.0      # For DataFrames
```

## Documentation

Created comprehensive documentation:

1. **`EVALUATION_GUIDE.md`** (700+ lines)
   - Complete usage guide
   - Statistical explanations
   - Example workflows
   - Troubleshooting

2. **`EVALUATION_IMPLEMENTATION_SUMMARY.md`** (this file)
   - Technical implementation details
   - Code structure
   - Research applications

3. **Updated `README.md`**
   - Added "Step 3: Evaluate Numerical Scores"
   - Quick reference
   - Link to detailed guide

## Summary

✅ **Fully Functional**: Extracts scores, performs t-tests, generates reports  
✅ **Format Complete**: All 4 formats supported  
✅ **CycleReviewer Special**: Handles 4 reviewers correctly  
✅ **Statistical**: Proper paired t-tests with effect sizes  
✅ **Documented**: Comprehensive guides and examples  
✅ **Production Ready**: Error handling, validation, output files

**Lines of Code**: ~450 lines  
**Documentation**: ~1,200 lines  
**Total**: ~1,650 lines

---

**Status**: ✅ Complete and ready for use  
**Next Step**: Run on actual review data to validate results

