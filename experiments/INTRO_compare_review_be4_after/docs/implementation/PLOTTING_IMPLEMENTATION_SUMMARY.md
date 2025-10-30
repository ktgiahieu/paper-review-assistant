# Plotting Implementation Summary

## Overview

Added comprehensive visualization capabilities to `evaluate_numerical_scores.py` that automatically generate publication-quality plots for statistical analysis results.

## What Was Added

### 1. New Function: `create_comparison_plots()`

**Location:** Lines 476-831 in `evaluate_numerical_scores.py`

**Purpose:** Generate 6 types of plots for each model type to visualize score comparisons between v1 and latest versions.

### 2. Plot Types

#### 1. Bar Comparison Plot (`{model}_bar_comparison.png`)
- **Subplots:** 2×2 grid (4 metrics)
- **Shows:** Mean scores ± std for v1 vs latest
- **Features:**
  - Blue bars (v1), Red bars (Latest)
  - Error bars (standard deviation)
  - Significance stars (*, **, ***, ns)
  - Y-axis starts at 0

#### 2. Scatter Plots (`{model}_scatter_plots.png`)
- **Subplots:** 2×2 grid (4 metrics)
- **Shows:** v1 scores (x-axis) vs latest scores (y-axis)
- **Features:**
  - Each point = one paper
  - Black dashed line = y=x (no change)
  - Red regression line with equation
  - Grid for easy reading

#### 3. Violin Plots (`{model}_violin_plots.png`)
- **Subplots:** 2×2 grid (4 metrics)
- **Shows:** Score distributions for v1 and latest
- **Features:**
  - Violin shape shows distribution density
  - White dot = median
  - Thick bar = IQR
  - Thin bar = mean
  - Blue (v1), Red (Latest)

#### 4. Effect Size & P-value Plot (`{model}_effect_sizes.png`)
- **Layout:** 1×2 side-by-side plots
- **Left:** Cohen's d effect sizes
  - Green = positive, Red = negative
  - Reference lines at ±0.2, ±0.5, ±0.8
- **Right:** -log10(p-value)
  - Green = significant, Gray = not significant
  - Reference lines at p=0.05 and p=0.01

#### 5. Score Change Heatmap (`{model}_heatmap_changes.png`)
- **Shows:** Per-paper score changes (Latest - v1)
- **Rows:** Papers (sorted by average change)
- **Columns:** Metrics
- **Colors:** Green (improvement), White (no change), Red (decline)
- **Limited to:** Top 25 + Bottom 25 papers (if >50 total)

#### 6. Difference Distribution Plot (`{model}_difference_distributions.png`) ⭐ NEW!
- **Subplots:** 2×2 grid (4 metrics)
- **Shows:** Distribution of score differences that the paired t-test analyzes
- **Features:**
  - Histogram with KDE overlay
  - Black dashed line at zero (H₀: no difference)
  - Red line at mean difference
  - Green shaded area for 95% CI
  - Statistics box (n, p-value, Cohen's d)
- **Purpose:** **KEY PLOT** - Directly visualizes what the paired t-test is testing

### 3. Dependencies Added

**`requirements.txt`:**
```
matplotlib>=3.7.0
seaborn>=0.12.0
```

### 4. Integration

Added to `main()` function as **Step 5** (lines 838-849):
- Calls `create_comparison_plots()` after statistical analysis
- Wrapped in try-except for graceful failure
- Prints success/failure messages

### 5. Documentation

Created comprehensive guides:
- **`EVALUATION_PLOTS_GUIDE.md`**: Complete user guide with examples
- **`EVALUATION_GUIDE.md`**: Updated to mention plotting feature
- **`PLOTTING_IMPLEMENTATION_SUMMARY.md`**: This document

## Configuration

### Plot Quality Settings

```python
plt.rcParams['figure.dpi'] = 150        # Screen display
plt.rcParams['savefig.dpi'] = 300       # Saved files (publication-ready)
plt.rcParams['font.size'] = 10          # Base font size
```

### Colors

```python
V1_COLOR = '#3498db'           # Blue
LATEST_COLOR = '#e74c3c'       # Red
POSITIVE_COLOR = '#2ecc71'     # Green
NEGATIVE_COLOR = '#e74c3c'     # Red
NS_COLOR = '#95a5a6'           # Gray
```

### Figure Sizes

- Bar plot: `(12, 10)`
- Scatter plots: `(12, 10)`
- Violin plots: `(12, 10)`
- Effect sizes: `(12, 5)`
- Heatmap: `(8, dynamic height based on # papers)`

## Usage

### Automatic Generation

Simply run the evaluator:
```bash
python evaluate_numerical_scores.py \
  --reviews_dir ./reviews_output \
  --output_dir ./evaluation_results
```

Plots are automatically generated in Step 5.

### Expected Output

```
================================================================================
Step 5: Generating visualization plots...
================================================================================

Generating plots for GenericStructured...
  Saved: evaluation_results/GenericStructured_bar_comparison.png
  Saved: evaluation_results/GenericStructured_scatter_plots.png
  Saved: evaluation_results/GenericStructured_violin_plots.png
  Saved: evaluation_results/GenericStructured_effect_sizes.png
  Saved: evaluation_results/GenericStructured_heatmap_changes.png
  Saved: evaluation_results/GenericStructured_difference_distributions.png

✅ Plots generated successfully!
```

## Error Handling

### Graceful Degradation

If plotting fails:
```python
try:
    create_comparison_plots(df_scores, results_by_model, output_dir)
    print("\n✅ Plots generated successfully!")
except Exception as e:
    print(f"\n⚠️  Warning: Failed to generate some plots: {e}")
    import traceback
    traceback.print_exc()
```

### Skips Empty Data

Each plot checks for valid data:
```python
if len(v1_data) == 0 or len(latest_data) == 0:
    continue  # Skip this metric
```

### Model Type Validation

```python
if not model_results or all('error' in r for r in model_results.values()):
    continue  # Skip this model type
```

## Features

### 1. Publication-Ready Quality
- 300 DPI resolution
- Vector-quality rendering
- Professional color schemes
- Clear labels and legends

### 2. Statistical Indicators
- Significance stars (*, **, ***)
- Reference lines (effect size thresholds)
- Confidence intervals (via error bars)
- Regression lines with equations

### 3. Multiple Perspectives
- **Bar plots**: Quick comparison
- **Scatter plots**: Individual papers
- **Violin plots**: Distributions
- **Effect sizes**: Practical significance
- **Heatmap**: Patterns across papers

### 4. Smart Defaults
- Automatic color coding
- Sensible axis limits
- Grid for readability
- Legend positioning

### 5. Customizable
- Easy to change colors
- Adjustable figure sizes
- Configurable DPI
- Can save as PDF instead of PNG

## Benefits

1. **Instant Insights:** Visualize results without manual plotting
2. **Multiple Views:** See data from different angles
3. **Publication-Ready:** No post-processing needed
4. **Error Detection:** Spot outliers and data issues visually
5. **Presentation-Friendly:** High-quality images for slides/papers

## Technical Details

### Data Aggregation

For plots, scores are aggregated:
```python
df_agg = df_model.groupby(['paper_id', 'version'])[metrics].mean().reset_index()
```

This averages:
- **Multiple reviewers** (CycleReviewer: 4 → 1)
- **Multiple runs** (3 runs → 1 average)

### Pivot for Paired Comparison

```python
pivot = df_agg.pivot_table(
    index='paper_id',
    columns='version',
    values=metric
)
```

Creates paired data for scatter plots and heatmaps.

### Heatmap Sorting

```python
changes_df['avg_change'] = changes_df.mean(axis=1)
changes_df_sorted = changes_df.sort_values('avg_change')
```

Shows most-improved papers at top, most-declined at bottom.

## Maintenance

### Adding New Plot Types

1. Add new plot function in `create_comparison_plots()`
2. Save with consistent naming: `{model_type}_{plot_type}.png`
3. Update documentation in `EVALUATION_PLOTS_GUIDE.md`
4. Update output list in `main()` function

### Modifying Existing Plots

Look for the specific plot section (commented):
```python
# 1. Bar plot: v1 vs latest comparison
# 2. Scatter plots: v1 vs latest
# 3. Violin plots: Distribution comparison
# 4. Effect size and p-value visualization
# 5. Score change heatmap (per paper)
```

## Summary

| Aspect | Details |
|--------|---------|
| **Files Modified** | `evaluate_numerical_scores.py`, `requirements.txt` |
| **Lines Added** | ~355 lines (plotting function) |
| **Dependencies** | matplotlib, seaborn, scipy (for KDE) |
| **Plot Types** | 6 per model type |
| **Quality** | 300 DPI, publication-ready |
| **Error Handling** | Graceful degradation |
| **Documentation** | Complete user guide |
| **Key Feature** | Difference distribution plot (paired t-test visualization) |

The plotting feature significantly enhances the evaluation workflow by providing immediate visual feedback on statistical results, including a direct visualization of what the paired t-test analyzes! 📊✨

