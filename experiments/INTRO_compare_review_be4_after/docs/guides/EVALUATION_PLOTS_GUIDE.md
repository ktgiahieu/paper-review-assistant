# Evaluation Visualization Plots Guide

## Overview

The `evaluate_numerical_scores.py` script now automatically generates **publication-quality visualizations** to help you understand the statistical results at a glance.

## Generated Plots

For each model type (e.g., "GenericStructured", "SEA-E", "CycleReviewer", "Anthropic"), the script generates **5 types of plots**:

### 1. Bar Comparison Plot
**File:** `{model_type}_bar_comparison.png`

Shows mean scores for v1 vs latest with error bars (standard deviation):
- **4 subplots**: Soundness, Presentation, Contribution, Rating
- **Color coding**: Blue (v1), Red (Latest)
- **Significance stars**: 
  - `***` = p < 0.001 (highly significant)
  - `**` = p < 0.01 (very significant)
  - `*` = p < 0.05 (significant)
  - `ns` = not significant

**Use case:** Quick visual comparison of scores between versions

### 2. Scatter Plots
**File:** `{model_type}_scatter_plots.png`

Shows paired v1 vs latest scores with regression lines:
- **4 subplots**: One per metric
- **Black dashed line**: y=x (no change)
- **Red line**: Regression fit with equation
- **Each point**: One paper

**Use case:** Identify papers that improved/worsened, see overall trend

### 3. Violin Plots
**File:** `{model_type}_violin_plots.png`

Shows score distributions for v1 vs latest:
- **4 subplots**: One per metric
- **Violin shape**: Distribution density
- **White dot**: Median
- **Thick bar**: Interquartile range
- **Thin bar**: Mean

**Use case:** Understand distribution shape, detect outliers, compare spreads

### 4. Effect Sizes & Significance
**File:** `{model_type}_effect_sizes.png`

Two side-by-side plots:

**Left plot - Cohen's d (Effect Size):**
- **Green bars**: Positive effect (latest > v1)
- **Red bars**: Negative effect (latest < v1)
- **Gray reference lines**: 
  - 0.2 = small effect
  - 0.5 = medium effect
  - 0.8 = large effect

**Right plot - Statistical Significance:**
- **Green bars**: Significant (p < 0.05)
- **Gray bars**: Not significant
- **Red line**: p = 0.05 threshold
- **Dark red line**: p = 0.01 threshold
- **Y-axis**: -log10(p-value), higher = more significant

**Use case:** Assess practical significance (effect size) vs statistical significance

### 5. Score Change Heatmap
**File:** `{model_type}_heatmap_changes.png`

Shows score changes (Latest - v1) for individual papers:
- **Rows**: Papers (sorted by average change)
- **Columns**: Metrics (Soundness, Presentation, Contribution, Rating)
- **Color**: 
  - Green = improvement
  - White = no change
  - Red = decline
- **Limited to 50 papers**: Top 25 improved + Top 25 declined

**Use case:** Identify which papers improved most, see patterns across metrics

### 6. Difference Distribution Plot (NEW!)
**File:** `{model_type}_difference_distributions.png`

**⭐ KEY PLOT FOR PAIRED T-TEST INTERPRETATION ⭐**

Shows the distribution of score differences (Latest - v1) that the paired t-test analyzes:
- **4 subplots**: One per metric
- **Histogram + KDE**: Shows distribution shape
- **Black dashed line**: Zero (null hypothesis H₀: Δ=0)
- **Red solid line**: Mean difference (what t-test tests)
- **Green shaded area**: 95% confidence interval
- **Statistics box**: Sample size, p-value, Cohen's d

**What to look for:**
- **Mean away from zero?** → Systematic difference
- **CI excludes zero?** → Significant difference
- **Distribution shape:** Normal? Skewed? Bimodal?
- **Spread:** Narrow (consistent changes) or wide (variable changes)?

**Use case:** Understand what the paired t-test is actually testing, see if differences are systematic or random

## Example Output

```
evaluation_results/
  evaluation_scores.csv
  evaluation_summary.csv
  evaluation_detailed_results.json
  GenericStructured_bar_comparison.png
  GenericStructured_scatter_plots.png
  GenericStructured_violin_plots.png
  GenericStructured_effect_sizes.png
  GenericStructured_heatmap_changes.png
  GenericStructured_difference_distributions.png  ← NEW!
```

## Usage

The plots are generated automatically when you run the evaluation:

```bash
python evaluate_numerical_scores.py \
  --reviews_dir ./reviews_vllm \
  --output_dir ./evaluation_results
```

Output:
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

## Customization

### Plot Quality Settings

In `evaluate_numerical_scores.py`, around line 487:

```python
plt.rcParams['figure.dpi'] = 150        # Screen display quality
plt.rcParams['savefig.dpi'] = 300       # Saved file quality (publication-ready)
plt.rcParams['font.size'] = 10          # Base font size
```

**For presentations:**
- Increase `savefig.dpi` to 600
- Increase `font.size` to 12

**For quick previews:**
- Decrease `savefig.dpi` to 150
- Decrease `font.size` to 8

### Color Schemes

Current colors (around lines 532, 638, 673):
- Blue: `#3498db` (v1)
- Red: `#e74c3c` (Latest)
- Green: `#2ecc71` (Positive effect)
- Gray: `#95a5a6` (Not significant)

**To change:** Search and replace color codes in the script.

### Heatmap Size Limit

By default, shows top 25 + bottom 25 papers (line 726):

```python
if len(changes_df_sorted) > 50:
    top_bottom = pd.concat([
        changes_df_sorted.head(25),  # Change to show more/fewer papers
        changes_df_sorted.tail(25)
    ])
```

## Dependencies

Required packages (added to `requirements.txt`):
```
matplotlib>=3.7.0
seaborn>=0.12.0
```

Install with:
```bash
pip install matplotlib seaborn
```

## Interpreting the Plots

### Example Scenario 1: Significant Improvement

**Bar plot:** Latest bars clearly higher than v1, with `**` or `***`  
**Scatter plot:** Most points above diagonal line  
**Violin plot:** Latest distribution shifted right  
**Effect size:** Positive Cohen's d > 0.5  
**Heatmap:** Mostly green (especially for top papers)  
**Difference dist:** Histogram centered right of zero, mean line away from zero, CI excludes zero

**Interpretation:** Paper revisions substantially improved review scores.

### Example Scenario 2: No Significant Change

**Bar plot:** Bars similar height, marked `ns`  
**Scatter plot:** Points cluster around diagonal  
**Violin plot:** Distributions overlap heavily  
**Effect size:** Cohen's d ≈ 0, p-value bar below red line  
**Heatmap:** Mix of red and green, no clear pattern  
**Difference dist:** Histogram centered at zero, CI includes zero, wide spread

**Interpretation:** Revisions didn't significantly affect review scores.

### Example Scenario 3: High Variance

**Bar plot:** Large error bars  
**Scatter plot:** Points widely scattered  
**Violin plot:** Wide, irregular distributions  
**Heatmap:** Mixture of strong green and red  
**Difference dist:** Very wide histogram, large CI, possibly bimodal distribution

**Interpretation:** Some papers improved dramatically, others didn't. Need to investigate individual cases.

### Example Scenario 4: Systematic but Small Effect

**Bar plot:** Small difference, marked `*` (barely significant)  
**Scatter plot:** Points slightly above diagonal  
**Effect size:** Cohen's d ≈ 0.2-0.3 (small)  
**Difference dist:** Narrow histogram slightly right of zero, CI barely excludes zero  

**Interpretation:** Revisions led to consistent but modest improvements across papers.

## Troubleshooting

### "Failed to generate some plots"

**Cause:** Usually missing data or library issues

**Solutions:**
1. Check if matplotlib/seaborn are installed: `pip install matplotlib seaborn`
2. Check if you have valid score data in both v1 and latest
3. Check terminal output for specific error message

### Empty/Blank Subplots

**Cause:** Missing data for that specific metric

**Solution:** Check `evaluation_scores.csv` to see which scores are NULL

### Plots Too Small/Too Large

**Cause:** Default figure sizes may not fit your data

**Solution:** Adjust `figsize` parameters in the code:
- Line 514: `figsize=(12, 10)` for bar plots
- Line 564: `figsize=(12, 10)` for scatter plots
- Line 616: `figsize=(12, 10)` for violin plots
- Line 666: `figsize=(12, 5)` for effect sizes
- Line 719: `figsize=(8, max(6, len(changes_df) * 0.15))` for heatmap

## Publication Use

These plots are publication-ready with:
- ✅ 300 DPI resolution
- ✅ Vector-quality rendering
- ✅ Clear labels and legends
- ✅ Statistical significance indicators
- ✅ Professional color schemes

**Recommended formats:**
- PNG: Good for presentations, emails
- PDF: Better for papers (edit in Inkscape/Illustrator)

**To save as PDF:** Change line 559, 611, etc.:
```python
plt.savefig(plot_path.with_suffix('.pdf'), bbox_inches='tight', format='pdf')
```

## Summary

The plotting feature provides:
- 📊 **6 plot types** per model (including key paired t-test visualization!)
- 🎨 **Publication-quality** visuals
- 📈 **Multiple perspectives** on the same data
- ✅ **Automatic generation** (no manual work)
- 🔧 **Customizable** (colors, sizes, formats)
- 🎯 **Direct t-test interpretation** (difference distribution plot)

Use these plots to:
1. **Quickly understand** statistical results
2. **Identify patterns** across papers
3. **Present findings** in papers/presentations
4. **Debug issues** (outliers, data quality)
5. **Visualize paired t-test assumptions** (normality, mean difference)

Happy analyzing! 📊✨

