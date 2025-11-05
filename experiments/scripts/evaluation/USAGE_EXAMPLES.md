# Usage Examples for Evaluation Scripts

## Quick Start: v1 vs Latest Comparison

### Example: Compare v1 vs latest scores from Gemini 2.5 Pro reviews

```bash
cd experiments/scripts/evaluation

# Basic usage
python3 evaluate_numerical_scores.py \
    --reviews_dir ../../sampled_data/reviews_gemini_2-5_pro/ICLR2024 \
    --output_dir ../../sampled_data/reviews_gemini_2-5_pro/evaluation_results \
    --num_runs 1
```

**Folder structure expected:**
```
reviews_gemini_2-5_pro/ICLR2024/
├── v1/
│   ├── 2Rwq6c3tvr/
│   │   └── review_run0.json
│   ├── ViNe1fjGME/
│   │   └── review_run0.json
│   └── ...
└── latest/
    ├── 2Rwq6c3tvr/
    │   └── review_run0.json
    ├── ViNe1fjGME/
    │   └── review_run0.json
    └── ...
```

**What this does:**
1. Scans `v1/` and `latest/` folders
2. Extracts scores from all `review_run*.json` files
3. Performs paired t-tests comparing v1 vs latest
4. Generates statistical analysis and plots

**Output files:**
- `evaluation_scores.csv` - All extracted scores
- `evaluation_summary.csv` - Summary statistics table
- `evaluation_detailed_results.json` - Full statistical results
- `default_bar_comparison.png` - Bar plots comparing v1 vs latest
- `default_scatter_plots.png` - Scatter plots with regression lines
- `default_violin_plots.png` - Distribution comparisons
- `default_effect_sizes.png` - Effect size visualization
- `default_heatmap_changes.png` - Per-paper score changes
- `default_difference_distributions.png` - Distribution of differences (key for paired t-test)

---

## Example Output

```
SOUNDNESS:
  Pairs: 25 (from 25 papers)
  v1:     6.680 ± 1.234
  Latest: 7.120 ± 1.456
  Diff:   0.440 ± 0.823
  t(24) = 2.678, p = 0.0132
  Cohen's d = 0.536
  95% CI: [0.095, 0.785]
  Interpretation: Latest version scored 0.440 points higher than v1 
                  (significant, p=0.0132). Effect size: medium (Cohen's d=0.536).
```

---

## Multiple Runs

If you have multiple runs per paper:

```bash
python3 evaluate_numerical_scores.py \
    --reviews_dir ../../sampled_data/reviews_gemini_2-5_pro/ICLR2024 \
    --output_dir ../../sampled_data/reviews_gemini_2-5_pro/evaluation_results \
    --num_runs 3
```

This will:
- Only include papers with complete runs (3 v1 + 3 latest)
- Average scores across runs for each paper
- Perform paired t-tests on the averaged scores

---

## Custom Output Location

```bash
python3 evaluate_numerical_scores.py \
    --reviews_dir ../../sampled_data/reviews_gemini_2-5_pro/ICLR2024 \
    --output_dir ./my_results \
    --output_prefix gemini_2_5_pro_analysis \
    --num_runs 1
```

---

## Compare AI vs Human Scores

```bash
python3 calculate_mse_mae.py \
    --csv_file ../../data/ICLR2024_pairs/filtered_pairs_with_human_scores.csv \
    --reviews_dir ../../sampled_data/reviews_gemini_2-5_pro/ICLR2024 \
    --output_dir ../../sampled_data/reviews_gemini_2-5_pro/ai_vs_human \
    --version latest
```

**For both versions:**
```bash
python3 calculate_mse_mae.py \
    --csv_file ../../data/ICLR2024_pairs/filtered_pairs_with_human_scores.csv \
    --reviews_dir ../../sampled_data/reviews_gemini_2-5_pro/ICLR2024 \
    --output_dir ../../sampled_data/reviews_gemini_2-5_pro/ai_vs_human \
    --version both
```

---

## Troubleshooting

### "No scores collected"
- Check that review files exist in the expected structure
- Verify review files have `"success": true`
- Check that scores are present in the JSON (soundness, presentation, contribution, rating)

### "Insufficient paired data"
- Ensure you have both v1 and latest versions for each paper
- Check that folder names match exactly (case-sensitive)

### "Missing v1 or latest version"
- The script needs both v1 and latest folders to compare
- Check that papers exist in both folders
