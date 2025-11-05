# Evaluation Scripts

This directory contains scripts for evaluating and analyzing paper review results.

## Scripts

### `evaluate_numerical_scores.py`
Performs paired t-tests comparing scores between different paper versions (e.g., v1 vs latest).

### `evaluate_author_affiliation_effect.py`
Analyzes the effect of author/affiliation modifications (triplet analysis: original, good, bad).

### `calculate_mse_mae.py`
Calculates MSE and MAE comparing AI-generated vs human review scores.

---

## Usage Examples

### Example 1: Compare v1 vs latest scores

**Folder structure:**
```
reviews_gemini_2-5_pro/ICLR2024/
├── v1/
│   ├── 2Rwq6c3tvr/
│   │   └── review_run0.json
│   └── ...
└── latest/
    ├── 2Rwq6c3tvr/
    │   └── review_run0.json
    └── ...
```

**Command:**
```bash
cd experiments/scripts/evaluation
python3 evaluate_numerical_scores.py \
    --reviews_dir ../../sampled_data/reviews_gemini_2-5_pro/ICLR2024 \
    --output_dir ../../sampled_data/reviews_gemini_2-5_pro/evaluation_results \
    --num_runs 1
```

**What it does:**
- Automatically detects the new folder structure (`v1/` and `latest/` folders)
- Extracts scores from all review files
- Performs paired t-tests comparing v1 vs latest for each metric
- Generates statistical analysis and visualization plots

**Output:**
- `evaluation_scores.csv` - Raw scores
- `evaluation_summary.csv` - Summary statistics
- `evaluation_detailed_results.json` - Full statistical results
- Plots: bar comparisons, scatter plots, violin plots, effect sizes, heatmaps

---

### Example 2: Custom output prefix

```bash
python3 evaluate_numerical_scores.py \
    --reviews_dir ../../sampled_data/reviews_gemini_2-5_pro/ICLR2024 \
    --output_dir ../../sampled_data/reviews_gemini_2-5_pro/evaluation_results \
    --output_prefix gemini_2_5_pro_v1_vs_latest \
    --num_runs 1
```

---

### Example 3: Compare AI vs Human scores

**Requirements:**
- CSV file with human scores (from `fetch_human_scores.py`)
- AI review files in the new folder structure

**Command:**
```bash
python3 calculate_mse_mae.py \
    --csv_file ../../data/ICLR2024_pairs/filtered_pairs_with_human_scores.csv \
    --reviews_dir ../../sampled_data/reviews_gemini_2-5_pro/ICLR2024 \
    --output_dir ../../sampled_data/reviews_gemini_2-5_pro/ai_vs_human \
    --version latest
```

**For both v1 and latest:**
```bash
python3 calculate_mse_mae.py \
    --csv_file ../../data/ICLR2024_pairs/filtered_pairs_with_human_scores.csv \
    --reviews_dir ../../sampled_data/reviews_gemini_2-5_pro/ICLR2024 \
    --output_dir ../../sampled_data/reviews_gemini_2-5_pro/ai_vs_human \
    --version both
```

---

### Example 4: Author/Affiliation Effect Analysis

**Folder structure:**
```
reviews_gemini_2-5_pro/ICLR2024/
├── latest/
│   ├── 2Rwq6c3tvr/
│   │   └── review_run0.json
│   └── ...
├── authors_affiliation_good/
│   ├── 2Rwq6c3tvr/
│   │   └── review_run0.json
│   └── ...
└── authors_affiliation_bad/
    ├── 2Rwq6c3tvr/
    │   └── review_run0.json
    └── ...
```

**Command:**
```bash
python3 evaluate_author_affiliation_effect.py \
    --reviews_dir ../../sampled_data/reviews_gemini_2-5_pro/ICLR2024 \
    --output_dir ../../sampled_data/reviews_gemini_2-5_pro/evaluation_results \
    --folders latest authors_affiliation_good authors_affiliation_bad
```

---

## Folder Structure Support

Both scripts automatically detect and support:

### New Structure (Recommended)
```
reviews_dir/
├── v1/
│   └── paper_id/
│       └── review_run0.json
└── latest/
    └── paper_id/
        └── review_run0.json
```

### Old Structure (Still Supported)
```
reviews_dir/
└── paper_id/
    ├── v1_review_run0.json
    └── latest_review_run0.json
```

The scripts automatically detect which structure you're using and process accordingly.

---

## Review JSON Format

The scripts expect review JSON files with the following structure:

```json
{
  "paper_id": "2Rwq6c3tvr",
  "folder": "v1",  // or "latest", "authors_affiliation_good", etc.
  "run_id": 0,
  "success": true,
  "soundness": 7,
  "presentation": 6,
  "contribution": 8,
  "rating": 7,
  // ... other fields
}
```

The scripts extract scores from:
- Direct fields: `soundness`, `presentation`, `contribution`, `rating`
- Mapped fields: `technical_quality_score` → `soundness`, `clarity_score` → `presentation`, etc.

