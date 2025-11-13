# Evaluation Scripts

This directory contains scripts for evaluating and analyzing paper review results.

## Scripts

### `run_all_evaluations.py` ⭐ **Recommended**
**Automatically runs all evaluation scripts.** This is the easiest way to run all evaluations at once.

### `evaluate_numerical_scores.py`
Performs paired t-tests comparing scores between different paper versions (e.g., v1 vs latest).

### `evaluate_good_bad_plant_effect.py`
Analyzes the effect of good/bad manipulations (triplet analysis: baseline, good, bad).
Supports any pattern (author/affiliation, abstract, etc.) via the `--pattern` argument.

### `calculate_mse_mae.py`
Calculates MSE and MAE comparing AI-generated vs human review scores.

---

## Quick Start: Run All Evaluations

The easiest way to run all evaluations is using `run_all_evaluations.py`:

```bash
cd experiments/scripts/evaluation

# Run all evaluations (skips human comparison if no CSV provided)
python run_all_evaluations.py \
    --reviews_dir ../../sampled_data/reviews_gemini_2-0_flash_lite/ICLR2024

# With human scores comparison
python run_all_evaluations.py \
    --reviews_dir ../../sampled_data/reviews_gemini_2-0_flash_lite/ICLR2024 \
    --human_scores_csv ../../sampled_data/ICLR2024/filtered_pairs_with_human_scores.csv
```

This will automatically:
1. Run v1 vs latest evaluation → saves to `evaluation_results/v1_latest/`
2. Run author/affiliation effect evaluation → saves to `evaluation_results/author_affiliation/`
3. Run abstract manipulation effect evaluation → saves to `evaluation_results/abstract/` (if available)
4. Run AI vs human comparison (if CSV provided) → saves to `evaluation_results/v1_human/`

See more examples below.

---

## Detailed Usage Examples

### Example 0: Run All Evaluations (Recommended)

**Basic usage:**
```bash
python run_all_evaluations.py \
    --reviews_dir ../../sampled_data/reviews_gemini_2-0_flash_lite/ICLR2024
```

**With human scores:**
```bash
python run_all_evaluations.py \
    --reviews_dir ../../sampled_data/reviews_gemini_2-0_flash_lite/ICLR2024 \
    --human_scores_csv ../../sampled_data/ICLR2024/filtered_pairs_with_human_scores.csv \
    --verbose
```

**Custom output directory:**
```bash
python run_all_evaluations.py \
    --reviews_dir ../../sampled_data/reviews_gemini_2-0_flash_lite/ICLR2024 \
    --output_dir ../../sampled_data/custom_evaluation_results
```

**Skip specific evaluations:**
```bash
python run_all_evaluations.py \
    --reviews_dir ../../sampled_data/reviews_gemini_2-0_flash_lite/ICLR2024 \
    --skip-author-affiliation  # Skip author/affiliation effect analysis
```

**Output structure:**
```
evaluation_results/
├── v1_latest/
│   ├── ICLR2024_scores.csv
│   ├── ICLR2024_summary.csv
│   └── ICLR2024_detailed_results.json
├── author_affiliation/
│   ├── score_differences.csv
│   ├── summary_statistics.csv
│   └── statistical_results.json
└── v1_human/
    ├── ai_vs_human_results.csv
    └── ai_vs_human_detailed.csv
```

---

### Example 1: Compare v1 vs latest scores (Individual Script)

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

**Command (author/affiliation - default pattern):**
```bash
python3 evaluate_good_bad_plant_effect.py \
    --reviews_dir ../../sampled_data/reviews_gemini_2-5_pro/ICLR2024 \
    --output_dir ../../sampled_data/reviews_gemini_2-5_pro/evaluation_results/author_affiliation \
    --pattern authors_affiliation \
    --baseline latest
```

**Command (abstract manipulation):**
```bash
python3 evaluate_good_bad_plant_effect.py \
    --reviews_dir ../../sampled_data/reviews_gemini_2-5_pro/ICLR2024 \
    --output_dir ../../sampled_data/reviews_gemini_2-5_pro/evaluation_results/abstract \
    --pattern abstract \
    --baseline latest
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

