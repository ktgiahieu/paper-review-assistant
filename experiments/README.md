# Experiments Directory

This directory contains all experiments for paper review analysis, organized into a shared structure.

## 📁 Directory Structure

```
experiments/
├── data/                    # Shared paper data (ICLR2024, etc.)
│   ├── ICLR2024_pairs/     # Paper pairs (v1 vs latest)
│   ├── ICLR2024_latest/    # Latest versions
│   └── ICLR2024_v1/        # v1 versions
│
├── sampled_data/            # Sampled subsets for testing
│   ├── ICLR2024/           # Sampled papers
│   └── reviews/            # Reviews of sampled papers
│
├── scripts/                 # Shared scripts (used by all experiments)
│   ├── review/             # Review generation scripts
│   ├── evaluation/         # Analysis and evaluation scripts
│   └── utils/              # Utility scripts
│
├── PLACEBO_plant_author_affiliation/  # Author/affiliation manipulation experiment
│   ├── scripts/            # Scripts specific to this experiment
│   │   ├── plant_author_affiliation.py
│   │   └── sample_papers.py
│   └── docs/
│
├── PLACEBO_plant_quality_errors/      # Quality/error manipulation experiment
│   ├── scripts/            # Scripts specific to this experiment
│   │   ├── plant_errors_and_placebo.py
│   │   └── sample_papers.py
│   └── docs/
│
└── PLACEBO_plant_v1_vs_latest/        # v1 vs latest comparison experiment
    ├── scripts/            # Scripts specific to this experiment
    ├── docs/
    └── output/
```

## 🎯 Experiment Folders

Each experiment folder contains scripts specific to creating/manipulating paper versions:

1. **PLACEBO_plant_author_affiliation/** - Creates versions with different author/affiliation combinations
2. **PLACEBO_plant_quality_errors/** - Creates versions with quality/error modifications
3. **PLACEBO_plant_v1_vs_latest/** - Compares original v1 vs latest versions

## 📝 Shared Scripts

All review generation and evaluation scripts are in the shared `scripts/` folder:

### Review Generation (`scripts/review/`)
- `review_papers_gemini.py` - Review papers using Gemini API (multiple keys, rate limiting)
- `review_paper_pairs.py` - Review paper pairs using Anthropic Claude API
- `review_paper_pairs_vllm.py` - Review papers using vLLM (multiple formats)
- `retry_failed_reviews.py` - Retry failed review generations

### Evaluation (`scripts/evaluation/`)
- `evaluate_good_bad_plant_effect.py` - Analyze good/bad manipulation effects (triplet analysis, supports any pattern)
- `evaluate_numerical_scores.py` - Extract and analyze numerical scores (paired t-tests)
- `calculate_mse_mae.py` - Calculate MSE/MAE comparing AI vs human scores
- `analyze_flaw_detection.py` - Analyze flaw detection with paired t-tests
- `evaluate_flaw_detection.py` - Evaluate whether AI reviews detect consensus flaws

### Utilities (`scripts/utils/`)
- `fetch_human_scores.py` - Fetch human review scores from OpenReview API
- `get_paper_pairs.py` - Extract paper pairs (v1 vs latest) from ICLR data
- `prepare_manual_gemini_prompts.py` - Prepare prompts for manual Gemini review
- `process_manual_gemini_outputs.py` - Process manual Gemini review outputs

## 🚀 Quick Start

### 1. Create Planted Versions

Choose an experiment and create modified versions:

```bash
# Author/affiliation experiment
cd PLACEBO_plant_author_affiliation/scripts
python3 plant_author_affiliation.py \
    --base_dir ../../data/ICLR2024 \
    --output_good_dir ../../data/ICLR2024/authors_affiliation_good \
    --output_bad_dir ../../data/ICLR2024/authors_affiliation_bad

# Quality/error experiment
cd PLACEBO_plant_quality_errors/scripts
python3 plant_errors_and_placebo.py ...
```

### 2. Sample Papers (Optional)

```bash
# From any experiment folder
python3 sample_papers.py \
    --base_dir ../../data/ICLR2024 \
    --n_samples 25 \
    --output_dir ../../sampled_data/ICLR2024
```

### 3. Generate Reviews

```bash
# Using Gemini (from experiments root)
cd scripts/review
python3 review_papers_gemini.py \
    --base_dir ../../sampled_data/ICLR2024 \
    --folders latest authors_affiliation_good authors_affiliation_bad \
    --output_dir ../../sampled_data/reviews/ICLR2024

# Using Anthropic Claude
python3 review_paper_pairs.py \
    --csv_file ../../data/ICLR2024_pairs/filtered_pairs.csv \
    --output_dir ../../sampled_data/reviews/ICLR2024
```

### 4. Evaluate Results

```bash
# Author/affiliation effect (triplet analysis)
cd scripts/evaluation
python3 evaluate_good_bad_plant_effect.py \
    --reviews_dir ../../sampled_data/reviews/ICLR2024

# Numerical scores comparison (v1 vs latest)
python3 evaluate_numerical_scores.py \
    --reviews_dir ../../sampled_data/reviews/ICLR2024 \
    --output_dir ../../sampled_data/evaluation_results
```

## 📊 Data Locations

All scripts use relative paths from the `experiments/` directory:

- **Original data**: `data/ICLR2024/` or `data/ICLR2024_pairs/`
- **Sampled data**: `sampled_data/ICLR2024/`
- **Reviews**: `sampled_data/reviews/ICLR2024/` or `data/ICLR2024/reviews/`
- **Results**: `sampled_data/evaluation_results/` or `data/ICLR2024/evaluation_results/`

## 🔄 Migration Notes

Previously, each experiment had its own review/evaluation scripts. These have been moved to the shared `scripts/` folder. Update any scripts that reference old paths:

- Old: `PLACEBO_planted_unauthorized_factor/scripts/review_papers_gemini.py`
- New: `scripts/review/review_papers_gemini.py`

- Old: `PLACEBO_planted_unauthorized_factor/scripts/evaluate_author_affiliation_effect.py`
- New: `scripts/evaluation/evaluate_good_bad_plant_effect.py`

