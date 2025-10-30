# Quick Start Guide

Welcome! This guide will help you get started with the reorganized project.

## 🗂️ New Folder Structure at a Glance

```
📁 Project Root
├── 📂 scripts/         → All Python scripts
│   ├── review/        → Generate reviews
│   ├── evaluation/    → Analyze results
│   └── utils/         → Prepare data
├── 📂 docs/           → All documentation
│   ├── guides/        → User guides
│   ├── implementation/→ Technical docs
│   └── fixes/         → Bug fix history
├── 📂 tests/          → Test scripts + outputs
└── 📂 data/           → Paper data
```

## 🚀 Common Tasks

### 1. Generate Reviews

**With vLLM:**
```bash
python scripts/review/review_paper_pairs_vllm.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ./reviews_llama3/ \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "Meta-Llama-3.1-70B-Instruct" \
  --num_runs 3 \
  --format default
```

**With Anthropic:**
```bash
python scripts/review/review_paper_pairs.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ./reviews_claude/ \
  --num_runs 3
```

**📖 Detailed guide:** `docs/guides/VLLM_GUIDE.md`

### 2. Evaluate Results

**Statistical analysis:**
```bash
python scripts/evaluation/evaluate_numerical_scores.py \
  --reviews_dir ./reviews_llama3/ \
  --output_dir ./evaluation_results/
```

**📖 Detailed guide:** `docs/guides/EVALUATION_GUIDE.md`

### 3. Compare with Human Reviews

**Fetch human scores:**
```bash
# Test API first
python tests/test_fetch_human_scores.py

# Fetch scores
python scripts/utils/fetch_human_scores.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv
```

**Calculate MSE/MAE:**
```bash
python scripts/evaluation/calculate_mse_mae.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs_with_human_scores.csv \
  --reviews_dir ./reviews_llama3/
```

**📖 Detailed guide:** `docs/guides/AI_VS_HUMAN_GUIDE.md`

### 4. Evaluate Flaw Detection

**Setup evaluator:**
```bash
python tests/test_flaw_detection_setup.py \
  --evaluator_endpoint "http://localhost:8000" \
  --evaluator_model "Qwen3-30B"
```

**Run evaluation:**
```bash
python scripts/evaluation/evaluate_flaw_detection.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --reviews_dir ./reviews_llama3/ \
  --evaluator_endpoint "http://localhost:8000"
```

**Analyze results:**
```bash
python scripts/evaluation/analyze_flaw_detection.py \
  --results_file ./flaw_detection_results/flaw_detection_detailed.json
```

**📖 Detailed guide:** `docs/guides/FLAW_DETECTION_GUIDE.md`

## 📚 Where to Find Things

### I want to...

| Task | Go to |
|------|-------|
| **Learn how to use a script** | `docs/guides/` |
| **Understand how something works** | `docs/implementation/` |
| **Fix an error** | `docs/fixes/` + `docs/guides/RETRY_GUIDE.md` |
| **Run a test** | `tests/` + `tests/README.md` |
| **See all available scripts** | `scripts/README.md` |
| **Get an overview** | Main `README.md` |

### Documentation by Role

**👤 Researcher/User:**
- Start: Main `README.md`
- Guides: `docs/guides/`
- Examples: Look for "Usage" sections in guides

**👨‍💻 Developer:**
- Architecture: `docs/implementation/`
- Bug fixes: `docs/fixes/`
- Tests: `tests/`

**🐛 Troubleshooter:**
- Retry guide: `docs/guides/RETRY_GUIDE.md`
- Known fixes: `docs/fixes/`
- Test scripts: `tests/test_*.py`

## 🎯 Complete Workflow Example

Here's a full end-to-end workflow:

```bash
# 1. Prepare data (if needed)
python scripts/utils/get_paper_pairs.py

# 2. Generate reviews (3 runs for variance)
python scripts/review/review_paper_pairs_vllm.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ./reviews_final/ \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "Meta-Llama-3.1-70B-Instruct" \
  --num_runs 3 \
  --format default

# 3. Evaluate numerical scores (v1 vs latest)
python scripts/evaluation/evaluate_numerical_scores.py \
  --reviews_dir ./reviews_final/ \
  --output_dir ./eval_numerical/

# 4. Fetch human scores
python scripts/utils/fetch_human_scores.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv

# 5. Compare AI vs Human
python scripts/evaluation/calculate_mse_mae.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs_with_human_scores.csv \
  --reviews_dir ./reviews_final/ \
  --output_dir ./eval_ai_vs_human/

# 6. Evaluate flaw detection
python scripts/evaluation/evaluate_flaw_detection.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --reviews_dir ./reviews_final/ \
  --evaluator_endpoint "http://localhost:8000" \
  --output_dir ./eval_flaw_detection/

# 7. Analyze flaw detection
python scripts/evaluation/analyze_flaw_detection.py \
  --results_file ./eval_flaw_detection/flaw_detection_detailed.json \
  --output_dir ./eval_flaw_analysis/
```

## ⚙️ Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Environment Variables
For Anthropic:
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

For vLLM:
- Start vLLM server on desired port
- Pass endpoint to scripts: `--vllm_endpoint "http://localhost:8000"`

## 💡 Tips

### Running from Different Directories

**From project root:**
```bash
python scripts/review/review_paper_pairs_vllm.py ...
```

**From scripts directory:**
```bash
cd scripts/review
python review_paper_pairs_vllm.py \
  --csv_file ../../data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ../../reviews/
```

### Testing Before Full Run

Use `--limit` flag to test on small sample:
```bash
python scripts/review/review_paper_pairs_vllm.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --limit 5 \
  ...
```

Use `--verbose` flag for detailed logging:
```bash
python scripts/review/review_paper_pairs_vllm.py \
  --verbose \
  ...
```

## 📞 Getting Help

1. **Check README files:**
   - Main `README.md` - Overview
   - `scripts/README.md` - Script reference
   - `docs/README.md` - Documentation index
   - `tests/README.md` - Test guide

2. **Read relevant guide:**
   - `docs/guides/VLLM_GUIDE.md` - vLLM usage
   - `docs/guides/EVALUATION_GUIDE.md` - Evaluation
   - `docs/guides/RETRY_GUIDE.md` - Troubleshooting
   - `docs/guides/FLAW_DETECTION_GUIDE.md` - Flaw detection

3. **Check implementation docs:**
   - `docs/implementation/` - Technical details
   - `docs/fixes/` - Known issues and solutions

## 🎉 You're Ready!

The project is now organized and ready to use. Happy researching! 🚀

**Key takeaway:** Everything is now in logical folders. Just remember:
- **Scripts** → `scripts/`
- **Docs** → `docs/`
- **Tests** → `tests/`
- **Data** → `data/`

