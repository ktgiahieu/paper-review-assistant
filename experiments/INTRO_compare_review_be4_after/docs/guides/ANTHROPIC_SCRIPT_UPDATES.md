# Anthropic Script Updates: CriticalNeurIPS Format + Gemini API Support

## Overview

The `review_paper_pairs.py` script (Anthropic version) has been enhanced with:

1. **CriticalNeurIPS Format** - Deep, multi-faceted critical reviews
2. **Gemini API Support** - Use Google's Gemini models alongside Anthropic

## What's New

### 1. Critical NeurIPS Review Format

The same sophisticated review format available in the vLLM script is now available for Anthropic and Gemini APIs.

**Key Features:**
- Multi-perspective analysis (Conceptual + Methodological)
- Evidence-based feedback with citations
- Scholarly rigor from first principles
- Markdown-formatted combined strengths/weaknesses
- Dedicated ethics/societal impact field

### 2. Gemini API Integration

You can now use Google's Gemini models for paper reviews.

**Supported Models:**
- `gemini-pro`
- `gemini-1.5-pro`
- `gemini-1.5-flash`
- Any Gemini model available via Google AI Studio

---

## Installation

### Add Gemini Support

```bash
# Install Google Generative AI library
pip install google-generativeai

# Or update all dependencies
pip install -r requirements.txt
```

### Set Up API Keys

Add to your `.env` file:

```bash
# Anthropic (existing)
ANTHROPIC_API_KEY=your_anthropic_key_here

# Gemini (new)
GEMINI_API_KEY=your_gemini_key_here
```

**Get Gemini API Key:**
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Click "Get API Key"
3. Create a new key or use existing
4. Add to `.env` file

---

## Usage

### Default Format with Anthropic (unchanged)

```bash
python scripts/review/review_paper_pairs.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ./reviews_anthropic/ \
  --model_name "claude-haiku-4-5-20251001" \
  --num_runs 3
```

### CriticalNeurIPS Format with Anthropic

```bash
python scripts/review/review_paper_pairs.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ./reviews_anthropic_critical/ \
  --model_name "claude-sonnet-4-20250514" \
  --format CriticalNeurIPS \
  --num_runs 3
```

### Default Format with Gemini

```bash
python scripts/review/review_paper_pairs.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ./reviews_gemini/ \
  --api gemini \
  --model_name "gemini-1.5-pro" \
  --num_runs 3
```

### CriticalNeurIPS Format with Gemini

```bash
python scripts/review/review_paper_pairs.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ./reviews_gemini_critical/ \
  --api gemini \
  --model_name "gemini-1.5-pro" \
  --format CriticalNeurIPS \
  --num_runs 3
```

---

## New Arguments

### `--api` (NEW)

Choose which API to use.

- **Options**: `anthropic`, `gemini`
- **Default**: `anthropic`
- **Example**: `--api gemini`

### `--format` (NEW)

Choose review format.

- **Options**: `default`, `CriticalNeurIPS`
- **Default**: `default`
- **Example**: `--format CriticalNeurIPS`

---

## Format Comparison

| Feature | Default | CriticalNeurIPS |
|---------|---------|-----------------|
| **Output** | JSON with lists | JSON with Markdown |
| **Strengths/Weaknesses** | Separate lists | Combined assessment |
| **Critique Style** | Balanced | Deep & critical |
| **Citations** | Not emphasized | Encouraged |
| **Ethics** | In comments | Dedicated field |
| **Scores** | 1-10 scale | 1-4 (S,P,C), 1-10 (overall) |
| **Best For** | Quick reviews | Research quality |

### Default Format Fields

```json
{
  "summary": "...",
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "clarity_score": 8,
  "novelty_score": 7,
  "technical_quality_score": 8,
  "experimental_rigor_score": 7,
  "overall_score": 8,
  "confidence": 4,
  "recommendation": "Accept",
  "detailed_comments": "..."
}
```

### CriticalNeurIPS Format Fields

```json
{
  "summary": "...",
  "strengths_and_weaknesses": "## Strengths\n\n...\n\n## Weaknesses\n\n...",
  "questions": "1. ...\n2. ...",
  "limitations_and_societal_impact": "...",
  "soundness": 3,
  "presentation": 3,
  "contribution": 4,
  "overall_score": 7,
  "confidence": 4
}
```

---

## API Comparison

| Feature | Anthropic | Gemini |
|---------|-----------|--------|
| **Context Window** | 200K tokens | 2M tokens (1.5-pro) |
| **Speed** | Fast | Variable |
| **Cost** | Moderate | Lower (Flash) / Higher (Pro) |
| **JSON Reliability** | Excellent | Good |
| **System Prompts** | Supported | Combined with user prompt |
| **Best Models** | Claude Sonnet 4 | Gemini 1.5 Pro |

### Recommended Model Combinations

**For Quality:**
```bash
# Best quality, higher cost
--api anthropic --model_name "claude-sonnet-4-20250514" --format CriticalNeurIPS

# Good quality, lower cost  
--api gemini --model_name "gemini-1.5-pro" --format CriticalNeurIPS
```

**For Speed:**
```bash
# Fastest Anthropic
--api anthropic --model_name "claude-haiku-4-5-20251001" --format default

# Fastest Gemini
--api gemini --model_name "gemini-1.5-flash" --format default
```

**For Long Papers:**
```bash
# Gemini Pro has 2M context window (much larger than Anthropic's 200K)
--api gemini --model_name "gemini-1.5-pro" --format CriticalNeurIPS
```

---

## Complete Examples

### Scenario 1: High-Quality Critical Reviews

```bash
# Use Claude Sonnet with CriticalNeurIPS format
python scripts/review/review_paper_pairs.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ./reviews_quality/ \
  --api anthropic \
  --model_name "claude-sonnet-4-20250514" \
  --format CriticalNeurIPS \
  --num_runs 5 \
  --max_workers 2
```

### Scenario 2: Cost-Effective Reviews

```bash
# Use Gemini Flash with default format
python scripts/review/review_paper_pairs.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ./reviews_budget/ \
  --api gemini \
  --model_name "gemini-1.5-flash" \
  --format default \
  --num_runs 3 \
  --max_workers 5
```

### Scenario 3: Comparing APIs

```bash
# Anthropic version
python scripts/review/review_paper_pairs.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ./reviews_anthropic_compare/ \
  --api anthropic \
  --model_name "claude-sonnet-4-20250514" \
  --format CriticalNeurIPS \
  --num_runs 3

# Gemini version (same format)
python scripts/review/review_paper_pairs.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ./reviews_gemini_compare/ \
  --api gemini \
  --model_name "gemini-1.5-pro" \
  --format CriticalNeurIPS \
  --num_runs 3

# Then evaluate both
python scripts/evaluation/evaluate_numerical_scores.py \
  --reviews_dir ./reviews_anthropic_compare/

python scripts/evaluation/evaluate_numerical_scores.py \
  --reviews_dir ./reviews_gemini_compare/
```

### Scenario 4: Testing with Small Sample

```bash
# Test on 5 papers first
python scripts/review/review_paper_pairs.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ./reviews_test/ \
  --api gemini \
  --model_name "gemini-1.5-flash" \
  --format CriticalNeurIPS \
  --limit 5 \
  --verbose
```

---

## Output Structure

Reviews are saved with `model_type` label indicating both API and format:

```
reviews_output/
├── paper_id_1/
│   ├── v1_review_run0.json       # model_type: "anthropic_CriticalNeurIPS"
│   ├── v1_review_run1.json
│   ├── latest_review_run0.json
│   └── latest_review_run1.json
```

### Model Type Labels

- `anthropic_default` - Anthropic with default format
- `anthropic_CriticalNeurIPS` - Anthropic with critical format
- `gemini_default` - Gemini with default format
- `gemini_CriticalNeurIPS` - Gemini with critical format

---

## Evaluation Compatibility

All format variations work with existing evaluation scripts:

### Numerical Scores

```bash
python scripts/evaluation/evaluate_numerical_scores.py \
  --reviews_dir ./reviews_anthropic_critical/
```

**Extracted Metrics:**
- Default format: Maps `technical_quality_score` → `soundness`, etc.
- CriticalNeurIPS: Direct `soundness`, `presentation`, `contribution`, `rating`

### Flaw Detection

```bash
python scripts/evaluation/evaluate_flaw_detection.py \
  --reviews_dir ./reviews_gemini_critical/
```

**Extracted Weaknesses:**
- Default format: From `weaknesses` list
- CriticalNeurIPS: From `strengths_and_weaknesses` field

### AI vs Human

```bash
python scripts/evaluation/calculate_mse_mae.py \
  --reviews_dir ./reviews_anthropic_critical/
```

Works with both formats (score mappings handled automatically).

---

## Troubleshooting

### Issue: "GEMINI_API_KEY not found"

**Solution**: Add to `.env` file:
```bash
GEMINI_API_KEY=your_key_here
```

### Issue: "google-generativeai not installed"

**Solution**:
```bash
pip install google-generativeai
```

### Issue: Gemini JSON parsing errors

**Cause**: Gemini sometimes includes markdown formatting

**Solution**: The script already handles this with JSON sanitization. If issues persist, try:
```bash
--format CriticalNeurIPS  # More explicit JSON schema
```

### Issue: Rate limiting with Gemini

**Solution**: Reduce max_workers:
```bash
--max_workers 2  # Instead of default 3
```

### Issue: Context length errors with Anthropic

**Cause**: Long papers exceed 200K token context

**Solution**: Use Gemini for long papers (2M context):
```bash
--api gemini --model_name "gemini-1.5-pro"
```

---

## Best Practices

### 1. Start Small

Always test on a few papers first:
```bash
--limit 5 --verbose
```

### 2. Choose Right Model for Task

| Task | Recommended |
|------|-------------|
| Highest quality | Anthropic Sonnet + CriticalNeurIPS |
| Long papers | Gemini 1.5 Pro (2M context) |
| Speed | Gemini Flash + default |
| Budget | Gemini Flash + default |

### 3. Use Multiple Runs for Critical Analysis

```bash
# For CriticalNeurIPS format, use more runs
--format CriticalNeurIPS --num_runs 5
```

### 4. Save API Credits

```bash
# Review only v1 first to test
--version v1 --limit 10

# Then add latest
--version latest --skip_existing
```

---

## Migration from Old Script

**Old (no format choice):**
```bash
python scripts/review/review_paper_pairs.py \
  --csv_file ./data/filtered_pairs.csv \
  --model_name "claude-haiku-4-5-20251001"
```

**New (explicit format):**
```bash
python scripts/review/review_paper_pairs.py \
  --csv_file ./data/filtered_pairs.csv \
  --api anthropic \
  --model_name "claude-haiku-4-5-20251001" \
  --format default
```

**Result**: Identical output (default values maintained)

---

## Summary

✅ **Backward Compatible**: Existing scripts/commands work unchanged
✅ **Two APIs**: Anthropic and Gemini both supported
✅ **Two Formats**: Default and CriticalNeurIPS
✅ **Evaluation Ready**: All outputs compatible with evaluation scripts
✅ **Flexible**: Mix and match API + format for your needs

### Quick Decision Guide

**Use Anthropic** when:
- You need highest reliability
- JSON format critical
- Speed important

**Use Gemini** when:
- Papers are very long (>100K tokens)
- Cost is a concern (Flash model)
- You need largest context window

**Use CriticalNeurIPS format** when:
- You want deep, scholarly critiques
- Evidence-based feedback important
- Research quality matters most

**Use Default format** when:
- You need quick, straightforward reviews
- Comparing with existing evaluations
- Speed/cost more important than depth

---

## Full Command Reference

```bash
python scripts/review/review_paper_pairs.py \
  --csv_file PATH                  # Required: path to filtered_pairs.csv
  --output_dir PATH                # Output directory (default: ./pair_reviews/)
  --api {anthropic,gemini}         # API to use (default: anthropic)
  --model_name MODEL               # Model name (required)
  --format {default,CriticalNeurIPS}  # Review format (default: default)
  --version {v1,latest,both}       # Which versions (default: both)
  --num_runs N                     # Number of runs (default: 1)
  --max_workers N                  # Concurrent workers (default: 3)
  --limit N                        # Limit papers for testing
  --skip_existing                  # Skip existing reviews
  --verbose                        # Verbose output
```

For complete CriticalNeurIPS format details, see:
- [`CRITICAL_NEURIPS_FORMAT.md`](../implementation/CRITICAL_NEURIPS_FORMAT.md)
- [`CRITICAL_NEURIPS_QUICK_START.md`](./CRITICAL_NEURIPS_QUICK_START.md)

