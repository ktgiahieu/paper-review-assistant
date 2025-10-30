# CriticalNeurIPS Format - Quick Start Guide

## TL;DR

The **CriticalNeurIPS** format produces exceptionally thorough, multi-faceted critiques that challenge papers from both conceptual and methodological angles.

## Quick Example

```bash
# Use with any model
python scripts/review/review_paper_pairs_vllm.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ./reviews_critical/ \
  --vllm_endpoint "http://localhost:8000/" \
  --model_name "Meta-Llama-3.1-70B-Instruct" \
  --format CriticalNeurIPS \
  --num_runs 3
```

## What Makes It Different?

### Other Formats
- SEA-E: Structured checklist
- CycleReviewer: 4 reviewers voting
- GenericStructured: Simple template

### CriticalNeurIPS
- **Deep critique**: Questions assumptions, cites literature
- **Multi-perspective**: Conceptual + Methodological analysis
- **Evidence-based**: Counter-examples, alternative hypotheses
- **Scholarly**: First principles, foundational references

## Reviewer Personas

### 1. The Conceptual Critic
- ❓ Questions core concepts
- 📚 Cites foundational literature
- 🔄 Re-frames weak arguments
- 🗺️ Provides citation roadmap

### 2. The Methodological Skeptic
- 🔬 Scrutinizes experimental design
- 🕵️ Identifies critical omissions
- ⚠️ Challenges unstated assumptions
- 🎯 Points out countervailing evidence

## Output Fields

```json
{
  "summary": "Neutral overview",
  "strengths_and_weaknesses": "Combined with Markdown",
  "questions": "3-5 actionable questions",
  "limitations_and_societal_impact": "Ethical considerations",
  "soundness": 1-4,
  "presentation": 1-4,
  "contribution": 1-4,
  "overall_score": 1-10,
  "confidence": 1-5
}
```

## Score Ranges

| Metric | Range | Meaning |
|--------|-------|---------|
| soundness, presentation, contribution | 1-4 | 4=excellent → 1=poor |
| overall_score | 1-10 | 10=award quality → 1=trivial |
| confidence | 1-5 | 5=certain → 1=guess |

## Usage Patterns

### Pattern 1: Auto-Detection

```bash
# Model name contains "criticalneurips"
--model_name "CriticalNeurIPS-70B"  # Auto-detected ✓
```

### Pattern 2: Explicit Format

```bash
# Any model + --format flag
--model_name "Meta-Llama-3.1-70B-Instruct" \
--format CriticalNeurIPS
```

### Pattern 3: Testing (Limited Papers)

```bash
# Test on 5 papers first
--limit 5 \
--format CriticalNeurIPS \
--verbose
```

### Pattern 4: Production (Multiple Runs)

```bash
# 5 runs for variance analysis
--num_runs 5 \
--format CriticalNeurIPS \
--max_workers 3
```

## Timeout

- **Default:** 600 seconds (10 minutes)
- **Why:** Critical analysis takes longer
- **Location:** `MODEL_TIMEOUTS["CriticalNeurIPS"]`

## Evaluation Compatibility

All standard evaluation scripts work:

```bash
# 1. Numerical scores
python scripts/evaluation/evaluate_numerical_scores.py \
  --reviews_dir ./reviews_critical/

# 2. Flaw detection
python scripts/evaluation/evaluate_flaw_detection.py \
  --reviews_dir ./reviews_critical/

# 3. AI vs Human
python scripts/evaluation/calculate_mse_mae.py \
  --reviews_dir ./reviews_critical/

# 4. Paired t-test for flaw detection
python scripts/evaluation/analyze_flaw_detection.py \
  --results_file ./flaw_detection_results/flaw_detection_detailed.json
```

## When to Use

### ✅ Use CriticalNeurIPS when:
- You want **deep, thorough critiques**
- You need **evidence-based feedback**
- You're studying **review quality**
- You have **large models** (70B+)
- You want **scholarly rigor**

### ❌ Avoid CriticalNeurIPS when:
- You need **quick reviews** (use GenericStructured)
- You want **multiple perspectives** (use CycleReviewer)
- You need **strict format** (use SEA-E)
- You have **small models** (<13B)
- You want **gentle feedback** (this format is critical!)

## Recommended Models

Best results with:
- Meta-Llama-3.1-70B-Instruct ⭐
- Qwen-72B-Instruct ⭐
- Mixtral-8x22B-Instruct
- GPT-4 (via compatible API)

Avoid with:
- Models < 13B parameters
- Models not instruction-tuned
- Models without academic training data

## Expected Characteristics

### Scores
- **Tend to be lower** than other formats (critical stance)
- **More differentiation** between good/bad papers
- **Higher variance** across runs (deeper analysis)

### Review Length
- **Longer** than other formats
- **More detailed** weaknesses
- **More actionable** questions

### Flaw Detection
- **Higher recall** for consensus flaws
- **Better at** methodological issues
- **Better at** unstated assumptions

## Common Issues

### Issue: Timeout after 10 minutes
**Solution:** Expected for complex papers. Check:
- Model speed (try faster model)
- Paper length (reduce figures with `--max_figures 0`)
- vLLM configuration (increase `max_tokens`)

### Issue: Reviews seem harsh
**Solution:** This is intentional! The format is critical.
- Compare within-format, not across formats
- Expect lower scores than GenericStructured
- This is a feature, not a bug

### Issue: JSON parsing errors
**Solution:** Robust sanitization is included, but:
- Check model instruction-following ability
- Try increasing timeout
- Use `--verbose` to debug

## File Locations

```
scripts/review/
  └── review_paper_pairs_vllm.py        # Main script

docs/implementation/
  └── CRITICAL_NEURIPS_FORMAT.md        # Full documentation

docs/guides/
  └── CRITICAL_NEURIPS_QUICK_START.md   # This guide
```

## Full Documentation

For complete details, see:
- [`CRITICAL_NEURIPS_FORMAT.md`](../implementation/CRITICAL_NEURIPS_FORMAT.md) - Complete technical documentation
- [`README.md`](../../README.md) - General usage guide
- [`MODEL_FORMATS.md`](../implementation/MODEL_FORMATS.md) - All format comparisons

## Example Workflow

```bash
# 1. Test on 5 papers
python scripts/review/review_paper_pairs_vllm.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ./reviews_critical_test/ \
  --vllm_endpoint "http://localhost:8000/" \
  --model_name "Meta-Llama-3.1-70B-Instruct" \
  --format CriticalNeurIPS \
  --limit 5 \
  --verbose

# 2. Check results
ls reviews_critical_test/*/v1_review_run0.json

# 3. Run full evaluation (125 papers, 3 runs = 750 reviews)
python scripts/review/review_paper_pairs_vllm.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ./reviews_critical_full/ \
  --vllm_endpoint "http://localhost:8000/" \
  --model_name "Meta-Llama-3.1-70B-Instruct" \
  --format CriticalNeurIPS \
  --num_runs 3 \
  --max_workers 3

# 4. Evaluate numerical scores
python scripts/evaluation/evaluate_numerical_scores.py \
  --reviews_dir ./reviews_critical_full/ \
  --output_dir ./evaluation_critical/

# 5. Evaluate flaw detection
python scripts/evaluation/evaluate_flaw_detection.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --reviews_dir ./reviews_critical_full/ \
  --output_dir ./flaw_detection_critical/

# 6. Analyze flaw detection results
python scripts/evaluation/analyze_flaw_detection.py \
  --results_file ./flaw_detection_critical/flaw_detection_detailed.json \
  --output_dir ./flaw_analysis_critical/
```

## Quick Comparison

| Aspect | CriticalNeurIPS | GenericStructured |
|--------|-----------------|-------------------|
| **Depth** | Very deep | Moderate |
| **Critique Style** | Multi-faceted | Standard |
| **Citations** | Encouraged | Not emphasized |
| **Time** | 10 min/review | 5 min/review |
| **Scores** | Lower (critical) | Higher (balanced) |
| **Best For** | Research quality | General use |

## Summary

**One-Liner:** CriticalNeurIPS = Deep, scholarly, multi-perspective critique with evidence-based feedback

**Use when:** You want the best possible AI-generated review, regardless of time/cost

**Avoid when:** You need quick, simple pass/fail decisions

**Expected result:** Thorough, rigorous, actionable reviews that push papers to improve

