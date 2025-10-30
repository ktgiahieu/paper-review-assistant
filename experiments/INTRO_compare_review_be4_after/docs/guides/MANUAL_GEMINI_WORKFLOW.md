# Manual Gemini Pro Review Workflow

## Overview

This guide explains how to conduct paper reviews using Gemini Pro via the web UI (without API access).

**Why use this workflow?**
- No API key needed
- Free Gemini Pro access via AI Studio
- Same quality reviews as API-based approach
- Full compatibility with evaluation pipeline

---

## Quick Start

### Step 1: Prepare Review Materials

```bash
python scripts/utils/prepare_manual_gemini_prompts.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ./manual_gemini_reviews/ \
  --format CriticalNeurIPS \
  --max_figures 10 \
  --num_runs 3
```

**Output:** Creates structured folders with prompts and figures ready for Gemini UI.

### Step 2: Complete Reviews Manually

For each paper folder:
1. Open `README.md` for instructions
2. Copy prompt from `input/prompt.txt`
3. Upload figures from `input/figures/`
4. Paste into [Gemini Pro UI](https://aistudio.google.com/)
5. Copy JSON output to `output/review.json`

### Step 3: Process Outputs

```bash
python scripts/utils/process_manual_gemini_outputs.py \
  --input_dir ./manual_gemini_reviews/ \
  --output_dir ./reviews_gemini_manual/
```

**Output:** Formatted reviews ready for evaluation scripts!

---

## Detailed Workflow

### Part 1: Preparation

#### Command Options

```bash
python scripts/utils/prepare_manual_gemini_prompts.py \
  --csv_file PATH                      # Required: filtered_pairs.csv
  --output_dir PATH                    # Output directory
  --format {default,CriticalNeurIPS}   # Review format
  --max_figures N                      # Max figures per paper
  --num_runs N                         # Review runs per paper
  --version {v1,latest,both}           # Which versions
  --limit N                            # Test with N papers
```

#### What It Creates

```
manual_gemini_reviews/
├── paper_id_1/
│   ├── v1_run0/
│   │   ├── README.md                # Step-by-step instructions
│   │   ├── input/
│   │   │   ├── prompt.txt           # Complete prompt for Gemini
│   │   │   ├── figure_list.txt      # Figure upload order
│   │   │   └── figures/             # Compressed figures
│   │   │       ├── figure_01.jpg
│   │   │       ├── figure_02.jpg
│   │   │       └── ...
│   │   └── output/
│   │       └── review.json          # Placeholder for your input
│   ├── v1_run1/
│   ├── latest_run0/
│   └── latest_run1/
├── paper_id_2/
└── ...
```

---

### Part 2: Manual Review Process

#### For Each Paper:

**1. Navigate to the Run Folder**

```bash
cd manual_gemini_reviews/paper_id_1/v1_run0/
```

**2. Read Instructions**

Open `README.md` - it contains complete step-by-step instructions.

**3. Open Gemini Pro**

- Go to: https://aistudio.google.com/
- Make sure you're using **Gemini 1.5 Pro** (supports images + 2M context)
- Start a new chat

**4. Upload Figures** (if any)

- Check `input/figures/` folder
- Upload figures in the order listed in `input/figure_list.txt`
- Click the image upload button in Gemini UI
- Upload `figure_01.jpg`, then `figure_02.jpg`, etc.

**5. Copy and Paste Prompt**

```bash
# Open the prompt file
cat input/prompt.txt

# Or on Mac/Linux:
pbcopy < input/prompt.txt  # Copies to clipboard
```

- Paste the entire prompt into Gemini's chat
- Press Enter

**6. Wait for Response**

- Gemini will generate the review (1-2 minutes for CriticalNeurIPS)
- The response should be a JSON object

**7. Extract JSON Output**

Gemini might return:
```
Here's my review:

```json
{
  "summary": "...",
  "strengths_and_weaknesses": "...",
  ...
}
```
```

**Copy ONLY the JSON part** (the `{...}` without the markdown):

```json
{
  "summary": "...",
  "strengths_and_weaknesses": "...",
  ...
}
```

**8. Save to Output File**

```bash
# Open output/review.json in your editor
# Replace EVERYTHING with the JSON from Gemini
# Save the file
```

**9. Repeat**

Move to next run folder and repeat for all papers.

---

### Part 3: Processing Outputs

#### After Completing All Reviews

```bash
python scripts/utils/process_manual_gemini_outputs.py \
  --input_dir ./manual_gemini_reviews/ \
  --output_dir ./reviews_gemini_manual/
```

#### What This Does

1. **Scans** all `output/review.json` files
2. **Skips** placeholders (not yet filled)
3. **Validates** JSON format and required fields
4. **Cleans** common formatting issues
5. **Adds** metadata (paper_id, version, model_type)
6. **Maps** scores for evaluation compatibility
7. **Saves** to standard review format

#### Output Structure

```
reviews_gemini_manual/
├── paper_id_1/
│   ├── v1_review_run0.json
│   ├── v1_review_run1.json
│   ├── latest_review_run0.json
│   └── latest_review_run1.json
├── paper_id_2/
└── ...
```

**Now compatible with all evaluation scripts!** ✅

---

## Tips & Best Practices

### 1. Start Small

Test with a few papers first:

```bash
--limit 3  # Just 3 papers for testing
```

### 2. Organize Your Work

Keep track of completed reviews:

```bash
# Create a checklist
find manual_gemini_reviews -name "README.md" | wc -l
```

### 3. Batch Upload Figures

For papers with many figures:
- Upload all figures at once
- Gemini can handle up to 10 images per chat

### 4. Save Gemini Chats

- Gemini AI Studio saves your chat history
- You can revisit if you need to re-copy output

### 5. Check JSON Validity

Before closing Gemini:
- Copy the JSON
- Paste into [JSONLint](https://jsonlint.com/) to verify
- Fix any issues before saving

### 6. Handle Long Papers

Gemini 1.5 Pro has 2M context:
- Can handle very long papers
- Much larger than Anthropic's 200K

### 7. Incremental Processing

You don't need to complete all reviews at once:

```bash
# Process what you've done so far
python scripts/utils/process_manual_gemini_outputs.py ...

# It will skip placeholders and process completed ones
```

---

## Common Issues & Solutions

### Issue: Gemini Returns Markdown-Formatted JSON

**Problem:**
```
Here's the review:

```json
{...}
```
```

**Solution:** Copy only the JSON part (between `{` and `}`), not the markdown.

---

### Issue: JSON Has Extra Text

**Problem:**
```
{
  "summary": "...",
  ...
}

I hope this review is helpful!
```

**Solution:** Copy only up to the final `}` of the JSON object.

---

### Issue: Invalid JSON Error

**Problem:** Processing script reports JSON decode error

**Solution:**
1. Copy the JSON to [JSONLint](https://jsonlint.com/)
2. Fix any errors (missing commas, quotes, etc.)
3. Save the corrected version

---

### Issue: Missing Required Fields

**Problem:** Validation error about missing fields

**Solution:**
1. Check `_expected_fields` in the placeholder
2. Re-run Gemini with clearer instructions
3. Or manually add missing fields with reasonable defaults

---

### Issue: Gemini Truncates Output

**Problem:** JSON is incomplete (ends mid-sentence)

**Solution:**
1. Ask Gemini to "continue" or "complete the JSON"
2. Combine the parts yourself
3. Or use a smaller paper/fewer figures

---

### Issue: Figures Too Large

**Problem:** Gemini UI rejects large images

**Solution:** 
The preparation script already compresses images to <4MB. If still issues:
- Check `input/figures/` folder
- All should be .jpg format, compressed
- Contact support if problems persist

---

## Format Comparison

### CriticalNeurIPS Format (Recommended)

**Output Fields:**
```json
{
  "summary": "string",
  "strengths_and_weaknesses": "string (Markdown)",
  "questions": "string",
  "limitations_and_societal_impact": "string",
  "soundness": 1-4,
  "presentation": 1-4,
  "contribution": 1-4,
  "overall_score": 1-10,
  "confidence": 1-5
}
```

**Best For:**
- Deep, scholarly reviews
- Evidence-based feedback
- Research quality

### Default Format

**Output Fields:**
```json
{
  "summary": "string",
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "clarity_score": 1-10,
  "novelty_score": 1-10,
  "technical_quality_score": 1-10,
  "experimental_rigor_score": 1-10,
  "overall_score": 1-10,
  "confidence": 1-5,
  "recommendation": "string",
  "detailed_comments": "string"
}
```

**Best For:**
- Quick, straightforward reviews
- Comparing with existing data

---

## Time Estimates

| Task | Time per Paper | Notes |
|------|----------------|-------|
| **Preparation** | ~5 seconds | Automated |
| **Manual Review** | 2-3 minutes | Upload figures + paste prompt + copy output |
| **Processing** | ~1 second | Automated |

**Example:**
- 125 papers × 2 versions × 3 runs = 750 reviews
- 750 × 2.5 min = **~31 hours** of manual work
- Spread over multiple sessions!

---

## Workflow Optimization

### Strategy 1: Parallel Work

Multiple people can work simultaneously:
- Person A: Papers 1-40
- Person B: Papers 41-80
- Person C: Papers 81-125

### Strategy 2: Batch by Version

1. Do all v1 reviews first
2. Then all latest reviews
3. Process after each batch

### Strategy 3: Daily Quota

Set a goal:
- 20 reviews per day
- Complete in ~38 days (for 750 reviews)
- ~1 hour per day

---

## Quality Control

### Before Processing

**Check Random Samples:**
```bash
# Look at a few review.json files
cat manual_gemini_reviews/*/v1_run0/output/review.json | head -50
```

**Ensure:**
- No placeholder text remaining
- Valid JSON format
- All required fields present

### After Processing

**Run Validation:**
```bash
python scripts/utils/process_manual_gemini_outputs.py \
  --input_dir ./manual_gemini_reviews/ \
  --output_dir ./reviews_gemini_manual/
```

**Check Output:**
- Should report success count
- Note any errors
- Re-do failed reviews

---

## Integration with Evaluation

### Run Standard Evaluations

**Numerical Scores:**
```bash
python scripts/evaluation/evaluate_numerical_scores.py \
  --reviews_dir ./reviews_gemini_manual/ \
  --output_dir ./evaluation_gemini_manual/
```

**Flaw Detection:**
```bash
python scripts/evaluation/evaluate_flaw_detection.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --reviews_dir ./reviews_gemini_manual/ \
  --output_dir ./flaw_detection_gemini_manual/
```

**AI vs Human:**
```bash
python scripts/evaluation/calculate_mse_mae.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs_with_human_scores.csv \
  --reviews_dir ./reviews_gemini_manual/ \
  --output_dir ./ai_vs_human_gemini_manual/
```

### All Work the Same!

The manual workflow produces identical output format to API-based reviews.

---

## Comparison: Manual vs API

| Feature | Manual UI | API-Based |
|---------|-----------|-----------|
| **Cost** | Free | Pay-per-use |
| **Setup** | None | API key needed |
| **Speed** | Slow (manual) | Fast (automated) |
| **Output Quality** | Same | Same |
| **Figures** | ✅ Yes | ✅ Yes |
| **Evaluation Compatible** | ✅ Yes | ✅ Yes |
| **Best For** | Small-medium datasets | Large datasets |

**Use Manual When:**
- No API budget
- Small number of papers (<100)
- Want complete control
- Learning the process

**Use API When:**
- Have API access
- Large dataset (>100 papers)
- Need speed
- Automated pipeline

---

## Full Example Walkthrough

### Scenario: 5 Papers, 2 Versions, 1 Run Each

**1. Prepare (10 seconds):**
```bash
python scripts/utils/prepare_manual_gemini_prompts.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ./manual_gemini_reviews/ \
  --format CriticalNeurIPS \
  --limit 5 \
  --num_runs 1
```

**2. Manual Reviews (20-30 minutes):**
- 10 reviews total (5 papers × 2 versions)
- ~2-3 min per review
- Follow README.md in each folder

**3. Process (1 second):**
```bash
python scripts/utils/process_manual_gemini_outputs.py \
  --input_dir ./manual_gemini_reviews/ \
  --output_dir ./reviews_gemini_manual/
```

**4. Evaluate (10 seconds):**
```bash
python scripts/evaluation/evaluate_numerical_scores.py \
  --reviews_dir ./reviews_gemini_manual/ \
  --output_dir ./evaluation_test/
```

**Total Time:** ~30 minutes for complete workflow!

---

## Checklist

### Before Starting
- [ ] Have access to Gemini AI Studio (https://aistudio.google.com/)
- [ ] Run preparation script
- [ ] Check that folders are created correctly
- [ ] Test with 1-2 papers first

### During Reviews
- [ ] Follow README.md in each folder
- [ ] Upload figures in correct order
- [ ] Copy complete JSON (no markdown)
- [ ] Verify JSON is valid
- [ ] Save to correct output/review.json

### After Completing
- [ ] Run processing script
- [ ] Check for errors
- [ ] Re-do any failed reviews
- [ ] Run evaluation scripts
- [ ] Verify results look reasonable

---

## Support & Troubleshooting

### Check Status

**How many reviews are done?**
```bash
# Count non-placeholder reviews
grep -L "_instructions" manual_gemini_reviews/*/*/output/review.json | wc -l
```

**Which are still pending?**
```bash
grep -l "_instructions" manual_gemini_reviews/*/*/output/review.json
```

### Re-run Specific Papers

If a paper failed:
1. Delete the output/review.json
2. Re-copy from placeholder
3. Redo the review
4. Re-run processing script

### Get Help

Common issues are documented above. For other problems:
1. Check the processing script output (shows specific errors)
2. Validate JSON at [JSONLint](https://jsonlint.com/)
3. Check placeholder for expected format
4. Review the prompt.txt to understand expected output

---

## Summary

✅ **No API Required** - Use free Gemini UI
✅ **Same Quality** - Identical to API-based reviews
✅ **Full Compatibility** - Works with all evaluation scripts
✅ **Flexible** - Do reviews at your own pace
✅ **Cost-Effective** - Completely free
✅ **Beginner-Friendly** - Clear instructions at every step

### Quick Commands

```bash
# 1. Prepare
python scripts/utils/prepare_manual_gemini_prompts.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ./manual_gemini_reviews/ \
  --format CriticalNeurIPS

# 2. Complete reviews manually (follow README.md files)

# 3. Process
python scripts/utils/process_manual_gemini_outputs.py \
  --input_dir ./manual_gemini_reviews/ \
  --output_dir ./reviews_gemini_manual/

# 4. Evaluate
python scripts/evaluation/evaluate_numerical_scores.py \
  --reviews_dir ./reviews_gemini_manual/
```

**You're ready to start!** 🚀

