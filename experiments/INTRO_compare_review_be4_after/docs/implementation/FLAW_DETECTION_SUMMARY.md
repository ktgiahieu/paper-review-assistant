# Flaw Detection Evaluation - Implementation Summary

## Overview

Added functionality to evaluate whether AI-generated reviews detect consensus flaws (ground truth weaknesses identified during the paper revision process) and compare flaw detection rates between v1 and latest paper versions.

## Motivation

**Research Question:** Do AI reviewers identify the same weaknesses that human reviewers found? Do paper revisions make these flaws more or less obvious?

**Approach:** Use an evaluator LLM to systematically check if each ground truth flaw (from `flaw_descriptions` in CSV) is mentioned in the AI review's weaknesses section, then perform statistical analysis.

## New Files

### 1. `evaluate_flaw_detection.py` (~450 lines)

**Purpose:** Evaluate flaw detection using an evaluator LLM

**Key Functions:**

```python
parse_flaw_descriptions(flaw_desc_str) → List[str]
  # Parse flaw_descriptions column (Python list as string)

load_ai_review(reviews_dir, paper_id, version, run_id) → Optional[Dict]
  # Load AI review JSON

extract_weaknesses(review_data) → str
  # Extract weaknesses section (handles all formats)

check_flaw_detection(flaw, weaknesses, evaluator_endpoint, evaluator_model) → Tuple[bool, str]
  # Ask evaluator LLM if flaw is detected
  # Returns (is_detected, reasoning)

evaluate_paper_review(paper_id, flaws, weaknesses, ...) → Dict
  # Evaluate all flaws for one review
  # Returns recall and detailed results

main()
  # Load CSV, iterate papers/versions, evaluate, save results
```

**Evaluator Prompt Design:**

- **System:** Define task (match ground truth flaw to weaknesses)
- **User:** Provide ground truth flaw + weaknesses section
- **Output:** JSON with `{"detected": bool, "reasoning": str}`
- **Temperature:** 0.0 for deterministic evaluation

**Output Files:**
1. `flaw_detection_detailed.json` - Full results with per-flaw reasoning
2. `flaw_detection_summary.csv` - Paper-level summary (recall per paper-version-run)
3. `flaw_detection_per_flaw.csv` - Individual flaw detection records

**Key Features:**
- Supports all review formats (SEA-E, CycleReviewer, GenericStructured, Anthropic)
- Handles multiple runs (evaluates each separately)
- Robust error handling (JSON parsing, API timeouts)
- Rate limiting (0.2s delay between flaw checks)
- Progress bar with tqdm

### 2. `analyze_flaw_detection.py` (~400 lines)

**Purpose:** Statistical analysis with paired t-tests

**Key Functions:**

```python
load_results(results_file) → List[Dict]
  # Load evaluation results from JSON

perform_paired_ttest(df_paired) → Dict
  # Compare v1 vs latest recall
  # Returns t-statistic, p-value, Cohen's d, CI, interpretation

create_comparison_plots(df_summary, results, output_dir)
  # Generate 4 types of plots

main()
  # Load results, aggregate across runs, perform t-test, plot, save
```

**Statistical Tests:**

- **Paired t-test:** Compare v1 vs latest recall (same papers)
- **Effect size:** Cohen's d = mean_diff / std_diff
- **Confidence interval:** 95% CI using t-distribution

**Output Files:**
1. `flaw_detection_ttest_results.json` - Full statistical results
2. `flaw_detection_comparison_summary.csv` - Summary table
3. `flaw_detection_paired_data.csv` - Paired recall values (for further analysis)
4. **4 plots** (see below)

### 3. `FLAW_DETECTION_GUIDE.md` (~600 lines)

**Purpose:** Comprehensive user documentation

**Contents:**
- Complete workflow (Step 1 & 2)
- Detailed usage examples
- Output file descriptions
- Metric interpretations (Recall, p-value, Cohen's d)
- Plot explanations
- Research questions addressed
- Troubleshooting guide
- Best practices

## Visualization Plots

### 1. Bar Comparison Plot
- Mean recall for v1 vs latest with error bars
- Significance stars (**, *, or ns)
- Sample size annotation

### 2. Scatter Plot
- X-axis: v1 recall, Y-axis: latest recall
- Red diagonal line (no change)
- Green regression line
- Statistics box (correlation, mean Δ, p-value)

### 3. Difference Distribution Plot ⭐
- Histogram + KDE of recall differences (Latest - v1)
- Black dashed line at zero (H₀)
- Red line at mean difference
- Green shaded 95% CI
- **KEY PLOT for understanding paired t-test!**

### 4. Violin Plot
- Full distribution comparison
- Shows variability and shape differences

## Recall Metric

**Formula:**
```
Recall = (Number of flaws detected) / (Total number of flaws)
```

**Range:** 0.0 to 1.0

**Interpretation:**
- 1.0 = Perfect detection (all flaws found)
- 0.5 = Half of flaws detected
- 0.0 = No flaws detected

**Example:**
- Ground truth: 4 flaws for a paper
- AI detected: 3 flaws
- Recall = 3/4 = 0.75

## Technical Implementation

### Evaluator LLM Integration

**API Call Pattern:**
```python
POST {evaluator_endpoint}/v1/chat/completions
{
  "model": "Qwen3-30B-A3B-Instruct-2507-FP8",
  "messages": [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
  ],
  "temperature": 0.0,
  "max_tokens": 500
}
```

**Response Parsing:**
```python
# Extract JSON from response
content = result['choices'][0]['message']['content']

# Handle markdown code blocks
if '```json' in content:
    content = content.split('```json')[1].split('```')[0]

result_json = json.loads(content)
is_detected = result_json['detected']
reasoning = result_json['reasoning']
```

### Flaw Description Parsing

```python
# CSV stores flaws as string representation of Python list
flaw_desc_str = "['Flaw 1...', 'Flaw 2...', 'Flaw 3...']"

# Parse using ast.literal_eval
flaws = ast.literal_eval(flaw_desc_str)
# → ['Flaw 1...', 'Flaw 2...', 'Flaw 3...']
```

### Weaknesses Extraction

Handles different review formats:

```python
if model_type == 'CycleReviewer':
    # Combine weaknesses from 4 reviewers
    all_weaknesses = []
    for reviewer in review_data['reviewers']:
        all_weaknesses.extend(reviewer['weaknesses'])
    weaknesses_text = "\n".join(all_weaknesses)

elif 'weaknesses' in review_data:
    # Direct extraction
    weaknesses = review_data['weaknesses']
    if isinstance(weaknesses, list):
        weaknesses_text = "\n".join(weaknesses)
    else:
        weaknesses_text = weaknesses
```

### Aggregation Across Runs

```python
# Multiple runs for same paper-version
# Aggregate by taking mean recall

df_agg = df_summary.groupby(['paper_id', 'version']).agg({
    'recall': 'mean',  # Average recall across runs
    'num_flaws': 'first',
    'num_detected': 'mean'
}).reset_index()
```

## Workflow Integration

```
┌─────────────────────────────────┐
│ filtered_pairs.csv              │
│ (with flaw_descriptions)        │
└──────────┬──────────────────────┘
           │
           ├────────────────┐
           │                │
           ▼                ▼
    ┌──────────┐      ┌──────────────┐
    │ Ground   │      │ AI Reviews   │
    │ Truth    │      │ (weaknesses) │
    │ Flaws    │      └──────┬───────┘
    └────┬─────┘             │
         │                   │
         └────────┬──────────┘
                  │
                  ▼
        ┌────────────────────────┐
        │ evaluate_flaw_         │
        │   detection.py         │
        │ (Evaluator LLM)        │
        └──────────┬─────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │ flaw_detection_          │
        │   detailed.json          │
        │ (recall per paper)       │
        └──────────┬───────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │ analyze_flaw_            │
        │   detection.py           │
        │ (Paired t-test)          │
        └──────────┬───────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │ Statistical Results      │
        │ + Plots                  │
        └──────────────────────────┘
```

## Dependencies

All dependencies already in `requirements.txt`:
- `requests` - For evaluator API calls
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `scipy` - Statistical tests (paired t-test)
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `tqdm` - Progress bars
- `ast` - Parse Python literals (for flaw lists)

## Performance

### Time Complexity

**evaluate_flaw_detection.py:**
- O(N × V × R × F) where:
  - N = number of papers
  - V = versions (2: v1, latest)
  - R = runs per version
  - F = flaws per paper (~4 average)
- **Example:** 122 papers × 2 versions × 1 run × 4 flaws = ~976 evaluator calls
- **Time:** ~15-30 minutes (2-3 seconds per evaluator call)

**analyze_flaw_detection.py:**
- O(N) for aggregation and plotting
- **Time:** < 10 seconds

### Memory Usage

- Minimal (< 100 MB)
- All data fits in memory
- JSON file size: ~5-10 MB for 1000 evaluations

### API Usage

- **Evaluator calls:** ~976 for 122 papers
- **Tokens per call:** ~300-500
- **Total tokens:** ~400K
- **Rate limiting:** 0.2s delay (5 calls/second max)

## Error Handling

### evaluate_flaw_detection.py

**JSON Parse Errors:**
```python
try:
    # Handle markdown code blocks
    if '```json' in content:
        content = content.split('```json')[1].split('```')[0]
    result_json = json.loads(content)
except json.JSONDecodeError as e:
    # Retry up to 3 times
    if attempt < max_retries - 1:
        time.sleep(2)
        continue
    return (False, f"JSON parse error: {str(e)}")
```

**API Timeouts:**
```python
try:
    response = requests.post(..., timeout=60)
except requests.exceptions.Timeout:
    # Exponential backoff retry
    if attempt < max_retries - 1:
        time.sleep(2 ** attempt)
        continue
```

**Missing Data:**
```python
# Papers without flaws
if not flaws:
    return {'num_flaws': 0, 'recall': None, ...}

# Reviews without weaknesses
if not weaknesses:
    return {'recall': 0.0, 'flaws_detailed': [...]}
```

### analyze_flaw_detection.py

**Missing Paired Data:**
```python
if 'v1' not in versions_present or 'latest' not in versions_present:
    print("Error: Need both v1 and latest versions!")
    return

df_paired = df_v1.merge(df_latest, on='paper_id', how='inner')
if len(df_paired) == 0:
    print("Error: No paired samples found!")
    return
```

## Research Applications

### Questions Answered

1. **Flaw Detection Rate**
   - Overall recall: How well do AI reviewers detect consensus flaws?
   - By metric: Which types of flaws are detected more/less?

2. **Impact of Revisions**
   - Paired t-test: Do revisions improve detectability?
   - Effect size: How large is the improvement?

3. **Consistency**
   - Across runs: Are results stable?
   - Across papers: Which papers show biggest changes?

4. **Flaw Analysis**
   - Which flaws are frequently missed?
   - Are certain flaw types harder to detect?

### Example Findings

```
v1 Recall:     0.631 ± 0.224
Latest Recall: 0.673 ± 0.211
Difference:    +0.042 ± 0.158
p-value:       0.0039 **

Interpretation:
- AI detected 63% of flaws in v1
- Improved to 67% in latest (4% absolute improvement)
- Statistically significant (p < 0.01)
- Small effect size (Cohen's d = 0.27)
- Conclusion: Revisions made flaws slightly more obvious
```

## Future Enhancements

Potential additions:

1. **Confidence Scores**
   - Ask evaluator for confidence level (0-100%)
   - Analyze correlation between confidence and accuracy

2. **Flaw Type Classification**
   - Categorize flaws (statistical, methodological, clarity, etc.)
   - Compare detection rates by flaw type

3. **Multi-Evaluator Ensemble**
   - Use multiple evaluator models
   - Aggregate decisions (majority vote)
   - Measure inter-evaluator agreement

4. **Partial Credit**
   - Instead of binary (detected/not detected)
   - Score on scale (0-1) for partial detection

5. **Causality Analysis**
   - Which specific paper changes led to improved detection?
   - Correlate with edit operations

## Summary

| Aspect | Details |
|--------|---------|
| **Files Added** | 3 (2 scripts + 1 guide) |
| **Lines of Code** | ~850 lines |
| **External APIs** | vLLM evaluator endpoint |
| **Metrics** | Recall, paired t-test, Cohen's d |
| **Plot Types** | 4 (bar, scatter, difference, violin) |
| **Documentation** | Complete guide + examples |
| **Integration** | Seamless with existing workflow |
| **Novel Feature** | LLM-as-judge for flaw detection evaluation |

This implementation enables systematic evaluation of AI reviewers' ability to detect consensus flaws, providing insights into both AI review quality and the impact of paper revisions! 🎯📊

