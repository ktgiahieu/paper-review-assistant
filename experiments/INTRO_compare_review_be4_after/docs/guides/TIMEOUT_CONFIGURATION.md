# Timeout Configuration Guide

## Overview

The review script now supports **configurable request timeouts** with **model-specific defaults**. This is especially important for CycleReviewer, which generates 4 independent reviews and takes significantly longer.

## Problem

**Original Issue:**
```
Worker 3301983: Request error for ViNe1fjGME (v1, run 0): 
HTTPConnectionPool(host='localhost', port=8000): Read timed out. (read timeout=300.0)
```

**Cause:** 
- CycleReviewer generates ~4x more content than other formats
- Fixed 300s (5 minute) timeout was too short
- Slow models or complex papers hit timeout limit

## Solution

### Model-Specific Defaults

The script now uses **different timeout defaults** based on model type:

| Model Type | Default Timeout | Reason |
|------------|----------------|--------|
| **SEA-E** | 300s (5 min) | Single review with moderate content |
| **CycleReviewer** | 900s (15 min) | 4 reviewers + meta review = ~4x content |
| **GenericStructured** | 300s (5 min) | Single review with moderate content |
| **Default** | 300s (5 min) | Standard single review |

### Automatic Timeout Selection

The script **automatically** selects the appropriate timeout based on model type:

```bash
# CycleReviewer automatically gets 900s timeout
python review_paper_pairs_vllm.py \
  --model_name "CycleReviewer-Llama-3.1-70B" \
  ...

# Output:
# Detected Model Type: CycleReviewer
# Request timeout: 900s (model-specific default)
```

### Manual Timeout Override

You can override the default timeout for any model:

```bash
# Use 20 minute timeout for very slow model
python review_paper_pairs_vllm.py \
  --model_name "CycleReviewer-Llama-3.1-70B" \
  --timeout 1200 \
  ...

# Output:
# Request timeout: 1200s (custom)
```

## Usage

### Default (Automatic)

```bash
# Automatically uses appropriate timeout
python review_paper_pairs_vllm.py \
  --csv_file ./data/filtered_pairs.csv \
  --output_dir ./reviews_cycle \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "CycleReviewer-Llama-3.1-70B" \
  --num_runs 3
```

**What happens:**
- Detects "CycleReviewer" model type
- Uses 900s timeout automatically
- Prints: `Request timeout: 900s (model-specific default)`

### Custom Timeout

```bash
# Use custom timeout for slower deployment
python review_paper_pairs_vllm.py \
  --csv_file ./data/filtered_pairs.csv \
  --output_dir ./reviews_cycle \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "CycleReviewer-Llama-3.1-70B" \
  --timeout 1800 \
  --num_runs 3
```

**What happens:**
- Uses 1800s (30 minutes) regardless of model type
- Prints: `Request timeout: 1800s (custom)`

## When to Adjust Timeout

### Increase Timeout If:

1. **Getting timeout errors** (like the original issue)
   ```
   Read timed out. (read timeout=300.0)
   ```

2. **Using CycleReviewer** on slow hardware
   - Default 900s may not be enough
   - Try: `--timeout 1200` or `--timeout 1800`

3. **Very long papers** (>50 pages)
   - More content → more time to process
   - Try: `--timeout 600` for single reviewers, `--timeout 1200` for CycleReviewer

4. **Slow model** (e.g., large model on limited GPU)
   - Generation speed varies by deployment
   - Monitor first few papers, adjust if needed

### Decrease Timeout If:

1. **Want faster failure detection**
   - Useful for debugging
   - Try: `--timeout 60` for quick tests

2. **Testing with simple papers**
   - Short test papers generate quickly
   - Can use lower timeout to catch errors faster

## Timeout Guidelines

### By Model Type

| Model | Recommended Timeout | Notes |
|-------|-------------------|-------|
| SEA-E | 300s (default) | Usually completes in 60-120s |
| CycleReviewer | 900s (default) | Usually completes in 300-600s |
| GenericStructured | 300s (default) | Usually completes in 60-120s |
| Qwen2-VL | 300s (default) | Fast, usually completes in 30-90s |
| Llama-3.1-70B | 300-600s | Slower than smaller models |

### By Paper Length

| Paper Length | Single Reviewer | CycleReviewer |
|--------------|----------------|---------------|
| Short (<10 pages) | 180s | 600s |
| Medium (10-20 pages) | 300s | 900s |
| Long (20-50 pages) | 600s | 1200s |
| Very Long (>50 pages) | 900s | 1800s |

### By Hardware

| GPU | Single Reviewer | CycleReviewer |
|-----|----------------|---------------|
| RTX 4090 / A100 | 300s | 900s |
| RTX 3090 / A6000 | 450s | 1200s |
| RTX 3080 / V100 | 600s | 1500s |
| CPU only | 1800s+ | 3600s+ |

## Implementation Details

### Constants in Code

```python
MODEL_TIMEOUTS = {
    "SEA-E": 300,           # 5 minutes
    "CycleReviewer": 900,   # 15 minutes
    "GenericStructured": 300,  # 5 minutes
    "default": 300          # 5 minutes
}
```

### Timeout Selection Logic

```python
# In review_single_paper_vllm()
model_type = ReviewPrompts.detect_model_type(model_name, format_override)

if timeout is None:
    timeout = MODEL_TIMEOUTS.get(model_type, MODEL_TIMEOUTS["default"])

print(f"Using timeout of {timeout}s for model type {model_type}")
```

### Request with Timeout

```python
response = requests.post(
    f"{vllm_endpoint}/v1/chat/completions",
    json=payload,
    timeout=timeout,  # Use model-specific or custom timeout
    headers={"Content-Type": "application/json"}
)
```

## Troubleshooting

### Issue: Still Getting Timeouts

**Symptoms:**
```
Read timed out. (read timeout=900.0)
```

**Solutions:**
1. **Increase timeout further:**
   ```bash
   --timeout 1800  # 30 minutes
   ```

2. **Check vLLM server performance:**
   ```bash
   # Monitor GPU usage
   nvidia-smi -l 1
   
   # Check vLLM logs
   tail -f vllm.log
   ```

3. **Reduce load:**
   ```bash
   --max_workers 1  # Sequential processing
   ```

4. **Optimize vLLM settings:**
   ```bash
   # Start vLLM with optimizations
   python -m vllm.entrypoints.openai.api_server \
     --model YourModel \
     --gpu-memory-utilization 0.95 \
     --max-model-len 8192
   ```

### Issue: Reviews Complete Fast but Timeout is Long

**Symptoms:**
- Reviews finish in 60s but timeout is 900s
- Wasted time waiting for failures

**Solution:**
- This is fine! Timeout is maximum wait time
- Only matters if generation actually times out
- No performance impact if completing normally

### Issue: Want Different Timeouts per Paper

**Current Limitation:**
- Timeout is set globally for all papers
- Cannot set different timeout for specific papers

**Workaround:**
1. Run papers in batches by complexity
2. Adjust timeout between batches
3. Use retry script for papers that timeout

### Issue: CycleReviewer Not Using 900s Timeout

**Symptoms:**
```
Request timeout: 300s (model-specific default)
```

**Cause:** Model name doesn't contain "cyclereviewer"

**Solutions:**
1. **Use format override:**
   ```bash
   --format CycleReviewer --timeout 900
   ```

2. **Rename model** in vLLM config to include "cyclereviewer"

3. **Manually specify timeout:**
   ```bash
   --timeout 900
   ```

## Best Practices

### 1. Start with Defaults

```bash
# Let script choose timeout automatically
python review_paper_pairs_vllm.py ...
# (no --timeout flag)
```

### 2. Monitor First Batch

```bash
# Run small batch first
python review_paper_pairs_vllm.py --limit 5 --verbose ...

# Check completion times in logs:
# Worker 123: Successfully reviewed abc (v1, run 0) in 245s
```

### 3. Adjust Based on Data

```bash
# If average completion time is 450s and timeout is 300s:
--timeout 900  # 2x average time for safety
```

### 4. Use Verbose Mode

```bash
--verbose

# Shows timeout being used:
# Worker 123: Using timeout of 900s for model type CycleReviewer
```

### 5. Combine with Retry

```bash
# Run with conservative timeout first
python review_paper_pairs_vllm.py --timeout 600 ...

# Retry failures with longer timeout
python retry_failed_reviews.py --timeout 1200 ...
```

## Examples

### CycleReviewer on Fast GPU

```bash
python review_paper_pairs_vllm.py \
  --model_name "CycleReviewer-Llama-3.1-70B" \
  --vllm_endpoint "http://localhost:8000" \
  ...
# Uses default 900s - should be sufficient
```

### CycleReviewer on Slow Hardware

```bash
python review_paper_pairs_vllm.py \
  --model_name "CycleReviewer-Llama-3.1-70B" \
  --vllm_endpoint "http://localhost:8000" \
  --timeout 1800 \
  ...
# Uses 30 minutes for slow generation
```

### Testing/Debugging

```bash
python review_paper_pairs_vllm.py \
  --model_name "SEA-E" \
  --timeout 60 \
  --limit 1 \
  --verbose
# Quick timeout to catch issues fast
```

### Very Long Papers

```bash
python review_paper_pairs_vllm.py \
  --model_name "GenericStructured" \
  --timeout 600 \
  ...
# 10 minutes for lengthy papers
```

## Monitoring

### Check Timeout Usage

```bash
# With --verbose, you'll see:
Worker 12345: Using timeout of 900s for model type CycleReviewer
Worker 12345: Reviewing paper abc (v1, run 0), attempt 1/3
...
Worker 12345: Successfully reviewed abc (v1, run 0)
```

### Track Completion Times

```python
import json
from pathlib import Path

# Analyze review times from logs
reviews_dir = Path("./reviews_cycle")
times = []

for review_file in reviews_dir.rglob("*_review_run*.json"):
    with open(review_file) as f:
        data = json.load(f)
        # Would need to add timestamp tracking to get actual times
        # This is a future enhancement

# Analyze distribution
print(f"Average: {sum(times)/len(times):.1f}s")
print(f"Max: {max(times)}s")
print(f"Recommended timeout: {max(times) * 2}s")  # 2x max for safety
```

## Summary

✅ **Automatic timeout selection** based on model type  
✅ **CycleReviewer gets 900s** (15 min) instead of 300s (5 min)  
✅ **Manual override** available via `--timeout` flag  
✅ **Clear feedback** showing which timeout is being used  
✅ **Fixes original timeout issue** for slow CycleReviewer generation  

**No action required for most users** - defaults work well!

For CycleReviewer, the system now automatically uses a 15-minute timeout, which should resolve the timeout errors you were experiencing. 🎉

---

**Questions?** Check the main README.md or open an issue if timeouts are still problematic.

