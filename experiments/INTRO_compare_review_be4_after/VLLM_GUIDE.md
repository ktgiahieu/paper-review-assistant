# vLLM Review Script - Quick Start Guide

## What's New in `review_paper_pairs_vllm.py`

This script extends the original `review_paper_pairs.py` with:

1. **📊 Multimodal Support**: Automatically extracts figures from papers and includes them in reviews
2. **🔄 Multiple Runs**: Run the same review multiple times to analyze LLM variance
3. **🖥️ Local Models**: Use your own vLLM-hosted models
4. **💰 Cost Savings**: Free inference if self-hosted

## Image Processing Features

The script uses sophisticated image processing from `review_with_anthropic.py`:

- **Automatic extraction**: Finds all images referenced in markdown
- **Smart resizing**: Reduces large images while maintaining quality
- **Format support**: PNG, JPG, JPEG, GIF, WebP
- **Size optimization**: Resizes images over 20MB to fit API limits
- **Base64 encoding**: Converts images to API-compatible format

## Multiple Runs Feature

Use `--num_runs N` to review each paper multiple times:

```bash
# Review each paper 3 times
python review_paper_pairs_vllm.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_vllm" \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "Qwen/Qwen2-VL-7B-Instruct" \
  --num_runs 3 \
  --max_figures 5
```

**Output Files**:
- `v1_review_run0.json`, `v1_review_run1.json`, `v1_review_run2.json`
- `latest_review_run0.json`, `latest_review_run1.json`, `latest_review_run2.json`

**Use Cases**:
- Measure consistency of LLM judgments
- Calculate confidence intervals for scores
- Identify high-variance papers (where LLM is uncertain)
- Study temperature/sampling effects

## Setting Up vLLM Server

### Option 1: Local vLLM (Single GPU)

```bash
# Install vLLM
pip install vllm

# Start server with Qwen2-VL
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.9
```

### Option 2: Multi-GPU vLLM

```bash
# Use tensor parallelism for larger models
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2-VL-72B-Instruct \
  --port 8000 \
  --tensor-parallel-size 4
```

### Option 3: Remote vLLM Server

If you have a remote vLLM server:

```bash
python review_paper_pairs_vllm.py \
  --vllm_endpoint "http://your-server:8000" \
  --model_name "Qwen/Qwen2-VL-7B-Instruct" \
  ...
```

## Image Processing Configuration

Control how many figures to include:

```bash
# Include up to 5 figures per paper (recommended)
--max_figures 5

# Include up to 10 figures (for figure-heavy papers)
--max_figures 10

# Disable images (text-only, faster)
--max_figures 0
```

**Image Processing Steps**:
1. Parse markdown for image references
2. Resolve relative paths to paper directory
3. Check file exists and is valid image format
4. Resize if > 20MB (maintains aspect ratio)
5. Encode to base64
6. Include in API request as OpenAI-compatible format

## Performance Tuning

### For Speed

```bash
--max_workers 10      # More parallel requests
--max_figures 0       # Skip images
--num_runs 1         # Single run per paper
```

### For Quality

```bash
--max_workers 3       # Fewer parallel requests (less GPU contention)
--max_figures 10      # Include more figures
--num_runs 5         # Multiple runs for reliability
```

### For GPU Memory

If you get OOM errors:

1. **Reduce batch size** in vLLM server:
   ```bash
   --max-num-batched-tokens 4096
   ```

2. **Reduce max_figures**:
   ```bash
   --max_figures 3
   ```

3. **Reduce concurrency**:
   ```bash
   --max_workers 1
   ```

## Example Workflows

### Workflow 1: Quick Test

```bash
# Test with 1 paper, no images
python review_paper_pairs_vllm.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./test" \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "Qwen/Qwen2-VL-7B-Instruct" \
  --limit 1 \
  --max_figures 0 \
  --verbose
```

### Workflow 2: Test with Images

```bash
# Test with 1 paper, 5 figures
python review_paper_pairs_vllm.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./test" \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "Qwen/Qwen2-VL-7B-Instruct" \
  --limit 1 \
  --max_figures 5 \
  --verbose
```

### Workflow 3: Variance Analysis

```bash
# Review 10 papers, 3 runs each, with figures
python review_paper_pairs_vllm.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./variance_study" \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "Qwen/Qwen2-VL-7B-Instruct" \
  --limit 10 \
  --num_runs 3 \
  --max_figures 5 \
  --max_workers 3 \
  --verbose
```

### Workflow 4: Full Production Run

```bash
# Review all papers, both versions, 3 runs, with figures
python review_paper_pairs_vllm.py \
  --csv_file "./data/ICLR2024_pairs/filtered_pairs.csv" \
  --output_dir "./reviews_vllm_production" \
  --vllm_endpoint "http://localhost:8000" \
  --model_name "Qwen/Qwen2-VL-7B-Instruct" \
  --version both \
  --num_runs 3 \
  --max_figures 5 \
  --max_workers 5
```

## Analyzing Multiple Runs

The `review_summary.csv` includes a `run_id` column. Use pandas to analyze:

```python
import pandas as pd
import numpy as np

# Load results
df = pd.read_csv("reviews_vllm_production/review_summary.csv")

# Calculate mean and std for each paper
stats = df.groupby('paper_id').agg({
    'v1_overall_score': ['mean', 'std'],
    'latest_overall_score': ['mean', 'std'],
    'overall_score_change': ['mean', 'std']
})

# Find high-variance papers (where LLM is uncertain)
high_variance = stats[stats[('v1_overall_score', 'std')] > 1.5]
print("Papers with high variance:", high_variance)

# Calculate confidence intervals
df['v1_overall_score'].groupby(df['paper_id']).apply(
    lambda x: (x.mean() - 1.96*x.std()/np.sqrt(len(x)),
               x.mean() + 1.96*x.std()/np.sqrt(len(x)))
)
```

## Troubleshooting

### Issue: "Could not connect to vLLM endpoint"

**Solution**: Check vLLM server is running:
```bash
curl http://localhost:8000/health
```

### Issue: "Pillow library not found"

**Solution**: Install Pillow:
```bash
pip install Pillow
```

### Issue: "Request timeout"

**Solution**: Increase timeout or reduce complexity:
- Use smaller model
- Reduce `--max_figures`
- Reduce paper length

### Issue: "Out of GPU memory"

**Solution**: 
- Restart vLLM with `--gpu-memory-utilization 0.8`
- Reduce `--max_figures`
- Reduce `--max_workers`
- Use smaller model

### Issue: Images not being included

**Solution**: Check:
1. Images exist in paper directory
2. `--max_figures > 0`
3. Image formats are supported (png, jpg, jpeg, gif, webp)
4. Model supports multimodal input

## Supported Models

Any vLLM-compatible multimodal model works. Recommended:

- **Qwen2-VL-7B-Instruct**: Good quality, moderate size
- **Qwen2-VL-72B-Instruct**: Best quality, large size
- **LLaVA-v1.6-34B**: Good alternative
- **InternVL2-8B**: Efficient option

**Note**: For text-only (no figures), any text model works (LLaMA, Mistral, etc.)

## Comparison to Anthropic Version

| Aspect | Anthropic | vLLM |
|--------|-----------|------|
| Setup | Easy | Moderate |
| Speed | Fast | Depends on GPU |
| Cost | ~$0.10-0.50/paper | Free (self-hosted) |
| Images | ❌ | ✅ |
| Multiple Runs | ❌ | ✅ |
| Model Choice | Claude only | Any vLLM model |
| Quality | Excellent | Model-dependent |

## Next Steps

1. **Start small**: Test with `--limit 1` first
2. **Add images**: Test with `--max_figures 5`
3. **Test variance**: Try `--num_runs 3`
4. **Scale up**: Remove `--limit` for full dataset
5. **Analyze**: Use pandas to study variance and consistency

