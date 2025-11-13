# Plant Errors and Placebo Generation Script

This script plants errors in research papers based on flaw descriptions from the CSV file and generates placebo/sham surgery versions.

## Features

- **Multi-threaded processing** with multiple Gemini API keys
- **Rate limiting** (30 RPM per key = 2 seconds between requests)
- **Error planting**: Modifies papers to introduce flaws based on descriptions
- **Placebo generation**: Creates sham surgery versions by learning writing style and rewriting original sections

## Requirements

Install required packages:
```bash
pip install pandas google-generativeai python-dotenv pydantic tqdm
```

## Environment Setup

Add your Gemini API keys to the `.env` file in the project root:
```
GEMINI_API_KEY_1=your_key_1
GEMINI_API_KEY_2=your_key_2
GEMINI_API_KEY_3=your_key_3
# ... add more keys as needed
```

## Usage

```bash
python plant_errors_and_placebo.py \
    --csv_file ../data/ICLR2024/filtered_pairs_with_human_scores.csv \
    --base_dir ../data/ICLR2024/latest \
    --output_dir ../data/ICLR2024 \
    --max_workers 3
```

### Arguments

- `--csv_file`: Path to the CSV file containing flaw descriptions (required)
- `--base_dir`: Base directory containing paper folders with `structured_paper_output/paper.md` files (required)
- `--output_dir`: Output directory where `planted_error/` and `sham_surgery/` folders will be created (default: same as base_dir)
- `--max_workers`: Maximum number of worker threads (default: number of API keys)

## Output Structure

The script creates the following structure:

```
output_dir/
├── planted_error/
│   ├── {paper_folder_name}/
│   │   ├── flaw_1.md
│   │   ├── flaw_2.md
│   │   └── ...
│   └── ...
├── sham_surgery/
│   ├── {paper_folder_name}/
│   │   ├── flaw_1.md
│   │   ├── flaw_2.md
│   │   └── ...
│   └── ...
└── planting_results.csv
```

## How It Works

1. **Error Planting**:
   - Reads flaw descriptions from the CSV
   - Uses Gemini to identify sections that need modification
   - Applies modifications to introduce the flaw
   - Saves the flawed version to `planted_error/`

2. **Placebo Generation**:
   - Extracts modified sections from the flawed paper
   - Analyzes the writing style of the modifications
   - Extracts original sections before modification
   - Rewrites original sections using the learned style (without dropping information)
   - Saves the placebo version to `sham_surgery/`

## Rate Limiting

The script automatically handles rate limiting:
- 30 RPM per API key = 2 seconds between requests
- Uses round-robin assignment to distribute load across keys
- Thread-safe rate limiting with locks

## Notes

- The script processes all flaw descriptions for each paper
- If error planting fails, the script continues to the next flaw
- If placebo generation fails, the planted error is still saved
- Results are saved to `planting_results.csv` with paths and metadata

