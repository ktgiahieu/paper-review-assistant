# Plant Abstract Manipulation

This script creates two manipulated versions of research paper abstracts:

1. **Good version**: Rewrites abstracts with buzzwords and superlatives, makes them twice as long, and adds code availability statements with GitHub links.
2. **Bad version**: Removes buzzwords and superlatives, makes abstracts concise (75% length), uses old scientific writing style (2000s), and removes code mentions.

## Requirements

- Python 3.7+
- `google-generativeai` package
- Multiple Gemini API keys (supports up to 9 keys for parallel processing)
- Environment variables: `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, etc. (or single `GEMINI_API_KEY`)

## Installation

```bash
pip install google-generativeai pandas tqdm pydantic python-dotenv
```

## Setup

Set up your Gemini API keys in your `.env` file or environment variables:

```bash
# Multiple keys for parallel processing (recommended)
GEMINI_API_KEY_1=your_key_1
GEMINI_API_KEY_2=your_key_2
GEMINI_API_KEY_3=your_key_3

# Or single key (sequential processing)
GEMINI_API_KEY=your_key
```

## Usage

### Basic Usage

Process all papers in the `latest/` folder:

```bash
python plant_abstract_manipulation.py \
    --base_dir ../../data/ICLR2024
```

### Process Specific Papers

```bash
python plant_abstract_manipulation.py \
    --base_dir ../../data/ICLR2024 \
    --paper_ids ViNe1fjGME 0akLDTFR9x AbXGwqb5Ht
```

### Process Papers from CSV

```bash
python plant_abstract_manipulation.py \
    --base_dir ../../data/ICLR2024 \
    --paper_ids_file ../../data/ICLR2024/filtered_pairs.csv
```

### Custom Output Directories

```bash
python plant_abstract_manipulation.py \
    --base_dir ../../data/ICLR2024 \
    --output_good_dir ../../data/ICLR2024/abstract_good \
    --output_bad_dir ../../data/ICLR2024/abstract_bad
```

### Advanced Options

```bash
python plant_abstract_manipulation.py \
    --base_dir ../../data/ICLR2024 \
    --model_name gemini-2.5-flash \
    --max_workers 5 \
    --verbose
```

## Arguments

- `--base_dir`: **Required**. Base directory containing the `latest/` folder with papers
- `--output_good_dir`: Output directory for good abstract versions (default: `base_dir/abstract_manipulation_good`)
- `--output_bad_dir`: Output directory for bad abstract versions (default: `base_dir/abstract_manipulation_bad`)
- `--paper_ids`: Optional list of specific paper IDs to process
- `--paper_ids_file`: Optional CSV file with paper IDs in first column
- `--model_name`: Gemini model name (default: `gemini-2.0-flash-lite`)
- `--max_workers`: Maximum number of worker threads (default: number of API keys)
- `--verbose`: Enable verbose output

## Output Structure

The script creates two directories:

```
base_dir/
├── abstract_manipulation_good/
│   └── paperid_arxiv_id/
│       └── structured_paper_output/
│           └── paper.md  (with enhanced abstract)
└── abstract_manipulation_bad/
    └── paperid_arxiv_id/
        └── structured_paper_output/
            └── paper.md  (with concise, old-style abstract)
```

## What Gets Modified

### Good Version
- ✅ Adds buzzwords and superlatives (state-of-the-art, groundbreaking, revolutionary, etc.)
- ✅ Makes abstract approximately **twice as long** as original
- ✅ Adds code availability line: "Code is available at https://github.com/..."
- ✅ Uses engaging, marketing-style language

### Bad Version
- ✅ Removes all buzzwords and superlatives
- ✅ Makes abstract approximately **75% of original length**
- ✅ Uses old scientific writing style (2000s era)
- ✅ Removes all code mentions and GitHub links
- ✅ Uses direct, formal, factual language

## Notes

- The script preserves the entire paper structure (figures, other sections, etc.) and only modifies the abstract
- If multiple API keys are provided, processing happens in parallel for faster execution
- Rate limiting is automatically handled based on the model's RPM limits
- The script includes retry logic for failed API calls (up to 3 attempts)

