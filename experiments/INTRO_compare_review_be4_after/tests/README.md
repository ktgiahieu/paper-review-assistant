# Tests Directory

Test scripts and test output data.

## Test Scripts

### Setup Tests
- **`test_fetch_human_scores.py`** - Verify OpenReview API connectivity and score extraction
- **`test_flaw_detection_setup.py`** - Verify evaluator LLM setup for flaw detection
- **`test_seae_parser_comprehensive.py`** - Test SEA-E parser with various formats

## Test Output Directories

### `reviews_test/`
Initial test reviews (small sample)

### `reviews_test_claude_haiku/`
Test reviews using Claude Haiku model

### `reviews_test_claude_sonnet/`
Test reviews using Claude Sonnet model

### `reviews_test_claude_sonnet_2/`
Additional Claude Sonnet test reviews

## Running Tests

### Test OpenReview API
```bash
python test_fetch_human_scores.py
```

Expected output: Successful connection and score extraction from a sample paper.

### Test Flaw Detection Setup
```bash
python test_flaw_detection_setup.py \
  --evaluator_endpoint "http://localhost:8000" \
  --evaluator_model "Qwen3-30B-A3B-Instruct-2507-FP8"
```

Expected output: Evaluator correctly identifies flaws in sample reviews.

### Test SEA-E Parser
```bash
python test_seae_parser_comprehensive.py
```

Expected output: Parser correctly handles various bullet point styles and formats.

## Test Data

Test review outputs are kept here for:
- **Regression testing** - Ensure changes don't break existing functionality
- **Format validation** - Verify output formats are correct
- **Quick testing** - Small datasets for fast iteration

## Cleanup

Test outputs can be deleted to save space:
```bash
rm -rf reviews_test*
```

But keep the test scripts! They're useful for validation.

## Adding New Tests

When adding new features:
1. Create a test script (`test_*.py`)
2. Add test data to appropriate subdirectory
3. Document expected behavior
4. Update this README

