# Documentation Directory

Comprehensive documentation for the paper review comparison system.

## Directory Structure

### 📖 `guides/` - User Guides
Step-by-step guides for using the system:

- **`VLLM_GUIDE.md`** - Complete guide for using vLLM-based review generation
- **`EVALUATION_GUIDE.md`** - Guide for evaluating numerical scores
- **`EVALUATION_PLOTS_GUIDE.md`** - Understanding evaluation plots
- **`RETRY_GUIDE.md`** - How to retry failed reviews
- **`FLAW_DETECTION_GUIDE.md`** - Evaluating consensus flaw detection
- **`AI_VS_HUMAN_GUIDE.md`** - Comparing AI with human review scores
- **`MODEL_FORMATS.md`** - Supported model formats (SEA-E, CycleReviewer, etc.)
- **`CONTEXT_LENGTH_MANAGEMENT.md`** - How context truncation works
- **`TIMEOUT_CONFIGURATION.md`** - Configuring model-specific timeouts

### 🔧 `implementation/` - Technical Documentation
Implementation details and summaries:

- **`SEA_E_IMPLEMENTATION_SUMMARY.md`** - SEA-E model format implementation
- **`CYCLEREVIEW_IMPLEMENTATION.md`** - CycleReviewer format implementation
- **`CYCLEREVIEW_CHANGES_SUMMARY.md`** - Changes for CycleReviewer support
- **`GENERIC_STRUCTURED_SUMMARY.md`** - GenericStructured format
- **`EVALUATION_IMPLEMENTATION_SUMMARY.md`** - Evaluation system design
- **`RETRY_IMPLEMENTATION_SUMMARY.md`** - Retry mechanism details
- **`PLOTTING_IMPLEMENTATION_SUMMARY.md`** - Plotting feature implementation
- **`AI_VS_HUMAN_SUMMARY.md`** - AI vs Human comparison system
- **`FLAW_DETECTION_SUMMARY.md`** - Flaw detection evaluation design
- **`IMPLEMENTATION_COMPLETE_SUMMARY.md`** - Complete feature overview

### 🐛 `fixes/` - Bug Fixes & Updates
Documentation of bug fixes and improvements:

- **`ADAPTIVE_TRUNCATION.md`** - Adaptive token estimation
- **`CONTEXT_TRUNCATION_SUMMARY.md`** - Smart truncation strategy
- **`SMART_TRUNCATION_SUMMARY.md`** - 3-stage truncation details
- **`SEAE_PARSER_IMPROVEMENTS.md`** - SEA-E parser enhancements
- **`AUTO_RETRY_SUMMARY.md`** - Automatic retry mechanism
- **`ANTHROPIC_MULTIRUN_SUMMARY.md`** - Multi-run support for Anthropic
- **`DEFAULT_FORMAT_FIX.md`** - Default format score validation fix

## Quick Reference

### For New Users
Start here:
1. **Main README** (in project root)
2. **`guides/VLLM_GUIDE.md`** or Anthropic documentation
3. **`guides/EVALUATION_GUIDE.md`**

### For Developers
See:
- **`implementation/`** for technical details
- **`fixes/`** for understanding bug fixes and improvements

### For Troubleshooting
Check:
- **`guides/RETRY_GUIDE.md`** - Handling failed reviews
- **`guides/CONTEXT_LENGTH_MANAGEMENT.md`** - Context length issues
- **`fixes/`** directory - Known issues and fixes

## Documentation Standards

### User Guides
- **Purpose**: Help users accomplish specific tasks
- **Format**: Step-by-step instructions with examples
- **Audience**: Researchers and practitioners

### Implementation Docs
- **Purpose**: Explain technical design and code structure
- **Format**: Architecture, code snippets, design decisions
- **Audience**: Developers and contributors

### Fix Documentation
- **Purpose**: Document bugs and their solutions
- **Format**: Problem, solution, code changes
- **Audience**: Maintainers and advanced users

## Contributing

When adding new features:
1. Add a user guide in `guides/`
2. Add implementation details in `implementation/`
3. Document any bug fixes in `fixes/`
4. Update the main README with quick start examples

