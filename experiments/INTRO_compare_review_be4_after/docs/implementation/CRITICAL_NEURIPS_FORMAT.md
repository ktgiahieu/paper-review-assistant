# CriticalNeurIPS Review Format

## Overview

The **CriticalNeurIPS** format implements a sophisticated, multi-faceted review template designed to produce exceptionally thorough, incisive, and constructive critiques. This format synthesizes multiple expert perspectives into a single, coherent review that challenges papers from conceptual, historical, and methodological angles.

## Key Features

### 1. Multi-Faceted Critical Approach

The reviewer adopts two complementary personas:

#### **The Conceptual Critic & Historian**
- Questions core concepts from first principles
- Situates the paper within broader scientific landscape
- Challenges authors' framing with evidence from foundational literature
- Provides constructive citation roadmap for improvement

#### **The Methodological Skeptic & Forensic Examiner**
- Forensically scrutinizes experimental design and methodology
- Identifies critical omissions and unstated assumptions
- Challenges validity of results based on methodological concerns
- Points out countervailing evidence not addressed

### 2. Structured JSON Output

The format produces a well-structured JSON review with these fields:

```json
{
  "summary": "Brief, neutral summary (not a critique)",
  "strengths_and_weaknesses": "Thorough assessment with Markdown formatting",
  "questions": "3-5 actionable questions that could change evaluation",
  "limitations_and_societal_impact": "Assessment of ethical considerations",
  "soundness": 1-4,           // Technical soundness
  "presentation": 1-4,        // Clarity and organization
  "contribution": 1-4,        // Significance and novelty
  "overall_score": 1-10,      // Overall recommendation
  "confidence": 1-5           // Reviewer confidence
}
```

### 3. Compatible Score Ranges

The scoring system aligns with standard academic review practices:

| Metric | Range | Description |
|--------|-------|-------------|
| **soundness** | 1-4 | 4=excellent, 3=good, 2=fair, 1=poor |
| **presentation** | 1-4 | 4=excellent, 3=good, 2=fair, 1=poor |
| **contribution** | 1-4 | 4=excellent, 3=good, 2=fair, 1=poor |
| **overall_score** | 1-10 | 10=Award quality, 8=Strong Accept, 6=Weak Accept, 5=Borderline, 4=Borderline reject, 2=Strong Reject, 1=Trivial/wrong |
| **confidence** | 1-5 | 5=Certain, 4=Confident, 3=Fairly confident, 2=Willing to defend, 1=Educated guess |

## Usage

### Auto-Detection

If your model name contains "criticalneurips", "critical-neurips", or "critical_neurips", the format is automatically detected:

```bash
python scripts/review/review_paper_pairs_vllm.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ./reviews_critical/ \
  --vllm_endpoint "http://localhost:8000/" \
  --model_name "CriticalNeurIPS-70B" \
  --num_runs 3
```

### Explicit Format Override

Use the `--format` flag to apply CriticalNeurIPS format to any model:

```bash
python scripts/review/review_paper_pairs_vllm.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ./reviews_critical_llama/ \
  --vllm_endpoint "http://localhost:8000/" \
  --model_name "Meta-Llama-3.1-70B-Instruct" \
  --format CriticalNeurIPS \
  --num_runs 3 \
  --verbose
```

### Advanced Options

```bash
# With multiple runs and limited papers for testing
python scripts/review/review_paper_pairs_vllm.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --output_dir ./reviews_critical_test/ \
  --vllm_endpoint "http://localhost:8000/" \
  --model_name "Qwen-72B-Instruct" \
  --format CriticalNeurIPS \
  --num_runs 5 \
  --limit 10 \
  --verbose
```

## Technical Implementation

### 1. Pydantic Model

```python
class CriticalNeurIPSReview(BaseModel):
    summary: str
    strengths_and_weaknesses: str
    questions: str
    limitations_and_societal_impact: str
    soundness: int (1-4)
    presentation: int (1-4)
    contribution: int (1-4)
    overall_score: int (1-10)
    confidence: int (1-5)
```

### 2. Timeout Configuration

CriticalNeurIPS reviews require more time due to their depth:

```python
MODEL_TIMEOUTS = {
    "CriticalNeurIPS": 600,  # 10 minutes
    # ... other models
}
```

This extended timeout (600 seconds = 10 minutes) accommodates:
- Deep conceptual analysis
- Extensive literature contextualization
- Thorough methodological scrutiny
- Comprehensive weakness identification

### 3. Parsing Logic

The implementation includes robust fallback parsing:

```python
try:
    # Strict Pydantic validation
    parsed_review = CriticalNeurIPSReview.model_validate_json(sanitized_json_content)
    review_data = parsed_review.model_dump()
except Exception as e:
    # Fallback to basic JSON parsing
    fallback_data = json.loads(sanitized_json_content)
    # Extract fields and provide defaults
```

### 4. Score Mapping

For compatibility with evaluation scripts:

```python
review_data["rating"] = review_data.get("overall_score")
# soundness, presentation, contribution already match standard names
```

## Output Format

### Successful Review

```json
{
  "summary": "This paper proposes a novel...",
  "strengths_and_weaknesses": "## Strengths\n\n1. The paper introduces...\n\n## Weaknesses\n\n1. The methodology assumes...",
  "questions": "1. How does the approach generalize to...?\n2. What is the computational complexity...?\n3. Can the authors clarify...",
  "limitations_and_societal_impact": "The paper adequately discusses computational limitations but could expand on potential biases in the training data that may affect fairness in deployment.",
  "soundness": 3,
  "presentation": 3,
  "contribution": 4,
  "overall_score": 7,
  "confidence": 4,
  "paper_id": "abc123",
  "version": "v1",
  "run_id": 0,
  "model_type": "CriticalNeurIPS",
  "success": true,
  "was_truncated": false,
  "chars_per_token_used": 3.0
}
```

### Key Output Features

1. **summary**: Neutral overview (not evaluative)
2. **strengths_and_weaknesses**: Combined assessment with Markdown formatting
3. **questions**: Actionable, specific queries for authors
4. **limitations_and_societal_impact**: Ethical and practical considerations
5. **Numerical scores**: Compatible with standard evaluation metrics

## Comparison with Other Formats

| Feature | CriticalNeurIPS | SEA-E | CycleReviewer | GenericStructured |
|---------|-----------------|-------|---------------|-------------------|
| **Approach** | Multi-faceted critical | Structured checklist | 4 reviewers + meta | Flexible JSON |
| **Strengths/Weaknesses** | Combined (Markdown) | Separate lists | Separate per reviewer | Separate lists |
| **Timeout** | 10 min | 5 min | 15 min | 5 min |
| **Scores** | 1-4 (S,P,C), 1-10 (R) | Text-based | Text-based | Text-based |
| **Unique Features** | Conceptual + methodological critique | Predefined format | Multiple perspectives | Generic template |
| **Best For** | Deep, critical analysis | Standardized reviews | Consensus building | General-purpose |

## Evaluation Compatibility

The CriticalNeurIPS format is fully compatible with:

### 1. Numerical Score Evaluation

```bash
python scripts/evaluation/evaluate_numerical_scores.py \
  --reviews_dir ./reviews_critical/ \
  --output_dir ./evaluation_critical/
```

**Extracted Metrics:**
- `soundness` (1-4)
- `presentation` (1-4)
- `contribution` (1-4)
- `rating` (1-10, from `overall_score`)

### 2. Flaw Detection Evaluation

```bash
python scripts/evaluation/evaluate_flaw_detection.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv \
  --reviews_dir ./reviews_critical/ \
  --output_dir ./flaw_detection_critical/
```

**Extracted Field:**
- `weaknesses` (from `strengths_and_weaknesses` field)

### 3. AI vs Human Comparison

```bash
python scripts/evaluation/calculate_mse_mae.py \
  --csv_file ./data/ICLR2024_pairs/filtered_pairs_with_human_scores.csv \
  --reviews_dir ./reviews_critical/ \
  --output_dir ./ai_vs_human_critical/
```

## Prompt Philosophy

### System Prompt

The CriticalNeurIPS system prompt establishes:

1. **High Standards**: "Top-tier academic reviewer for NeurIPS"
2. **Multi-Perspective**: Conceptual + Methodological angles
3. **Constructive Critique**: Challenge with evidence, not just criticism
4. **Scholarly Rigor**: Citations, counter-examples, foundational literature

### User Prompt

The user prompt emphasizes:

1. **Clarity**: Explicit JSON schema with exact field requirements
2. **Specificity**: Precise score ranges and meanings
3. **Actionability**: Questions should potentially change evaluation
4. **Balance**: Neutral summary, thorough assessment, constructive feedback

## Best Practices

### When to Use CriticalNeurIPS

✅ **Use when you want:**
- Deep, thorough critiques
- Conceptual and methodological challenges
- Evidence-based feedback with citations
- High-quality, publication-ready reviews

❌ **Avoid when you want:**
- Quick, lightweight reviews
- Multiple independent perspectives (use CycleReviewer)
- Highly standardized format (use SEA-E)
- Simple pass/fail decisions

### Recommended Models

This format works best with:

- **Large models** (70B+ parameters)
- **Instruction-tuned** models
- Models with **strong reasoning** capabilities
- Models trained on **academic/scientific** text

**Examples:**
- Meta-Llama-3.1-70B-Instruct
- Qwen-72B-Instruct
- Mixtral-8x22B-Instruct
- GPT-4 (via compatible API)

## Troubleshooting

### Issue: Reviews are too verbose

**Solution:** The format is designed for depth. If this is problematic:
- Use a smaller model
- Add token limits to the prompt
- Switch to GenericStructured format

### Issue: JSON parsing fails

**Solution:** The implementation includes robust sanitization:
- Automatic JSON cleanup
- Fallback to basic parsing
- Default values for missing fields

### Issue: Timeout errors

**Solution:** CriticalNeurIPS has a 10-minute timeout by default:
- Ensure your model can handle long outputs
- Check vLLM server configuration
- Consider reducing paper length (fewer figures)

### Issue: Scores seem harsh

**Solution:** This is intentional - the format encourages critical analysis:
- Reviews may score lower than other formats
- This is expected for thorough critique
- Compare within-format, not across formats

## Research Applications

### 1. Comparing Review Quality

Compare CriticalNeurIPS vs other formats:

```bash
# Generate reviews with different formats
python scripts/review/review_paper_pairs_vllm.py --format CriticalNeurIPS ...
python scripts/review/review_paper_pairs_vllm.py --format GenericStructured ...

# Evaluate both
python scripts/evaluation/evaluate_numerical_scores.py --reviews_dir ./reviews_critical/
python scripts/evaluation/evaluate_numerical_scores.py --reviews_dir ./reviews_generic/

# Compare flaw detection rates
python scripts/evaluation/evaluate_flaw_detection.py --reviews_dir ./reviews_critical/
python scripts/evaluation/evaluate_flaw_detection.py --reviews_dir ./reviews_generic/
```

### 2. Studying Reviewer Personas

Analyze how the critical persona affects:
- Score distributions
- Flaw detection recall
- Differentiation between v1 and latest versions
- Agreement with human reviewers

### 3. Model Capabilities

Test which models can:
- Follow complex multi-faceted instructions
- Provide evidence-based critiques
- Generate well-structured Markdown
- Maintain consistency across runs

## Example Output

### Sample Review

**Paper:** "Generative Sliced MMD Flows"

```json
{
  "summary": "This paper proposes Generative Sliced Maximum Mean Discrepancy (MMD) Flows using Riesz kernels to train generative models. The authors demonstrate that Riesz kernels enable the MMD to coincide with sliced MMD, reducing computational complexity from quadratic to near-linear time. They provide theoretical guarantees and empirical validation on image generation tasks.",
  
  "strengths_and_weaknesses": "## Strengths\n\n1. **Novel Theoretical Contribution**: The paper establishes a formal connection between Riesz kernels and sliced MMD that was previously unexplored in the literature, building on foundational work by Gretton et al. (2012) on kernel methods.\n\n2. **Computational Efficiency**: The proposed sorting algorithm for gradient computation represents a significant practical advancement, reducing complexity from O(n²) to O(n log n), making the approach viable for large-scale applications.\n\n3. **Rigorous Mathematical Treatment**: The authors provide formal proofs of convergence rates and error bounds, demonstrating that stochastic approximations converge at √(1/L) rate where L is the number of projections.\n\n## Weaknesses\n\n1. **Limited Theoretical Scope**: The analysis assumes global Lipschitz continuity and diagonal noise, which are restrictive assumptions that may not hold for many real-world distributions. The authors cite Tabak & Turner (2013) on continuous normalizing flows but do not adequately address how violations of these assumptions affect their theoretical guarantees.\n\n2. **Experimental Limitations**: The empirical evaluation is confined to relatively simple image datasets (MNIST, FashionMNIST, CIFAR10). There is no comparison with recent state-of-the-art generative models like diffusion models (Ho et al., 2020) or modern GAN architectures (Karras et al., 2020), making it difficult to assess practical competitiveness.\n\n3. **Unstated Modeling Assumptions**: The paper does not discuss how the choice of Riesz kernel parameter α affects model behavior, nor does it provide guidance for practitioners on selecting this hyperparameter. This omission undermines the practical applicability of the method.",
  
  "questions": "1. How does the approach perform when the global Lipschitz assumption is violated? Can the authors provide empirical evidence or theoretical bounds on degradation?\n\n2. What is the computational and memory overhead compared to recent diffusion models? A detailed complexity analysis including constant factors would strengthen the practical claims.\n\n3. Can the authors provide ablation studies showing sensitivity to the Riesz kernel parameter α across different data distributions?\n\n4. How does the method handle multimodal distributions where slicing may fail to capture complex dependencies? Are there theoretical or empirical failure modes?\n\n5. The paper mentions momentum MMD flows but provides limited analysis. What is the theoretical justification for faster convergence, and how does it relate to accelerated gradient methods in optimization literature?",
  
  "limitations_and_societal_impact": "The paper briefly mentions computational limitations but does not adequately address potential negative societal impacts. Generative models can be misused for creating deepfakes or generating misleading content. The authors should discuss: (1) potential misuse scenarios, (2) whether the improved efficiency makes harmful applications more accessible, and (3) possible technical safeguards. Additionally, the environmental cost of training large-scale generative models deserves mention given recent concerns about carbon footprint in ML research.",
  
  "soundness": 3,
  "presentation": 3,
  "contribution": 3,
  "overall_score": 6,
  "confidence": 4
}
```

## Summary

The CriticalNeurIPS format represents a sophisticated approach to AI-generated peer review that:

✅ Combines conceptual and methodological critique
✅ Produces evidence-based, scholarly feedback  
✅ Generates structured, evaluation-compatible output
✅ Encourages deep engagement with papers
✅ Provides actionable, constructive suggestions

It's ideal for researchers who want to push LLMs to produce the highest-quality, most rigorous reviews possible.

