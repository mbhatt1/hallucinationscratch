# PCIB Detector Examples

This directory contains example scripts demonstrating how to use the PCIB detector.

## Prerequisites

```bash
# Install the package
pip install -e ..

# Set your OpenAI API key
export OPENAI_API_KEY=your_key_here
```

## Examples

### 1. Basic Usage ([`basic_usage.py`](basic_usage.py))

Simple detection with different types of answers:
- ✅ Grounded answer (supported by evidence)
- 🚨 Hallucinated answer (contradicts evidence)
- 💭 Opinion-based answer (no factual claims)

```bash
python basic_usage.py
```

### 2. Batch Processing ([`batch_processing.py`](batch_processing.py))

Process multiple answer-evidence pairs efficiently:

```bash
python batch_processing.py
```

### 3. Advanced Configuration ([`advanced_config.py`](advanced_config.py))

Demonstrates:
- Ensemble verification for better accuracy
- Custom threshold tuning
- Signal analysis and interpretation

```bash
python advanced_config.py
```

### 4. Multi-Provider Support ([`multi_provider.py`](multi_provider.py))

Compare detection across different LLM providers:
- OpenAI (GPT-4o-mini)
- Anthropic (Claude Sonnet)
- Google Gemini

```bash
# Requires API keys for all providers
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...

python multi_provider.py
```

### 5. Trace Validation ⭐ NEW ([`trace_validation_example.py`](trace_validation_example.py))

Advanced reasoning trace validation and rationalization detection:
- Enable trace validation to catch sophisticated hallucinations
- Detect post-hoc rationalization (fabricated reasoning)
- Measure trace consistency and conclusion support
- Compare detection with/without trace validation

```bash
python trace_validation_example.py
```

**Features:**
- **Trace Consistency**: Checks logical coherence of reasoning
- **Trace Support**: Verifies reasoning supports conclusion
- **Rationalization Detection**: Compares forward vs backward reasoning traces
- **Cost**: ~3x API calls per claim (use selectively for high-stakes verification)

See [TRACE_VALIDATION.md](../TRACE_VALIDATION.md) for detailed documentation.

## Expected Output

Each example will show:
- **Flagged**: Whether hallucination was detected
- **Score**: Numerical risk score (0-10+, higher = more risky)
- **Claims**: Extracted atomic claims and their individual scores
- **Signals**: PC+IB signals (uptake, stress, conflict)

## Notes

- **Response API**: These examples use OpenAI's Responses API with structured outputs
- **Rate Limits**: The detector respects OpenAI rate limits with automatic backoff
- **Cost**: Typical cost is ~$0.001-0.003 per answer (with gpt-4o-mini)
