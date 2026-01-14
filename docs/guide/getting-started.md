# Getting Started

Get up and running with PCIB Detector in minutes.

## Installation

::: code-group
```bash [pip]
pip install pcib-detector
```

```bash [from source]
git clone https://github.com/mbhatt1/hallucinationscratch.git
cd hallucinationscratch/pcib_detector
pip install -e .
```
:::

## Prerequisites

- **Python 3.8+**
- **API Key** from at least one provider:
  - [OpenAI API Key](https://platform.openai.com/api-keys)
  - [Anthropic API Key](https://console.anthropic.com/)
  - [Google AI API Key](https://makersuite.google.com/app/apikey)

## Environment Setup

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

Or create a `.env` file:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

## Your First Detection

Create a file `example.py`:

```python
from pcib_detector import PCIBDetector, Config
import asyncio

async def main():
    # Initialize detector
    config = Config(
        provider="openai",
        model="gpt-4o-mini",
        enable_trace_validation=True
    )
    detector = PCIBDetector(config)
    
    # Example: Check a factual statement
    result = await detector.detect_hallucination(
        answer="The Eiffel Tower was completed in 1889.",
        evidence="The Eiffel Tower opened on March 31, 1889."
    )
    
    print(f"🎯 Hallucination detected: {result.flagged}")
    print(f"📊 Confidence score: {result.score:.3f}")
    print(f"🔍 Signals:")
    print(f"  - Evidence Uptake: {result.signals.uptake:.3f}")
    print(f"  - Bottleneck Stress: {result.signals.stress:.3f}")
    print(f"  - Conflict: {result.signals.conflict:.3f}")
    
    # Print individual claim scores
    if result.claims:
        print(f"\n📝 Claims analyzed:")
        for i, claim in enumerate(result.claims, 1):
            print(f"  {i}. [{claim.score:.3f}] {claim.text[:60]}...")

asyncio.run(main())
```

Run it:

```bash
python example.py
```

## Expected Output

```
🎯 Hallucination detected: False
📊 Confidence score: 0.127
🔍 Signals:
  - Evidence Uptake: 0.823
  - Bottleneck Stress: 0.156
  - Conflict: 0.089

📝 Claims analyzed:
  1. [0.112] The Eiffel Tower was completed in 1889.
```

::: tip Interpretation
- **Low score (< 0.5)**: Factual, well-grounded answer
- **High score (> 0.5)**: Potential hallucination detected
- **High Uptake**: Answer strongly depends on evidence (good!)
- **Low Stress**: Answer stable under perturbation (good!)
- **Low Conflict**: No contradictions found (good!)
:::

## Next Steps

<div class="tip custom-block" style="padding-top: 8px">

- [Configuration →](/guide/configuration) - Customize detection behavior
- [Multi-Signal Detection →](/guide/multi-signal) - Understand the signals
- [Batch Processing →](/guide/batch-processing) - Process multiple examples
- [Provider Setup →](/guide/providers) - Use different LLM backends

</div>

## Common Patterns

### Detect with Context

```python
result = await detector.detect_hallucination(
    question="When was the Eiffel Tower built?",
    answer="It was completed in 1887.",  # Wrong year!
    evidence="The Eiffel Tower opened on March 31, 1889."
)

print(result.flagged)  # True - hallucination detected
```

### Batch Processing

```python
examples = [
    {"answer": "Paris is the capital of France.", "evidence": "..."},
    {"answer": "The Earth is flat.", "evidence": "..."},
    # ... more examples
]

results = await detector.detect_batch(examples)
for result in results:
    print(f"{result.answer[:30]}... -> {result.flagged}")
```

### Custom Configuration

```python
config = Config(
    provider="anthropic",
    model="claude-3-sonnet-20240229",
    enable_trace_validation=True,
    max_concurrent=5,
    temperature=0.0
)
```

## Troubleshooting

### API Key Not Found

```
Error: OpenAI API key not found
```

**Solution**: Set the environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

### Rate Limit Errors

```
Error: Rate limit exceeded
```

**Solution**: Reduce concurrency:
```python
config = Config(max_concurrent=1)
```

### Timeout Errors

```
Error: Request timeout
```

**Solution**: Increase timeout in config:
```python
config = Config(timeout=120.0)  # 2 minutes
```
