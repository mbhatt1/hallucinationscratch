# Quick Start Guide

**Get started with PCIB hallucination detection in 5 minutes**

## Prerequisites

- Python 3.8 or higher
- An OpenAI API key (or Anthropic/Gemini)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/pcib-detector.git
cd pcib-detector

# Install package
cd pcib_detector
pip install -e .

# Or install directly from PyPI (when published)
pip install pcib-detector
```

## Set Up API Key

```bash
# Option 1: Environment variable (recommended)
export OPENAI_API_KEY=sk-your-key-here

# Option 2: In your Python code
import os
os.environ["OPENAI_API_KEY"] = "sk-your-key-here"

# Option 3: Pass directly to Config
config = Config(api_key="sk-your-key-here")
```

## Your First Detection

Create a file called `first_detection.py`:

```python
from pcib_detector import PCIBDetector, Config
import asyncio

async def main():
    # Initialize detector
    detector = PCIBDetector(Config(
        provider="openai",
        model="gpt-4o-mini"
    ))
    
    # Check a claim
    result = await detector.detect_hallucination(
        answer="Paris is the capital of Germany.",
        evidence="Berlin is the capital and largest city of Germany."
    )
    
    # Print results
    print(f"🚨 Hallucination detected: {result.flagged}")
    print(f"📊 Confidence score: {result.score:.3f}")
    
    if result.claims:
        print("\nClaim analysis:")
        for claim in result.claims:
            print(f"  • {claim.text}")
            print(f"    Score: {claim.score:.3f}")
            print(f"    Signals:")
            print(f"      - Contradict: {claim.signals.post.contradict:.2f}")
            print(f"      - Entail: {claim.signals.post.entail:.2f}")
            print(f"      - Evidence uptake: {claim.signals.uptake_kl:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
python first_detection.py
```

Expected output:

```
🚨 Hallucination detected: True
📊 Confidence score: 0.892

Claim analysis:
  • Paris is the capital of Germany
    Score: 0.892
    Signals:
      - Contradict: 0.95
      - Entail: 0.03
      - Evidence uptake: 1.234
```

## Common Use Cases

### 1. Check Multiple Examples

```python
from pcib_detector import PCIBDetector, Config
import asyncio

async def check_batch():
    detector = PCIBDetector(Config(provider="openai"))
    
    examples = [
        {
            "answer": "The Eiffel Tower is 324 meters tall.",
            "evidence": "The Eiffel Tower stands at 330 meters including antennas."
        },
        {
            "answer": "Python was created in 1991.",
            "evidence": "Python was first released in February 1991."
        }
    ]
    
    results = await detector.detect_batch(examples)
    
    for i, result in enumerate(results):
        print(f"\nExample {i+1}:")
        print(f"  Flagged: {result.flagged}")
        print(f"  Score: {result.score:.3f}")

asyncio.run(check_batch())
```

### 2. Enable Trace Validation

```python
config = Config(
    provider="openai",
    model="gpt-4o-mini",
    enable_trace_validation=True,      # ✅ Enable reasoning validation
    detect_rationalization=True         # ✅ Detect post-hoc justification
)

detector = PCIBDetector(config)

result = await detector.detect_hallucination(
    answer="Your answer here",
    evidence="Your evidence here",
    return_details=True  # ✅ Get detailed trace metrics
)

# Access trace validation metrics
for claim in result.claims:
    print(f"Trace consistency: {claim.signals.trace_consistency:.2f}")
    print(f"Rationalization score: {claim.signals.rationalization_score:.2f}")
```

### 3. Use Different Providers

```python
# OpenAI
detector_openai = PCIBDetector(Config(
    provider="openai",
    model="gpt-4o-mini"
))

# Anthropic Claude
detector_claude = PCIBDetector(Config(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022"
))

# Google Gemini
detector_gemini = PCIBDetector(Config(
    provider="gemini",
    model="gemini-1.5-flash"
))
```

### 4. Adjust Detection Sensitivity

```python
config = Config(
    provider="openai",
    max_claims=8,              # Check more claims (default: 4)
    distractor_chars=2000,     # Larger bottleneck test (default: 1500)
    n_ensemble=3,              # Ensemble verification (default: 1)
    temperature=0.1            # Lower temperature = more consistent (default: 0.0)
)

detector = PCIBDetector(config)
```

### 5. Command Line Usage

```bash
# Evaluate a dataset
pcib-eval \
    --dataset PatronusAI/HaluBench \
    --limit 100 \
    --model gpt-4o-mini \
    --output results.jsonl

# Run ablation study
python ablation_study.py \
    --limit 500 \
    --model gpt-4o-mini
```

## Understanding the Output

### DetectionResult Fields

```python
result = await detector.detect_hallucination(...)

# Main fields
result.flagged         # bool: True if hallucination detected
result.score          # float: 0-1, higher = more likely hallucination
result.claims         # List[ClaimResult]: Per-claim analysis
result.signals        # dict: Aggregate signals (if return_details=True)
```

### ClaimResult Fields

```python
claim = result.claims[0]

claim.text            # str: The extracted claim
claim.score           # float: Claim-level hallucination score
claim.flagged         # bool: Whether this claim is flagged
claim.signals         # ClaimSignals: Detailed signals
```

### Signal Interpretation

| Signal | Range | Interpretation |
|--------|-------|----------------|
| `post.contradict` | 0-1 | Evidence contradicts claim (higher = more contradiction) |
| `post.entail` | 0-1 | Evidence supports claim (lower = less support) |
| `uptake_kl` | 0-∞ | Evidence uptake strength (very low = prior-driven) |
| `stress_js` | 0-1 | Bottleneck sensitivity (high = unstable) |
| `conflict_js` | 0-1 | Contradiction sensitivity (low = ignoring conflicts) |
| `trace_consistency` | 0-1 | Reasoning consistency (low = inconsistent) |
| `rationalization_score` | 0-1 | Post-hoc justification likelihood (high = rationalized) |

**Red flags for hallucination**:
- High `contradict` (> 0.5)
- Low `entail` (< 0.3)
- Very low `uptake_kl` (< 0.1) with high confidence
- High `stress_js` (> 0.15)
- Low `conflict_js` (< 0.1) when confident
- Low `trace_consistency` (< 0.5)
- High `rationalization_score` (> 0.4)

## Configuration Reference

### Basic Config

```python
config = Config(
    provider="openai",           # "openai", "anthropic", or "gemini"
    model="gpt-4o-mini",        # Model identifier
    api_key=None,               # API key (or use env var)
    temperature=0.0,            # Sampling temperature
    max_concurrent=10           # Concurrent API calls
)
```

### Advanced Config

```python
config = Config(
    # Core settings
    max_claims=4,                      # Max claims to extract
    distractor_chars=1500,             # Bottleneck test size
    
    # Trace validation
    enable_trace_validation=True,      # Enable reasoning validation
    detect_rationalization=True,       # Detect post-hoc justification
    trace_temperature=0.3,             # Trace generation temperature
    
    # Ensemble
    n_ensemble=1,                      # Number of verification samples
    ensemble_temperature=0.3           # Ensemble sampling temperature
)
```

## Next Steps

1. **📚 Read the full documentation**: [`README.md`](README.md)
2. **🔧 Explore examples**: [`pcib_detector/examples/`](pcib_detector/examples/)
3. **🧪 Run ablation study**: [`ABLATION_STUDY.md`](ABLATION_STUDY.md)
4. **🔌 Try different providers**: [`PROVIDERS.md`](pcib_detector/PROVIDERS.md)
5. **🤖 Set up MCP**: [`MCP_SETUP.md`](pcib_detector/MCP_SETUP.md)
6. **🧠 Learn trace validation**: [`TRACE_VALIDATION.md`](pcib_detector/TRACE_VALIDATION.md)

## Troubleshooting

### Import Error

```
ModuleNotFoundError: No module named 'pcib_detector'
```

**Solution**: Install the package:
```bash
cd pcib_detector
pip install -e .
```

### API Key Error

```
AuthenticationError: Incorrect API key provided
```

**Solution**: Check your API key:
```bash
echo $OPENAI_API_KEY
# Should print: sk-...
```

### Rate Limit Error

```
RateLimitError: Rate limit reached
```

**Solution**: Reduce concurrency:
```python
config = Config(max_concurrent=3)  # Down from 10
```

### Timeout Error

```
asyncio.TimeoutError
```

**Solution**: Increase timeout (in custom code):
```python
result = await asyncio.wait_for(
    detector.detect_hallucination(...),
    timeout=600.0  # 10 minutes
)
```

## Getting Help

- **Documentation**: See [`README.md`](README.md) and [`pcib_detector/README.md`](pcib_detector/README.md)
- **Examples**: Check [`pcib_detector/examples/`](pcib_detector/examples/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/pcib-detector/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/pcib-detector/discussions)

## What's Next?

Ready for more advanced usage? Check out:

- **[Complete Examples](pcib_detector/examples/README.md)** - Batch processing, calibration, custom configs
- **[Trace Validation](pcib_detector/TRACE_VALIDATION.md)** - Deep dive into reasoning validation
- **[Multi-Provider Setup](pcib_detector/PROVIDERS.md)** - Use Anthropic and Gemini
- **[Ablation Study](ABLATION_STUDY.md)** - Generate paper-ready evaluation results

---

**Happy detecting! 🔍**
