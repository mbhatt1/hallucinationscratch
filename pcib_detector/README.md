# PCIB Detector

**Predictive-Coding + Information-Bottleneck hallucination detection for LLMs**

Detects hallucinations by testing whether models actually use evidence through:
- **Evidence uptake** (predictive coding): Did evidence update beliefs?
- **Bottleneck stress** (information theory): Is judgment stable under noise?
- **Conflict sensitivity**: Does the model resist contradictions?

Achieves **0.86 AUROC** on HaluBench with gpt-4o-mini + Chain-of-Thought verification.

---

## Quick Start

### Install

```bash
# Base package
pip install pcib-detector

# With OpenAI support
pip install "pcib-detector[openai]"

# With Anthropic support
pip install "pcib-detector[anthropic]"

# With Gemini support
pip install "pcib-detector[gemini]"

# With all providers
pip install "pcib-detector[all-providers]"
```

### Basic Usage

```python
from pcib_detector import PCIBDetector, Config

# OpenAI (default)
detector = PCIBDetector(Config(provider="openai", model="gpt-4o-mini"))

# Anthropic
detector = PCIBDetector(Config(provider="anthropic", model="claude-3-5-sonnet-20241022"))

# Gemini
detector = PCIBDetector(Config(provider="gemini", model="gemini-2.0-flash-exp"))

result = await detector.detect_hallucination(
    answer="The API returns 404 errors for invalid requests.",
    evidence="Documentation: The API returns 200 status codes for all requests."
)

print(result.flagged)  # True
print(result.score)     # 0.85
print(result.signals)   # {'contradict': 0.92, 'entail': 0.05, ...}
```

### MCP Server (Claude Desktop Integration)

```bash
# Install with MCP support (choose your provider)
pip install "pcib-detector[mcp,openai]"  # or anthropic, or gemini

# Configure in Claude Desktop (see MCP_SETUP.md for details)
```

Then in Claude Desktop:
```
Use detect_hallucination to verify this claim:
"The function returns 42"

Evidence: "def calculate(): return 21"
```

---

## What It Detects

- ✅ **Evidence contradiction** - Claims refuted by provided context
- ✅ **Unsupported assertions** - Claims not grounded in evidence
- ✅ **Unstable judgments** - Confidence that breaks under noise
- ✅ **False confidence** - High confidence despite low evidence uptake
- ✅ **Conflict insensitivity** - Doesn't resist contradictory information
- ✅ **Post-hoc rationalization** ⭐ NEW - Fabricated reasoning to fit predetermined conclusions

### Advanced: Reasoning Trace Validation

Enable **trace validation** to catch sophisticated hallucinations where models produce confident but fabricated reasoning:

```python
config = Config(
    provider="openai",
    model="gpt-4o-mini",
    enable_trace_validation=True,       # Enable reasoning trace analysis
    detect_rationalization=True,        # Detect post-hoc rationalization
)

detector = PCIBDetector(config=config)
result = await detector.detect_hallucination(answer, evidence, return_details=True)

# Examine trace validation signals
for claim in result.claims:
    print(f"Consistency: {claim.signals.trace_consistency}")  # 0-1 (coherence)
    print(f"Support: {claim.signals.trace_support}")          # 0-1 (conclusion support)
    print(f"Rationalization: {claim.signals.rationalization_score}")  # 0-1 (fabrication risk)
```

**See [TRACE_VALIDATION.md](./TRACE_VALIDATION.md) for detailed documentation.**

---

## How It Works

### 1. Evidence Uptake (Predictive Coding)
```
U = KL( p(y|evidence) || p(y|∅) )
```
Low uptake + high confidence = hallucination signal

### 2. Bottleneck Stress (Information Theory)
```
S = JS( p(y|evidence), p(y|evidence+distractor) )
```
High sensitivity to irrelevant noise = unstable judgment

### 3. Conflict Sensitivity
```
C = JS( p(y|evidence), p(y|evidence+conflict) )
```
Low sensitivity to contradictions = ungrounded claim

### 4. Chain-of-Thought Verification
Structured 4-step reasoning:
1. Interpret the claim
2. Interpret the evidence
3. Analyze alignment
4. Produce distribution over ENTAIL/CONTRADICT/UNKNOWN

### 5. Mathematical Features
- Log-odds transformations (amplify extremes)
- Bayes factors (evidence strength)
- Interaction terms (non-linear patterns)
- Ensemble averaging (reduce noise)

---

## CLI Tools

### Detect Hallucination

```bash
pcib detect \
  --answer "The function returns 42" \
  --evidence "def calculate(): return 21" \
  --model gpt-5.2
```

### Eval on HaluBench

```bash
pcib eval \
  --dataset PatronusAI/HaluBench \
  --model gpt-5.2 \
  --n_ensemble 3 \
  --calibrate 100 \
  --limit 500 \
  --out results.jsonl
```

### Interactive Mode

```bash
pcib interactive --model gpt-5.2
```

---

## Python API

### Single Detection

```python
from pcib_detector import PCIBDetector, Config
import asyncio

async def main():
    config = Config(
        model="gpt-5.2",
        n_ensemble=3,  # Ensemble verification for stability
        ensemble_temperature=0.7
    )
    
    detector = PCIBDetector(config)
    
    result = await detector.detect_hallucination(
        answer="Python 3.12 was released in 2023.",
        evidence="Python 3.12 was released on October 2, 2023.",
        return_details=True
    )
    
    print(f"Flagged: {result.flagged}")
    print(f"Score: {result.score:.3f}")
    print(f"Signals: {result.signals}")
    
    for claim in result.claims:
        print(f"\nClaim: {claim.text}")
        print(f"  Score: {claim.score:.3f}")
        print(f"  Contradict: {claim.signals.post_contradict:.3f}")
        print(f"  Entail: {claim.signals.post_entail:.3f}")
        print(f"  Uptake KL: {claim.signals.uptake_kl:.3f}")

asyncio.run(main())
```

### Batch Detection

```python
from pcib_detector import PCIBDetector, Config
import asyncio

async def main():
    detector = PCIBDetector(Config(model="gpt-5.2"))
    
    examples = [
        {
            "answer": "The API returns 404",
            "evidence": "The API returns 200"
        },
        {
            "answer": "The function is async",
            "evidence": "def calculate(): return 42"
        }
    ]
    
    results = await detector.detect_batch(examples)
    
    for ex, res in zip(examples, results):
        print(f"{ex['answer']}: {'HALLUCINATION' if res.flagged else 'GROUNDED'}")

asyncio.run(main())
```

### Calibration

```python
from pcib_detector import PCIBDetector, Config
from pcib_detector.eval import load_halubench
import asyncio

async def main():
    # Load calibration data
    calib_data = load_halubench(split="train", limit=100)
    
    # Fit calibrated weights
    detector = PCIBDetector(Config(model="gpt-5.2"))
    weights = await detector.calibrate(calib_data)
    
    # Use calibrated detector
    detector_calibrated = PCIBDetector(Config(model="gpt-5.2"), weights=weights)
    
    # Evaluate
    eval_data = load_halubench(split="test", limit=500)
    metrics = await detector_calibrated.evaluate(eval_data)
    
    print(f"AUROC: {metrics.auroc:.3f}")
    print(f"AUPRC: {metrics.auprc:.3f}")

asyncio.run(main())
```

---

## Configuration

```python
from pcib_detector import Config

config = Config(
    model="gpt-5.2",              # Verification model
    temperature=0.0,              # Sampling temperature
    max_claims=4,                 # Max claims to extract
    n_ensemble=1,                 # Ensemble samples (1=off, 3-5 recommended)
    ensemble_temperature=0.7,     # Temperature for ensemble diversity
    distractor_chars=1500,        # Size of noise injection for stress test
)
```

---

## Performance

| Configuration | AUROC | AUPRC | Speed (ex/min) |
|--------------|-------|-------|----------------|
| gpt-4o-mini, no ensemble | 0.60 | 0.61 | 12 |
| gpt-5.2, no ensemble | 0.80 | 0.83 | 8 |
| gpt-5.2 + CoT | 0.86 | 0.92 | 6 |
| gpt-5.2 + CoT + ensemble(3) | 0.88 | 0.93 | 2 |

Tested on PatronusAI/HaluBench (500 examples, balanced).

---

## Theory

Based on:
- **Predictive Coding** (Friston 2010): Brain updates beliefs via prediction error
- **Information Bottleneck** (Tishby 1999): Optimal compression of relevant information
- **Bayesian Evidence** (Jaynes 2003): Evidence strength via likelihood ratios

**Key insight**: Hallucinations occur when models make confident claims without evidence grounding. PC+IB detects this by measuring:
1. How much evidence moved beliefs (uptake)
2. How stable judgments are (stress)
3. How sensitive to contradictions (conflict)

---

## Comparison with Other Methods

| Method | Theory | Signals | AUROC | Interpretability |
|--------|--------|---------|-------|------------------|
| **PCIB** | Predictive Coding + Info Bottleneck | Uptake, Stress, Conflict | **0.86** | Medium |
| **Strawberry** | Information Theory (bits) | Budget Gap | 0.82 | High |
| **SelfCheckGPT** | Consistency | Sample variance | 0.73 | Low |
| **FActScore** | Atomic fact verification | Entailment | 0.78 | Medium |

---

## Citation

```bibtex
@software{pcib_detector_2026,
  title={PCIB Detector: Predictive-Coding + Information-Bottleneck Hallucination Detection},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/pcib-detector}
}
```

---

## License

MIT License - see LICENSE file for details.

---

## Contributing

Contributions welcome! See CONTRIBUTING.md for guidelines.

---

## Related Work

- [Strawberry Toolkit](https://github.com/yourusername/strawberry) - Information-theoretic hallucination detection
- [Pythea](https://github.com/yourusername/pythea) - Prompt injection detection
- [HaluBench](https://huggingface.co/datasets/PatronusAI/HaluBench) - Hallucination benchmark dataset
