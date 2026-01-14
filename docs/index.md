---
layout: home

hero:
  name: PCIB Detector
  text: Hallucination Detection for LLMs
  tagline: Predictive Coding + Information Bottleneck framework for detecting hallucinations in Large Language Model outputs
  image:
    src: /logo.svg
    alt: PCIB Detector
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: View on GitHub
      link: https://github.com/mbhatt1/hallucinationscratch
    - theme: alt
      text: Research Paper
      link: /research/paper

features:
  - icon: 🎯
    title: Multi-Signal Detection
    details: Combines Evidence Uptake, Bottleneck Stress, Conflict Sensitivity, and Trace Validation for robust hallucination detection
    
  - icon: 🔌
    title: Multi-Provider Support
    details: Works seamlessly with OpenAI (GPT-4), Anthropic (Claude), and Google Gemini APIs
    
  - icon: 🧪
    title: Research-Backed
    details: Based on neuroscience principles of Predictive Coding and Information Bottleneck theory
    
  - icon: ⚡
    title: Production Ready
    details: Fast, lightweight (<1M parameters), fully interpretable, and suitable for production deployment
    
  - icon: 📊
    title: High Performance
    details: AUROC 0.8669 on HaluBench with 75× less training data than competing methods
    
  - icon: 🤖
    title: MCP Integration
    details: Model Context Protocol support for Claude Desktop and other MCP clients
---

## Quick Example

```python
from pcib_detector import PCIBDetector, Config
import asyncio

# Initialize detector
config = Config(provider="openai", model="gpt-4o-mini")
detector = PCIBDetector(config)

# Detect hallucinations
async def check():
    result = await detector.detect_hallucination(
        answer="The Eiffel Tower was completed in 1889.",
        evidence="The Eiffel Tower opened on March 31, 1889."
    )
    
    print(f"Hallucination detected: {result.flagged}")
    print(f"Confidence: {result.score:.3f}")

asyncio.run(check())
```

## Why PCIB?

<div class="vp-doc">

| Feature | PCIB | LLM Judges (Lynx 70B) | Self-Consistency |
|---------|------|----------------------|------------------|
| **AUROC** | 0.8669 | 0.874 | 0.750 |
| **Training Data** | 200 samples | 15,000 samples | Zero-shot |
| **Inference Speed** | 5ms | 5s | 10s |
| **Parameters** | <1M | 70B | Variable |
| **Interpretable** | ✅ Yes | ❌ No | ❌ No |
| **Cost per 1M** | $0.15 | $3.00 | $2.00 |

</div>

## Key Signals

### 🔍 Evidence Uptake
Measures how much the LLM's belief changes when given evidence (KL divergence). Factual answers should strongly depend on context.

### 🧊 Bottleneck Stress  
Tests judgment stability when semantic noise is added. Hallucinations degrade faster than facts under perturbation.

### ⚔️ Conflict Sensitivity
Evaluates how the model responds to contradictory information. Hallucinations show inconsistent conflict patterns.

### 🔗 Trace Validation
Checks reasoning consistency by comparing forward and backward reasoning traces. Detects post-hoc rationalization.

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

## Research Impact

- **75× less training data** than Lynx (200 vs 15,000 samples)
- **1000× faster inference** (5ms vs 5s)  
- **Fully interpretable** signals grounded in neuroscience
- **Competitive performance** (0.8669 AUROC on HaluBench)

::: tip Published Research
Our methodology is detailed in the paper "Predictive Coding and Information Bottleneck for Hallucination Detection in Large Language Models" (2024).

[Read the paper →](/research/paper)
:::
