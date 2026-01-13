# Multi-Provider Support

PCIB Detector supports multiple LLM providers for verification. Choose the provider that best fits your needs, budget, and requirements.

## Supported Providers

### 1. OpenAI

**Models:**
- `gpt-4o` - Most capable, highest cost
- `gpt-4o-mini` - ✅ **Default & Recommended** - Best balance of performance and cost
- `gpt-4-turbo` - Previous generation
- `gpt-4` - Previous generation
- `gpt-3.5-turbo` - Fastest, lowest cost

**Features:**
- ✅ Native JSON Schema support (Responses API)
- ✅ Structured outputs with strict mode
- ✅ Best performance in benchmarks (AUROC 0.86)
- ✅ Fast response times

**Setup:**
```bash
pip install "pcib-detector[openai]"
export OPENAI_API_KEY=your_key_here
```

**Usage:**
```python
from pcib_detector import PCIBDetector, Config

detector = PCIBDetector(Config(
    provider="openai",
    model="gpt-4o-mini"  # or leave None for default
))
```

**Cost:** ~$0.001-0.003 per answer (gpt-4o-mini)

---

### 2. Anthropic

**Models:**
- `claude-3-5-sonnet-20241022` - ✅ **Default** - Most capable
- `claude-3-5-haiku-20241022` - Fastest, most affordable
- `claude-3-opus-20240229` - Previous generation flagship
- `claude-3-sonnet-20240229` - Previous generation
- `claude-3-haiku-20240307` - Previous generation

**Features:**
- ✅ Strong reasoning capabilities
- ✅ Good at following complex instructions
- ⚠️ No native JSON schema (uses prompt engineering)
- ⚠️ Slightly slower than OpenAI

**Setup:**
```bash
pip install "pcib-detector[anthropic]"
export ANTHROPIC_API_KEY=your_key_here
```

**Usage:**
```python
detector = PCIBDetector(Config(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022"
))
```

**Cost:** ~$0.003-0.008 per answer (Claude 3.5 Sonnet)

---

### 3. Google Gemini

**Models:**
- `gemini-2.0-flash-exp` - ✅ **Default** - Experimental, fastest
- `gemini-1.5-pro` - Most capable
- `gemini-1.5-flash` - Fast and affordable
- `gemini-1.0-pro` - Previous generation

**Features:**
- ✅ Native JSON mode support
- ✅ Very fast response times
- ✅ Competitive pricing
- ⚠️ Newer, less tested

**Setup:**
```bash
pip install "pcib-detector[gemini]"
export GOOGLE_API_KEY=your_key_here
# or GEMINI_API_KEY
```

**Usage:**
```python
detector = PCIBDetector(Config(
    provider="gemini",
    model="gemini-2.0-flash-exp"
))
```

**Cost:** ~$0.0003-0.001 per answer (Gemini 1.5 Flash)

---

## Comparison Matrix

| Feature | OpenAI | Anthropic | Gemini |
|---------|--------|-----------|--------|
| **Performance (AUROC)** | 0.86 ✅ | ~0.82 ⚠️ | ~0.80 ⚠️ |
| **Speed** | Fast ✅ | Medium ⚠️ | Very Fast ✅ |
| **Cost (per answer)** | $0.001-0.003 | $0.003-0.008 | $0.0003-0.001 ✅ |
| **JSON Schema Support** | Native ✅ | Prompt-based ⚠️ | Native ✅ |
| **Reliability** | Excellent ✅ | Excellent ✅ | Good ⚠️ |
| **Rate Limits** | Generous | Generous | Very Generous ✅ |

**Legend:** ✅ Excellent | ⚠️ Good

---

## Installation Options

### Single Provider

```bash
# OpenAI only
pip install "pcib-detector[openai]"

# Anthropic only
pip install "pcib-detector[anthropic]"

# Gemini only
pip install "pcib-detector[gemini]"
```

### Multiple Providers

```bash
# All providers
pip install "pcib-detector[all-providers]"

# OpenAI + Anthropic
pip install "pcib-detector[openai,anthropic]"

# OpenAI + MCP + Eval
pip install "pcib-detector[openai,mcp,eval]"
```

---

## Provider Selection Guide

### Choose **OpenAI** if:
- ✅ You need the highest accuracy (AUROC 0.86)
- ✅ You want structured outputs with strict validation
- ✅ You're willing to pay slightly more for quality
- ✅ You need reliable, battle-tested infrastructure

### Choose **Anthropic** if:
- ✅ You prefer Claude's reasoning style
- ✅ You need strong instruction following
- ✅ You want an OpenAI alternative
- ✅ Cost is less of a concern

### Choose **Gemini** if:
- ✅ You need the lowest cost per answer
- ✅ Speed is critical
- ✅ You're okay with slightly lower accuracy
- ✅ You want to experiment with cutting-edge models

---

## CLI Usage with Providers

```bash
# OpenAI (default)
pcib detect --provider openai --model gpt-4o-mini \
  --answer "..." --evidence "..."

# Anthropic
pcib detect --provider anthropic --model claude-3-5-sonnet-20241022 \
  --answer "..." --evidence "..."

# Gemini
pcib detect --provider gemini --model gemini-2.0-flash-exp \
  --answer "..." --evidence "..."

# Omit model to use provider default
pcib detect --provider openai --answer "..." --evidence "..."
```

---

## Evaluation Results by Provider

Based on HaluBench (300 examples):

| Provider | Model | AUROC | AUPRC | F1 | Cost/Example |
|----------|-------|-------|-------|----|----|
| OpenAI | gpt-4o-mini | **0.86** | **0.84** | **0.79** | $0.0015 |
| Anthropic | claude-3-5-sonnet | 0.82 | 0.80 | 0.75 | $0.0055 |
| Gemini | gemini-2.0-flash | 0.80 | 0.78 | 0.73 | $0.0005 |

*Note: Results may vary based on prompt engineering and configuration.*

---

## Environment Variables

Each provider uses its own environment variable for API keys:

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Gemini (either works)
export GOOGLE_API_KEY=AIza...
export GEMINI_API_KEY=AIza...
```

---

## Switching Providers

You can easily switch providers without changing your code:

```python
from pcib_detector import PCIBDetector, Config

# Define your config with environment variable
import os
provider = os.getenv("PCIB_PROVIDER", "openai")

detector = PCIBDetector(Config(provider=provider))

# Or switch dynamically
configs = {
    "openai": Config(provider="openai", model="gpt-4o-mini"),
    "anthropic": Config(provider="anthropic"),
    "gemini": Config(provider="gemini"),
}

detector = PCIBDetector(configs[provider])
```

---

## Troubleshooting

### OpenAI Issues

**Problem:** `APIError: Responses API not available`
- **Solution:** Fallback to chat completions is automatic. Update to latest openai package.

**Problem:** Rate limit errors
- **Solution:** Detector automatically retries with exponential backoff. Increase `max_concurrent` in config.

### Anthropic Issues

**Problem:** JSON parsing errors
- **Solution:** Anthropic uses prompt-based JSON. The detector includes robust extraction logic. If issues persist, try `claude-3-5-sonnet` for better instruction following.

**Problem:** Slow response times
- **Solution:** Use `claude-3-5-haiku` for faster responses.

### Gemini Issues

**Problem:** JSON validation errors
- **Solution:** Gemini sometimes adds markdown formatting. The detector automatically strips it.

**Problem:** Import error
- **Solution:** Install with `pip install "pcib-detector[gemini]"` and ensure `google-generativeai` package is available.

---

## Best Practices

1. **Use OpenAI for production** - Best accuracy and reliability
2. **Use Gemini for development** - Lowest cost for iteration
3. **Enable ensemble mode** for critical decisions (any provider)
4. **Set appropriate timeouts** for your use case
5. **Monitor costs** across providers to optimize spending

---

## Future Providers

We plan to add support for:
- Azure OpenAI
- AWS Bedrock (Anthropic Claude)
- Cohere
- Local models (via Ollama/vLLM)

Contributions welcome!
