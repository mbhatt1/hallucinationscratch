# API Reference

**Complete API documentation for PCIB Detector**

## Core Classes

### PCIBDetector

Main detector class for hallucination detection.

```python
from pcib_detector import PCIBDetector, Config

detector = PCIBDetector(config: Optional[Config] = None, weights: Optional[np.ndarray] = None)
```

#### Constructor Parameters

- **config** (`Config`, optional): Configuration object. Uses defaults if `None`.
- **weights** (`np.ndarray`, optional): Calibrated weights for scoring. Uses heuristic if `None`.

#### Methods

##### `async detect_hallucination()`

Detect hallucinations in an answer given evidence.

```python
result = await detector.detect_hallucination(
    answer: str,
    evidence: str,
    return_details: bool = False,
    threshold: float = 0.5
) -> DetectionResult
```

**Parameters:**
- `answer` (str): The answer text to check
- `evidence` (str): The evidence/context to verify against
- `return_details` (bool): Whether to include detailed per-claim results
- `threshold` (float): Score threshold for flagging (0-1)

**Returns:** `DetectionResult`

**Example:**
```python
result = await detector.detect_hallucination(
    answer="Paris is the capital of Germany.",
    evidence="Berlin is the capital of Germany.",
    return_details=True,
    threshold=0.5
)

print(f"Flagged: {result.flagged}")
print(f"Score: {result.score:.3f}")
```

##### `async detect_batch()`

Detect hallucinations in a batch of examples.

```python
results = await detector.detect_batch(
    examples: List[dict],
    return_details: bool = False,
    threshold: float = 0.5
) -> List[DetectionResult]
```

**Parameters:**
- `examples` (List[dict]): List of `{answer: str, evidence: str}` dicts
- `return_details` (bool): Whether to include detailed results
- `threshold` (float): Score threshold for flagging

**Returns:** `List[DetectionResult]`

**Example:**
```python
examples = [
    {"answer": "...", "evidence": "..."},
    {"answer": "...", "evidence": "..."}
]
results = await detector.detect_batch(examples)
```

##### `async calibrate()`

Calibrate detector weights from labeled examples.

```python
weights = await detector.calibrate(
    examples: List[dict]
) -> np.ndarray
```

**Parameters:**
- `examples` (List[dict]): List of `{answer: str, evidence: str, label: int}` dicts
  - `label=1`: hallucination
  - `label=0`: grounded

**Returns:** `np.ndarray` - Calibrated weights

**Example:**
```python
labeled_data = [
    {"answer": "...", "evidence": "...", "label": 1},
    {"answer": "...", "evidence": "...", "label": 0}
]
weights = await detector.calibrate(labeled_data)
detector.weights = weights  # Apply calibrated weights
```

---

### Config

Configuration for the detector.

```python
from pcib_detector import Config

config = Config(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_concurrent: int = 10,
    max_claims: int = 4,
    distractor_chars: int = 1500,
    enable_trace_validation: bool = False,
    detect_rationalization: bool = False,
    trace_temperature: float = 0.3,
    n_ensemble: int = 1,
    ensemble_temperature: float = 0.3
)
```

#### Parameters

**Provider Settings:**
- `provider` (str): LLM provider - `"openai"`, `"anthropic"`, or `"gemini"`
- `model` (str, optional): Model identifier (uses provider default if `None`)
- `api_key` (str, optional): API key (reads from environment if `None`)

**Core Settings:**
- `temperature` (float): Sampling temperature for verification (default: 0.0)
- `max_concurrent` (int): Maximum concurrent API calls (default: 10)
- `max_claims` (int): Maximum claims to extract (default: 4)
- `distractor_chars` (int): Bottleneck test size (default: 1500)

**Trace Validation:**
- `enable_trace_validation` (bool): Enable reasoning validation (default: False)
- `detect_rationalization` (bool): Detect post-hoc justification (default: False)
- `trace_temperature` (float): Temperature for trace generation (default: 0.3)

**Ensemble:**
- `n_ensemble` (int): Number of verification samples (default: 1)
- `ensemble_temperature` (float): Temperature for ensemble (default: 0.3)

#### Examples

**Basic configuration:**
```python
config = Config(provider="openai", model="gpt-4o-mini")
```

**With trace validation:**
```python
config = Config(
    provider="openai",
    enable_trace_validation=True,
    detect_rationalization=True
)
```

**Ensemble with custom settings:**
```python
config = Config(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    n_ensemble=3,
    max_claims=8,
    max_concurrent=5
)
```

---

## Data Structures

### DetectionResult

Result of hallucination detection.

```python
@dataclass
class DetectionResult:
    flagged: bool                           # True if hallucination detected
    score: float                            # Overall score (0-1)
    claims: List[ClaimResult]              # Per-claim results
    signals: Optional[Dict[str, float]]    # Aggregate signals
    answer: Optional[str]                   # Original answer
    evidence: Optional[str]                 # Original evidence
```

#### Fields

- **flagged** (bool): Whether hallucination was detected
- **score** (float): Detection confidence (0-1, higher = more likely hallucination)
- **claims** (List[ClaimResult]): Per-claim analysis (if `return_details=True`)
- **signals** (dict, optional): Aggregate signal metrics
- **answer** (str, optional): Original answer text
- **evidence** (str, optional): Original evidence text

---

### ClaimResult

Per-claim analysis result.

```python
@dataclass
class ClaimResult:
    text: str              # Claim text
    score: float           # Claim score
    signals: ClaimSignals  # Detection signals
    flagged: bool          # Whether claim is flagged
```

#### Fields

- **text** (str): The extracted claim
- **score** (float): Hallucination score for this claim
- **signals** (ClaimSignals): All detection signals
- **flagged** (bool): Whether this claim exceeds threshold

---

### ClaimSignals

All detection signals for a claim.

```python
@dataclass
class ClaimSignals:
    prior: Belief                        # Prior belief (no evidence)
    post: Belief                         # Posterior belief (with evidence)
    uptake_kl: float                     # Evidence uptake (KL divergence)
    stress_js: float                     # Bottleneck stress (JS divergence)
    conflict_js: float                   # Conflict sensitivity
    post_entropy: float                  # Posterior entropy
    trace_consistency: Optional[float]   # Reasoning consistency (0-1)
    trace_support: Optional[float]       # Trace-conclusion support (0-1)
    rationalization_score: Optional[float]  # Post-hoc rationalization (0-1)
    trace_length: Optional[int]          # Trace word count
```

#### Signal Interpretation

| Signal | Range | Good | Bad |
|--------|-------|------|-----|
| `prior` | Belief | Uniform (~0.33 each) | N/A |
| `post.entail` | 0-1 | High (>0.7) | Low (<0.3) |
| `post.contradict` | 0-1 | Low (<0.2) | High (>0.5) |
| `post.unknown` | 0-1 | Low (<0.3) | High (>0.5) |
| `uptake_kl` | 0-∞ | Moderate (0.3-1.0) | Very low (<0.1) or very high (>2.0) |
| `stress_js` | 0-1 | Low (<0.1) | High (>0.15) |
| `conflict_js` | 0-1 | Moderate-high (>0.15) | Low (<0.08) |
| `post_entropy` | 0-log(3) | Low (decisive) | High (uncertain) |
| `trace_consistency` | 0-1 | High (>0.7) | Low (<0.5) |
| `trace_support` | 0-1 | High (>0.7) | Low (<0.5) |
| `rationalization_score` | 0-1 | Low (<0.3) | High (>0.5) |

---

### Belief

Probability distribution over labels.

```python
@dataclass
class Belief:
    dist: Dict[str, float]  # Distribution over ENTAIL/CONTRADICT/UNKNOWN
    entail: float           # P(ENTAIL)
    contradict: float       # P(CONTRADICT)
    unknown: float          # P(UNKNOWN)
```

Labels:
- **ENTAIL**: Evidence supports/implies the claim
- **CONTRADICT**: Evidence refutes the claim
- **UNKNOWN**: Evidence is insufficient or ambiguous

---

## Backend Interface

### Backend (Abstract Base Class)

```python
class Backend(ABC):
    @abstractmethod
    async def call_json_schema(
        self,
        *,
        model: str,
        prompt: str,
        schema_name: str,
        schema: Dict[str, Any],
        max_output_tokens: int = 600,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Call LLM with structured JSON output."""
        pass
    
    @abstractmethod
    async def call_text(
        self,
        *,
        prompt: str,
        max_tokens: int = 600,
        temperature: float = 0.0,
    ) -> str:
        """Call LLM for text generation."""
        pass
    
    @abstractmethod
    def get_default_model(self) -> str:
        """Get default model for this provider."""
        pass
    
    @abstractmethod
    def validate_model(self, model: str) -> bool:
        """Check if model is supported."""
        pass
```

### Creating Custom Backends

```python
from pcib_detector.backends import Backend

class CustomBackend(Backend):
    async def call_json_schema(self, *, model, prompt, schema_name, schema, **kwargs):
        # Implement your provider's structured output API
        ...
    
    async def call_text(self, *, prompt, **kwargs):
        # Implement your provider's text generation API
        ...
    
    def get_default_model(self):
        return "your-default-model"
    
    def validate_model(self, model):
        return model in ["model-1", "model-2"]
```

---

## Utility Functions

### Math Utilities

```python
from pcib_detector.math_utils import kl_cat, js_cat, entropy, normalize_dist

# KL divergence for categorical distributions
kl = kl_cat(p={"A": 0.7, "B": 0.3}, q={"A": 0.5, "B": 0.5})

# JS divergence (symmetric)
js = js_cat(p={"A": 0.7, "B": 0.3}, q={"A": 0.5, "B": 0.5})

# Shannon entropy
h = entropy({"A": 0.5, "B": 0.3, "C": 0.2})

# Normalize distribution
dist = normalize_dist({"A": 3, "B": 1, "C": 1})  # -> {"A": 0.6, "B": 0.2, "C": 0.2}
```

### Perturbations

```python
from pcib_detector.perturbations import make_distractor, make_conflict_snippet

# Generate distractor text (for bottleneck test)
distractor = make_distractor(n_chars=1500)

# Generate conflicting snippet
conflict = make_conflict_snippet(
    claim="It happened in 2020",
    evidence="The event occurred..."
)
```

---

## Command Line Interface

### pcib-eval

Evaluate detector on a dataset.

```bash
pcib-eval \
    --dataset DATASET_ID \
    --split SPLIT \
    --limit N \
    --model MODEL \
    --output OUTPUT_FILE \
    --enable-traces \
    --detect-rationalization
```

**Options:**
- `--dataset`: Hugging Face dataset ID (default: PatronusAI/HaluBench)
- `--split`: Dataset split (default: auto-detect)
- `--limit`: Number of examples (default: 100)
- `--model`: Model to use (default: gpt-4o-mini)
- `--output`: Output file path (default: results.jsonl)
- `--enable-traces`: Enable trace validation
- `--detect-rationalization`: Enable rationalization detection

### pcib-mcp

Start Model Context Protocol server.

```bash
pcib-mcp [--port PORT] [--host HOST]
```

**Options:**
- `--port`: Server port (default: 3000)
- `--host`: Server host (default: localhost)

---

## Environment Variables

```bash
# API Keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...

# Optional: Override default settings
export PCIB_MAX_CONCURRENT=5
export PCIB_TEMPERATURE=0.0
```

---

## Error Handling

### Common Exceptions

```python
from openai import AuthenticationError, RateLimitError
from anthropic import APIError

try:
    result = await detector.detect_hallucination(...)
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded - wait and retry")
except APIError as e:
    print(f"API error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Timeouts

```python
import asyncio

try:
    result = await asyncio.wait_for(
        detector.detect_hallucination(...),
        timeout=300.0  # 5 minutes
    )
except asyncio.TimeoutError:
    print("Detection timed out")
```

---

## Type Hints

All public APIs include full type hints:

```python
from typing import List, Optional, Dict, Any
from pcib_detector import PCIBDetector, Config, DetectionResult

async def my_function(
    detector: PCIBDetector,
    answer: str,
    evidence: str
) -> DetectionResult:
    result: DetectionResult = await detector.detect_hallucination(
        answer=answer,
        evidence=evidence
    )
    return result
```

---

## Constants

```python
from pcib_detector.math_utils import LABELS

# Available labels
LABELS = ["ENTAIL", "CONTRADICT", "UNKNOWN"]
```

---

## Version Information

```python
import pcib_detector

print(pcib_detector.__version__)  # "1.0.0"
```

---

## See Also

- **[Quick Start Guide](QUICKSTART.md)** - Get started in 5 minutes
- **[Main README](README.md)** - Overview and features
- **[Examples](pcib_detector/examples/)** - Code examples
- **[Ablation Study](ABLATION_STUDY.md)** - Evaluation framework

---

**Last Updated**: January 2026  
**Version**: 1.0.0
