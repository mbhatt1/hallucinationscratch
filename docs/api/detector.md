# PCIBDetector

The main class for hallucination detection.

## Constructor

```python
from pcib_detector import PCIBDetector, Config

detector = PCIBDetector(config: Config)
```

### Parameters

- **config** (`Config`): Configuration object specifying provider, model, and detection parameters

## Methods

### detect_hallucination

Detect hallucinations in a single answer.

```python
async def detect_hallucination(
    self,
    answer: str,
    evidence: str,
    question: Optional[str] = None
) -> DetectionResult
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `answer` | `str` | ✅ Yes | The LLM-generated answer to check |
| `evidence` | `str` | ✅ Yes | The source context/evidence |
| `question` | `str` | ❌ No | Optional question that prompted the answer |

#### Returns

`DetectionResult` object containing:

```python
@dataclass
class DetectionResult:
    flagged: bool                    # True if hallucination detected
    score: float                     # Hallucination probability [0, 1]
    signals: SignalScores           # Individual signal scores
    claims: List[ClaimResult]       # Per-claim analysis
    explanation: str                # Human-readable explanation
    metadata: Dict[str, Any]        # Additional metadata
```

#### Example

```python
result = await detector.detect_hallucination(
    question="When was the Eiffel Tower built?",
    answer="The Eiffel Tower was completed in 1889.",
    evidence="The Eiffel Tower opened on March 31, 1889."
)

print(f"Hallucination: {result.flagged}")
print(f"Score: {result.score:.3f}")
print(f"Explanation: {result.explanation}")
```

### detect_batch

Process multiple examples in parallel.

```python
async def detect_batch(
    self,
    examples: List[Dict[str, str]],
    max_concurrent: Optional[int] = None
) -> List[DetectionResult]
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `examples` | `List[Dict]` | ✅ Yes | List of dicts with 'answer', 'evidence', optional 'question' |
| `max_concurrent` | `int` | ❌ No | Override config's max_concurrent setting |

#### Returns

`List[DetectionResult]` - Results in same order as input

#### Example

```python
examples = [
    {
        "question": "What is the capital of France?",
        "answer": "The capital is Paris.",
        "evidence": "Paris is France's capital city..."
    },
    {
        "answer": "The Earth is flat.",
        "evidence": "The Earth is an oblate spheroid..."
    }
]

results = await detector.detect_batch(examples, max_concurrent=5)

for i, result in enumerate(results):
    print(f"Example {i+1}: {result.flagged} (score: {result.score:.3f})")
```

## Signal Scores

The `SignalScores` object contains individual signal measurements:

```python
@dataclass
class SignalScores:
    uptake: float              # Evidence uptake [0, 1]
    stress: float              # Bottleneck stress [0, 1]  
    conflict: float            # Conflict score [0, 1]
    rationalization: float     # Rationalization score [0, 1]
    entity_uptake: float       # Entity-focused uptake [0, 1]
    context_adherence: float   # Context adherence [0, 1]
    falsifiability: float      # Falsifiability score [0, 1]
```

### Interpretation

| Signal | Low (Good) | High (Bad) |
|--------|------------|------------|
| **Uptake** | Ignores evidence | Strongly evidence-based ✅ |
| **Stress** | Stable under noise ✅ | Fragile, inconsistent |
| **Conflict** | Internally consistent ✅ | Self-contradictory |
| **Rationalization** | Coherent reasoning ✅ | Post-hoc justification |

## Claim Results

Individual claims are analyzed separately:

```python
@dataclass
class ClaimResult:
    text: str                  # The extracted claim
    score: float              # Hallucination score for this claim
    signals: SignalScores     # Signal breakdown for this claim
    entailment: str           # 'entailed', 'neutral', 'contradiction'
    confidence: float         # Model confidence [0, 1]
```

### Example

```python
result = await detector.detect_hallucination(...)

for claim in result.claims:
    print(f"[{claim.score:.3f}] {claim.text}")
    print(f"  Entailment: {claim.entailment}")
    print(f"  Confidence: {claim.confidence:.3f}")
```

## Error Handling

The detector raises specific exceptions:

```python
from pcib_detector.exceptions import (
    PCIBError,           # Base exception
    APIError,            # Provider API errors
    ConfigError,         # Configuration errors
    ValidationError      # Input validation errors
)

try:
    result = await detector.detect_hallucination(answer, evidence)
except APIError as e:
    print(f"API error: {e}")
    # Handle rate limits, timeouts, etc.
except ValidationError as e:
    print(f"Invalid input: {e}")
    # Handle malformed inputs
except PCIBError as e:
    print(f"General error: {e}")
```

## Advanced Usage

### Custom Thresholds

```python
result = await detector.detect_hallucination(answer, evidence)

# Custom threshold
is_hallucination = result.score > 0.7  # More strict

# Signal-specific checks
high_stress = result.signals.stress > 0.6
low_uptake = result.signals.uptake < 0.3
```

### Accessing Raw Data

```python
result = await detector.detect_hallucination(answer, evidence)

# Raw signal data
raw_uptake = result.metadata.get('raw_uptake')
kl_divergence = result.metadata.get('kl_divergence')

# Timing information
duration = result.metadata.get('duration_ms')
api_calls = result.metadata.get('api_calls')
```

### Partial Signal Computation

```python
# Only compute specific signals (not yet implemented)
config = Config(
    enabled_signals=['uptake', 'stress'],  # Skip trace validation
    enable_trace_validation=False
)
```

## Performance Tips

### Batching

```python
# Process in batches for better throughput
for batch in chunks(large_dataset, size=100):
    results = await detector.detect_batch(batch, max_concurrent=10)
    save_results(results)
```

### Caching

```python
# Cache results to avoid recomputation
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_detect(answer: str, evidence: str):
    return asyncio.run(detector.detect_hallucination(answer, evidence))
```

### Timeout Handling

```python
import asyncio

try:
    result = await asyncio.wait_for(
        detector.detect_hallucination(answer, evidence),
        timeout=30.0  # 30 seconds
    )
except asyncio.TimeoutError:
    print("Detection timed out")
```

## See Also

- [Configuration →](/api/config) - Config options
- [Results →](/api/results) - Result object details
- [Backends →](/api/backends/openai) - Provider-specific docs
