# Reasoning Trace Validation

PCIB now includes advanced **reasoning trace validation** and **post-hoc rationalization detection** inspired by Strawberry's approach to Chain-of-Thought (CoT) verification.

## Overview

Trace validation adds a meta-reasoning layer that examines **how** the model arrives at its judgments, not just the final probability distributions. This catches sophisticated hallucinations where models produce confident but fabricated reasoning.

## What It Detects

### 1. **Trace Consistency**
Measures logical coherence of the reasoning trace.

**Catches:**
- Circular reasoning (restating claim as evidence)
- Internal contradictions
- Non-sequiturs

**Score:** 0.0 (inconsistent) to 1.0 (perfectly consistent)

### 2. **Trace Support**
Measures how well the reasoning actually supports the stated conclusion.

**Catches:**
- Disconnected reasoning (conclusion doesn't follow from steps)
- Evidence misinterpretation
- Logical leaps

**Score:** 0.0 (unsupported) to 1.0 (fully supports)

### 3. **Post-Hoc Rationalization** ⭐
Compares forward reasoning (claim → analysis → judgment) vs. backward reasoning (judgment given → justification).

**Catches:**
- Fabricated reasoning to fit a predetermined conclusion
- Confirmation bias
- Overconfident hallucinations with plausible-sounding explanations

**Score:** 0.0 (genuine reasoning) to 1.0 (likely rationalization)

**How it works:**
1. **Forward trace:** Model receives claim + evidence → generates reasoning → produces judgment
2. **Backward trace:** Model receives claim + evidence + predetermined conclusion → explains why
3. **Divergence:** If forward and backward traces differ significantly (by key phrase overlap), reasoning is likely post-hoc

This is the **most powerful signal** for detecting confident but false claims.

## Usage

### Basic Configuration

```python
from pcib_detector import PCIBDetector, Config

config = Config(
    provider="openai",
    model="gpt-4o-mini",
    enable_trace_validation=True,       # Enable feature
    detect_rationalization=True,        # Enable rationalization detection
    trace_temperature=0.3,              # Slight diversity in traces
)

detector = PCIBDetector(config=config)
```

### Examining Results

```python
result = await detector.detect_hallucination(
    answer="The French Revolution began in 1791.",
    evidence="The French Revolution began in 1789...",
    return_details=True
)

for claim in result.claims:
    if claim.signals.trace_consistency is not None:
        print(f"Consistency: {claim.signals.trace_consistency:.3f}")
        print(f"Support: {claim.signals.trace_support:.3f}")
        print(f"Rationalization: {claim.signals.rationalization_score:.3f}")
```

### Interpreting Scores

| Metric | Low (< 0.3) | Medium (0.3-0.7) | High (> 0.7) |
|--------|-------------|------------------|--------------|
| **Consistency** | ⚠️ Contradictory | Somewhat unclear | ✅ Coherent |
| **Support** | ⚠️ Disconnected | Partial support | ✅ Well-supported |
| **Rationalization** | ✅ Genuine | Uncertain | ⚠️ Likely fabricated |

## Example: Detecting Rationalization

```python
# Hallucinated answer with confident reasoning
answer = """
The Eiffel Tower was completed in 1891. This was two years after the
Paris Exposition of 1889, allowing for final adjustments and public opening.
"""

evidence = """
The Eiffel Tower was completed in March 1889 for the 1889 World's Fair.
"""

result = await detector.detect_hallucination(answer, evidence, return_details=True)

# High rationalization score indicates fabricated reasoning
# Model confidently generated plausible-sounding but false justification
```

## Performance Considerations

### Cost & Latency
Trace validation adds **~3x API calls per claim**:
- 1 forward trace generation
- 1 backward trace generation (if rationalization detection enabled)
- 1 consistency check

**Recommendation:** Use selectively on high-stakes claims or when base PCIB signals are ambiguous.

### When to Enable

✅ **Enable when:**
- Dealing with complex factual claims
- Model might use sophisticated but flawed reasoning
- Need explainability (WHY was this flagged?)
- False positives are costly

❌ **Disable when:**
- High-volume batch processing
- Simple factual checks (dates, numbers)
- Cost/latency is critical
- Base PCIB signals are already decisive

## Configuration Options

```python
Config(
    # Core trace validation
    enable_trace_validation=True,      # Master switch
    
    # Rationalization detection
    detect_rationalization=True,        # Compare forward/backward traces
    
    # Generation settings
    trace_temperature=0.3,              # 0.0=deterministic, 0.7=diverse
)
```

### Temperature Recommendations

- **0.0-0.2:** Deterministic reasoning (default for consistency)
- **0.3-0.5:** Slight diversity (recommended for rationalization detection)
- **0.6-0.9:** High diversity (for ensemble-style validation)

## Integration with Scoring

Trace signals are incorporated into the heuristic scorer:

```python
# Base PCIB score
base_score = 3.0 * contradict + 2.0 * (1.0 - entail)

# Trace penalty (weighted sum)
trace_penalty = (
    0.8 * (1.0 - consistency) +      # Inconsistent reasoning
    0.6 * (1.0 - support) +          # Weak support
    1.2 * rationalization            # Post-hoc fabrication (strongest)
)

# Combined score
final_score = base_score + 0.5 * trace_penalty
```

Rationalization has the **highest weight** (1.2) because it's the strongest signal for confident hallucinations.

## Feature Vectors for Calibration

When using calibrated weights (via `detector.calibrate()`), trace features are automatically included:

**Base features (13):** contradict, entail, unknown, uptake, stress, conflict, Bayes factors, interactions

**+ Trace features (6):**
- `trace_inconsistency` = 1.0 - consistency
- `trace_lack_support` = 1.0 - support
- `rationalization` = rationalization score
- `entail × inconsistency` (interaction)
- `entail × rationalization` (interaction)
- `lack_support × high_entail` (interaction)

Total: **19 features** when trace validation is enabled.

## Comparison with Strawberry

| Feature | Strawberry | PCIB Trace Validation |
|---------|------------|----------------------|
| **CoT generation** | ✅ Multi-step reasoning | ✅ Forward + backward traces |
| **Trace budgeting** | ✅ Information-theoretic | ❌ (planned) |
| **Rationalization** | ✅ Implicit via consistency | ✅ **Explicit forward/backward comparison** |
| **Stage 2A/2B checks** | ✅ Safety validation | ❌ (not applicable) |
| **Backend support** | OpenAI, Azure, vLLM | OpenAI, Anthropic, Gemini |
| **Integration** | Standalone | **Integrated with PCIB signals** |

**Key difference:** PCIB combines trace validation with PC+IB signals (uptake, stress, conflict) for a multi-layered approach. Strawberry focuses purely on trace-level validation.

## Troubleshooting

### High latency
- Disable `detect_rationalization` to skip backward traces (2x faster)
- Reduce `max_claims` in config
- Use only for ambiguous cases after base PCIB check

### Low signal
- Increase `trace_temperature` to 0.5-0.7 for more diverse traces
- Check that evidence is substantial (trace validation needs context)
- Some models may produce overly cautious traces

### API errors
- Ensure `call_text()` is implemented for your backend (added in this release)
- Check max_tokens settings if traces are truncated
- Gemini users: may need higher token limits for trace generation

## Example Scripts

See `examples/trace_validation_example.py` for a complete walkthrough:

```bash
cd pcib_detector
python examples/trace_validation_example.py
```

## Research Background

This feature is inspired by:

1. **Strawberry's CoT Detector** (Pythea toolkit): Reasoning trace validation for o1-style models
2. **Self-Consistency** (Wang et al.): Sample multiple reasoning paths and check agreement
3. **Backward Reasoning** (Saparov & He, 2023): Forward/backward divergence indicates confabulation

## Future Enhancements

Planned features:
- [ ] Trace budgeting (information-theoretic quantification)
- [ ] Multi-sample trace consistency (ensemble validation)
- [ ] Attention-based trace grounding (verify trace uses evidence)
- [ ] Adversarial trace testing (inject misleading context)

## References

- Strawberry (Pythea): https://github.com/PatronusAI/pythea
- Self-Consistency: https://arxiv.org/abs/2203.11171
- Backward Reasoning: https://arxiv.org/abs/2308.04445
