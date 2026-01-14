# Multi-Signal Detection

PCIB uses multiple complementary signals to detect hallucinations. Each signal captures a different aspect of factual generation.

## Signal Overview

| Signal | Principle | Measures | Good Value | Bad Value |
|--------|-----------|----------|------------|-----------|
| **Uptake** | Predictive Coding | Context dependency | High (>0.7) | Low (<0.3) |
| **Stress** | Information Bottleneck | Semantic stability | Low (<0.3) | High (>0.7) |
| **Conflict** | Logical Consistency | Self-contradiction | Low (<0.3) | High (>0.7) |
| **Trace** | Reasoning Coherence | Trace consistency | High (>0.7) | Low (<0.3) |

## 1. Evidence Uptake 📊

### What it Measures
How much the LLM's belief changes when given evidence vs. without evidence.

### Theory: Predictive Coding
From neuroscience: Brains minimize prediction error by updating beliefs based on sensory input. When an LLM hallucinates, it ignores the "sensory input" (context) and generates from parametric memory alone.

### Formula
$$
U = D_{KL}(P(A|Q,C) \parallel P(A|Q))
$$

Where:
- $P(A|Q,C)$ = probability of answer given question + context
- $P(A|Q)$ = probability of answer given only question

### Implementation
```python
# Get log-likelihoods without context
logprobs_prior = model.get_logprobs(answer, context=question)

# Get log-likelihoods with context
logprobs_post = model.get_logprobs(answer, context=f"{question}\n{evidence}")

# Compute KL divergence
kl_div = sum((logprobs_post[i] - logprobs_prior[i]) * exp(logprobs_post[i])
             for i in range(len(answer_tokens)))
```

### Example: Factual Answer

```python
Q: "When was the Eiffel Tower built?"
Evidence: "The Eiffel Tower opened on March 31, 1889."
Answer: "The Eiffel Tower was built in 1889."

# Without evidence: P(1889) = 0.4  (model knows this fact)
# With evidence:    P(1889) = 0.95 (strongly reinforced)
# Uptake = 0.85 ✅ HIGH (good!)
```

### Example: Hallucination

```python
Q: "When was the Eiffel Tower built?"
Evidence: "The Eiffel Tower opened on March 31, 1889."
Answer: "The Eiffel Tower was built in 1887."  # WRONG!

# Without evidence: P(1887) = 0.05 (wrong but model committed)
# With evidence:    P(1887) = 0.06 (barely changed)
# Uptake = 0.12 ⚠️ LOW (hallucination!)
```

### Enhancement: Entity-Focused Uptake

Standard KL divergence treats all tokens equally, but hallucinations concentrate in named entities and numbers.

```python
# Extract high-value tokens
entities = extract_entities(answer)  # "1889", "Eiffel Tower"

# Weight uptake by entity density
entity_density = len(entities) / len(tokens)
uptake_entity = uptake_base * (1 + 2.0 * entity_density)
```

## 2. Bottleneck Stress 🧊

### What it Measures
How stable the judgment remains when semantic noise is added.

### Theory: Information Bottleneck
From information theory: Robust representations are invariant to nuisance transformations. Facts are densely connected in latent space; hallucinations are isolated points easily perturbed.

### Formula
$$
S = \frac{1}{|C|} \sum_{i=1}^{|C|} \mathbb{E}_k [\text{JS}(p_i \parallel p_i^{(k)})]
$$

Where:
- $C$ = set of extracted claims
- $p_i$ = entailment distribution for original claim
- $p_i^{(k)}$ = distribution for perturbed claim $k$
- $\text{JS}$ = Jensen-Shannon divergence

### Implementation
```python
for claim in extract_claims(answer):
    # Generate semantic perturbations
    perturbations = [
        paraphrase(claim),
        substitute_entities(claim),
        negate_polarity(claim)
    ]
    
    # Get entailment distributions
    p_original = nli_model(evidence, claim)
    p_perturbed = [nli_model(evidence, p) for p in perturbations]
    
    # Compute JS divergence (symmetric KL)
    stress = mean([js_divergence(p_original, p) for p in p_perturbed])
```

### Example: Factual (Low Stress)

```python
Claim: "The Eiffel Tower opened in 1889."

Perturbations:
1. "The tower in Paris opened in 1889."       → entailment: 0.95
2. "In 1889, the Eiffel Tower was opened."   → entailment: 0.94
3. "The Eiffel Tower launched in 1889."      → entailment: 0.92

Stress = 0.08 ✅ LOW (stable under perturbation!)
```

### Example: Hallucination (High Stress)

```python
Claim: "The Eiffel Tower opened in 1887."  # WRONG DATE

Perturbations:
1. "The tower in Paris opened in 1887."      → contradiction: 0.78
2. "In 1887, the Eiffel Tower was opened."  → contradiction: 0.82
3. "The Eiffel Tower launched in 1887."     → neutral: 0.45

Stress = 0.73 ⚠️ HIGH (fragile, inconsistent!)
```

### Enhancement: Context Adherence

Measures how well the answer is grounded in available context:

```python
context_length = len(evidence.split())
adherence = (1 / (1 + stress)) * min(1.0, context_length / 200)

# High stress + short context = low adherence (bad)
# Low stress + rich context = high adherence (good)
```

## 3. Conflict Sensitivity ⚔️

### What it Measures
Whether the answer contradicts itself when perturbed.

### Implementation
```python
for claim in extract_claims(answer):
    perturbations = generate_perturbations(claim)
    
    # Check if ANY perturbation contradicts the original answer
    contradictions = [
        nli_model(answer, perturbed).contradiction
        for perturbed in perturbations
    ]
    
    conflict = max(contradictions)  # Worst-case
```

### Example: Consistent (Low Conflict)

```python
Answer: "The Eiffel Tower is in Paris, France."

Perturbations:
- "The tower is located in Paris."          → entails ✅
- "It's in the French capital."             → entails ✅
- "Located in Paris, not London."           → entails ✅

Conflict = 0.05 ✅ LOW (logically consistent!)
```

### Example: Self-Contradictory (High Conflict)

```python
Answer: "The Eiffel Tower is in Paris and was never in France."

Perturbations:
- "The tower is in France."                 → contradicts ⚠️
- "Paris is in France."                     → contradicts ⚠️
- "It's in Paris but not France."           → contradicts ⚠️

Conflict = 0.89 ⚠️ HIGH (self-contradictory!)
```

### Enhancement: Falsifiability Score

Combines conflict with linguistic confidence markers:

```python
definitive = count_words(answer, ['definitely', 'certainly', 'clearly'])
hedge = count_words(answer, ['possibly', 'maybe', 'perhaps'])

falsifiability = conflict * (1 + 0.1 * (definitive - hedge))

# High conflict + definitive language = high falsifiability (bad)
```

## 4. Trace Validation 🔗

### What it Measures
Consistency between forward and backward reasoning traces.

### Theory
If the LLM truly "believes" a claim, it should be able to explain WHY it's true, and that explanation should be consistent across multiple generations.

### Implementation
```python
# Generate multiple reasoning traces
traces = []
for _ in range(5):
    trace = model.generate(
        f"Explain why the following is true: {claim}\nEvidence: {evidence}"
    )
    traces.append(trace)

# Measure semantic overlap (Jaccard similarity)
consistency = pairwise_jaccard_similarity(traces)
```

### Example: Genuine Reasoning

```python
Claim: "The Eiffel Tower opened in 1889."

Traces:
1. "The evidence explicitly states it opened March 31, 1889."
2. "The date 1889 is mentioned in the provided text."
3. "According to the context, the opening was in 1889."

Consistency = 0.87 ✅ HIGH (coherent reasoning!)
```

### Example: Post-Hoc Rationalization

```python
Claim: "The Eiffel Tower opened in 1887."  # WRONG

Traces:
1. "Construction began in 1887, so that's the opening year."
2. "Historical records show 1887 as the completion date."
3. "The tower was functional by 1887."

Consistency = 0.34 ⚠️ LOW (fabricated, inconsistent!)
```

## Signal Aggregation

### Theory-Guided Baseline

Harmonic mean (weakest-link principle):

```python
esi = 3 / (1/uptake + 1/(1-stress) + 1/(1-conflict))
threshold = 0.5
flagged = esi < threshold
```

### Supervised Stacking

Learn optimal weights via Random Forest:

```python
features = [
    uptake, stress, conflict, trace,
    uptake * stress,  # interactions
    stress / uptake,  # ratios
    uptake ** 2       # polynomials
]

model = RandomForestClassifier(n_estimators=100)
model.fit(features, labels)
prediction = model.predict_proba(features)[1]
```

## Practical Usage

### Access Individual Signals

```python
result = await detector.detect_hallucination(answer, evidence)

print(f"Uptake: {result.signals.uptake:.3f}")
print(f"Stress: {result.signals.stress:.3f}")
print(f"Conflict: {result.signals.conflict:.3f}")
print(f"Trace: {result.signals.rationalization:.3f}")
```

### Custom Thresholds

```python
# Strict detection
high_risk = (
    result.signals.uptake < 0.3 or
    result.signals.stress > 0.7 or
    result.signals.conflict > 0.6
)

# Conservative detection
low_risk = (
    result.signals.uptake < 0.1 and
    result.signals.stress > 0.9
)
```

### Signal-Specific Debugging

```python
if result.flagged:
    if result.signals.uptake < 0.3:
        print("⚠️ Low uptake: Answer ignores evidence")
    if result.signals.stress > 0.7:
        print("⚠️ High stress: Unstable under perturbation")
    if result.signals.conflict > 0.6:
        print("⚠️ High conflict: Self-contradictory")
```

## Next Steps

- [Configuration →](/guide/configuration) - Customize signal weights
- [API Reference →](/api/detector) - Full API documentation
- [Research →](/research/methodology) - Theoretical details
