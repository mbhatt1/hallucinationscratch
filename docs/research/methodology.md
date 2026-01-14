# Research Methodology

This page explains the theoretical foundations and empirical methodology behind PCIB Detector.

## Theoretical Framework

PCIB combines two foundational theories from cognitive neuroscience and information theory:

### Predictive Coding

**Origin**: Neuroscience (Friston, 2010; Rao & Ballard, 1999)

**Core Principle**: Intelligent systems minimize prediction error by constantly updating beliefs based on sensory evidence.

**Application to LLMs**: When an LLM generates text:
- **Factual answer**: Posterior belief $P(A|Q,C)$ differs significantly from prior $P(A|Q)$
- **Hallucination**: Model ignores context and generates from parametric priors

We quantify this via **Evidence Uptake**:

$$
U = D_{KL}(P(A|Q,C) \parallel P(A|Q)) = \sum_i P(a_i|Q,C) \log \frac{P(a_i|Q,C)}{P(a_i|Q)}
$$

Where:
- $Q$ = question
- $C$ = evidence/context  
- $A$ = answer
- $a_i$ = tokens in answer

**Interpretation**:
- High $U$ → Answer strongly depends on evidence (✅ factual)
- Low $U$ → Answer independent of context (⚠️ hallucination)

### Information Bottleneck

**Origin**: Information Theory (Tishby et al., 2000)

**Core Principle**: Optimal representations compress input to retain only task-relevant information, discarding noise.

**Application to LLMs**: 
- **Factual knowledge**: Robust compression, invariant to semantic perturbations
- **Hallucinations**: Fragile representations, sensitive to noise

We test this via **Bottleneck Stress**:

$$
S = \frac{1}{|C|} \sum_{i=1}^{|C|} \mathbb{E}_k [\text{JS}(p_i \parallel p_i^{(k)})]
$$

Where:
- $C$ = set of extracted claims
- $p_i$ = entailment distribution for claim $c_i$
- $p_i^{(k)}$ = distribution for perturbed claim $c_i^{(k)}$
- $\text{JS}$ = Jensen-Shannon divergence

**Perturbations**:
- Paraphrasing (semantic invariance)
- Entity substitution
- Negation tests

**Interpretation**:
- Low $S$ → Stable under perturbation (✅ factual)
- High $S$ → Fragile, noise-sensitive (⚠️ hallucination)

## Signal Architecture

### 1. Evidence Uptake (U)

**Computation**:
```python
# Get log-likelihoods
P_prior = model(answer | question)           # Without evidence
P_post = model(answer | question, evidence)  # With evidence

# Compute KL divergence
uptake = sum((P_post - P_prior) * P_post)
```

**Enhancement: Entity-Focused Uptake**
Standard KL treats all tokens equally, but hallucinations concentrate in high-value tokens:

$$
U_{entity} = U_{base} \times \left(1 + 2.0 \cdot \frac{|\text{entities}|}{|\text{tokens}|}\right)
$$

### 2. Bottleneck Stress (S)

**Computation**:
```python
for claim in extract_claims(answer):
    # Generate perturbations
    perturbed = [paraphrase(claim) for _ in range(5)]
    
    # Get entailment distributions
    p_orig = entailment_model(evidence, claim)
    p_pert = [entailment_model(evidence, p) for p in perturbed]
    
    # Compute JS divergence
    stress = mean([js_divergence(p_orig, p) for p in p_pert])
```

**Enhancement: Context Adherence**
Measures grounding strength:

$$
A_{context} = \frac{1}{1 + S} \cdot \min\left(1, \frac{|C_{words}|}{200}\right)
$$

### 3. Conflict Sensitivity (C)

Measures logical consistency under perturbation:

$$
C = \frac{1}{|C|} \sum_{i=1}^{|C|} \max_k [P_{NLI}(\text{contradiction} | A, c_i^{(k)})]
$$

**Enhancement: Falsifiability Score**
Combines conflict with linguistic confidence:

$$
F = C_{base} \times (1 + 0.1 \cdot (n_{definitive} - n_{hedge}))
$$

Where:
- $n_{definitive}$ = count of "definitely", "certainly", "clearly"
- $n_{hedge}$ = count of "possibly", "maybe", "perhaps"

### 4. Trace Validation (R)

Measures reasoning consistency:

$$
R = \frac{2}{M(M-1)} \sum_{j<k} \text{Jaccard}(T_j, T_k)
$$

Where $T_m$ are reasoning traces explaining why claims are true.

## Supervised Learning

We train lightweight classifiers on extracted features:

### Feature Engineering

From 4 base signals + 3 enhancements:
- Raw signals: $[U, S, C, R, U_{entity}, A_{context}, F]$
- Interactions: $U \times S$, $C \times R$, etc.
- Ratios: $U/S$, $C/U$, etc.
- Polynomials: $U^2$, $S^2$, etc.

**Result**: ~20-60 engineered features

### Model Architecture

**Stacking Ensemble**:
1. **Base Models**: Random Forest, Gradient Boosting, Neural Network
2. **Meta-Learner**: Logistic Regression on base predictions
3. **Optimal Threshold**: Learned via Youden's J statistic

```python
# Stacking pipeline
base_models = [
    RandomForestClassifier(n_estimators=100),
    GradientBoostingClassifier(n_estimators=100),
    MLPClassifier(hidden_layers=(64, 32))
]

meta_model = LogisticRegression()

# Train
base_preds = [model.fit_predict(X_train) for model in base_models]
meta_model.fit(base_preds, y_train)
```

## Evaluation Protocol

### Dataset

**HaluBench** (PatronusAI, 2024):
- $n=200$ examples (perfectly balanced)
- RAG setting: question, context, answer, ground truth label
- Domains: QA, summarization, dialogue

### Metrics

| Metric | Description | Why Important |
|--------|-------------|---------------|
| **AUROC** | Area under ROC curve | Threshold-independent performance |
| **AUPRC** | Area under PR curve | Better for imbalanced data |
| **Accuracy** | Correct predictions at optimal threshold | Practical performance |
| **F1 Score** | Harmonic mean of precision/recall | Balanced measure |

### Ablation Study

Systematic evaluation of signal contributions:

```python
configurations = [
    'baseline_theory_guided',    # No ML, just theory
    'base_signals',              # U, S, C, R only
    'base_supervised',           # + supervised learning
    'improved_signals',          # + entity, adherence, falsifiability
    'improved_supervised'        # Full system
]
```

## Results Summary

| Configuration | AUROC | Accuracy | Params | Inference |
|---------------|-------|----------|--------|-----------|
| Theory-Guided | 0.8017 | 77.0% | 0 | 5ms |
| Base Supervised | 0.8274 | 79.0% | <1M | 5ms |
| **Improved Supervised** | **0.8669** | **81.5%** | **<1M** | **5ms** |
| Lynx (70B) | 0.874 | 87.4% | 70B | 5s |

**Key Findings**:
1. Theory-guided baseline achieves 0.8017 AUROC (no training!)
2. Supervised learning adds +2.6% AUROC
3. SOTA enhancements add +4.0% AUROC
4. **Total gain**: +8.5% from theory to improved

## Comparison with Baselines

### vs. Self-Consistency Methods

| Method | AUROC | Cost | Latency |
|--------|-------|------|---------|
| SelfCheckGPT | 0.750 | High (10+ samples) | 10s |
| Semantic Entropy | 0.780 | High | 15s |
| **PCIB** | **0.8669** | **Low** | **5ms** |

### vs. LLM Judges

| Method | AUROC | Training Data | Cost/1M |
|--------|-------|---------------|---------|
| Lynx (70B) | 0.874 | 15,000 | $3.00 |
| Claude Judge | 0.850 | 0 (prompt) | $15.00 |
| **PCIB** | **0.8669** | **200** | **$0.15** |

**Advantages**:
- 75× less training data
- 20× lower cost
- 1000× faster
- Fully interpretable

## Negative Results

### Rationalization Signal Fails

**Hypothesis**: Checking reasoning consistency should detect hallucinations.

**Result**: Rationalization signal shows **no discriminative power** (AUROC ≈ 0.5).

**Explanation**: LLMs exhibit **sycophancy** - they generate coherent post-hoc rationalizations for false premises, making trace consistency useless.

**Citation**: Turpin et al. (2023) - "Language Models Don't Always Say What They Think"

## References

1. Friston, K. (2010). The free-energy principle: A unified brain theory? *Nature Reviews Neuroscience*.
2. Tishby, N., & Zaslavsky, N. (2015). Deep learning and the information bottleneck principle. *ITW*.
3. Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*.
4. Turpin, M. et al. (2023). Language models don't always say what they think. *arXiv*.

## Reproducibility

All experiments are reproducible:

```bash
# Run full ablation study
python ablation_study.py --limit 500 --model gpt-4o-mini

# Run SOTA comparison
python pcib_signal_stacking_sota.py

# Generate paper figures
python generate_tikz_data.py
```

Outputs:
- Raw data (JSON)
- LaTeX tables
- Publication figures
- Statistical tests
