# Ablation Study Guide

**Complete one-click ablation study for PCIB detector paper generation**

## Overview

The [`ablation_study.py`](ablation_study.py) script provides a comprehensive ablation study framework that:

1. ✅ Evaluates multiple detector configurations in **parallel**
2. ✅ Stratifies dataset by answer length (short/medium/long)
3. ✅ Computes metrics with **bootstrap confidence intervals**
4. ✅ Generates **LaTeX tables** ready for paper submission
5. ✅ Creates **publication-quality plots** (PDF + PNG)
6. ✅ Performs **statistical significance tests**
7. ✅ Provides **executive summary** with key findings
8. ✅ Saves **complete raw data** with UUID traceability

## Quick Start

```bash
# Set your API key
export OPENAI_API_KEY=sk-...

# Run complete study (500 examples, ~30 minutes)
python ablation_study.py --limit 500 --model gpt-4o-mini

# Output directory: ablation_results/
```

## Usage

### Basic Command

```bash
python ablation_study.py --limit <N> --model <MODEL>
```

### All Options

```bash
python ablation_study.py \
    --dataset PatronusAI/HaluBench \
    --split test \
    --limit 500 \
    --model gpt-4o-mini \
    --output-dir ablation_results \
    --no-plots \
    --no-stratify
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `PatronusAI/HaluBench` | Hugging Face dataset ID |
| `--split` | (auto) | Dataset split (auto-detects test/train) |
| `--limit` | `500` | Number of examples per configuration |
| `--model` | `gpt-4o-mini` | OpenAI model to use |
| `--output-dir` | `ablation_results` | Output directory |
| `--no-plots` | (flag) | Skip plot generation |
| `--no-stratify` | (flag) | Skip dataset stratification |

## Configurations Evaluated

The script evaluates **6 configurations** across **4 stratifications** (24 parallel tasks):

### 1. Baseline
- **Name**: `baseline`
- **Description**: Base PCIB (uptake + stress + conflict)
- **Cost**: 3× API calls
- **Purpose**: Establish baseline performance

### 2. PCIB + Traces (All Signals)
- **Name**: `pcib_plus_traces`
- **Description**: PCIB with full trace validation + rationalization detection
- **Cost**: 6× API calls
- **Purpose**: Measure full system performance

### 3. PCIB + Traces (No Rationalization)
- **Name**: `pcib_no_rationalization`
- **Description**: PCIB with trace validation but no rationalization detection
- **Cost**: 5× API calls
- **Purpose**: Isolate rationalization detection value

### 4-6. Ensemble Variants
- **Names**: `ensemble_1`, `ensemble_3`, `ensemble_5`
- **Description**: PCIB with n verification samples per claim
- **Cost**: 3n× API calls
- **Purpose**: Measure ensemble improvement vs cost

### Stratifications

Each configuration is evaluated on:

1. **All**: Full balanced dataset (500 examples)
2. **Short**: Answers < 50 words
3. **Medium**: Answers 50-150 words  
4. **Long**: Answers > 150 words

## Output Files

The script generates a comprehensive set of outputs in the specified `--output-dir`:

### 1. Raw Data (JSON)

**File**: `raw_data_{model}_{uuid}_{timestamp}.json`

Complete results including:
- All metrics with confidence intervals
- Per-example predictions
- Per-claim signals
- Configuration details
- Run metadata (UUID, timestamp, model, dataset)

```json
{
  "all": {
    "baseline": {
      "description": "Base PCIB...",
      "metrics": {
        "n_examples": 500,
        "auroc": 0.8567,
        "auroc_ci": [0.8234, 0.8891],
        "auprc": 0.8234,
        ...
      },
      "examples": [...]
    },
    ...
  },
  "_metadata": {
    "run_id": "a1b2c3d4",
    "timestamp": "20260111_123456",
    "model": "gpt-4o-mini",
    ...
  }
}
```

### 2. Metrics Summary (JSON)

**File**: `metrics.json`

Condensed metrics-only version (lighter, for quick reference):

```json
{
  "all": {
    "baseline": {
      "description": "...",
      "metrics": { ... }
    }
  }
}
```

### 3. LaTeX Table

**File**: `table_ablations.tex`

Publication-ready table with:
- AUROC with 95% CI
- AUPRC with 95% CI
- F1 score with 95% CI
- Relative cost (API calls)

```latex
\begin{table}[t]
\centering
\caption{Ablation Study Results on HaluBench...}
\begin{tabular}{l c c c c}
\toprule
Configuration & AUROC & AUPRC & F1 & Cost \\
\midrule
\textbf{Baseline} \\
Baseline & 0.857 (0.823-0.889) & ... & ... & 3.0x \\
\textbf{Trace Validation} \\
PCIB + Traces & 0.892 (0.861-0.923) & ... & ... & 6.0x \\
...
\end{tabular}
\end{table}
```

**Usage**: Copy directly into your LaTeX paper!

### 4. Methodology Section

**File**: `methodology_section.tex`

LaTeX text describing the ablation study:

```latex
\subsection{Ablation Study}

We conduct a systematic ablation study...

\begin{itemize}
\item \textbf{Baseline}:
\begin{itemize}
\item \textit{Baseline}: Base PCIB...
\end{itemize}
...
\end{itemize}
```

**Usage**: Insert into your paper's methodology section.

### 5. Executive Summary

**File**: `executive_summary.txt`

Plain-text summary with:
- Best configuration
- Improvement over baseline
- Trace validation impact
- Rationalization detection value
- Performance by answer length
- Recommendations for paper

```
================================================================================
PCIB ABLATION STUDY - EXECUTIVE SUMMARY
================================================================================

KEY FINDINGS:
================================================================================

1. BEST CONFIGURATION: PCIB_PLUS_TRACES
   - AUROC: 0.8923 (95% CI: 0.8612-0.9234)
   - Improvement over baseline: +4.2%

2. TRACE VALIDATION IMPACT:
   - Baseline AUROC: 0.8567
   - With traces: 0.8923
   - Δ AUROC: +0.0356 (4.2% improvement)
   - Cost: +100% API calls
   - Cost-benefit: 0.036 AUROC per 100% cost

3. RATIONALIZATION DETECTION VALUE:
   - With rationalization: 0.8923
   - Without: 0.8834
   - Δ AUROC: +0.0089
   - ✅ Rationalization detection is VALUABLE

...

RECOMMENDATIONS FOR PAPER:
================================================================================

1. MAIN CLAIM: PCIB + trace validation achieves state-of-the-art performance
2. ABLATION: Each component contributes meaningfully
3. COST-PERFORMANCE TRADEOFF: Quantified and justified
4. ROBUSTNESS: Consistent across answer lengths
```

### 6. Plots (PDF + PNG)

#### a) Performance Comparison

**File**: `figure_performance_comparison.pdf` / `.png`

Three-panel bar chart showing AUROC, AUPRC, and F1 for all configurations.

#### b) Cost-Performance Tradeoff

**File**: `figure_cost_vs_performance.pdf` / `.png`

Scatter plot with:
- X-axis: Relative cost (API calls multiplier)
- Y-axis: AUROC
- Error bars: 95% CI
- Each configuration as a point

Shows which configurations offer best value.

#### c) Stratified Performance

**File**: `figure_stratified_performance.pdf` / `.png`

Grouped bar chart showing AUROC by answer length category (short/medium/long).

Demonstrates robustness across different answer types.

## Performance & Timing

### Expected Runtime

| Examples | Configurations | Total Tasks | Sequential | **Parallel** |
|----------|---------------|-------------|-----------|------------|
| 10 | 6 | 24 | ~10 min | **~1 min** |
| 50 | 6 | 24 | ~60 min | **~5 min** |
| 500 | 6 | 24 | ~12 hrs | **~30 min** |
| 1000 | 6 | 24 | ~24 hrs | **~60 min** |

*With 4 stratifications (all, short, medium, long)*

**Speedup**: ~24× through parallel evaluation!

### API Costs (OpenAI gpt-4o-mini)

| Examples | Baseline | Full System | All Configs |
|----------|----------|-------------|-------------|
| 10 | ~$0.05 | ~$0.10 | ~$0.20 |
| 50 | ~$0.25 | ~$0.50 | ~$1.00 |
| 500 | ~$2.50 | ~$5.00 | ~$10.00 |
| 1000 | ~$5.00 | ~$10.00 | ~$20.00 |

*Approximate costs; actual varies by answer/evidence length*

### Memory Usage

- **Typical**: 1-2 GB RAM
- **Peak**: 3-4 GB RAM (all tasks in memory)
- **Disk**: ~10-50 MB per 500 examples

## Implementation Details

### Parallel Execution

The script uses `asyncio.gather()` to run all configurations and stratifications in parallel:

```python
# All 6 configs × 4 stratifications = 24 tasks run simultaneously
async def evaluate_stratification(strat_name, subset):
    # Create tasks for all configs
    tasks = [evaluate_configuration(cfg, subset) for cfg in all_configs]
    # Run in parallel
    results = await asyncio.gather(*tasks)
    return results

# All stratifications run in parallel too
strat_tasks = [evaluate_stratification(s, data) for s in strats]
all_results = await asyncio.gather(*strat_tasks)
```

**Benefits**:
- ~24× speedup over sequential execution
- Maximizes API throughput
- Automatic error isolation (one failure doesn't crash others)

### Dataset Stratification

Balanced sampling ensures long answers are represented:

```python
# Target: 33% from each category
per_category = limit // 3

short_sample = random.sample(short, per_category)    # <50 words
medium_sample = random.sample(medium, per_category)  # 50-150 words
long_sample = random.sample(long, per_category)      # >150 words

all_examples = short_sample + medium_sample + long_sample
```

This prevents the dataset from being dominated by short answers (which are most common in HaluBench).

### Bootstrap Confidence Intervals

Metrics include 95% CI via bootstrap resampling:

```python
def bootstrap_ci(y_true, y_score, metric_fn, n_bootstrap=1000):
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        scores.append(metric_fn(y_true[idx], y_score[idx]))
    
    return (percentile(scores, 2.5), percentile(scores, 97.5))
```

Provides robust uncertainty estimates for paper claims.

### Error Handling

Each configuration evaluation includes:

1. **Timeouts**: 5-minute limit per example
2. **Exception catching**: Individual failures logged but don't crash study
3. **Empty dataset handling**: Graceful handling of missing stratifications
4. **List conversion**: Ensures datasets are reusable across tasks

```python
try:
    result = await asyncio.wait_for(
        detector.detect_hallucination(...),
        timeout=300.0
    )
except asyncio.TimeoutError:
    print(f"Timeout on example {idx}, skipping...")
    continue
except Exception as e:
    print(f"Error on example {idx}: {e}")
    traceback.print_exc()
    continue
```

## Customization

### Adding New Configurations

Edit the `get_ablation_configs()` function:

```python
def get_ablation_configs(model: str) -> List[AblationConfig]:
    configs = []
    
    # Your custom configuration
    configs.append(AblationConfig(
        name="my_custom_config",
        description="My detector variant",
        pcib_config=PCIBConfig(
            provider="openai",
            model=model,
            # Your custom settings
            max_claims=10,
            distractor_chars=2000,
            enable_trace_validation=True,
        ),
        category="Custom",
        api_calls_multiplier=7.0
    ))
    
    return configs
```

### Changing Metrics

To add custom metrics, modify `compute_metrics()`:

```python
def compute_metrics(y_true, y_score) -> MetricResult:
    # Standard metrics
    roc = auroc(y_true, y_score)
    ...
    
    # Your custom metric
    custom_metric = my_custom_score(y_true, y_score)
    
    return MetricResult(
        # Add to dataclass first
        custom_metric=custom_metric,
        ...
    )
```

### Custom Plots

Add to the plotting section:

```python
def plot_my_custom_viz(results, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    # Your visualization code
    plt.savefig(output_dir / "figure_my_viz.pdf")
```

## Troubleshooting

### Issue: Script hangs

**Cause**: Deadlock in async tasks or API rate limiting

**Solution**:
```python
# Reduce concurrency in config
PCIBConfig(max_concurrent=2)
```

### Issue: Out of memory

**Cause**: Too many parallel tasks

**Solution**:
```bash
# Reduce limit or disable stratification
python ablation_study.py --limit 100 --no-stratify
```

### Issue: API rate limits

**Cause**: Too many concurrent API calls

**Solution**:
```python
# In Config creation, reduce concurrency
config = PCIBConfig(
    provider="openai",
    model=model,
    max_concurrent=3  # Down from 10
)
```

### Issue: Missing plots

**Cause**: matplotlib not installed

**Solution**:
```bash
pip install matplotlib seaborn

# Or skip plots
python ablation_study.py --no-plots
```

### Issue: Wrong model in filename

**Cause**: Filename uses `args.model` which may contain slashes

**Solution**: Already handled via `model_clean`:
```python
model_clean = args.model.replace("/", "_").replace(":", "_")
filename = f"raw_data_{model_clean}_{uuid}_{timestamp}.json"
```

## Best Practices

### 1. Start Small

```bash
# Test with 10 examples first (~1 minute)
python ablation_study.py --limit 10 --no-plots

# Then scale up
python ablation_study.py --limit 500
```

### 2. Save Intermediate Results

The script automatically saves raw data with UUID. Keep these files for:
- Reproducibility
- Reanalysis without re-running
- Sharing with collaborators

### 3. Version Control Outputs

```bash
# Create a results branch
git checkout -b ablation-results-2024-01

# Commit outputs
git add ablation_results/
git commit -m "Ablation study results (500 examples, gpt-4o-mini)"
```

### 4. Document Your Runs

Keep a log file:

```bash
python ablation_study.py --limit 500 2>&1 | tee ablation.log
```

## Paper Integration Checklist

- [ ] Run ablation study with appropriate `--limit` (≥500 recommended)
- [ ] Review executive summary for key claims
- [ ] Copy `table_ablations.tex` to paper
- [ ] Copy `methodology_section.tex` to paper
- [ ] Include `figure_*.pdf` plots in paper
- [ ] Cite exact run ID and timestamp in paper
- [ ] Archive raw data for reproducibility
- [ ] Document model and dataset versions

## Example Output

```bash
$ python ablation_study.py --limit 500 --model gpt-4o-mini

================================================================================
PCIB ABLATION STUDY - COMPLETE PAPER GENERATION
================================================================================
Run ID: a1b2c3d4
Timestamp: 20260111_123456
Dataset: PatronusAI/HaluBench
Limit: 500 examples per configuration
Model: gpt-4o-mini
Output: ablation_results
================================================================================

📚 Loading dataset...
✅ Loaded 14900 examples from split: test

📊 Categorizing dataset by answer length...
   Short (<50w): 13814
   Medium (50-150w): 877
   Long (>150w): 209
✅ Balanced sampling: 167 short, 167 medium, 166 long
📦 Final dataset: 500 examples total

================================================================================
🚀 RUNNING ALL CONFIGURATIONS IN PARALLEL
================================================================================
Stratifications: ['short', 'medium', 'long', 'all']
Configurations per stratification: 6
Total parallel tasks: 24
⚠️  Progress bars may overlap - this is normal
================================================================================

[... parallel progress bars ...]

✅ all/baseline: AUROC=0.8567 [0.8234-0.8891], AUPRC=0.8234, F1=0.7891
✅ all/pcib_plus_traces: AUROC=0.8923 [0.8612-0.9234], AUPRC=0.8612, F1=0.8234
...

================================================================================
GENERATING OUTPUTS
================================================================================

💾 Raw data saved: ablation_results/raw_data_gpt-4o-mini_a1b2c3d4_20260111_123456.json
💾 Metrics saved: ablation_results/metrics.json
📄 LaTeX table saved to: ablation_results/table_ablations.tex
📄 Methodology section saved to: ablation_results/methodology_section.tex
📋 Executive summary saved to: ablation_results/executive_summary.txt
📊 Performance comparison saved
📊 Cost-performance tradeoff saved
📊 Stratified performance saved

================================================================================
✅ ABLATION STUDY COMPLETE - PAPER-READY OUTPUTS
================================================================================

📁 Output directory: ablation_results/

Generated files:
  📊 metrics.json                          - Complete metrics with CIs
  📄 table_ablations.tex                   - LaTeX table (copy to paper)
  📄 methodology_section.tex               - LaTeX methodology text
  📋 executive_summary.txt                 - Key findings and recommendations
  📈 figure_performance_comparison.pdf     - Performance bar charts
  📈 figure_cost_vs_performance.pdf        - Cost-benefit analysis
  📈 figure_stratified_performance.pdf     - Performance by length

🎓 Ready for paper submission!
```

## Support

For issues with the ablation study script:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review your API key and rate limits
3. Try with `--limit 10` to isolate issues
4. Open an issue with the run log: `python ablation_study.py ... 2>&1 | tee issue.log`

---

**Script**: [`ablation_study.py`](ablation_study.py)  
**Version**: 1.0.0  
**Last Updated**: January 2026
