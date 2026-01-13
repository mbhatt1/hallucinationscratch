# Class Imbalance Analysis - Summary Report

## Overview

This document summarizes the class imbalance analysis performed on the PCIB detector evaluation results, addressing the concern about whether high AUPRC metrics might be artifactually inflated due to class imbalance.

## The Problem

When evaluating binary classifiers, a common pitfall occurs when the positive class dominates the dataset (e.g., 95% positive, 5% negative). In such cases:

- **AUROC** remains near **0.5** (random baseline) if the model isn't learning
- **AUPRC** appears artificially **high** (e.g., 0.95) simply because predicting "positive" is usually correct
- This creates the misleading pattern: **Low AUROC + High AUPRC**

### Example Scenario (from your description)

If you have:
- 95% positive examples (hallucinations)
- 5% negative examples (non-hallucinations)

Then a poor/random model might show:
- AUROC ≈ **0.50** (can't distinguish classes)
- AUPRC ≈ **0.95** (misleadingly high, just because positive is common)
- AUPRC baseline = **0.95** (the % of positive examples)

**The fix:** Treat the minority class as "positive" for evaluation purposes.

## Analysis of Your Datasets

### Results

We analyzed two evaluation result files:

| File | N | Positive | Negative | % Positive | AUROC | AUPRC | Status |
|------|---|----------|----------|------------|-------|-------|--------|
| `pc_ib_results.jsonl` | 24 | 12 | 12 | 50.0% | 0.7292 | 0.7200 | ✓ Balanced |
| `pc_ib_results_fixed.jsonl` | 200 | 100 | 100 | 50.0% | 0.8017 | 0.7387 | ✓ Balanced |

### Key Findings

✅ **No class imbalance detected** - Both datasets are perfectly balanced (50/50)

✅ **AUPRC baseline is 0.5** - Same as AUROC baseline (no artificial inflation)

✅ **Good model performance**:
- AUROC well above 0.5 baseline (+0.23 to +0.30)
- AUPRC well above 0.5 baseline (+0.22 to +0.24)
- Clear score separation between classes (0.35 margin)

✅ **No problematic pattern** - The "low AUROC + high AUPRC" pattern is NOT present

## Score Distribution Analysis

**Large dataset (n=200):**
- **Positive class (hallucinations):** mean = 0.742, std = 0.261
- **Negative class (non-hallucinations):** mean = 0.390, std = 0.242
- **Separation:** 0.352 (good discriminative ability)

This distributional separation confirms the model has learned meaningful representations.

## Detailed Metrics (pc_ib_results_fixed.jsonl)

| Metric | Value |
|--------|-------|
| AUROC | 0.8017 |
| AUPRC | 0.7387 |
| AUPRC Baseline | 0.5000 |
| AUPRC Gain | +0.2387 |
| Accuracy | 78.0% |
| F1-Score | 0.7732 |
| Sensitivity (Recall) | 0.7500 |
| Specificity | 0.8100 |
| Precision (PPV) | 0.7979 |
| Best Threshold | 0.516 |
| Youden's J | 0.570 |

### Confusion Matrix (threshold = 0.5)

```
                Predicted Neg | Predicted Pos
Actual Neg          81        |      19
Actual Pos          25        |      75
```

## Visualizations

Generated ROC and Precision-Recall curves showing:
- ROC curves well above diagonal (random baseline)
- PR curves well above 0.5 baseline
- Consistent performance across different dataset sizes

**Output files:**
- `metrics_output/roc_pr_curves.png` - Combined ROC/PR visualization
- `metrics_output/detailed_metrics.json` - Full metrics with thresholds

## What Was Added to the Paper

### New Section (Section 4.1)

Added "Model Validation and Class Balance Analysis" subsection to the Results section, including:

1. **Explanation of the class imbalance issue** - What it is and why it matters
2. **Analysis of our datasets** - Confirming they are balanced
3. **ROC/PR curve visualization** - Figure showing genuine performance
4. **Score distribution analysis** - Demonstrating clear class separation
5. **Citation** - Added reference to Saito & Rehmsmeier (2015) on PR curves and imbalanced datasets

### Modified Files

1. **`paper/main.tex`** - Added new subsection with analysis and figure
2. **`paper/references.bib`** - Added citation for class imbalance paper
3. **`paper/roc_pr_curves.png`** - Copied visualization to paper directory

### Paper Status

- ✅ PDF compiled successfully (`paper/main.pdf`, 452 KB)
- ✅ All references resolved
- ✅ Figures included and properly referenced

## Tools Created

### 1. `analyze_class_imbalance.py`

Comprehensive class imbalance detector that:
- Analyzes JSONL result files
- Detects problematic patterns (low AUROC + high AUPRC)
- Computes baselines and gains
- Identifies severe/moderate imbalance
- Provides actionable recommendations

**Usage:**
```bash
python3 analyze_class_imbalance.py
```

### 2. `plot_metrics.py`

Metrics computation and visualization tool that:
- Computes AUROC, AUPRC, confusion matrices
- Finds optimal thresholds (Youden's J, F1)
- Generates ROC and PR curve plots
- Saves detailed metrics to JSON
- Provides summary tables

**Usage:**
```bash
python3 plot_metrics.py
```

## Recommendations

### For Current Project

✅ **Your metrics are valid and interpretable** - No action needed on class imbalance

The balanced datasets ensure that:
- AUROC and AUPRC are both meaningful
- No artificial metric inflation
- Model discrimination ability is genuine

### For Future Projects

Use the provided tools to check for class imbalance:

1. Run `analyze_class_imbalance.py` to detect issues
2. Use `plot_metrics.py` to visualize ROC/PR curves
3. Check for the "low AUROC + high AUPRC" pattern
4. If imbalance exists, consider treating minority class as "positive"

### When Class Imbalance is Detected

If you find severe imbalance (e.g., 90% one class):
- **Option 1:** Flip labels so minority = "positive"
- **Option 2:** Use balanced accuracy or F1 on minority class
- **Option 3:** Apply resampling/reweighting techniques
- **Always report:** Class distribution and AUPRC baseline

## Conclusion

Your evaluation results **do not** exhibit the problematic class imbalance pattern described. The datasets are perfectly balanced (50/50), and both AUROC and AUPRC demonstrate genuine model performance above their respective baselines. The paper has been updated to include this validation, ensuring readers understand the metrics are reliable and not artifactually inflated.
