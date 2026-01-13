#!/usr/bin/env python3
"""
Compute comprehensive metrics and produce ROC/PR curves for evaluation results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    roc_auc_score, confusion_matrix, classification_report
)


def load_results(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load labels and scores from JSONL results file."""
    labels = []
    scores = []
    
    with open(filepath, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'label' in data and 'score' in data:
                    labels.append(data['label'])
                    scores.append(data['score'])
            except:
                continue
    
    return np.array(labels), np.array(scores)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Compute comprehensive classification metrics."""
    # Binary predictions at threshold
    y_pred = (y_score >= threshold).astype(int)
    
    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # PR curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0.0
    
    # Find best threshold by Youden's J statistic (sensitivity + specificity - 1)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = roc_thresholds[best_idx]
    best_sensitivity = tpr[best_idx]
    best_specificity = 1 - fpr[best_idx]
    
    # Find best F1 threshold
    y_pred_best_f1 = (y_score >= best_threshold).astype(int)
    best_f1 = f1
    for thr in np.linspace(0, 1, 100):
        y_pred_temp = (y_score >= thr).astype(int)
        tp_temp = np.sum((y_pred_temp == 1) & (y_true == 1))
        fp_temp = np.sum((y_pred_temp == 1) & (y_true == 0))
        fn_temp = np.sum((y_pred_temp == 0) & (y_true == 1))
        prec_temp = tp_temp / max(1, tp_temp + fp_temp)
        rec_temp = tp_temp / max(1, tp_temp + fn_temp)
        f1_temp = 2 * prec_temp * rec_temp / max(1e-10, prec_temp + rec_temp)
        if f1_temp > best_f1:
            best_f1 = f1_temp
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': roc_thresholds,
        'precision': precision,
        'recall': recall,
        'pr_thresholds': pr_thresholds,
        'confusion_matrix': {'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)},
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'f1': f1,
        'best_threshold': best_threshold,
        'best_sensitivity': best_sensitivity,
        'best_specificity': best_specificity,
        'best_f1': best_f1,
        'threshold_used': threshold,
    }


def plot_roc_pr_curves(results: Dict[str, Dict[str, Any]], output_dir: str = 'metrics_output'):
    """Plot ROC and PR curves for all result files."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    for idx, (filename, metrics) in enumerate(results.items()):
        label = Path(filename).stem
        color = colors[idx]
        
        # ROC curve
        ax1.plot(
            metrics['fpr'], 
            metrics['tpr'], 
            color=color,
            lw=2,
            label=f'{label} (AUC = {metrics["roc_auc"]:.3f})'
        )
        
        # PR curve
        ax2.plot(
            metrics['recall'],
            metrics['precision'],
            color=color,
            lw=2,
            label=f'{label} (AP = {metrics["pr_auc"]:.3f})'
        )
    
    # ROC plot formatting
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # PR plot formatting
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'roc_pr_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved ROC/PR curves to: {output_path}")
    
    plt.close()


def print_metrics_report(filename: str, metrics: Dict[str, Any], y_true: np.ndarray):
    """Print detailed metrics report."""
    print("=" * 80)
    print(f"FILE: {filename}")
    print("=" * 80)
    
    n_total = len(y_true)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    print(f"\nDATASET:")
    print(f"  Total examples: {n_total}")
    print(f"  Positive (hallucinations): {n_pos} ({n_pos/n_total*100:.1f}%)")
    print(f"  Negative (non-hallucinations): {n_neg} ({n_neg/n_total*100:.1f}%)")
    
    print(f"\nDISCRIMINATION METRICS:")
    print(f"  AUROC (ROC AUC): {metrics['roc_auc']:.4f}")
    print(f"  AUPRC (PR AUC):  {metrics['pr_auc']:.4f}")
    print(f"  AUPRC Baseline:  {n_pos/n_total:.4f} (% positive)")
    print(f"  AUPRC Gain:      {metrics['pr_auc'] - n_pos/n_total:+.4f}")
    
    print(f"\nCLASSIFICATION METRICS (threshold = {metrics['threshold_used']:.3f}):")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f} (Recall/TPR)")
    print(f"  Specificity: {metrics['specificity']:.4f} (TNR)")
    print(f"  Precision:   {metrics['ppv']:.4f} (PPV)")
    print(f"  NPV:         {metrics['npv']:.4f}")
    print(f"  F1-Score:    {metrics['f1']:.4f}")
    
    print(f"\nOPTIMAL THRESHOLDS:")
    print(f"  Best ROC threshold: {metrics['best_threshold']:.4f}")
    print(f"    → Sensitivity: {metrics['best_sensitivity']:.4f}")
    print(f"    → Specificity: {metrics['best_specificity']:.4f}")
    print(f"    → Youden's J: {metrics['best_sensitivity'] + metrics['best_specificity'] - 1:.4f}")
    print(f"  Best F1-Score: {metrics['best_f1']:.4f}")
    
    cm = metrics['confusion_matrix']
    print(f"\nCONFUSION MATRIX:")
    print(f"                 Predicted Neg | Predicted Pos")
    print(f"  Actual Neg         {cm['TN']:4d}     |     {cm['FP']:4d}")
    print(f"  Actual Pos         {cm['FN']:4d}     |     {cm['TP']:4d}")
    
    print()


def main():
    """Main function to compute metrics and generate plots."""
    # Find result files
    result_files = [
        'pc_ib_results.jsonl',
        'pc_ib_results_fixed.jsonl',
    ]
    
    all_metrics = {}
    
    print("\n" + "=" * 80)
    print("COMPUTING COMPREHENSIVE METRICS")
    print("=" * 80 + "\n")
    
    for filepath in result_files:
        if not Path(filepath).exists():
            continue
        
        print(f"Loading {filepath}...")
        labels, scores = load_results(filepath)
        
        if len(labels) == 0:
            print(f"  Skipping: no data\n")
            continue
        
        print(f"  Found {len(labels)} examples")
        
        # Compute metrics
        metrics = compute_metrics(labels, scores, threshold=0.5)
        all_metrics[filepath] = metrics
        
        # Print detailed report
        print_metrics_report(filepath, metrics, labels)
    
    if all_metrics:
        # Plot ROC and PR curves
        print("\n" + "=" * 80)
        print("GENERATING PLOTS")
        print("=" * 80 + "\n")
        plot_roc_pr_curves(all_metrics)
        
        # Create summary table
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80 + "\n")
        
        print(f"{'File':<30} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'Acc':>8} {'N':>6}")
        print("-" * 80)
        
        for filepath, metrics in all_metrics.items():
            filename = Path(filepath).stem
            print(f"{filename:<30} {metrics['roc_auc']:>8.4f} {metrics['pr_auc']:>8.4f} "
                  f"{metrics['f1']:>8.4f} {metrics['accuracy']:>8.4f} {len(labels):>6}")
        
        print("\n")
        
        # Save metrics to JSON
        output_dir = Path('metrics_output')
        output_dir.mkdir(exist_ok=True)
        
        metrics_output = {}
        for filepath, metrics in all_metrics.items():
            # Convert numpy arrays to lists for JSON serialization
            metrics_copy = metrics.copy()
            metrics_copy['fpr'] = metrics['fpr'].tolist()
            metrics_copy['tpr'] = metrics['tpr'].tolist()
            metrics_copy['roc_thresholds'] = metrics['roc_thresholds'].tolist()
            metrics_copy['precision'] = metrics['precision'].tolist()
            metrics_copy['recall'] = metrics['recall'].tolist()
            metrics_copy['pr_thresholds'] = metrics['pr_thresholds'].tolist()
            metrics_output[filepath] = metrics_copy
        
        output_path = output_dir / 'detailed_metrics.json'
        with open(output_path, 'w') as f:
            json.dump(metrics_output, f, indent=2)
        
        print(f"✓ Saved detailed metrics to: {output_path}\n")


if __name__ == "__main__":
    main()
