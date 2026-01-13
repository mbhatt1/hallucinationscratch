#!/usr/bin/env python3
"""
Analyze evaluation results for class imbalance issues.

This script checks if your dataset exhibits the pattern of:
- Low AUROC (near 0.5) indicating poor discrimination
- High AUPRC due to majority class dominance (misleading metric)
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Any
import sys


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


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC using rank-based method."""
    order = np.argsort(y_score)
    y = y_true[order]
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    
    ranks = np.arange(1, len(y) + 1)
    rank_sum_pos = np.sum(ranks[y == 1])
    u = rank_sum_pos - (n_pos * (n_pos + 1)) / 2
    
    return float(u / (n_pos * n_neg))


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUPRC (precision-recall curve area)."""
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = 0
    fp = 0
    precisions = []
    recalls = []
    n_pos = np.sum(y_true == 1)
    
    if n_pos == 0:
        return float('nan')
    
    for i in range(len(y)):
        if y[i] == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / max(1, tp + fp)
        recall = tp / n_pos
        precisions.append(precision)
        recalls.append(recall)
    
    area = 0.0
    prev_r = 0.0
    prev_p = 1.0
    
    for p, r in zip(precisions, recalls):
        area += (r - prev_r) * ((p + prev_p) / 2.0)
        prev_r = r
        prev_p = p
    
    return float(area)


def analyze_class_imbalance(labels: np.ndarray, scores: np.ndarray, filename: str) -> Dict[str, Any]:
    """Analyze dataset for class imbalance patterns."""
    n_total = len(labels)
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    pct_pos = (n_pos / n_total) * 100
    
    # Compute metrics
    auroc = compute_auroc(labels, scores)
    auprc = compute_auprc(labels, scores)
    
    # Baselines
    auroc_baseline = 0.5
    auprc_baseline = n_pos / n_total
    
    # Score statistics by class
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    
    pos_mean = np.mean(pos_scores) if len(pos_scores) > 0 else float('nan')
    pos_std = np.std(pos_scores) if len(pos_scores) > 0 else float('nan')
    neg_mean = np.mean(neg_scores) if len(neg_scores) > 0 else float('nan')
    neg_std = np.std(neg_scores) if len(neg_scores) > 0 else float('nan')
    
    # Detect problematic patterns
    issues = []
    warnings = []
    
    # Pattern 1: Low AUROC + High AUPRC (the main issue from the user's description)
    if auroc < 0.6 and auprc > 0.85:
        issues.append("LOW_AUROC_HIGH_AUPRC")
        warnings.append(
            f"⚠️  CRITICAL: Low AUROC ({auroc:.4f}) but high AUPRC ({auprc:.4f}) detected!\n"
            f"    This pattern indicates the model is NOT learning (AUROC near random {auroc_baseline:.2f}),\n"
            f"    but AUPRC appears high only because the positive class is the majority ({pct_pos:.1f}%).\n"
            f"    The AUPRC baseline for this dataset is {auprc_baseline:.4f}, so the actual performance\n"
            f"    above baseline is only {auprc - auprc_baseline:.4f}."
        )
    
    # Pattern 2: Severe class imbalance
    if pct_pos > 85 or pct_pos < 15:
        issues.append("SEVERE_IMBALANCE")
        minority_class = "negative" if pct_pos > 50 else "positive"
        warnings.append(
            f"⚠️  SEVERE CLASS IMBALANCE: {pct_pos:.1f}% positive, {100-pct_pos:.1f}% negative\n"
            f"    The {minority_class} class is severely underrepresented.\n"
            f"    Consider treating the minority class as 'positive' for evaluation purposes."
        )
    elif pct_pos > 70 or pct_pos < 30:
        issues.append("MODERATE_IMBALANCE")
        warnings.append(
            f"⚠️  MODERATE CLASS IMBALANCE: {pct_pos:.1f}% positive\n"
            f"    AUPRC baseline is {auprc_baseline:.4f} (not 0.5), so interpret AUPRC carefully."
        )
    
    # Pattern 3: Low AUROC (poor discrimination)
    if auroc < 0.6:
        issues.append("POOR_DISCRIMINATION")
        warnings.append(
            f"⚠️  POOR DISCRIMINATION: AUROC = {auroc:.4f} (close to random baseline {auroc_baseline:.2f})\n"
            f"    The model cannot effectively distinguish between positive and negative classes."
        )
    
    # Pattern 4: Overlapping score distributions
    if abs(pos_mean - neg_mean) < 0.2:
        issues.append("OVERLAPPING_DISTRIBUTIONS")
        warnings.append(
            f"⚠️  OVERLAPPING SCORE DISTRIBUTIONS:\n"
            f"    Positive class: mean={pos_mean:.4f}, std={pos_std:.4f}\n"
            f"    Negative class: mean={neg_mean:.4f}, std={neg_std:.4f}\n"
            f"    The score distributions are too similar, indicating weak signal."
        )
    
    # Good performance
    if auroc >= 0.7 and not issues:
        warnings.append(
            f"✓ GOOD PERFORMANCE: AUROC = {auroc:.4f}, AUPRC = {auprc:.4f}\n"
            f"  Classes are reasonably balanced ({pct_pos:.1f}% positive)\n"
            f"  Model shows good discrimination ability."
        )
    
    return {
        'filename': filename,
        'n_total': n_total,
        'n_positive': n_pos,
        'n_negative': n_neg,
        'pct_positive': pct_pos,
        'auroc': auroc,
        'auprc': auprc,
        'auroc_baseline': auroc_baseline,
        'auprc_baseline': auprc_baseline,
        'pos_mean': pos_mean,
        'pos_std': pos_std,
        'neg_mean': neg_mean,
        'neg_std': neg_std,
        'issues': issues,
        'warnings': warnings,
    }


def print_analysis(analysis: Dict[str, Any]):
    """Print formatted analysis results."""
    print("=" * 80)
    print(f"FILE: {analysis['filename']}")
    print("=" * 80)
    
    print(f"\nDATASET STATISTICS:")
    print(f"  Total examples: {analysis['n_total']}")
    print(f"  Positive (1 = hallucination): {analysis['n_positive']} ({analysis['pct_positive']:.1f}%)")
    print(f"  Negative (0 = no hallucination): {analysis['n_negative']} ({100 - analysis['pct_positive']:.1f}%)")
    
    print(f"\nMETRIC BASELINES:")
    print(f"  AUROC baseline (always): {analysis['auroc_baseline']:.4f}")
    print(f"  AUPRC baseline (% positive): {analysis['auprc_baseline']:.4f}")
    
    print(f"\nACTUAL METRICS:")
    print(f"  AUROC: {analysis['auroc']:.4f}")
    print(f"  AUPRC: {analysis['auprc']:.4f}")
    print(f"  AUROC above baseline: {analysis['auroc'] - analysis['auroc_baseline']:+.4f}")
    print(f"  AUPRC above baseline: {analysis['auprc'] - analysis['auprc_baseline']:+.4f}")
    
    print(f"\nSCORE DISTRIBUTIONS:")
    print(f"  Positive class: mean={analysis['pos_mean']:.4f}, std={analysis['pos_std']:.4f}")
    print(f"  Negative class: mean={analysis['neg_mean']:.4f}, std={analysis['neg_std']:.4f}")
    print(f"  Separation: {abs(analysis['pos_mean'] - analysis['neg_mean']):.4f}")
    
    if analysis['issues']:
        print(f"\nISSUES DETECTED: {', '.join(analysis['issues'])}")
    else:
        print(f"\nISSUES DETECTED: None")
    
    print(f"\nANALYSIS:")
    for warning in analysis['warnings']:
        print(f"\n{warning}")
    
    print("\n")


def main():
    """Main analysis function."""
    # Find all result files
    result_files = [
        'pc_ib_results.jsonl',
        'pc_ib_results_fixed.jsonl',
    ]
    
    # Add ablation results
    ablation_dir = Path('ablation_results/foo')
    if ablation_dir.exists():
        for raw_file in ablation_dir.glob('raw_data_*.json'):
            result_files.append(str(raw_file))
    
    analyses = []
    
    for filepath in result_files:
        if not Path(filepath).exists():
            continue
        
        print(f"\nAnalyzing {filepath}...")
        
        try:
            # Handle both JSONL and JSON formats
            if filepath.endswith('.jsonl'):
                labels, scores = load_results(filepath)
            else:
                # For raw_data JSON files from ablation study
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if 'predictions' in data:
                        labels = np.array([p['label'] for p in data['predictions']])
                        scores = np.array([p['score'] for p in data['predictions']])
                    else:
                        print(f"  Skipping {filepath}: unsupported format")
                        continue
            
            if len(labels) == 0:
                print(f"  Skipping {filepath}: no data found")
                continue
            
            analysis = analyze_class_imbalance(labels, scores, filepath)
            analyses.append(analysis)
            print_analysis(analysis)
            
        except Exception as e:
            print(f"  Error analyzing {filepath}: {e}")
            continue
    
    # Summary
    if analyses:
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        total_issues = sum(len(a['issues']) for a in analyses)
        files_with_issues = sum(1 for a in analyses if a['issues'])
        
        print(f"\nAnalyzed {len(analyses)} file(s)")
        print(f"Files with issues: {files_with_issues}/{len(analyses)}")
        print(f"Total issues found: {total_issues}")
        
        # Count issue types
        issue_counts = Counter()
        for analysis in analyses:
            for issue in analysis['issues']:
                issue_counts[issue] += 1
        
        if issue_counts:
            print(f"\nIssue breakdown:")
            for issue, count in issue_counts.most_common():
                print(f"  {issue}: {count}")
        
        # Recommendations
        print(f"\n{'=' * 80}")
        print("RECOMMENDATIONS")
        print("=" * 80)
        
        has_critical = any('LOW_AUROC_HIGH_AUPRC' in a['issues'] for a in analyses)
        has_imbalance = any('SEVERE_IMBALANCE' in a['issues'] or 'MODERATE_IMBALANCE' in a['issues'] 
                           for a in analyses)
        has_poor_disc = any('POOR_DISCRIMINATION' in a['issues'] for a in analyses)
        
        if has_critical:
            print("\n🔴 CRITICAL: Class Imbalance Masking Poor Performance")
            print("   Your high AUPRC is misleading due to class imbalance.")
            print("   Action: Flip labels so the minority class is 'positive' for evaluation.")
            print("   This will reveal the true (lower) AUPRC that aligns with the low AUROC.")
        
        if has_imbalance and not has_critical:
            print("\n⚠️  Class Imbalance Detected")
            print("   Action: Consider using the minority class as 'positive' for evaluation,")
            print("   or use additional metrics like balanced accuracy, F1-score on minority class.")
        
        if has_poor_disc:
            print("\n⚠️  Poor Model Discrimination")
            print("   Your model is performing near random (AUROC ≈ 0.5).")
            print("   Action: Review model architecture, features, training data quality.")
        
        if not has_critical and not has_imbalance and not has_poor_disc:
            print("\n✓ No major class imbalance issues detected.")
            print("  Your metrics appear to be reliable and interpretable.")
    
    print("\n")


if __name__ == "__main__":
    main()
