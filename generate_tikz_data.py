
#!/usr/bin/env python3
"""
Generate TikZ plot data from evaluation results for direct embedding in LaTeX.
"""

import json
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

def load_data(filepath):
    """Load labels and scores."""
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

def generate_tikz_coordinates(x, y, max_points=50):
    """Generate TikZ coordinate string, downsampling if needed."""
    # Downsample to max_points for cleaner LaTeX
    if len(x) > max_points:
        indices = np.linspace(0, len(x)-1, max_points, dtype=int)
        x = x[indices]
        y = y[indices]
    
    coords = []
    for xi, yi in zip(x, y):
        coords.append(f"({xi:.4f},{yi:.4f})")
    
    return " ".join(coords)

def main():
    # Load large dataset
    labels, scores = load_data('pc_ib_results_fixed.jsonl')
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    
    # Compute PR curve
    precision, recall, _ = precision_recall_curve(labels, scores)
    
    # Generate TikZ code
    print("% TikZ data for ROC and PR curves")
    print("% Generated from pc_ib_results_fixed.jsonl (n=200)")
    print()
    
    print("% ROC Curve coordinates (FPR, TPR)")
    print(f"% AUROC: {np.trapz(tpr, fpr):.4f}")
    roc_coords = generate_tikz_coordinates(fpr, tpr)
    print(f"\\def\\roccoords{{{roc_coords}}}")
    print()
    
    print("% PR Curve coordinates (Recall, Precision)")
    print(f"% AUPRC: {np.trapz(precision, recall):.4f}")
    pr_coords = generate_tikz_coordinates(recall, precision)
    print(f"\\def\\prcoords{{{pr_coords}}}")
    print()
    
    # Also save to file
    with open('paper/tikz_plot_data.tex', 'w') as f:
        f.write("% TikZ data for ROC and PR curves\n")
        f.write("% Generated from pc_ib_results_fixed.jsonl (n=200)\n\n")
        f.write(f"% ROC Curve: AUROC = {np.trapz(tpr, fpr):.4f}\n")
        f.write(f"\\def\\roccoords{{{roc_coords}}}\n\n")
        f.write(f"% PR Curve: AUPRC = {np.trapz(precision, recall):.4f}\n")
        f.write(f"\\def\\prcoords{{{pr_coords}}}\n")
    
    print("✓ Saved to paper/tikz_plot_data.tex")

if __name__ == '__main__':
    main()
