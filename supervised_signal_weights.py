#!/usr/bin/env python3
"""
Supervised learning to find optimal signal weights for PCIB detector.
Compares baseline (current unsupervised) vs. learned weights.
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. Data Loading and Feature Extraction
# -------------------------------------------------------------------------
def load_data(filepath='pc_ib_results_fixed.jsonl'):
    """Load data and extract signal features from nested structure."""
    data = []
    
    with open(filepath, 'r') as f:
        for line in f:
            entry = json.loads(line)
            
            # Current score (from existing aggregation)
            current_score = entry.get('score', 0)
            
            # Extract signals from claims
            claims = entry.get('claims', [])
            
            if not claims:
                continue
            
            # Aggregate signals across all claims for this example
            uptake_vals = []
            stress_vals = []
            conflict_vals = []
            
            for claim in claims:
                # Prior and post probabilities
                prior = claim.get('prior', {})
                post = claim.get('post', {})
                
                # Uptake: KL divergence (higher = more context uptake)
                uptake_kl = claim.get('uptake_kl', 0)
                uptake_vals.append(uptake_kl)
                
                # Stress: JS divergence under perturbation (higher = less stable)
                stress_js = claim.get('stress_js', 0)
                stress_vals.append(stress_js)
                
                # Conflict: contradiction signal (higher = more contradictory)
                conflict_js = claim.get('conflict_js', 0)
                conflict_vals.append(conflict_js)
            
            # Aggregate across claims (mean)
            row = {
                'label': entry.get('label', 0),  # 1 = hallucination, 0 = factual
                'current_score': current_score,
                'uptake': np.mean(uptake_vals) if uptake_vals else 0,
                'stress': np.mean(stress_vals) if stress_vals else 0,
                'conflict': np.mean(conflict_vals) if conflict_vals else 0,
                'n_claims': len(claims)
            }
            data.append(row)
    
    return pd.DataFrame(data)

# -------------------------------------------------------------------------
# 2. Supervised Model Training
# -------------------------------------------------------------------------
def train_supervised_model(df, features=['uptake', 'stress', 'conflict']):
    """Train logistic regression with cross-validation to avoid overfitting."""
    X = df[features].values
    y = df['label'].values
    
    print(f"Training on {len(df)} examples")
    print(f"Class distribution: {np.sum(y==1)} positive, {np.sum(y==0)} negative")
    
    # Logistic regression with balanced class weights
    model = LogisticRegression(
        class_weight='balanced',
        solver='liblinear',
        random_state=42,
        max_iter=1000
    )
    
    # Cross-validation predictions (prevents overfitting assessment)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_scores_cv = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
    
    # Also get CV AUROC scores per fold for variance
    cv_aurocs = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    cv_auprcs = cross_val_score(model, X, y, cv=cv, scoring='average_precision')
    
    # Train on full data to extract weights (for interpretation only)
    model.fit(X, y)
    
    return model, y_scores_cv, cv_aurocs, cv_auprcs

# -------------------------------------------------------------------------
# 3. Evaluation
# -------------------------------------------------------------------------
def evaluate_method(y_true, y_scores, method_name):
    """Evaluate and print performance metrics."""
    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)
    
    # Baseline is 0.5 for both metrics (balanced dataset)
    auprc_baseline = np.mean(y_true)
    
    print(f"\n{'='*60}")
    print(f"{method_name}")
    print(f"{'='*60}")
    print(f"AUROC: {auroc:.4f} (baseline: 0.5000, gain: {auroc-0.5:+.4f})")
    print(f"AUPRC: {auprc:.4f} (baseline: {auprc_baseline:.4f}, gain: {auprc-auprc_baseline:+.4f})")
    
    # Find optimal threshold (Youden's J)
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    # Metrics at optimal threshold
    y_pred = (y_scores >= best_threshold).astype(int)
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\nAt optimal threshold ({best_threshold:.4f}):")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"  Youden's J:  {j_scores[best_idx]:.4f}")
    
    return auroc, auprc

# -------------------------------------------------------------------------
# 4. Main Analysis
# -------------------------------------------------------------------------
def main():
    print("Loading data...")
    df = load_data('pc_ib_results_fixed.jsonl')
    
    print(f"\nLoaded {len(df)} samples")
    print(f"Features: {df.columns.tolist()}")
    print(f"\nFeature statistics:")
    print(df[['uptake', 'stress', 'conflict', 'current_score']].describe())
    
    # Method 1: Current approach (existing scores)
    print("\n" + "="*60)
    print("METHOD 1: Current Unsupervised Aggregation")
    print("="*60)
    auroc_current, auprc_current = evaluate_method(
        df['label'], 
        df['current_score'], 
        "Current PCIB Scores"
    )
    
    # Method 2: Supervised learning
    print("\n" + "="*60)
    print("METHOD 2: Supervised Signal Aggregation (Logistic Regression)")
    print("="*60)
    
    features = ['uptake', 'stress', 'conflict']
    model, y_scores_supervised, cv_aurocs, cv_auprcs = train_supervised_model(df, features)
    
    auroc_supervised, auprc_supervised = evaluate_method(
        df['label'],
        y_scores_supervised,
        "Supervised PCIB (Cross-Validated)"
    )
    
    # Show learned weights
    print(f"\n{'='*60}")
    print("LEARNED FEATURE WEIGHTS")
    print(f"{'='*60}")
    weights_df = pd.DataFrame({
        'Feature': features,
        'Weight': model.coef_[0],
        'Abs_Weight': np.abs(model.coef_[0])
    }).sort_values('Abs_Weight', ascending=False)
    
    print(weights_df[['Feature', 'Weight']])
    print(f"\nIntercept: {model.intercept_[0]:.4f}")
    
    print(f"\nInterpretation:")
    print(f"  - Positive weights increase hallucination probability")
    print(f"  - Negative weights decrease hallucination probability")
    print(f"  - Magnitude indicates importance")
    
    # Cross-validation variance
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION STABILITY")
    print(f"{'='*60}")
    print(f"Supervised AUROC across 5 folds: {cv_aurocs.mean():.4f} ± {cv_aurocs.std():.4f}")
    print(f"Supervised AUPRC across 5 folds: {cv_auprcs.mean():.4f} ± {cv_auprcs.std():.4f}")
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Method':<30} {'AUROC':>10} {'AUPRC':>10} {'AUROC Gain':>12}")
    print("-" * 64)
    print(f"{'Current (Unsupervised)':<30} {auroc_current:>10.4f} {auprc_current:>10.4f} {auroc_current-0.5:>12.4f}")
    print(f"{'Supervised (LogReg)':<30} {auroc_supervised:>10.4f} {auprc_supervised:>10.4f} {auroc_supervised-0.5:>12.4f}")
    print(f"{'Improvement':<30} {auroc_supervised-auroc_current:>10.4f} {auprc_supervised-auprc_current:>10.4f}")
    
    improvement_pct = ((auroc_supervised - auroc_current) / (auroc_current - 0.5)) * 100
    print(f"\nRelative AUROC improvement: {improvement_pct:+.2f}%")
    
    # Save results
    results = {
        'current': {
            'auroc': float(auroc_current),
            'auprc': float(auprc_current)
        },
        'supervised': {
            'auroc': float(auroc_supervised),
            'auprc': float(auprc_supervised),
            'cv_auroc_mean': float(cv_aurocs.mean()),
            'cv_auroc_std': float(cv_aurocs.std()),
            'weights': {feat: float(w) for feat, w in zip(features, model.coef_[0])},
            'intercept': float(model.intercept_[0])
        },
        'improvement': {
            'auroc_delta': float(auroc_supervised - auroc_current),
            'auprc_delta': float(auprc_supervised - auprc_current),
            'auroc_improvement_pct': float(improvement_pct)
        }
    }
    
    with open('supervised_weights_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to supervised_weights_results.json")

if __name__ == '__main__':
    main()
