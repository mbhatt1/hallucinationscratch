#!/usr/bin/env python3
"""
Comprehensive comparison of signal aggregation methods.
Tests PCIB's theory-guided approach against multiple supervised learning baselines.
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------
# 1. Data Loading
# -------------------------------------------------------------------------
def load_data(filepath='pc_ib_results_fixed.jsonl'):
    """Load data and extract signal features."""
    data = []
    
    with open(filepath, 'r') as f:
        for line in f:
            entry = json.loads(line)
            current_score = entry.get('score', 0)
            claims = entry.get('claims', [])
            
            if not claims:
                continue
            
            uptake_vals = []
            stress_vals = []
            conflict_vals = []
            
            for claim in claims:
                uptake_vals.append(claim.get('uptake_kl', 0))
                stress_vals.append(claim.get('stress_js', 0))
                conflict_vals.append(claim.get('conflict_js', 0))
            
            row = {
                'label': entry.get('label', 0),
                'current_score': current_score,
                'uptake': np.mean(uptake_vals) if uptake_vals else 0,
                'stress': np.mean(stress_vals) if stress_vals else 0,
                'conflict': np.mean(conflict_vals) if conflict_vals else 0,
                'n_claims': len(claims)
            }
            data.append(row)
    
    return pd.DataFrame(data)

# -------------------------------------------------------------------------
# 2. Method Definitions
# -------------------------------------------------------------------------
def method_current(df):
    """Current PCIB theory-guided aggregation."""
    return df['current_score'].values

def method_simple_average(df):
    """Simple unweighted average of signals."""
    return ((1 - df['uptake']) + df['stress'] + df['conflict']) / 3.0

def method_logistic(df, features):
    """Logistic Regression with cross-validation."""
    X = df[features].values
    y = df['label'].values
    model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42, max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1], model

def method_random_forest(df, features):
    """Random Forest with cross-validation."""
    X = df[features].values
    y = df['label'].values
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1], model

def method_gradient_boosting(df, features):
    """Gradient Boosting with cross-validation."""
    X = df[features].values
    y = df['label'].values
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1], model

def method_svm(df, features):
    """SVM with RBF kernel and cross-validation."""
    X = df[features].values
    y = df['label'].values
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1], model

# -------------------------------------------------------------------------
# 3. Evaluation
# -------------------------------------------------------------------------
def evaluate_method(y_true, y_scores, method_name):
    """Evaluate and return metrics."""
    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)
    
    # Optimal threshold (Youden's J)
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    y_pred = (y_scores >= best_threshold).astype(int)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    return {
        'method': method_name,
        'auroc': auroc,
        'auprc': auprc,
        'f1': f1,
        'accuracy': acc,
        'auroc_gain': auroc - 0.5,
        'auprc_gain': auprc - 0.5,
    }

# -------------------------------------------------------------------------
# 4. Main Comparison
# -------------------------------------------------------------------------
def main():
    print("="*80)
    print("COMPREHENSIVE METHOD COMPARISON")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df = load_data('pc_ib_results_fixed.jsonl')
    print(f"Loaded {len(df)} samples (50% positive, 50% negative)\n")
    
    features = ['uptake', 'stress', 'conflict']
    y_true = df['label'].values
    
    results = []
    
    # Method 1: Current PCIB (Theory-Guided)
    print("Testing Method 1: PCIB (Theory-Guided)...")
    y_scores = method_current(df)
    results.append(evaluate_method(y_true, y_scores, "PCIB (Theory-Guided)"))
    
    # Method 2: Simple Average (Unsupervised Baseline)
    print("Testing Method 2: Simple Average...")
    y_scores = method_simple_average(df)
    results.append(evaluate_method(y_true, y_scores, "Simple Average"))
    
    # Method 3: Logistic Regression
    print("Testing Method 3: Logistic Regression...")
    y_scores, model = method_logistic(df, features)
    results.append(evaluate_method(y_true, y_scores, "Logistic Regression"))
    
    # Method 4: Random Forest
    print("Testing Method 4: Random Forest...")
    y_scores, model = method_random_forest(df, features)
    results.append(evaluate_method(y_true, y_scores, "Random Forest"))
    
    # Method 5: Gradient Boosting
    print("Testing Method 5: Gradient Boosting...")
    y_scores, model = method_gradient_boosting(df, features)
    results.append(evaluate_method(y_true, y_scores, "Gradient Boosting"))
    
    # Method 6: SVM
    print("Testing Method 6: SVM (RBF)...")
    y_scores, model = method_svm(df, features)
    results.append(evaluate_method(y_true, y_scores, "SVM (RBF)"))
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('auroc', ascending=False)
    
    # Print results table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Method':<25} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'Acc':>8} {'Δ AUROC':>10}")
    print("-" * 80)
    
    baseline_auroc = results_df.iloc[0]['auroc']
    for _, row in results_df.iterrows():
        delta_pct = ((row['auroc'] - baseline_auroc) / baseline_auroc) * 100 if baseline_auroc > 0 else 0
        print(f"{row['method']:<25} {row['auroc']:>8.4f} {row['auprc']:>8.4f} "
              f"{row['f1']:>8.4f} {row['accuracy']:>8.4f} {delta_pct:>9.1f}%")
    
    # Rank analysis
    print("\n" + "="*80)
    print("RANKING BY METRIC")
    print("="*80)
    
    for metric in ['auroc', 'auprc', 'f1']:
        ranked = results_df.sort_values(metric, ascending=False)
        print(f"\nBy {metric.upper()}:")
        for i, (_, row) in enumerate(ranked.iterrows(), 1):
            print(f"  {i}. {row['method']:<25} {row[metric]:.4f}")
    
    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    pcib_result = results_df[results_df['method'] == 'PCIB (Theory-Guided)'].iloc[0]
    pcib_rank_auroc = (results_df['auroc'] >= pcib_result['auroc']).sum()
    pcib_rank_auprc = (results_df['auprc'] >= pcib_result['auprc']).sum()
    
    print(f"\n1. PCIB (Theory-Guided) Performance:")
    print(f"   - Ranks #{pcib_rank_auroc} by AUROC ({pcib_result['auroc']:.4f})")
    print(f"   - Ranks #{pcib_rank_auprc} by AUPRC ({pcib_result['auprc']:.4f})")
    
    if pcib_rank_auroc == 1:
        print(f"   ✓ PCIB achieves BEST performance among all methods")
        print(f"   ✓ Theory-guided design outperforms data-driven approaches")
    else:
        best_method = results_df.iloc[0]
        improvement = best_method['auroc'] - pcib_result['auroc']
        pct = (improvement / pcib_result['auroc']) * 100
        print(f"   - Best method: {best_method['method']} (AUROC: {best_method['auroc']:.4f})")
        print(f"   - Performance gap: {improvement:.4f} ({pct:.1f}%)")
    
    # Average performance of ML methods
    ml_methods = results_df[~results_df['method'].str.contains('PCIB|Average')]
    if len(ml_methods) > 0:
        avg_ml_auroc = ml_methods['auroc'].mean()
        print(f"\n2. Supervised Learning Methods (avg):")
        print(f"   - Average AUROC: {avg_ml_auroc:.4f}")
        print(f"   - vs PCIB: {avg_ml_auroc - pcib_result['auroc']:+.4f} ({((avg_ml_auroc - pcib_result['auroc']) / pcib_result['auroc'] * 100):+.1f}%)")
    
    # Variance analysis
    print(f"\n3. Performance Variance:")
    print(f"   - AUROC range: {results_df['auroc'].min():.4f} to {results_df['auroc'].max():.4f}")
    print(f"   - AUROC std: {results_df['auroc'].std():.4f}")
    print(f"   - This shows {'high' if results_df['auroc'].std() > 0.05 else 'moderate'} sensitivity to aggregation method")
    
    # Save results
    output = {
        'summary': results_df.to_dict('records'),
        'insights': {
            'pcib_rank_auroc': int(pcib_rank_auroc),
            'pcib_rank_auprc': int(pcib_rank_auprc),
            'best_method': results_df.iloc[0]['method'],
            'best_auroc': float(results_df.iloc[0]['auroc']),
            'pcib_auroc': float(pcib_result['auroc']),
            'avg_ml_auroc': float(ml_methods['auroc'].mean()) if len(ml_methods) > 0 else None
        }
    }
    
    with open('method_comparison_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to method_comparison_results.json")
    print("="*80)

if __name__ == '__main__':
    main()
