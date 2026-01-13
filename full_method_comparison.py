#!/usr/bin/env python3
"""
Full comparison: PCIB vs. supervised AND unsupervised aggregation methods.
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from scipy.stats import gmean
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
# 2. UNSUPERVISED Methods
# -------------------------------------------------------------------------
def method_current(df):
    """Current PCIB theory-guided aggregation."""
    return df['current_score'].values, "UNSUPERVISED"

def method_simple_average(df):
    """Simple unweighted average (inverted uptake)."""
    return ((1 - df['uptake']) + df['stress'] + df['conflict']) / 3.0, "UNSUPERVISED"

def method_arithmetic_mean(df):
    """Arithmetic mean without inversion."""
    return (df['uptake'] + df['stress'] + df['conflict']) / 3.0, "UNSUPERVISED"

def method_weighted_variance(df):
    """Weight by inverse variance (high variance = less weight)."""
    features = df[['uptake', 'stress', 'conflict']]
    variances = features.var()
    weights = 1 / (variances + 1e-6)
    weights = weights / weights.sum()
    
    # Invert uptake since high uptake = low risk
    weighted = (1 - df['uptake']) * weights['uptake'] + \
               df['stress'] * weights['stress'] + \
               df['conflict'] * weights['conflict']
    return weighted.values, "UNSUPERVISED"

def method_max_signal(df):
    """Take maximum of signals (conservative)."""
    return np.maximum.reduce([1 - df['uptake'], df['stress'], df['conflict']]), "UNSUPERVISED"

def method_product(df):
    """Product of signals (penalizes low signals)."""
    # Add small epsilon to avoid zeros
    return ((1 - df['uptake'] + 0.01) * (df['stress'] + 0.01) * (df['conflict'] + 0.01)).values, "UNSUPERVISED"

def method_geometric_mean(df):
    """Geometric mean of signals."""
    signals = np.column_stack([1 - df['uptake'] + 0.1, df['stress'] + 0.1, df['conflict'] + 0.1])
    result = gmean(signals, axis=1)
    # Handle any NaN/inf
    result = np.nan_to_num(result, nan=0.5, posinf=1.0, neginf=0.0)
    return result, "UNSUPERVISED"

def method_harmonic_mean(df):
    """Harmonic mean of signals (emphasizes small values)."""
    signals = np.column_stack([1 - df['uptake'] + 0.1, df['stress'] + 0.1, df['conflict'] + 0.1])
    result = 3 / np.sum(1 / signals, axis=1)
    result = np.nan_to_num(result, nan=0.5, posinf=1.0, neginf=0.0)
    return result, "UNSUPERVISED"

def method_pca_first_component(df):
    """PCA: use first principal component."""
    features = df[['uptake', 'stress', 'conflict']].values
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(features).flatten()
    # Normalize to [0, 1]
    pc1 = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-6)
    return pc1, "UNSUPERVISED"

def method_l2_norm(df):
    """L2 norm (Euclidean distance from origin)."""
    signals = np.column_stack([1 - df['uptake'], df['stress'], df['conflict']])
    return np.linalg.norm(signals, axis=1), "UNSUPERVISED"

# -------------------------------------------------------------------------
# 3. SUPERVISED Methods
# -------------------------------------------------------------------------
def method_logistic(df, features):
    """Logistic Regression with cross-validation."""
    X = df[features].values
    y = df['label'].values
    model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42, max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1], "SUPERVISED"

def method_random_forest(df, features):
    """Random Forest with cross-validation."""
    X = df[features].values
    y = df['label'].values
    model = RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_split=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1], "SUPERVISED"

def method_gradient_boosting(df, features):
    """Gradient Boosting with cross-validation."""
    X = df[features].values
    y = df['label'].values
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1], "SUPERVISED"

def method_svm(df, features):
    """SVM with RBF kernel."""
    X = df[features].values
    y = df['label'].values
    model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced',
                probability=True, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1], "SUPERVISED"

# -------------------------------------------------------------------------
# 4. Evaluation
# -------------------------------------------------------------------------
def evaluate_method(y_true, y_scores, method_name, method_type):
    """Evaluate and return metrics."""
    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)
    
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
        'type': method_type,
        'auroc': auroc,
        'auprc': auprc,
        'f1': f1,
        'accuracy': acc,
        'auroc_gain': auroc - 0.5,
        'auprc_gain': auprc - 0.5,
    }

# -------------------------------------------------------------------------
# 5. Main Comparison
# -------------------------------------------------------------------------
def main():
    print("="*80)
    print("COMPREHENSIVE METHOD COMPARISON")
    print("Unsupervised + Supervised Aggregation Methods")
    print("="*80)
    
    df = load_data('pc_ib_results_fixed.jsonl')
    print(f"\nLoaded {len(df)} samples (50% positive, 50% negative)\n")
    
    features = ['uptake', 'stress', 'conflict']
    y_true = df['label'].values
    
    results = []
    
    # UNSUPERVISED METHODS
    print("Testing UNSUPERVISED methods...")
    print("-" * 80)
    
    unsupervised_methods = [
        ("PCIB (Theory-Guided)", lambda d: method_current(d)),
        ("Simple Average (Inverted Uptake)", lambda d: method_simple_average(d)),
        ("Arithmetic Mean", lambda d: method_arithmetic_mean(d)),
        ("Variance-Weighted", lambda d: method_weighted_variance(d)),
        ("Max Signal", lambda d: method_max_signal(d)),
        ("Product of Signals", lambda d: method_product(d)),
        ("Geometric Mean", lambda d: method_geometric_mean(d)),
        ("Harmonic Mean", lambda d: method_harmonic_mean(d)),
        ("PCA (1st Component)", lambda d: method_pca_first_component(d)),
        ("L2 Norm", lambda d: method_l2_norm(d)),
    ]
    
    for name, method_func in unsupervised_methods:
        print(f"  {name}...")
        y_scores, method_type = method_func(df)
        results.append(evaluate_method(y_true, y_scores, name, method_type))
    
    # SUPERVISED METHODS
    print("\nTesting SUPERVISED methods...")
    print("-" * 80)
    
    supervised_methods = [
        ("Logistic Regression", lambda d, f: method_logistic(d, f)),
        ("Random Forest", lambda d, f: method_random_forest(d, f)),
        ("Gradient Boosting", lambda d, f: method_gradient_boosting(d, f)),
        ("SVM (RBF)", lambda d, f: method_svm(d, f)),
    ]
    
    for name, method_func in supervised_methods:
        print(f"  {name}...")
        y_scores, method_type = method_func(df, features)
        results.append(evaluate_method(y_true, y_scores, name, method_type))
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Sort by AUROC
    results_df_sorted = results_df.sort_values('auroc', ascending=False)
    
    # Print results by category
    print("\n" + "="*80)
    print("RESULTS: UNSUPERVISED METHODS")
    print("="*80)
    print(f"\n{'Method':<30} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'Acc':>8}")
    print("-" * 80)
    
    unsup = results_df[results_df['type'] == 'UNSUPERVISED'].sort_values('auroc', ascending=False)
    for _, row in unsup.iterrows():
        star = "***" if row['method'] == 'PCIB (Theory-Guided)' else "   "
        print(f"{row['method']:<30} {row['auroc']:>8.4f} {row['auprc']:>8.4f} "
              f"{row['f1']:>8.4f} {row['accuracy']:>8.4f} {star}")
    
    print("\n" + "="*80)
    print("RESULTS: SUPERVISED METHODS")
    print("="*80)
    print(f"\n{'Method':<30} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'Acc':>8}")
    print("-" * 80)
    
    sup = results_df[results_df['type'] == 'SUPERVISED'].sort_values('auroc', ascending=False)
    for _, row in sup.iterrows():
        print(f"{row['method']:<30} {row['auroc']:>8.4f} {row['auprc']:>8.4f} "
              f"{row['f1']:>8.4f} {row['accuracy']:>8.4f}")
    
    print("\n" + "="*80)
    print("OVERALL RANKING (by AUROC)")
    print("="*80)
    print(f"\n{'Rank':<6} {'Method':<35} {'Type':<13} {'AUROC':>8}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(results_df_sorted.iterrows(), 1):
        star = " ⭐" if row['method'] == 'PCIB (Theory-Guided)' else ""
        print(f"{i:<6} {row['method']:<35} {row['type']:<13} {row['auroc']:>8.4f}{star}")
    
    # Statistical summary
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    
    pcib = results_df[results_df['method'] == 'PCIB (Theory-Guided)'].iloc[0]
    pcib_rank = (results_df['auroc'] >= pcib['auroc']).sum()
    
    unsup_avg = unsup[unsup['method'] != 'PCIB (Theory-Guided)']['auroc'].mean()
    sup_avg = sup['auroc'].mean()
    
    print(f"\nPCIB (Theory-Guided):")
    print(f"  Rank: #{pcib_rank} out of {len(results_df)} methods")
    print(f"  AUROC: {pcib['auroc']:.4f}")
    print(f"  AUPRC: {pcib['auprc']:.4f}")
    
    print(f"\nUnsupervised Methods (excluding PCIB):")
    print(f"  Average AUROC: {unsup_avg:.4f}")
    print(f"  PCIB advantage: {pcib['auroc'] - unsup_avg:+.4f} ({(pcib['auroc'] - unsup_avg)/unsup_avg*100:+.1f}%)")
    
    print(f"\nSupervised Methods:")
    print(f"  Average AUROC: {sup_avg:.4f}")
    print(f"  PCIB advantage: {pcib['auroc'] - sup_avg:+.4f} ({(pcib['auroc'] - sup_avg)/sup_avg*100:+.1f}%)")
    
    print(f"\nBest Alternative Methods:")
    best_unsup = unsup[unsup['method'] != 'PCIB (Theory-Guided)'].iloc[0]
    best_sup = sup.iloc[0]
    print(f"  Best Unsupervised: {best_unsup['method']} (AUROC: {best_unsup['auroc']:.4f})")
    print(f"  Best Supervised: {best_sup['method']} (AUROC: {best_sup['auroc']:.4f})")
    
    # Save results
    output = {
        'all_results': results_df.to_dict('records'),
        'summary': {
            'pcib_rank': int(pcib_rank),
            'pcib_auroc': float(pcib['auroc']),
            'unsupervised_avg_auroc': float(unsup_avg),
            'supervised_avg_auroc': float(sup_avg),
            'best_unsupervised': best_unsup['method'],
            'best_supervised': best_sup['method'],
        }
    }
    
    with open('full_method_comparison.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to full_method_comparison.json")
    print("="*80)

if __name__ == '__main__':
    main()
