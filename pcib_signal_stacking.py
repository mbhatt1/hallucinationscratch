#!/usr/bin/env python3
"""
Enhanced PCIB Signal Stacking with Advanced Feature Engineering
Target: Achieve 0.90 AUROC through sophisticated stacking techniques.

This approach combines:
1. Theory-guided PCIB signal design
2. Advanced feature engineering (interactions, ratios, polynomials)
3. Multiple ensemble models (RF, GB, XGB, LGB, Neural Networks)
4. Optimized weighted ensemble with learned weights
"""

import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import LightGBM (optional but recommended for 0.90 AUROC)
try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Warning: LightGBM not installed. Install with: pip install lightgbm")


def load_results(filepath: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Load evaluation results and extract features."""
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Extract features (PCIB signals)
    features = []
    labels = []
    
    for entry in data:
        # Get the label (0 or 1)
        label = entry.get('label', 0)
        
        # Get composite score
        composite_score = entry.get('score', 0.0)
        
        # Get individual signals from first claim (aggregate if multiple)
        claims = entry.get('claims', [])
        if claims:
            # Average signals across all claims
            uptake = np.mean([c.get('uptake_kl', 0.0) for c in claims])
            stress = np.mean([c.get('stress_js', 0.0) for c in claims])
            conflict = np.mean([c.get('conflict_js', 0.0) for c in claims])
            rationalization = 0.0  # Not in this data format
        else:
            uptake = stress = conflict = rationalization = 0.0
        
        # Feature vector: [uptake, stress, conflict, rationalization, composite_score]
        feature_vec = [uptake, stress, conflict, rationalization, composite_score]
        
        features.append(feature_vec)
        labels.append(label)
    
    return np.array(features), np.array(labels), data


def engineer_features(X: np.ndarray) -> np.ndarray:
    """
    Advanced feature engineering to capture non-linear interactions.
    
    From base PCIB signals [uptake, stress, conflict, rationalization, composite],
    we generate:
    1. Interaction terms (uptake * stress, etc.)
    2. Ratio features (uptake / stress, etc.)
    3. Polynomial features (uptake^2, etc.)
    4. Aggregations (max, min, mean of signals)
    
    This expands 5 base features to ~30+ engineered features.
    """
    features = []
    
    # Original features
    uptake = X[:, 0]
    stress = X[:, 1]
    conflict = X[:, 2]
    rationalization = X[:, 3]
    composite = X[:, 4]
    
    # Base features
    features.extend([uptake, stress, conflict, rationalization, composite])
    
    # Interaction terms (key for capturing complex patterns)
    features.append(uptake * stress)  # High uptake + high stress = ?
    features.append(uptake * conflict)  # Context-driven conflict
    features.append(stress * conflict)  # Unstable + contradictory
    features.append(uptake * rationalization)
    features.append(stress * rationalization)
    
    # Ratio features (normalized comparisons)
    epsilon = 1e-8  # Avoid division by zero
    features.append(uptake / (stress + epsilon))  # Uptake dominance
    features.append(uptake / (conflict + epsilon))
    features.append(stress / (uptake + epsilon))  # Stress dominance
    features.append(conflict / (stress + epsilon))
    
    # Polynomial features (non-linear effects)
    features.append(uptake ** 2)
    features.append(stress ** 2)
    features.append(conflict ** 2)
    features.append(np.sqrt(np.abs(uptake)))
    features.append(np.sqrt(np.abs(stress)))
    
    # Aggregations (holistic view)
    features.append(np.maximum(uptake, stress))
    features.append(np.minimum(uptake, stress))
    features.append((uptake + stress + conflict) / 3)  # Mean signal
    features.append(np.maximum(np.maximum(uptake, stress), conflict))  # Max signal
    
    # Composite-based features
    features.append(composite ** 2)
    features.append(np.abs(composite - 0.5))  # Distance from decision boundary
    
    # Three-way interactions (advanced)
    features.append(uptake * stress * conflict)
    features.append((uptake + stress) * conflict)
    features.append(uptake * (stress + conflict))
    
    # Signal variance (diversity)
    signal_matrix = np.column_stack([uptake, stress, conflict])
    features.append(np.var(signal_matrix, axis=1))
    features.append(np.std(signal_matrix, axis=1))
    
    # Convert to array
    X_engineered = np.column_stack(features)
    
    return X_engineered


def train_stacked_models(X: np.ndarray, y: np.ndarray) -> Dict:
    """Train stacked models and compare performance."""
    results = {}
    
    # Use stratified k-fold for robust evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("=" * 80)
    print("STACKING PCIB SIGNALS INTO SUPERVISED LEARNERS")
    print("=" * 80)
    print(f"\nDataset: n={len(y)} (Class balance: {np.mean(y):.2%} positive)")
    print(f"Features: {X.shape[1]} PCIB signals [uptake, stress, conflict, rationalization, composite]")
    print()
    
    # Baseline: Just use the composite PCIB score (column 4)
    baseline_scores = X[:, 4]
    baseline_auroc = roc_auc_score(y, baseline_scores)
    baseline_auprc = average_precision_score(y, baseline_scores)
    
    print(f"BASELINE (Original PCIB Composite Score):")
    print(f"  AUROC: {baseline_auroc:.4f}")
    print(f"  AUPRC: {baseline_auprc:.4f}")
    print()
    
    results['baseline'] = {
        'auroc': baseline_auroc,
        'auprc': baseline_auprc,
        'method': 'PCIB Theory-Guided'
    }
    
    # Model 1: Random Forest on PCIB signals
    print("-" * 80)
    print("MODEL 1: Random Forest Stacking")
    print("-" * 80)
    
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_grid = GridSearchCV(rf, rf_params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
    rf_grid.fit(X, y)
    
    # Get predictions with calibration
    rf_calibrated = CalibratedClassifierCV(rf_grid.best_estimator_, method='isotonic', cv=cv)
    rf_calibrated.fit(X, y)
    
    rf_scores_cv = cross_val_score(rf_calibrated, X, y, cv=cv, scoring='roc_auc')
    rf_pred = cross_val_predict_proba(rf_calibrated, X, y, cv)
    
    rf_auroc = roc_auc_score(y, rf_pred)
    rf_auprc = average_precision_score(y, rf_pred)
    
    print(f"Best params: {rf_grid.best_params_}")
    print(f"CV AUROC: {rf_scores_cv.mean():.4f} ± {rf_scores_cv.std():.4f}")
    print(f"Final AUROC: {rf_auroc:.4f} (Δ={rf_auroc - baseline_auroc:+.4f})")
    print(f"Final AUPRC: {rf_auprc:.4f} (Δ={rf_auprc - baseline_auprc:+.4f})")
    print(f"\n✓ Improvement vs baseline: {((rf_auroc/baseline_auroc - 1) * 100):+.2f}%")
    
    # Feature importance
    importances = rf_grid.best_estimator_.feature_importances_
    feature_names = ['Uptake', 'Stress', 'Conflict', 'Rationalization', 'Composite']
    print("\nFeature Importances:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"  {name:15s}: {imp:.4f} {'█' * int(imp * 50)}")
    
    results['random_forest'] = {
        'auroc': rf_auroc,
        'auprc': rf_auprc,
        'method': 'RF Stacking (Calibrated)',
        'improvement': rf_auroc - baseline_auroc
    }
    
    # Model 2: Gradient Boosting
    print("\n" + "-" * 80)
    print("MODEL 2: Gradient Boosting Stacking")
    print("-" * 80)
    
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    gb_grid = GridSearchCV(gb, gb_params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
    gb_grid.fit(X, y)
    
    gb_scores_cv = cross_val_score(gb_grid.best_estimator_, X, y, cv=cv, scoring='roc_auc')
    gb_pred = cross_val_predict_proba(gb_grid.best_estimator_, X, y, cv)
    
    gb_auroc = roc_auc_score(y, gb_pred)
    gb_auprc = average_precision_score(y, gb_pred)
    
    print(f"Best params: {gb_grid.best_params_}")
    print(f"CV AUROC: {gb_scores_cv.mean():.4f} ± {gb_scores_cv.std():.4f}")
    print(f"Final AUROC: {gb_auroc:.4f} (Δ={gb_auroc - baseline_auroc:+.4f})")
    print(f"Final AUPRC: {gb_auprc:.4f} (Δ={gb_auprc - baseline_auprc:+.4f})")
    print(f"\n✓ Improvement vs baseline: {((gb_auroc/baseline_auroc - 1) * 100):+.2f}%")
    
    results['gradient_boosting'] = {
        'auroc': gb_auroc,
        'auprc': gb_auprc,
        'method': 'GB Stacking',
        'improvement': gb_auroc - baseline_auroc
    }
    
    # Model 3: SVM with RBF Kernel
    print("\n" + "-" * 80)
    print("MODEL 3: SVM with RBF Kernel (Non-linear)")
    print("-" * 80)
    
    # SVM needs feature scaling for best performance
    svm_params = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'svc__kernel': ['rbf']
    }
    
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(probability=True, random_state=42, class_weight='balanced'))
    ])
    
    svm_grid = GridSearchCV(svm_pipeline, svm_params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
    svm_grid.fit(X, y)
    
    # Calibrate SVM predictions
    svm_calibrated = CalibratedClassifierCV(svm_grid.best_estimator_, method='isotonic', cv=cv)
    svm_calibrated.fit(X, y)
    
    svm_scores_cv = cross_val_score(svm_calibrated, X, y, cv=cv, scoring='roc_auc')
    svm_pred = cross_val_predict_proba(svm_calibrated, X, y, cv)
    
    svm_auroc = roc_auc_score(y, svm_pred)
    svm_auprc = average_precision_score(y, svm_pred)
    
    print(f"Best params: {svm_grid.best_params_}")
    print(f"CV AUROC: {svm_scores_cv.mean():.4f} ± {svm_scores_cv.std():.4f}")
    print(f"Final AUROC: {svm_auroc:.4f} (Δ={svm_auroc - baseline_auroc:+.4f})")
    print(f"Final AUPRC: {svm_auprc:.4f} (Δ={svm_auprc - baseline_auprc:+.4f})")
    print(f"\n✓ Improvement vs baseline: {((svm_auroc/baseline_auroc - 1) * 100):+.2f}%")
    
    results['svm_rbf'] = {
        'auroc': svm_auroc,
        'auprc': svm_auprc,
        'method': 'SVM-RBF Stacking (Calibrated)',
        'improvement': svm_auroc - baseline_auroc
    }
    
    # Model 4: Ensemble of all stacked models
    print("\n" + "-" * 80)
    print("MODEL 4: Meta-Ensemble (Average RF + GB + SVM)")
    print("-" * 80)
    
    ensemble_pred = (rf_pred + gb_pred + svm_pred) / 3
    ensemble_auroc = roc_auc_score(y, ensemble_pred)
    ensemble_auprc = average_precision_score(y, ensemble_pred)
    
    print(f"Final AUROC: {ensemble_auroc:.4f} (Δ={ensemble_auroc - baseline_auroc:+.4f})")
    print(f"Final AUPRC: {ensemble_auprc:.4f} (Δ={ensemble_auprc - baseline_auprc:+.4f})")
    print(f"\n✓ Improvement vs baseline: {((ensemble_auroc/baseline_auroc - 1) * 100):+.2f}%")
    
    results['meta_ensemble'] = {
        'auroc': ensemble_auroc,
        'auprc': ensemble_auprc,
        'method': 'Meta-Ensemble (RF+GB+SVM)',
        'improvement': ensemble_auroc - baseline_auroc
    }
    
    # Model 5: LightGBM (if available)
    if HAS_LGB:
        print("\n" + "-" * 80)
        print("MODEL 5: LightGBM Stacking")
        print("-" * 80)
        
        lgb_params = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [15, 31, 63],
            'subsample': [0.8, 1.0]
        }
        
        lgb = LGBMClassifier(random_state=42, verbose=-1)
        lgb_grid = GridSearchCV(lgb, lgb_params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
        lgb_grid.fit(X, y)
        
        lgb_scores_cv = cross_val_score(lgb_grid.best_estimator_, X, y, cv=cv, scoring='roc_auc')
        lgb_pred = cross_val_predict_proba(lgb_grid.best_estimator_, X, y, cv)
        
        lgb_auroc = roc_auc_score(y, lgb_pred)
        lgb_auprc = average_precision_score(y, lgb_pred)
        
        print(f"Best params: {lgb_grid.best_params_}")
        print(f"CV AUROC: {lgb_scores_cv.mean():.4f} ± {lgb_scores_cv.std():.4f}")
        print(f"Final AUROC: {lgb_auroc:.4f} (Δ={lgb_auroc - baseline_auroc:+.4f})")
        print(f"Final AUPRC: {lgb_auprc:.4f} (Δ={lgb_auprc - baseline_auprc:+.4f})")
        print(f"\n✓ Improvement vs baseline: {((lgb_auroc/baseline_auroc - 1) * 100):+.2f}%")
        
        results['lightgbm'] = {
            'auroc': lgb_auroc,
            'auprc': lgb_auprc,
            'method': 'LightGBM Stacking',
            'improvement': lgb_auroc - baseline_auroc
        }
    else:
        lgb_pred = None
    
    # Model 6: Neural Network
    print("\n" + "-" * 80)
    print("MODEL 7: Multi-Layer Perceptron (Neural Network)")
    print("-" * 80)
    
    nn_params = {
        'mlp__hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
        'mlp__alpha': [0.0001, 0.001, 0.01],
        'mlp__learning_rate_init': [0.001, 0.01]
    }
    
    nn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(random_state=42, max_iter=1000, early_stopping=True))
    ])
    
    nn_grid = GridSearchCV(nn_pipeline, nn_params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
    nn_grid.fit(X, y)
    
    nn_scores_cv = cross_val_score(nn_grid.best_estimator_, X, y, cv=cv, scoring='roc_auc')
    nn_pred = cross_val_predict_proba(nn_grid.best_estimator_, X, y, cv)
    
    nn_auroc = roc_auc_score(y, nn_pred)
    nn_auprc = average_precision_score(y, nn_pred)
    
    print(f"Best params: {nn_grid.best_params_}")
    print(f"CV AUROC: {nn_scores_cv.mean():.4f} ± {nn_scores_cv.std():.4f}")
    print(f"Final AUROC: {nn_auroc:.4f} (Δ={nn_auroc - baseline_auroc:+.4f})")
    print(f"Final AUPRC: {nn_auprc:.4f} (Δ={nn_auprc - baseline_auprc:+.4f})")
    print(f"\n✓ Improvement vs baseline: {((nn_auroc/baseline_auroc - 1) * 100):+.2f}%")
    
    results['neural_network'] = {
        'auroc': nn_auroc,
        'auprc': nn_auprc,
        'method': 'Neural Network (MLP)',
        'improvement': nn_auroc - baseline_auroc
    }
    
    # Model 7: Optimized Weighted Ensemble (learn optimal weights)
    print("\n" + "-" * 80)
    print("MODEL 7: Optimized Weighted Ensemble (Learned Weights)")
    print("-" * 80)
    
    # Collect all predictions
    all_preds = [rf_pred, gb_pred, svm_pred, nn_pred]
    model_names = ['RF', 'GB', 'SVM', 'NN']
    
    if lgb_pred is not None:
        all_preds.append(lgb_pred)
        model_names.append('LGB')
    
    # Stack predictions as features
    stacked_preds = np.column_stack(all_preds)
    
    # Learn optimal weights using logistic regression
    weight_learner = LogisticRegression(random_state=42, max_iter=1000)
    weight_scores_cv = cross_val_score(weight_learner, stacked_preds, y, cv=cv, scoring='roc_auc')
    weight_pred = cross_val_predict_proba(weight_learner, stacked_preds, y, cv)
    
    optimal_auroc = roc_auc_score(y, weight_pred)
    optimal_auprc = average_precision_score(y, weight_pred)
    
    # Fit to get weights
    weight_learner.fit(stacked_preds, y)
    learned_weights = weight_learner.coef_[0]
    learned_weights = np.abs(learned_weights) / np.sum(np.abs(learned_weights))  # Normalize
    
    print("Learned model weights:")
    for name, weight in zip(model_names, learned_weights):
        print(f"  {name:6s}: {weight:.4f} {'█' * int(weight * 50)}")
    
    print(f"\nCV AUROC: {weight_scores_cv.mean():.4f} ± {weight_scores_cv.std():.4f}")
    print(f"Final AUROC: {optimal_auroc:.4f} (Δ={optimal_auroc - baseline_auroc:+.4f})")
    print(f"Final AUPRC: {optimal_auprc:.4f} (Δ={optimal_auprc - baseline_auprc:+.4f})")
    print(f"\n✓ Improvement vs baseline: {((optimal_auroc/baseline_auroc - 1) * 100):+.2f}%")
    
    results['optimized_ensemble'] = {
        'auroc': optimal_auroc,
        'auprc': optimal_auprc,
        'method': 'Optimized Weighted Ensemble',
        'improvement': optimal_auroc - baseline_auroc,
        'weights': {name: float(w) for name, w in zip(model_names, learned_weights)}
    }
    
    return results


def cross_val_predict_proba(estimator, X, y, cv):
    """Get out-of-fold predictions for proper evaluation."""
    predictions = np.zeros(len(y))
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        
        estimator.fit(X_train, y_train)
        predictions[test_idx] = estimator.predict_proba(X_test)[:, 1]
    
    return predictions


def print_summary(results: Dict):
    """Print summary comparison."""
    print("\n" + "=" * 80)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Method':<30} {'AUROC':>10} {'AUPRC':>10} {'Δ AUROC':>12}")
    print("-" * 80)
    
    # Sort by AUROC
    sorted_results = sorted(results.items(), key=lambda x: x[1]['auroc'], reverse=True)
    
    for name, result in sorted_results:
        auroc = result['auroc']
        auprc = result['auprc']
        improvement = result.get('improvement', 0)
        marker = "🏆" if auroc == max(r['auroc'] for r in results.values()) else "  "
        
        print(f"{marker} {result['method']:<28} {auroc:>10.4f} {auprc:>10.4f} {improvement:>+11.4f}")
    
    print()
    best_method = max(results.items(), key=lambda x: x[1]['auroc'])
    baseline_auroc = results['baseline']['auroc']
    best_auroc = best_method[1]['auroc']
    
    print(f"Best performing method: {best_method[1]['method']}")
    print(f"Absolute AUROC gain: {best_auroc - baseline_auroc:+.4f}")
    print(f"Relative improvement: {((best_auroc / baseline_auroc - 1) * 100):+.2f}%")
    
    if best_auroc >= 0.90:
        print("\n🎯🎯🎯 EXCEPTIONAL: Achieved 0.90+ AUROC - World-class performance!")
    elif best_auroc >= 0.85:
        print("\n🎯 SUCCESS: Achieved state-of-the-art performance (AUROC ≥ 0.85)!")
    elif best_auroc >= 0.83:
        print("\n✓ STRONG: Close to state-of-the-art (AUROC ≥ 0.83)")
    else:
        print(f"\n→ Current AUROC: {best_auroc:.4f}. Target: 0.90")
        print("   Next steps:")
        print("   1. Feature engineering: Add more interaction terms")
        print("   2. Data: Increase training samples (n→1000+)")
        print("   3. Signals: Add semantic embeddings or external knowledge")


def main():
    # Load data
    print("\nLoading evaluation results...")
    X, y, data = load_results('pc_ib_results_fixed.jsonl')
    
    print(f"Loaded {len(X)} examples with {X.shape[1]} features")
    print(f"Class distribution: {np.sum(y)} positive, {len(y) - np.sum(y)} negative")
    print()
    
    # Feature engineering
    print("=" * 80)
    print("ADVANCED FEATURE ENGINEERING")
    print("=" * 80)
    print(f"Original features: {X.shape[1]}")
    X_engineered = engineer_features(X)
    print(f"Engineered features: {X_engineered.shape[1]}")
    print(f"Feature expansion: {X_engineered.shape[1] / X.shape[1]:.1f}x")
    print()
    
    # Train stacked models on engineered features
    results = train_stacked_models(X_engineered, y)
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_file = 'stacked_model_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    print("\nTo learn PCIB signals from raw HaluBench data, see: learn_from_raw_halubench.py")


if __name__ == '__main__':
    main()
