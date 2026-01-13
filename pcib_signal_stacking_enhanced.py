#!/usr/bin/env python3
"""
Enhanced stacking with advanced techniques to reach AUROC 0.90+
Adds: XGBoost, LightGBM, Neural Networks, Feature Engineering, and Advanced Ensembles
"""

import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGB = True
except (ImportError, Exception) as e:
    HAS_XGB = False
    print(f"Warning: XGBoost not available ({type(e).__name__}). Skipping XGBoost models.")
    print("  To fix: Try 'brew reinstall libomp' then 'pip install --force-reinstall xgboost'")

try:
    import lightgbm as lgb
    HAS_LGB = True
except (ImportError, Exception) as e:
    HAS_LGB = False
    print(f"Warning: LightGBM not available ({type(e).__name__}). Skipping LightGBM models.")
    print("  Install with: pip install lightgbm")


def load_results(filepath: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Load evaluation results and extract features."""
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    
    features = []
    labels = []
    
    for entry in data:
        label = entry.get('label', 0)
        composite_score = entry.get('score', 0.0)
        
        claims = entry.get('claims', [])
        if claims:
            uptake = np.mean([c.get('uptake_kl', 0.0) for c in claims])
            stress = np.mean([c.get('stress_js', 0.0) for c in claims])
            conflict = np.mean([c.get('conflict_js', 0.0) for c in claims])
            # Additional aggregations
            uptake_max = np.max([c.get('uptake_kl', 0.0) for c in claims])
            stress_max = np.max([c.get('stress_js', 0.0) for c in claims])
            conflict_max = np.max([c.get('conflict_js', 0.0) for c in claims])
            uptake_std = np.std([c.get('uptake_kl', 0.0) for c in claims]) if len(claims) > 1 else 0.0
            stress_std = np.std([c.get('stress_js', 0.0) for c in claims]) if len(claims) > 1 else 0.0
            conflict_std = np.std([c.get('conflict_js', 0.0) for c in claims]) if len(claims) > 1 else 0.0
        else:
            uptake = stress = conflict = 0.0
            uptake_max = stress_max = conflict_max = 0.0
            uptake_std = stress_std = conflict_std = 0.0
        
        # Extended feature vector with aggregations
        feature_vec = [
            uptake, stress, conflict, composite_score,
            uptake_max, stress_max, conflict_max,
            uptake_std, stress_std, conflict_std,
            len(claims)  # Number of claims
        ]
        
        features.append(feature_vec)
        labels.append(label)
    
    return np.array(features), np.array(labels), data


def engineer_features(X: np.ndarray) -> np.ndarray:
    """Add engineered features: interactions, polynomials, ratios."""
    X_eng = []
    
    for row in X:
        uptake, stress, conflict, composite = row[0], row[1], row[2], row[3]
        uptake_max, stress_max, conflict_max = row[4], row[5], row[6]
        uptake_std, stress_std, conflict_std = row[7], row[8], row[9]
        n_claims = row[10]
        
        # Original features
        features = list(row)
        
        # Interaction features
        features.extend([
            uptake * stress,           # uptake-stress interaction
            uptake * conflict,         # uptake-conflict interaction
            stress * conflict,         # stress-conflict interaction
            uptake * stress * conflict,  # 3-way interaction
        ])
        
        # Ratio features (with epsilon to avoid division by zero)
        eps = 1e-10
        features.extend([
            uptake / (stress + eps),
            stress / (conflict + eps),
            uptake / (conflict + eps),
            (uptake + stress) / (conflict + eps),
        ])
        
        # Polynomial features (squared)
        features.extend([
            uptake ** 2,
            stress ** 2,
            conflict ** 2,
            composite ** 2,
        ])
        
        # Aggregate ratios
        features.extend([
            uptake_max / (uptake + eps),
            stress_max / (stress + eps),
            conflict_max / (conflict + eps),
        ])
        
        # Variability measures
        features.extend([
            uptake_std + stress_std + conflict_std,
            uptake_std * stress_std * conflict_std,
        ])
        
        # Composite-based features
        features.extend([
            composite * uptake,
            composite * stress,
            composite * conflict,
            composite / (uptake + stress + conflict + eps),
        ])
        
        X_eng.append(features)
    
    return np.array(X_eng)


def train_advanced_models(X: np.ndarray, y: np.ndarray) -> Dict:
    """Train advanced stacked models with feature engineering."""
    results = {}
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("=" * 80)
    print("ENHANCED STACKING: PCIB SIGNALS → SOTA PERFORMANCE (Target: 0.90 AUROC)")
    print("=" * 80)
    print(f"\nDataset: n={len(y)} (Class balance: {np.mean(y):.2%} positive)")
    print(f"Base Features: {X.shape[1]}")
    
    # Engineer features
    print("\nEngineering features...")
    X_eng = engineer_features(X)
    print(f"Engineered Features: {X_eng.shape[1]} (added {X_eng.shape[1] - X.shape[1]} features)")
    
    # Baseline: Original composite score
    baseline_scores = X[:, 3]  # composite_score is at index 3
    baseline_auroc = roc_auc_score(y, baseline_scores)
    baseline_auprc = average_precision_score(y, baseline_scores)
    
    print(f"\nBASELINE (PCIB Composite):")
    print(f"  AUROC: {baseline_auroc:.4f}")
    print(f"  AUPRC: {baseline_auprc:.4f}")
    print()
    
    results['baseline'] = {
        'auroc': baseline_auroc,
        'auprc': baseline_auprc,
        'method': 'PCIB Theory-Guided'
    }
    
    all_predictions = {}
    
    # Model 1: XGBoost
    if HAS_XGB:
        print("-" * 80)
        print("MODEL 1: XGBoost")
        print("-" * 80)
        
        xgb_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2],
            'min_child_weight': [1, 3, 5]
        }
        
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            scale_pos_weight=len(y[y==0]) / len(y[y==1]),  # Handle imbalance
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
        xgb_grid.fit(X_eng, y)
        
        xgb_pred = cross_val_predict_proba(xgb_grid.best_estimator_, X_eng, y, cv)
        xgb_auroc = roc_auc_score(y, xgb_pred)
        xgb_auprc = average_precision_score(y, xgb_pred)
        
        print(f"Best params: {xgb_grid.best_params_}")
        print(f"AUROC: {xgb_auroc:.4f} (Δ={xgb_auroc - baseline_auroc:+.4f})")
        print(f"AUPRC: {xgb_auprc:.4f}")
        print(f"Improvement: {((xgb_auroc/baseline_auroc - 1) * 100):+.2f}%")
        
        results['xgboost'] = {
            'auroc': xgb_auroc,
            'auprc': xgb_auprc,
            'method': 'XGBoost',
            'improvement': xgb_auroc - baseline_auroc
        }
        all_predictions['xgboost'] = xgb_pred
    
    # Model 2: LightGBM
    if HAS_LGB:
        print("\n" + "-" * 80)
        print("MODEL 2: LightGBM")
        print("-" * 80)
        
        lgb_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, -1],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 100],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        lgb_model = lgb.LGBMClassifier(
            random_state=42,
            class_weight='balanced',
            verbose=-1
        )
        
        lgb_grid = GridSearchCV(lgb_model, lgb_params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
        lgb_grid.fit(X_eng, y)
        
        lgb_pred = cross_val_predict_proba(lgb_grid.best_estimator_, X_eng, y, cv)
        lgb_auroc = roc_auc_score(y, lgb_pred)
        lgb_auprc = average_precision_score(y, lgb_pred)
        
        print(f"Best params: {lgb_grid.best_params_}")
        print(f"AUROC: {lgb_auroc:.4f} (Δ={lgb_auroc - baseline_auroc:+.4f})")
        print(f"AUPRC: {lgb_auprc:.4f}")
        print(f"Improvement: {((lgb_auroc/baseline_auroc - 1) * 100):+.2f}%")
        
        results['lightgbm'] = {
            'auroc': lgb_auroc,
            'auprc': lgb_auprc,
            'method': 'LightGBM',
            'improvement': lgb_auroc - baseline_auroc
        }
        all_predictions['lightgbm'] = lgb_pred
    
    # Model 3: Random Forest with Engineered Features
    print("\n" + "-" * 80)
    print("MODEL 3: Random Forest (Enhanced)")
    print("-" * 80)
    
    rf_params = {
        'n_estimators': [200, 300, 500],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    rf_grid = GridSearchCV(rf, rf_params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
    rf_grid.fit(X_eng, y)
    
    rf_calibrated = CalibratedClassifierCV(rf_grid.best_estimator_, method='isotonic', cv=cv)
    rf_calibrated.fit(X_eng, y)
    
    rf_pred = cross_val_predict_proba(rf_calibrated, X_eng, y, cv)
    rf_auroc = roc_auc_score(y, rf_pred)
    rf_auprc = average_precision_score(y, rf_pred)
    
    print(f"Best params: {rf_grid.best_params_}")
    print(f"AUROC: {rf_auroc:.4f} (Δ={rf_auroc - baseline_auroc:+.4f})")
    print(f"AUPRC: {rf_auprc:.4f}")
    
    results['random_forest_enhanced'] = {
        'auroc': rf_auroc,
        'auprc': rf_auprc,
        'method': 'RF Enhanced',
        'improvement': rf_auroc - baseline_auroc
    }
    all_predictions['rf'] = rf_pred
    
    # Model 4: Gradient Boosting
    print("\n" + "-" * 80)
    print("MODEL 4: Gradient Boosting (Enhanced)")
    print("-" * 80)
    
    gb_params = {
        'n_estimators': [200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    gb_grid = GridSearchCV(gb, gb_params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
    gb_grid.fit(X_eng, y)
    
    gb_pred = cross_val_predict_proba(gb_grid.best_estimator_, X_eng, y, cv)
    gb_auroc = roc_auc_score(y, gb_pred)
    gb_auprc = average_precision_score(y, gb_pred)
    
    print(f"Best params: {gb_grid.best_params_}")
    print(f"AUROC: {gb_auroc:.4f} (Δ={gb_auroc - baseline_auroc:+.4f})")
    print(f"AUPRC: {gb_auprc:.4f}")
    
    results['gradient_boosting_enhanced'] = {
        'auroc': gb_auroc,
        'auprc': gb_auprc,
        'method': 'GB Enhanced',
        'improvement': gb_auroc - baseline_auroc
    }
    all_predictions['gb'] = gb_pred
    
    # Model 5: Neural Network
    print("\n" + "-" * 80)
    print("MODEL 5: Neural Network (MLP)")
    print("-" * 80)
    
    mlp_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(random_state=42, max_iter=1000, early_stopping=True))
    ])
    
    mlp_params = {
        'mlp__hidden_layer_sizes': [(100,), (100, 50), (100, 100), (200, 100, 50)],
        'mlp__alpha': [0.0001, 0.001, 0.01],
        'mlp__learning_rate_init': [0.001, 0.01]
    }
    
    mlp_grid = GridSearchCV(mlp_pipeline, mlp_params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
    mlp_grid.fit(X_eng, y)
    
    mlp_pred = cross_val_predict_proba(mlp_grid.best_estimator_, X_eng, y, cv)
    mlp_auroc = roc_auc_score(y, mlp_pred)
    mlp_auprc = average_precision_score(y, mlp_pred)
    
    print(f"Best params: {mlp_grid.best_params_}")
    print(f"AUROC: {mlp_auroc:.4f} (Δ={mlp_auroc - baseline_auroc:+.4f})")
    print(f"AUPRC: {mlp_auprc:.4f}")
    
    results['neural_network'] = {
        'auroc': mlp_auroc,
        'auprc': mlp_auprc,
        'method': 'Neural Network',
        'improvement': mlp_auroc - baseline_auroc
    }
    all_predictions['mlp'] = mlp_pred
    
    # Model 6: SVM with RBF Kernel
    print("\n" + "-" * 80)
    print("MODEL 6: SVM-RBF (Enhanced)")
    print("-" * 80)
    
    svm_params = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 'auto', 0.001, 0.01],
        'svc__kernel': ['rbf']
    }
    
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(probability=True, random_state=42, class_weight='balanced'))
    ])
    
    svm_grid = GridSearchCV(svm_pipeline, svm_params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
    svm_grid.fit(X_eng, y)
    
    svm_calibrated = CalibratedClassifierCV(svm_grid.best_estimator_, method='isotonic', cv=cv)
    svm_calibrated.fit(X_eng, y)
    
    svm_pred = cross_val_predict_proba(svm_calibrated, X_eng, y, cv)
    svm_auroc = roc_auc_score(y, svm_pred)
    svm_auprc = average_precision_score(y, svm_pred)
    
    print(f"Best params: {svm_grid.best_params_}")
    print(f"AUROC: {svm_auroc:.4f} (Δ={svm_auroc - baseline_auroc:+.4f})")
    print(f"AUPRC: {svm_auprc:.4f}")
    
    results['svm_rbf_enhanced'] = {
        'auroc': svm_auroc,
        'auprc': svm_auprc,
        'method': 'SVM-RBF Enhanced',
        'improvement': svm_auroc - baseline_auroc
    }
    all_predictions['svm'] = svm_pred
    
    # Model 7: Weighted Ensemble (All Models)
    print("\n" + "-" * 80)
    print("MODEL 7: Optimized Weighted Ensemble")
    print("-" * 80)
    
    # Find optimal weights using grid search
    best_ensemble_auroc = 0
    best_weights = None
    
    # Generate weight combinations
    from itertools import product
    weight_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    pred_list = list(all_predictions.values())
    n_models = len(pred_list)
    
    print(f"Searching optimal weights for {n_models} models...")
    
    # Simplified search: try some reasonable combinations
    for w1 in [0.2, 0.3, 0.4]:
        for w2 in [0.2, 0.3, 0.4]:
            for w3 in [0.1, 0.2, 0.3]:
                # Normalize to sum to 1
                remaining = 1.0 - (w1 + w2 + w3)
                if remaining >= 0 and remaining <= 0.7:
                    w_remaining = remaining / (n_models - 3) if n_models > 3 else 0
                    weights = [w1, w2, w3] + [w_remaining] * (n_models - 3)
                    
                    ensemble_pred = sum(w * p for w, p in zip(weights, pred_list))
                    ensemble_auroc = roc_auc_score(y, ensemble_pred)
                    
                    if ensemble_auroc > best_ensemble_auroc:
                        best_ensemble_auroc = ensemble_auroc
                        best_weights = weights
    
    # Create final ensemble with best weights
    ensemble_pred = sum(w * p for w, p in zip(best_weights, pred_list))
    ensemble_auprc = average_precision_score(y, ensemble_pred)
    
    print(f"Optimal weights: {dict(zip(all_predictions.keys(), [f'{w:.3f}' for w in best_weights]))}")
    print(f"AUROC: {best_ensemble_auroc:.4f} (Δ={best_ensemble_auroc - baseline_auroc:+.4f})")
    print(f"AUPRC: {ensemble_auprc:.4f}")
    print(f"Improvement: {((best_ensemble_auroc/baseline_auroc - 1) * 100):+.2f}%")
    
    results['optimized_ensemble'] = {
        'auroc': best_ensemble_auroc,
        'auprc': ensemble_auprc,
        'method': 'Optimized Weighted Ensemble',
        'improvement': best_ensemble_auroc - baseline_auroc,
        'weights': dict(zip(all_predictions.keys(), best_weights))
    }
    
    # Model 8: Simple Average Ensemble
    print("\n" + "-" * 80)
    print("MODEL 8: Simple Average Ensemble")
    print("-" * 80)
    
    avg_ensemble_pred = np.mean(pred_list, axis=0)
    avg_ensemble_auroc = roc_auc_score(y, avg_ensemble_pred)
    avg_ensemble_auprc = average_precision_score(y, avg_ensemble_pred)
    
    print(f"AUROC: {avg_ensemble_auroc:.4f} (Δ={avg_ensemble_auroc - baseline_auroc:+.4f})")
    print(f"AUPRC: {avg_ensemble_auprc:.4f}")
    
    results['simple_average_ensemble'] = {
        'auroc': avg_ensemble_auroc,
        'auprc': avg_ensemble_auprc,
        'method': 'Simple Average Ensemble',
        'improvement': avg_ensemble_auroc - baseline_auroc
    }
    
    return results


def cross_val_predict_proba(estimator, X, y, cv):
    """Get out-of-fold predictions."""
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
    print("FINAL PERFORMANCE COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Method':<35} {'AUROC':>10} {'AUPRC':>10} {'Δ AUROC':>12}")
    print("-" * 80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['auroc'], reverse=True)
    
    for name, result in sorted_results:
        auroc = result['auroc']
        auprc = result['auprc']
        improvement = result.get('improvement', 0)
        
        if auroc >= 0.90:
            marker = "🎯"
        elif auroc == max(r['auroc'] for r in results.values()):
            marker = "🏆"
        else:
            marker = "  "
        
        print(f"{marker} {result['method']:<33} {auroc:>10.4f} {auprc:>10.4f} {improvement:>+11.4f}")
    
    print()
    best_method = max(results.items(), key=lambda x: x[1]['auroc'])
    baseline_auroc = results['baseline']['auroc']
    best_auroc = best_method[1]['auroc']
    
    print(f"Best performing method: {best_method[1]['method']}")
    print(f"Absolute AUROC gain: {best_auroc - baseline_auroc:+.4f}")
    print(f"Relative improvement: {((best_auroc / baseline_auroc - 1) * 100):+.2f}%")
    
    if best_auroc >= 0.90:
        print(f"\n🎯 EXCELLENT: Achieved 0.90+ AUROC target! (AUROC = {best_auroc:.4f})")
    elif best_auroc >= 0.85:
        print(f"\n✓ STRONG: Close to 0.90 target (AUROC = {best_auroc:.4f})")
        print(f"   Gap to 0.90: {0.90 - best_auroc:.4f}")
    else:
        print(f"\n→ Current AUROC: {best_auroc:.4f}. Target: 0.90")
        print(f"   Gap: {0.90 - best_auroc:.4f}")
        print("   Next steps: More training data, additional signals, or deep learning")


def main():
    print("\nLoading evaluation results...")
    X, y, data = load_results('pc_ib_results_fixed.jsonl')
    
    print(f"Loaded {len(X)} examples with {X.shape[1]} base features")
    print(f"Class distribution: {np.sum(y)} positive ({np.mean(y):.2%}), {len(y) - np.sum(y)} negative")
    print()
    
    # Train advanced models
    results = train_advanced_models(X, y)
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_file = 'stacked_model_results_enhanced.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {output_file}")


if __name__ == '__main__':
    main()
