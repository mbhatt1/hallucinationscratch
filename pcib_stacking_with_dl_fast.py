#!/usr/bin/env python3
"""
Fast stacking with Deep Learning: RF + GB + SVM + NN
Optimized for speed with progress indicators.
"""

import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False


def load_training_data(filepath='pc_ib_results_fixed.jsonl'):
    """Load n=200 training data."""
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
        else:
            uptake = stress = conflict = 0.0
        
        feature_vec = [uptake, stress, conflict, 0.0, composite_score]
        features.append(feature_vec)
        labels.append(label)
    
    return np.array(features), np.array(labels)


def create_simple_nn(input_dim=5):
    """Create a simple neural network (much faster than complex one)."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),  # Faster learning
        loss='binary_crossentropy',
        metrics=['AUC']
    )
    
    return model


def cross_val_predict_proba_nn_fast(X, y, cv):
    """Fast neural network cross-validation with single model per fold."""
    print("Training Neural Network...")
    predictions = np.zeros(len(y))
    
    fold_num = 1
    for train_idx, test_idx in cv.split(X, y):
        print(f"  Fold {fold_num}/5... ", end='', flush=True)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train simple NN with fewer epochs
        tf.random.set_seed(42)
        model = create_simple_nn(input_dim=X.shape[1])
        
        model.fit(
            X_train_scaled, y_train,
            epochs=50,  # Much faster than 200
            batch_size=32,
            verbose=0,
            validation_split=0.2
        )
        
        predictions[test_idx] = model.predict(X_test_scaled, verbose=0).flatten()
        print("✓")
        fold_num += 1
    
    return predictions


def cross_val_predict_proba(estimator, X, y, cv, name="Model"):
    """Get out-of-fold predictions with progress indicator."""
    print(f"Training {name}...")
    predictions = np.zeros(len(y))
    
    fold_num = 1
    for train_idx, test_idx in cv.split(X, y):
        print(f"  Fold {fold_num}/5... ", end='', flush=True)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        
        estimator.fit(X_train, y_train)
        predictions[test_idx] = estimator.predict_proba(X_test)[:, 1]
        print("✓")
        fold_num += 1
    
    return predictions


def main():
    print("\n" + "=" * 80)
    print("FAST STACKING: RF + GB + SVM + DEEP LEARNING")
    print("=" * 80)
    
    print("\nLoading training data...")
    X, y = load_training_data()
    print(f"✓ Loaded {len(X)} examples (Class balance: {np.mean(y):.2%} positive)\n")
    
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Baseline
    baseline_scores = X[:, 4]
    baseline_auroc = roc_auc_score(y, baseline_scores)
    baseline_auprc = average_precision_score(y, baseline_scores)
    
    print(f"BASELINE (PCIB):")
    print(f"  AUROC: {baseline_auroc:.4f}")
    print(f"  AUPRC: {baseline_auprc:.4f}\n")
    
    results['baseline'] = {'auroc': baseline_auroc, 'auprc': baseline_auprc}
    
    # Model 1: Random Forest (no grid search for speed)
    print("-" * 80)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    rf_pred = cross_val_predict_proba(rf, X, y, cv, "Random Forest")
    rf_auroc = roc_auc_score(y, rf_pred)
    rf_auprc = average_precision_score(y, rf_pred)
    
    print(f"✓ AUROC: {rf_auroc:.4f} (Δ={rf_auroc - baseline_auroc:+.4f})")
    print(f"  AUPRC: {rf_auprc:.4f}\n")
    
    results['random_forest'] = {'auroc': rf_auroc, 'auprc': rf_auprc}
    
    # Model 2: Gradient Boosting
    print("-" * 80)
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
    
    gb_pred = cross_val_predict_proba(gb, X, y, cv, "Gradient Boosting")
    gb_auroc = roc_auc_score(y, gb_pred)
    gb_auprc = average_precision_score(y, gb_pred)
    
    print(f"✓ AUROC: {gb_auroc:.4f} (Δ={gb_auroc - baseline_auroc:+.4f})")
    print(f"  AUPRC: {gb_auprc:.4f}\n")
    
    results['gradient_boosting'] = {'auroc': gb_auroc, 'auprc': gb_auprc}
    
    # Model 3: SVM-RBF
    print("-" * 80)
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(
            C=10,
            gamma='scale',
            kernel='rbf',
            probability=True,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    
    svm_pred = cross_val_predict_proba(svm_pipeline, X, y, cv, "SVM-RBF")
    svm_auroc = roc_auc_score(y, svm_pred)
    svm_auprc = average_precision_score(y, svm_pred)
    
    print(f"✓ AUROC: {svm_auroc:.4f} (Δ={svm_auroc - baseline_auroc:+.4f})")
    print(f"  AUPRC: {svm_auprc:.4f}\n")
    
    results['svm_rbf'] = {'auroc': svm_auroc, 'auprc': svm_auprc}
    
    # Model 4: Neural Network
    if HAS_TF:
        print("-" * 80)
        nn_pred = cross_val_predict_proba_nn_fast(X, y, cv)
        nn_auroc = roc_auc_score(y, nn_pred)
        nn_auprc = average_precision_score(y, nn_pred)
        
        print(f"✓ AUROC: {nn_auroc:.4f} (Δ={nn_auroc - baseline_auroc:+.4f})")
        print(f"  AUPRC: {nn_auprc:.4f}\n")
        
        results['neural_network'] = {'auroc': nn_auroc, 'auprc': nn_auprc}
        
        # Meta-Ensemble: RF + GB + SVM + NN
        print("=" * 80)
        print("META-ENSEMBLE (RF + GB + SVM + NN)")
        print("=" * 80)
        
        ensemble_pred = (rf_pred + gb_pred + svm_pred + nn_pred) / 4
        ensemble_auroc = roc_auc_score(y, ensemble_pred)
        ensemble_auprc = average_precision_score(y, ensemble_pred)
        
        print(f"\n✓ AUROC: {ensemble_auroc:.4f} (Δ={ensemble_auroc - baseline_auroc:+.4f})")
        print(f"  AUPRC: {ensemble_auprc:.4f}")
        print(f"  Improvement vs baseline: {((ensemble_auroc/baseline_auroc - 1) * 100):+.2f}%")
        
        results['meta_ensemble_with_dl'] = {'auroc': ensemble_auroc, 'auprc': ensemble_auprc}
        
        if ensemble_auroc >= 0.95:
            print("\n🎯 TARGET ACHIEVED: AUROC ≥ 0.95!")
        elif ensemble_auroc >= 0.90:
            print("\n✓ EXCELLENT: AUROC ≥ 0.90")
        elif ensemble_auroc >= 0.85:
            print("\n✓ STATE-OF-THE-ART: AUROC ≥ 0.85")
    else:
        print("⚠️  TensorFlow not installed, skipping Neural Network")
        
        # 3-way ensemble
        ensemble_pred = (rf_pred + gb_pred + svm_pred) / 3
        ensemble_auroc = roc_auc_score(y, ensemble_pred)
        ensemble_auprc = average_precision_score(y, ensemble_pred)
        
        results['meta_ensemble'] = {'auroc': ensemble_auroc, 'auprc': ensemble_auprc}
    
    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Method':<40} {'AUROC':>10} {'AUPRC':>10}")
    print("-" * 80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['auroc'], reverse=True)
    
    for name, result in sorted_results:
        auroc = result['auroc']
        auprc = result['auprc']
        marker = "🏆" if auroc == max(r['auroc'] for r in results.values()) else "  "
        display_name = name.replace('_', ' ').title()
        print(f"{marker} {display_name:<38} {auroc:>10.4f} {auprc:>10.4f}")
    
    # Save results
    output_file = 'stacking_with_dl_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")


if __name__ == '__main__':
    main()
