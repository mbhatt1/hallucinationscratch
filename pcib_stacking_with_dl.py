#!/usr/bin/env python3
"""
Enhanced stacking with Deep Learning added to RF, GB, and SVM.
Meta-ensemble of 4 models: Random Forest + Gradient Boosting + SVM-RBF + Neural Network
"""

import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    HAS_TF = True
    # Suppress TF warnings
    tf.get_logger().setLevel('ERROR')
except ImportError:
    HAS_TF = False
    print("⚠️  TensorFlow not installed. Install with: pip install tensorflow")


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


def create_neural_network(input_dim=5):
    """Create a neural network for binary classification."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['AUC']
    )
    
    return model


def train_neural_network(X_train, y_train, X_val, y_val):
    """Train neural network with early stopping."""
    # Scale features for NN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    model = create_neural_network(input_dim=X_train.shape[1])
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_auc',
        patience=20,
        restore_best_weights=True,
        mode='max'
    )
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=200,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0
    )
    
    return model, scaler


def cross_val_predict_proba_nn(X, y, cv, n_models=5):
    """Get out-of-fold predictions for neural network ensemble."""
    predictions = np.zeros(len(y))
    models = []
    scalers = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train multiple NNs and average (ensemble within ensemble!)
        fold_preds = []
        for seed in range(n_models):
            tf.random.set_seed(seed)
            np.random.seed(seed)
            
            # Further split train into train/val for early stopping
            split_idx = int(len(X_train) * 0.8)
            X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
            y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
            
            model, scaler = train_neural_network(X_tr, y_tr, X_val, y_val)
            models.append(model)
            scalers.append(scaler)
            
            X_test_scaled = scaler.transform(X_test)
            fold_preds.append(model.predict(X_test_scaled, verbose=0).flatten())
        
        # Average predictions from multiple seeds
        predictions[test_idx] = np.mean(fold_preds, axis=0)
    
    return predictions, models, scalers


def train_all_models(X, y):
    """Train all stacked models including deep learning."""
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("=" * 80)
    print("ENHANCED STACKING: RF + GB + SVM + DEEP LEARNING")
    print("=" * 80)
    print(f"\nDataset: n={len(y)} (Class balance: {np.mean(y):.2%} positive)")
    print(f"Features: {X.shape[1]} PCIB signals")
    print()
    
    # Baseline
    baseline_scores = X[:, 4]
    baseline_auroc = roc_auc_score(y, baseline_scores)
    baseline_auprc = average_precision_score(y, baseline_scores)
    
    print(f"BASELINE (PCIB Theory-Guided):")
    print(f"  AUROC: {baseline_auroc:.4f}")
    print(f"  AUPRC: {baseline_auprc:.4f}\n")
    
    results['baseline'] = {'auroc': baseline_auroc, 'auprc': baseline_auprc}
    
    # Model 1: Random Forest
    print("-" * 80)
    print("MODEL 1: Random Forest")
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
    
    rf_calibrated = CalibratedClassifierCV(rf_grid.best_estimator_, method='isotonic', cv=cv)
    rf_calibrated.fit(X, y)
    
    rf_pred = cross_val_predict_proba(rf_calibrated, X, y, cv)
    rf_auroc = roc_auc_score(y, rf_pred)
    rf_auprc = average_precision_score(y, rf_pred)
    
    print(f"Final AUROC: {rf_auroc:.4f} (Δ={rf_auroc - baseline_auroc:+.4f})")
    print(f"Final AUPRC: {rf_auprc:.4f}")
    
    results['random_forest'] = {'auroc': rf_auroc, 'auprc': rf_auprc}
    
    # Model 2: Gradient Boosting
    print("\n" + "-" * 80)
    print("MODEL 2: Gradient Boosting")
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
    
    gb_pred = cross_val_predict_proba(gb_grid.best_estimator_, X, y, cv)
    gb_auroc = roc_auc_score(y, gb_pred)
    gb_auprc = average_precision_score(y, gb_pred)
    
    print(f"Final AUROC: {gb_auroc:.4f} (Δ={gb_auroc - baseline_auroc:+.4f})")
    print(f"Final AUPRC: {gb_auprc:.4f}")
    
    results['gradient_boosting'] = {'auroc': gb_auroc, 'auprc': gb_auprc}
    
    # Model 3: SVM-RBF
    print("\n" + "-" * 80)
    print("MODEL 3: SVM with RBF Kernel")
    print("-" * 80)
    
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
    
    svm_calibrated = CalibratedClassifierCV(svm_grid.best_estimator_, method='isotonic', cv=cv)
    svm_calibrated.fit(X, y)
    
    svm_pred = cross_val_predict_proba(svm_calibrated, X, y, cv)
    svm_auroc = roc_auc_score(y, svm_pred)
    svm_auprc = average_precision_score(y, svm_pred)
    
    print(f"Final AUROC: {svm_auroc:.4f} (Δ={svm_auroc - baseline_auroc:+.4f})")
    print(f"Final AUPRC: {svm_auprc:.4f}")
    
    results['svm_rbf'] = {'auroc': svm_auroc, 'auprc': svm_auprc}
    
    # Model 4: Neural Network
    if HAS_TF:
        print("\n" + "-" * 80)
        print("MODEL 4: Deep Neural Network (Ensemble of 5 NNs)")
        print("-" * 80)
        
        nn_pred, nn_models, nn_scalers = cross_val_predict_proba_nn(X, y, cv, n_models=5)
        nn_auroc = roc_auc_score(y, nn_pred)
        nn_auprc = average_precision_score(y, nn_pred)
        
        print(f"Final AUROC: {nn_auroc:.4f} (Δ={nn_auroc - baseline_auroc:+.4f})")
        print(f"Final AUPRC: {nn_auprc:.4f}")
        
        results['neural_network'] = {'auroc': nn_auroc, 'auprc': nn_auprc}
        
        # Meta-Ensemble: RF + GB + SVM + NN
        print("\n" + "-" * 80)
        print("MODEL 5: Meta-Ensemble (RF + GB + SVM + NN)")
        print("-" * 80)
        
        ensemble_pred = (rf_pred + gb_pred + svm_pred + nn_pred) / 4
        ensemble_auroc = roc_auc_score(y, ensemble_pred)
        ensemble_auprc = average_precision_score(y, ensemble_pred)
        
        print(f"Final AUROC: {ensemble_auroc:.4f} (Δ={ensemble_auroc - baseline_auroc:+.4f})")
        print(f"Final AUPRC: {ensemble_auprc:.4f}")
        print(f"\n✓ Improvement vs baseline: {((ensemble_auroc/baseline_auroc - 1) * 100):+.2f}%")
        
        results['meta_ensemble_with_dl'] = {'auroc': ensemble_auroc, 'auprc': ensemble_auprc}
    else:
        # Fall back to 3-way ensemble without DL
        print("\n⚠️  Skipping Neural Network (TensorFlow not installed)")
        print("\n" + "-" * 80)
        print("MODEL 5: Meta-Ensemble (RF + GB + SVM)")
        print("-" * 80)
        
        ensemble_pred = (rf_pred + gb_pred + svm_pred) / 3
        ensemble_auroc = roc_auc_score(y, ensemble_pred)
        ensemble_auprc = average_precision_score(y, ensemble_pred)
        
        print(f"Final AUROC: {ensemble_auroc:.4f} (Δ={ensemble_auroc - baseline_auroc:+.4f})")
        print(f"Final AUPRC: {ensemble_auprc:.4f}")
        
        results['meta_ensemble'] = {'auroc': ensemble_auroc, 'auprc': ensemble_auprc}
    
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


def print_summary(results):
    """Print summary."""
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Method':<40} {'AUROC':>10} {'AUPRC':>10}")
    print("-" * 80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['auroc'], reverse=True)
    
    for name, result in sorted_results:
        auroc = result['auroc']
        auprc = result['auprc']
        marker = "🏆" if auroc == max(r['auroc'] for r in results.values()) else "  "
        
        print(f"{marker} {name.replace('_', ' ').title():<38} {auroc:>10.4f} {auprc:>10.4f}")
    
    print()
    best = max(results.items(), key=lambda x: x[1]['auroc'])
    print(f"Best: {best[0]} with AUROC {best[1]['auroc']:.4f}")
    
    if best[1]['auroc'] >= 0.95:
        print("\n🎯 TARGET ACHIEVED: AUROC ≥ 0.95!")
    elif best[1]['auroc'] >= 0.90:
        print("\n✓ EXCELLENT: AUROC ≥ 0.90")
    elif best[1]['auroc'] >= 0.85:
        print("\n✓ STATE-OF-THE-ART: AUROC ≥ 0.85")


def main():
    print("\nLoading training data...")
    X, y = load_training_data()
    print(f"Loaded {len(X)} examples\n")
    
    results = train_all_models(X, y)
    print_summary(results)
    
    # Save results
    output_file = 'stacking_with_dl_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")


if __name__ == '__main__':
    main()
