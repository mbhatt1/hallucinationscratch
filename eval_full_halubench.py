#!/usr/bin/env python3
"""
Train stacked models on n=200, evaluate on full HaluBench dataset.
Tests generalization of learned stacking to larger unseen data.
"""

import json
import numpy as np
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
import sys
import os

# Add PCIB detector to path
sys.path.insert(0, 'pcib_detector/src')
from pcib_detector import PCIBDetector
from pcib_detector.backends.openai_backend import OpenAIBackend

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


def train_models(X_train, y_train):
    """Train stacked models on training data."""
    print("Training stacked models on n=200...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {}
    
    # Random Forest
    print("  Training Random Forest...")
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_grid = GridSearchCV(rf, rf_params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
    rf_grid.fit(X_train, y_train)
    rf_calibrated = CalibratedClassifierCV(rf_grid.best_estimator_, method='isotonic', cv=cv)
    rf_calibrated.fit(X_train, y_train)
    models['random_forest'] = rf_calibrated
    print(f"    Best params: {rf_grid.best_params_}")
    
    # Gradient Boosting
    print("  Training Gradient Boosting...")
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }
    gb = GradientBoostingClassifier(random_state=42)
    gb_grid = GridSearchCV(gb, gb_params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
    gb_grid.fit(X_train, y_train)
    models['gradient_boosting'] = gb_grid.best_estimator_
    print(f"    Best params: {gb_grid.best_params_}")
    
    # SVM with RBF kernel
    print("  Training SVM-RBF...")
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
    svm_grid.fit(X_train, y_train)
    svm_calibrated = CalibratedClassifierCV(svm_grid.best_estimator_, method='isotonic', cv=cv)
    svm_calibrated.fit(X_train, y_train)
    models['svm_rbf'] = svm_calibrated
    print(f"    Best params: {svm_grid.best_params_}")
    
    print("✓ Training complete\n")
    return models


def evaluate_on_full_dataset(models, dataset_name="PatronusAI/HaluBench"):
    """Evaluate trained models on full HaluBench dataset."""
    print(f"Loading full HaluBench dataset...")
    
    # Load dataset from HuggingFace
    try:
        dataset = load_dataset(dataset_name, split="train")  # or appropriate split
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to test split...")
        dataset = load_dataset(dataset_name, split="test")
    
    print(f"Dataset loaded: {len(dataset)} examples")
    
    # Initialize PCIB detector
    print("Initializing PCIB detector...")
    backend = OpenAIBackend()
    detector = PCIBDetector(backend=backend)
    
    # Run PCIB on full dataset to get signals
    print("Running PCIB evaluation on full dataset (this may take a while)...")
    
    all_features = []
    all_labels = []
    all_pcib_scores = []
    
    for i, example in enumerate(dataset):
        if i % 100 == 0:
            print(f"  Processing example {i}/{len(dataset)}...")
        
        # Extract question, answer, context
        question = example.get('question', '')
        answer = example.get('answer', '')
        context = example.get('context', example.get('passage', ''))
        label = 1 if str(example.get('label', '')).lower() in ['fail', 'failed', '1', 'true'] else 0
        
        # Run PCIB detection
        try:
            result = detector.detect(question=question, answer=answer, context=context)
            
            # Extract signals
            uptake = result.signals.get('uptake', 0.0)
            stress = result.signals.get('stress', 0.0)
            conflict = result.signals.get('conflict', 0.0)
            composite = result.risk_score
            
            feature_vec = [uptake, stress, conflict, 0.0, composite]
            all_features.append(feature_vec)
            all_labels.append(label)
            all_pcib_scores.append(composite)
            
        except Exception as e:
            print(f"  Error on example {i}: {e}")
            continue
    
    X_test = np.array(all_features)
    y_test = np.array(all_labels)
    
    print(f"\n✓ Evaluation dataset prepared: {len(X_test)} examples")
    print(f"  Class balance: {np.mean(y_test):.2%} positive\n")
    
    # Evaluate each model
    results = {}
    
    # Baseline: PCIB theory-guided
    pcib_auroc = roc_auc_score(y_test, all_pcib_scores)
    pcib_auprc = average_precision_score(y_test, all_pcib_scores)
    results['pcib_baseline'] = {
        'auroc': pcib_auroc,
        'auprc': pcib_auprc,
        'method': 'PCIB Theory-Guided (Full Dataset)'
    }
    print(f"PCIB Baseline (Full Dataset):")
    print(f"  AUROC: {pcib_auroc:.4f}")
    print(f"  AUPRC: {pcib_auprc:.4f}\n")
    
    # Evaluate stacked models (trained on n=200)
    for model_name, model in models.items():
        print(f"Evaluating {model_name} (trained on n=200, tested on full dataset)...")
        pred_proba = model.predict_proba(X_test)[:, 1]
        
        auroc = roc_auc_score(y_test, pred_proba)
        auprc = average_precision_score(y_test, pred_proba)
        
        results[model_name] = {
            'auroc': auroc,
            'auprc': auprc,
            'improvement': auroc - pcib_auroc
        }
        
        print(f"  AUROC: {auroc:.4f} (Δ={auroc - pcib_auroc:+.4f})")
        print(f"  AUPRC: {auprc:.4f}\n")
    
    # Meta-ensemble
    print("Evaluating Meta-Ensemble...")
    rf_pred = models['random_forest'].predict_proba(X_test)[:, 1]
    gb_pred = models['gradient_boosting'].predict_proba(X_test)[:, 1]
    svm_pred = models['svm_rbf'].predict_proba(X_test)[:, 1]
    
    ensemble_pred = (rf_pred + gb_pred + svm_pred) / 3
    ensemble_auroc = roc_auc_score(y_test, ensemble_pred)
    ensemble_auprc = average_precision_score(y_test, ensemble_pred)
    
    results['meta_ensemble'] = {
        'auroc': ensemble_auroc,
        'auprc': ensemble_auprc,
        'improvement': ensemble_auroc - pcib_auroc
    }
    
    print(f"  AUROC: {ensemble_auroc:.4f} (Δ={ensemble_auroc - pcib_auroc:+.4f})")
    print(f"  AUPRC: {ensemble_auprc:.4f}\n")
    
    return results


def print_summary(results):
    """Print summary of results."""
    print("=" * 80)
    print("FINAL RESULTS: Full HaluBench Evaluation")
    print("=" * 80)
    print()
    print(f"{'Method':<40} {'AUROC':>10} {'AUPRC':>10} {'Δ AUROC':>12}")
    print("-" * 80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['auroc'], reverse=True)
    
    for name, result in sorted_results:
        auroc = result['auroc']
        auprc = result['auprc']
        improvement = result.get('improvement', 0)
        marker = "🏆" if auroc == max(r['auroc'] for r in results.values()) else "  "
        method_name = result.get('method', name.replace('_', ' ').title())
        
        print(f"{marker} {method_name:<38} {auroc:>10.4f} {auprc:>10.4f} {improvement:>+11.4f}")
    
    print()
    best = max(results.items(), key=lambda x: x[1]['auroc'])
    print(f"Best model: {best[0]} with AUROC {best[1]['auroc']:.4f}")
    
    if best[1]['auroc'] >= 0.85:
        print("\n🎯 SUCCESS: Stacked model achieves SOTA on full dataset!")
    elif best[1]['auroc'] >= 0.83:
        print("\n✓ STRONG: Near SOTA generalization to full dataset")


def main():
    print("=" * 80)
    print("TRAIN ON n=200, EVALUATE ON FULL HALUBENCH")
    print("=" * 80)
    print()
    
    # Load training data (n=200)
    print("Loading training data (n=200)...")
    X_train, y_train = load_training_data()
    print(f"  Training set: {len(X_train)} examples")
    print(f"  Class balance: {np.mean(y_train):.2%} positive\n")
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate on full dataset
    results = evaluate_on_full_dataset(models)
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_file = 'full_halubench_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")


if __name__ == '__main__':
    main()
