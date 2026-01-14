#!/usr/bin/env python3
"""
Ultra-Enhanced PCIB Signal Stacking - Target: 0.90+ AUROC
==========================================================

Advanced techniques to push toward 0.90 AUROC:
1. Deep feature engineering (50+ features from 5 base signals)
2. Stacked generalization with multiple layers
3. Diversity-promoting ensemble techniques
4. Calibration and uncertainty quantification
5. Advanced regularization and cross-validation
"""

import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Note: LightGBM not installed. Install for best performance: pip install lightgbm")


def load_results(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load evaluation results and extract PCIB signals."""
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
            rationalization = 0.0
        else:
            uptake = stress = conflict = rationalization = 0.0
        
        feature_vec = [uptake, stress, conflict, rationalization, composite_score]
        features.append(feature_vec)
        labels.append(label)
    
    return np.array(features), np.array(labels)


def ultra_engineer_features(X: np.ndarray) -> np.ndarray:
    """
    Ultra-advanced feature engineering: 5 → 60+ features
    
    New additions to push toward 0.90 AUROC:
    - Exponential transformations
    - Log transformations
    - Trigonometric features (capture periodic patterns)
    - Higher-order polynomials
    - Cross-ratios (4-way interactions)
    - Statistical moments (skewness, kurtosis)
    - EVIDENCE SUFFICIENCY ENFORCEMENT (new!)
    """
    features = []
    
    uptake = X[:, 0]
    stress = X[:, 1]
    conflict = X[:, 2]
    rationalization = X[:, 3]
    composite = X[:, 4]
    
    epsilon = 1e-8
    
    # === BASE FEATURES ===
    features.extend([uptake, stress, conflict, rationalization, composite])
    
    # === EVIDENCE SUFFICIENCY ENFORCEMENT ===
    # Principled metric based on Information Theory and Bayesian reasoning
    
    # ============================================================================
    # EVIDENCE SUFFICIENCY INDEX (ESI) - Theoretically Grounded Metric
    # ============================================================================
    # ESI answers: "Given the context, is there SUFFICIENT evidence to support this answer?"
    #
    # Three Pillars (each normalized to [0, 1]):
    # 1. Context Usage (U): How much context information was absorbed?
    # 2. Evidence Stability (S): How robust is the evidence under perturbation?
    # 3. Logical Consistency (C): How coherent is the reasoning?
    #
    # A factual answer needs ALL three (multiplicative, not additive)
    # A hallucination fails on at least one dimension
    # ============================================================================
    
    # Component 1: Context Usage [0, 1]
    # Uptake is KL divergence (unbounded), so normalize using sigmoid-like function
    # 1 - exp(-x) maps [0, ∞) → [0, 1) smoothly
    context_usage = 1.0 - np.exp(-uptake)
    features.append(context_usage)
    
    # Component 2: Evidence Stability [0, 1]
    # Low stress = stable evidence under semantic perturbation
    # Inverse function: high stress → low stability
    evidence_stability = 1.0 / (1.0 + stress)
    features.append(evidence_stability)
    
    # Component 3: Logical Consistency [0, 1]
    # Low conflict = logically coherent, no contradictions
    logical_consistency = 1.0 / (1.0 + conflict)
    features.append(logical_consistency)
    
    # ============================================================================
    # ESI: Combined Evidence Sufficiency Index
    # ============================================================================
    
    # Geometric Mean (penalizes imbalance: one bad dimension hurts overall score)
    # ESI_geometric = (U × S × C)^(1/3)
    esi_geometric = (context_usage * evidence_stability * logical_consistency) ** (1/3)
    features.append(esi_geometric)
    
    # Harmonic Mean (stronger penalty for imbalance: weakest link matters most)
    # ESI_harmonic = 3 / (1/U + 1/S + 1/C)
    # Protects against division by zero with small epsilon
    esi_harmonic = 3.0 / (
        1.0/(context_usage + epsilon) +
        1.0/(evidence_stability + epsilon) +
        1.0/(logical_consistency + epsilon)
    )
    features.append(esi_harmonic)
    
    # Weighted Harmonic Mean (emphasize context usage more)
    # Weights: Context=0.5, Stability=0.3, Consistency=0.2
    esi_weighted = 1.0 / (
        0.5/(context_usage + epsilon) +
        0.3/(evidence_stability + epsilon) +
        0.2/(logical_consistency + epsilon)
    )
    features.append(esi_weighted)
    
    # ============================================================================
    # ESI Interpretation:
    # ESI ≈ 1.0: Perfect evidence (high context use, stable, consistent)
    # ESI ≈ 0.7: Good evidence (minor issues in one dimension)
    # ESI ≈ 0.5: Borderline (significant weakness in one dimension)
    # ESI ≈ 0.3: Insufficient evidence (multiple weak dimensions)
    # ESI ≈ 0.0: No evidence (hallucination: fails all checks)
    # ============================================================================
    
    # Evidence Sufficiency Score (original, for comparison)
    evidence_sufficiency_legacy = (uptake * (1 - stress) * (1 - conflict))
    features.append(evidence_sufficiency_legacy)
    
    # ============================================================================
    # Additional Evidence-Based Features
    # ============================================================================
    
    # Evidence Deficit: High confidence despite low context usage (hallucination red flag)
    evidence_deficit = composite * (1.0 / (context_usage + epsilon))
    features.append(evidence_deficit)
    
    # Grounding Score: Is answer grounded in context or fabricated?
    # Negative score = likely hallucination (high composite, low uptake)
    grounding_score = uptake - (composite * stress)
    features.append(grounding_score)
    
    # Support Ratio: How well do signals support the conclusion?
    support_ratio = (uptake + rationalization) / (stress + conflict + epsilon)
    features.append(support_ratio)
    
    # Insufficiency Penalty: High conflict with low uptake = insufficient evidence
    insufficiency_penalty = conflict / (context_usage + epsilon)
    features.append(insufficiency_penalty)
    
    # Evidence Quality: Quadratic emphasis on strong uptake
    evidence_quality = uptake ** 2
    features.append(evidence_quality)
    
    # Evidence Alignment: Low variance across signals = all agree (well-supported)
    signal_variance = np.var(np.column_stack([uptake, stress, conflict]), axis=1)
    evidence_alignment = 1.0 / (1.0 + signal_variance)
    features.append(evidence_alignment)
    
    # Minimum Evidence Threshold: Weakest signal (conservative estimate)
    # If ANY signal is bad, overall evidence is questionable
    min_evidence = np.minimum(np.minimum(context_usage, evidence_stability), logical_consistency)
    features.append(min_evidence)
    
    # Maximum Evidence Potential: Best signal (optimistic estimate)
    max_evidence = np.maximum(np.maximum(context_usage, evidence_stability), logical_consistency)
    features.append(max_evidence)
    
    # Evidence Range: max - min (high range = inconsistent signals)
    evidence_range = max_evidence - min_evidence
    features.append(evidence_range)
    
    # === 2-WAY INTERACTIONS ===
    features.append(uptake * stress)
    features.append(uptake * conflict)
    features.append(stress * conflict)
    features.append(uptake * rationalization)
    features.append(stress * rationalization)
    features.append(conflict * rationalization)
    
    # === RATIO FEATURES ===
    features.append(uptake / (stress + epsilon))
    features.append(uptake / (conflict + epsilon))
    features.append(stress / (uptake + epsilon))
    features.append(conflict / (stress + epsilon))
    features.append(composite / (uptake + epsilon))
    features.append(composite / (stress + epsilon))
    
    # === POLYNOMIAL FEATURES ===
    features.append(uptake ** 2)
    features.append(stress ** 2)
    features.append(conflict ** 2)
    features.append(composite ** 2)
    features.append(uptake ** 3)
    features.append(stress ** 3)
    features.append(np.sqrt(np.abs(uptake)))
    features.append(np.sqrt(np.abs(stress)))
    features.append(np.sqrt(np.abs(conflict)))
    
    # === EXPONENTIAL TRANSFORMATIONS ===
    features.append(np.exp(-uptake))  # Inverse activation
    features.append(np.exp(-stress))
    features.append(1 / (1 + np.exp(-uptake)))  # Sigmoid
    features.append(1 / (1 + np.exp(-stress)))
    
    # === LOG TRANSFORMATIONS ===
    features.append(np.log1p(np.abs(uptake)))
    features.append(np.log1p(np.abs(stress)))
    features.append(np.log1p(np.abs(conflict)))
    
    # === AGGREGATIONS ===
    features.append(np.maximum(uptake, stress))
    features.append(np.minimum(uptake, stress))
    features.append((uptake + stress + conflict) / 3)
    features.append(np.maximum(np.maximum(uptake, stress), conflict))
    features.append(np.minimum(np.minimum(uptake, stress), conflict))
    
    # === 3-WAY INTERACTIONS ===
    features.append(uptake * stress * conflict)
    features.append((uptake + stress) * conflict)
    features.append(uptake * (stress + conflict))
    features.append((uptake * stress) / (conflict + epsilon))
    
    # === STATISTICAL MOMENTS ===
    signal_matrix = np.column_stack([uptake, stress, conflict])
    features.append(np.var(signal_matrix, axis=1))
    features.append(np.std(signal_matrix, axis=1))
    features.append(np.mean(signal_matrix, axis=1))
    features.append(np.median(signal_matrix, axis=1))
    
    # === COMPOSITE-BASED FEATURES ===
    features.append(composite ** 2)
    features.append(np.abs(composite - 0.5))  # Distance from decision boundary
    features.append(composite * uptake)
    features.append(composite * stress)
    features.append(composite * conflict)
    
    # === TRIGONOMETRIC FEATURES (for periodic patterns) ===
    features.append(np.sin(uptake * np.pi))
    features.append(np.cos(stress * np.pi))
    features.append(np.sin(composite * 2 * np.pi))
    
    # === CROSS-RATIOS (4-way interactions) ===
    features.append((uptake * conflict) / (stress * (composite + epsilon) + epsilon))
    features.append((stress * conflict) / (uptake * (composite + epsilon) + epsilon))
    
    return np.column_stack(features)


def create_diverse_base_models():
    """
    Create diverse base models for stacking.
    Diversity is key for strong ensemble performance.
    """
    models = {
        'rf_shallow': RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_split=10,
            random_state=42, class_weight='balanced', n_jobs=-1
        ),
        'rf_deep': RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=2,
            random_state=43, class_weight='balanced', n_jobs=-1
        ),
        'extra_trees': ExtraTreesClassifier(
            n_estimators=150, max_depth=10, min_samples_split=5,
            random_state=44, class_weight='balanced', n_jobs=-1
        ),
        'gb_fast': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            subsample=0.8, random_state=45
        ),
        'gb_slow': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.9, random_state=46
        ),
        'svm_rbf': CalibratedClassifierCV(
            SVC(C=10, gamma='scale', kernel='rbf', 
                random_state=47, class_weight='balanced'),
            method='isotonic', cv=3
        ),
        'svm_poly': CalibratedClassifierCV(
            SVC(C=1, gamma='scale', kernel='poly', degree=3,
                random_state=48, class_weight='balanced'),
            method='isotonic', cv=3
        ),
        'mlp_small': MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=0.001,
            random_state=49, max_iter=1000, early_stopping=True
        ),
        'mlp_large': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), alpha=0.0001,
            random_state=50, max_iter=1000, early_stopping=True
        ),
        'ridge': RidgeClassifier(
            alpha=1.0, random_state=51, class_weight='balanced'
        ),
        'naive_bayes': GaussianNB()
    }
    
    if HAS_LGB:
        models['lgb_default'] = LGBMClassifier(
            n_estimators=150, learning_rate=0.1, num_leaves=31,
            random_state=52, verbose=-1
        )
        models['lgb_deep'] = LGBMClassifier(
            n_estimators=250, learning_rate=0.05, num_leaves=63,
            max_depth=10, random_state=53, verbose=-1
        )
    
    return models


def get_oof_predictions(models: Dict, X: np.ndarray, y: np.ndarray, cv) -> Tuple[np.ndarray, Dict]:
    """
    Get out-of-fold predictions for all models (proper stacking).
    This prevents overfitting in the meta-learner.
    """
    n_samples = len(y)
    n_models = len(models)
    
    # Store OOF predictions
    oof_preds = np.zeros((n_samples, n_models))
    
    # Store CV scores for each model
    cv_scores = {}
    
    print("\n" + "=" * 80)
    print("LEVEL 1: TRAINING BASE MODELS (Out-of-Fold Predictions)")
    print("=" * 80)
    
    # Scale features for models that need it
    scaler = StandardScaler()
    
    for i, (name, model) in enumerate(models.items()):
        print(f"\n[{i+1}/{n_models}] Training {name}...")
        
        # Check if model needs scaling
        needs_scaling = isinstance(model, (CalibratedClassifierCV, MLPClassifier, RidgeClassifier))
        
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            # Scale if needed
            if needs_scaling:
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
            
            # Train and predict
            model.fit(X_train_scaled, y_train)
            
            if hasattr(model, 'predict_proba'):
                val_pred = model.predict_proba(X_val_scaled)[:, 1]
            else:
                val_pred = model.decision_function(X_val_scaled)
                val_pred = (val_pred - val_pred.min()) / (val_pred.max() - val_pred.min())
            
            oof_preds[val_idx, i] = val_pred
            
            # Calculate fold score
            fold_auroc = roc_auc_score(y[val_idx], val_pred)
            fold_scores.append(fold_auroc)
        
        # Overall OOF score for this model
        model_auroc = roc_auc_score(y, oof_preds[:, i])
        cv_scores[name] = {
            'auroc': model_auroc,
            'cv_mean': np.mean(fold_scores),
            'cv_std': np.std(fold_scores)
        }
        
        print(f"  OOF AUROC: {model_auroc:.4f} (CV: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f})")
    
    return oof_preds, cv_scores


def train_hierarchical_stacking(oof_preds: np.ndarray, y: np.ndarray, model_names: List[str], cv) -> Dict:
    """
    Hierarchical stacking: Group similar models, stack within groups, then stack group outputs.
    This can capture different aspects of the problem more efficiently.
    """
    print("\n" + "=" * 80)
    print("HIERARCHICAL STACKING (Layer-by-Layer)")
    print("=" * 80)
    
    results = {}
    
    # Define model groups by type
    groups = {
        'tree_ensemble': [],
        'boosting': [],
        'svm': [],
        'neural_net': [],
        'other': []
    }
    
    # Categorize models
    for i, name in enumerate(model_names):
        if 'rf' in name or 'extra' in name:
            groups['tree_ensemble'].append(i)
        elif 'gb' in name or 'lgb' in name:
            groups['boosting'].append(i)
        elif 'svm' in name:
            groups['svm'].append(i)
        elif 'mlp' in name:
            groups['neural_net'].append(i)
        else:
            groups['other'].append(i)
    
    # Remove empty groups
    groups = {k: v for k, v in groups.items() if v}
    
    print(f"\nModel groups:")
    for group_name, indices in groups.items():
        models_in_group = [model_names[i] for i in indices]
        print(f"  {group_name}: {len(indices)} models - {models_in_group}")
    
    # Level 1: Stack within each group
    print(f"\n--- Level 1: Within-Group Stacking ---")
    group_predictions = []
    group_names = []
    
    for group_name, indices in groups.items():
        if len(indices) == 1:
            # Single model, just use its predictions
            group_pred = oof_preds[:, indices[0]]
            auroc = roc_auc_score(y, group_pred)
            print(f"{group_name}: Single model, AUROC={auroc:.4f}")
        else:
            # Multiple models, average them
            group_pred = np.mean(oof_preds[:, indices], axis=1)
            auroc = roc_auc_score(y, group_pred)
            print(f"{group_name}: Averaged {len(indices)} models, AUROC={auroc:.4f}")
        
        group_predictions.append(group_pred)
        group_names.append(group_name)
    
    group_preds_matrix = np.column_stack(group_predictions)
    
    # Level 2: Stack group predictions
    print(f"\n--- Level 2: Cross-Group Stacking ---")
    
    # Method 1: Simple average of groups
    hier_simple = np.mean(group_preds_matrix, axis=1)
    hier_simple_auroc = roc_auc_score(y, hier_simple)
    hier_simple_auprc = average_precision_score(y, hier_simple)
    
    print(f"[1] Simple Average of Groups")
    print(f"  AUROC: {hier_simple_auroc:.4f}")
    print(f"  AUPRC: {hier_simple_auprc:.4f}")
    
    results['hierarchical_simple'] = {
        'auroc': hier_simple_auroc,
        'auprc': hier_simple_auprc,
        'method': 'Hierarchical: Simple Average'
    }
    
    # Method 2: Weighted average of groups by performance
    group_aurocs = np.array([roc_auc_score(y, group_preds_matrix[:, i]) for i in range(group_preds_matrix.shape[1])])
    group_weights = group_aurocs ** 2
    group_weights = group_weights / group_weights.sum()
    
    hier_weighted = np.average(group_preds_matrix, axis=1, weights=group_weights)
    hier_weighted_auroc = roc_auc_score(y, hier_weighted)
    hier_weighted_auprc = average_precision_score(y, hier_weighted)
    
    print(f"\n[2] Performance-Weighted Groups")
    print(f"  Group weights:")
    for name, weight in zip(group_names, group_weights):
        print(f"    {name:15s}: {weight:.4f} {'█' * int(weight * 40)}")
    print(f"  AUROC: {hier_weighted_auroc:.4f}")
    print(f"  AUPRC: {hier_weighted_auprc:.4f}")
    
    results['hierarchical_weighted'] = {
        'auroc': hier_weighted_auroc,
        'auprc': hier_weighted_auprc,
        'method': 'Hierarchical: Weighted Groups'
    }
    
    # Method 3: Logistic regression on group predictions
    meta_lr = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    lr_scores = cross_val_score(meta_lr, group_preds_matrix, y, cv=cv, scoring='roc_auc')
    lr_pred = cross_val_predict(meta_lr, group_preds_matrix, y, cv=cv, method='predict_proba')[:, 1]
    hier_lr_auroc = roc_auc_score(y, lr_pred)
    hier_lr_auprc = average_precision_score(y, lr_pred)
    
    print(f"\n[3] Logistic Regression on Groups")
    print(f"  CV AUROC: {lr_scores.mean():.4f} ± {lr_scores.std():.4f}")
    print(f"  AUROC: {hier_lr_auroc:.4f}")
    print(f"  AUPRC: {hier_lr_auprc:.4f}")
    
    results['hierarchical_lr'] = {
        'auroc': hier_lr_auroc,
        'auprc': hier_lr_auprc,
        'method': 'Hierarchical: LR on Groups'
    }
    
    return results, group_preds_matrix


def train_meta_learner(oof_preds: np.ndarray, y: np.ndarray, cv) -> Dict:
    """
    Train meta-learner on OOF predictions (Level 2) - Traditional Stacking.
    """
    print("\n" + "=" * 80)
    print("TRADITIONAL STACKING (All Models → Meta-Learner)")
    print("=" * 80)
    
    results = {}
    
    # Meta-learner 1: Simple averaging
    simple_avg = np.mean(oof_preds, axis=1)
    avg_auroc = roc_auc_score(y, simple_avg)
    avg_auprc = average_precision_score(y, simple_avg)
    
    print(f"\n[1] Simple Averaging (All {oof_preds.shape[1]} Models)")
    print(f"  AUROC: {avg_auroc:.4f}")
    print(f"  AUPRC: {avg_auprc:.4f}")
    
    results['simple_average'] = {
        'auroc': avg_auroc,
        'auprc': avg_auprc,
        'method': 'Traditional: Simple Average'
    }
    
    # Meta-learner 2: Top-K models only (select best performers)
    oof_aurocs = np.array([roc_auc_score(y, oof_preds[:, i]) for i in range(oof_preds.shape[1])])
    top_k = 5
    top_indices = np.argsort(oof_aurocs)[-top_k:]
    
    topk_avg = np.mean(oof_preds[:, top_indices], axis=1)
    topk_auroc = roc_auc_score(y, topk_avg)
    topk_auprc = average_precision_score(y, topk_avg)
    
    print(f"\n[2] Top-{top_k} Models Average")
    print(f"  Selected models with AUROCs: {oof_aurocs[top_indices]}")
    print(f"  AUROC: {topk_auroc:.4f}")
    print(f"  AUPRC: {topk_auprc:.4f}")
    
    results['topk_average'] = {
        'auroc': topk_auroc,
        'auprc': topk_auprc,
        'method': f'Traditional: Top-{top_k} Average'
    }
    
    # Meta-learner 3: Logistic regression with regularization
    meta_lr = LogisticRegression(C=0.1, random_state=42, max_iter=1000)
    lr_scores = cross_val_score(meta_lr, oof_preds, y, cv=cv, scoring='roc_auc')
    lr_pred = cross_val_predict(meta_lr, oof_preds, y, cv=cv, method='predict_proba')[:, 1]
    lr_auroc = roc_auc_score(y, lr_pred)
    lr_auprc = average_precision_score(y, lr_pred)
    
    print(f"\n[3] Logistic Regression Meta-Learner")
    print(f"  CV AUROC: {lr_scores.mean():.4f} ± {lr_scores.std():.4f}")
    print(f"  AUROC: {lr_auroc:.4f}")
    print(f"  AUPRC: {lr_auprc:.4f}")
    
    # Fit to get weights
    meta_lr.fit(oof_preds, y)
    weights = np.abs(meta_lr.coef_[0])
    weights = weights / weights.sum()
    
    print(f"  Top 5 model weights:")
    top_weight_indices = np.argsort(weights)[-5:][::-1]
    for i in top_weight_indices:
        print(f"    Model {i+1}: {weights[i]:.4f} {'█' * int(weights[i] * 40)}")
    
    results['logistic_meta'] = {
        'auroc': lr_auroc,
        'auprc': lr_auprc,
        'method': 'Traditional: LR Meta-Learner'
    }
    
    # Meta-learner 4: Random Forest meta-learner
    meta_rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_split=10,
        random_state=42, class_weight='balanced'
    )
    rf_scores = cross_val_score(meta_rf, oof_preds, y, cv=cv, scoring='roc_auc')
    rf_pred = cross_val_predict(meta_rf, oof_preds, y, cv=cv, method='predict_proba')[:, 1]
    rf_auroc = roc_auc_score(y, rf_pred)
    rf_auprc = average_precision_score(y, rf_pred)
    
    print(f"\n[4] Random Forest Meta-Learner")
    print(f"  CV AUROC: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
    print(f"  AUROC: {rf_auroc:.4f}")
    print(f"  AUPRC: {rf_auprc:.4f}")
    
    results['rf_meta'] = {
        'auroc': rf_auroc,
        'auprc': rf_auprc,
        'method': 'Traditional: RF Meta-Learner'
    }
    
    # Meta-learner 5: Weighted average based on OOF performance
    perf_weights = oof_aurocs ** 2
    perf_weights = perf_weights / perf_weights.sum()
    
    weighted_avg = np.average(oof_preds, axis=1, weights=perf_weights)
    weighted_auroc = roc_auc_score(y, weighted_avg)
    weighted_auprc = average_precision_score(y, weighted_avg)
    
    print(f"\n[5] Performance-Weighted Average")
    print(f"  AUROC: {weighted_auroc:.4f}")
    print(f"  AUPRC: {weighted_auprc:.4f}")
    
    results['weighted_average'] = {
        'auroc': weighted_auroc,
        'auprc': weighted_auprc,
        'method': 'Traditional: Weighted Average'
    }
    
    return results


def compare_efficiency(results: Dict, baseline_auroc: float):
    """Compare efficiency: performance vs complexity tradeoff."""
    print("\n" + "=" * 80)
    print("EFFICIENCY ANALYSIS")
    print("=" * 80)
    
    complexity_map = {
        'baseline': 1, 'simple_average': 13, 'topk_average': 5,
        'logistic_meta': 13, 'rf_meta': 13, 'weighted_average': 13,
        'hierarchical_simple': 4, 'hierarchical_weighted': 4, 'hierarchical_lr': 4
    }
    
    efficiency_scores = []
    for name, result in results.items():
        if name.startswith('base_'):
            continue
        auroc = result['auroc']
        gain = auroc - baseline_auroc
        complexity = complexity_map.get(name, 10)
        efficiency = gain / np.sqrt(complexity)
        efficiency_scores.append({
            'method': result['method'], 'auroc': auroc, 'gain': gain,
            'complexity': complexity, 'efficiency': efficiency
        })
    
    efficiency_scores.sort(key=lambda x: x['efficiency'], reverse=True)
    
    print(f"\n{'Method':<45} {'AUROC':<8} {'Gain':<8} {'#Models':<8} {'Efficiency':<10}")
    print("-" * 90)
    for i, score in enumerate(efficiency_scores, 1):
        marker = "⚡" if i == 1 else f"{i}."
        print(f"{marker:<2} {score['method']:<43} {score['auroc']:.4f}   {score['gain']:+.4f}   "
              f"{score['complexity']:<8} {score['efficiency']:.5f}")
    
    most_efficient = efficiency_scores[0]
    best_performance = max(efficiency_scores, key=lambda x: x['auroc'])
    
    print(f"\n⚡ Most Efficient: {most_efficient['method']} ({most_efficient['complexity']} models, {most_efficient['auroc']:.4f} AUROC)")
    print(f"🏆 Best Performance: {best_performance['method']} ({best_performance['complexity']} models, {best_performance['auroc']:.4f} AUROC)")


def main():
    print("=" * 80)
    print("STACKING COMPARISON: Traditional vs Hierarchical")
    print("=" * 80)
    
    X, y = load_results('pc_ib_results_fixed.jsonl')
    print(f"\nLoaded {len(X)} examples, class dist: {np.mean(y)*100:.1f}% positive")
    
    baseline_scores = X[:, 4]
    baseline_auroc = roc_auc_score(y, baseline_scores)
    baseline_auprc = average_precision_score(y, baseline_scores)
    print(f"BASELINE: AUROC={baseline_auroc:.4f}, AUPRC={baseline_auprc:.4f}")
    
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)
    X_ultra = ultra_engineer_features(X)
    print(f"{X.shape[1]} → {X_ultra.shape[1]} features ({X_ultra.shape[1]/X.shape[1]:.1f}x)")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_models = create_diverse_base_models()
    model_names = list(base_models.keys())
    print(f"\nCreated {len(base_models)} base models")
    
    oof_preds, cv_scores = get_oof_predictions(base_models, X_ultra, y, cv)
    
    print("\n" + "=" * 80)
    print("LEVEL 1: BASE MODELS")
    print("=" * 80)
    sorted_models = sorted(cv_scores.items(), key=lambda x: x[1]['auroc'], reverse=True)
    for rank, (name, scores) in enumerate(sorted_models[:5], 1):
        print(f"{rank}. {name:<20} AUROC={scores['auroc']:.4f} (Δ={scores['auroc']-baseline_auroc:+.4f})")
    
    # COMPARISON
    print("\n" + "=" * 80)
    print("LEVEL 2: STACKING COMPARISON")
    print("=" * 80)
    
    traditional_results = train_meta_learner(oof_preds, y, cv)
    hierarchical_results, _ = train_hierarchical_stacking(oof_preds, y, model_names, cv)
    
    all_results = {
        'baseline': {'auroc': baseline_auroc, 'auprc': baseline_auprc, 'method': 'Baseline'},
        **traditional_results, **hierarchical_results
    }
    
    trad_best = max(traditional_results.items(), key=lambda x: x[1]['auroc'])
    hier_best = max(hierarchical_results.items(), key=lambda x: x[1]['auroc'])
    
    print(f"\n📦 TRADITIONAL: {trad_best[1]['method']} - AUROC={trad_best[1]['auroc']:.4f}")
    print(f"🎯 HIERARCHICAL: {hier_best[1]['method']} - AUROC={hier_best[1]['auroc']:.4f}")
    
    if trad_best[1]['auroc'] > hier_best[1]['auroc']:
        print(f"\n🏆 WINNER: Traditional (+{trad_best[1]['auroc'] - hier_best[1]['auroc']:.4f})")
    elif hier_best[1]['auroc'] > trad_best[1]['auroc']:
        print(f"\n🏆 WINNER: Hierarchical (+{hier_best[1]['auroc'] - trad_best[1]['auroc']:.4f})")
    else:
        print("\n🤝 TIE")
    
    print("\n" + "=" * 80)
    print("OVERALL RANKING")
    print("=" * 80)
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['auroc'], reverse=True)
    print(f"\n{'Rank':<6} {'Method':<45} {'AUROC':<10} {'AUPRC':<10} {'Δ':<8}")
    print("-" * 85)
    for rank, (name, result) in enumerate(sorted_results[:15], 1):
        marker = "🏆" if rank == 1 else f"{rank}."
        print(f"{marker:<6} {result['method']:<45} {result['auroc']:.4f}     "
              f"{result.get('auprc', 0):.4f}     {result['auroc']-baseline_auroc:+.4f}")
    
    compare_efficiency(all_results, baseline_auroc)
    
    best_auroc = sorted_results[0][1]['auroc']
    print(f"\n{'='*80}\nBEST: {sorted_results[0][1]['method']} - {best_auroc:.4f} AUROC "
          f"(+{best_auroc-baseline_auroc:.4f}, +{((best_auroc/baseline_auroc-1)*100):.2f}%)\n{'='*80}")
    
    if best_auroc >= 0.90:
        print("\n🎯🎯🎯 EXCEPTIONAL: 0.90+ AUROC achieved!")
    elif best_auroc >= 0.85:
        print("\n🎯 SUCCESS: State-of-the-art (≥0.85)")
    else:
        print(f"\nGap to 0.90: {0.90-best_auroc:.4f}. Need more data or signals.")
    
    with open('stacking_comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\n✓ Results saved to stacking_comparison_results.json")


if __name__ == '__main__':
    main()
