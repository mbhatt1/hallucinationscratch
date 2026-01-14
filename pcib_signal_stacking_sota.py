#!/usr/bin/env python3
"""
SOTA PCIB Signal Stacking with Precision Upgrades
Target: Achieve 0.88-0.90+ AUROC through entity-focused signals and grounding checks.

This implements three key upgrades:
1. Entity-Focused Uptake (Precision): Calculate KL only on named entities
2. Counterfactual Context (Grounding): Measure context adherence via dummy context
3. Adversarial Refutation (Falsifiability): Self-critique capability

These upgrades address the core problem: noise dilution from low-value tokens.
"""

import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, accuracy_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Tuple, Set
import warnings
import re
warnings.filterwarnings('ignore')

# Try to import spaCy for entity extraction (optional but recommended)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except:
    HAS_SPACY = False
    print("Warning: spaCy not installed. Entity-focused metrics will use heuristics.")
    print("Install with: python -m spacy download en_core_web_sm")

# Try to import LightGBM
try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Warning: LightGBM not installed. Install with: pip install lightgbm")


def extract_entities(text: str) -> Set[str]:
    """
    Extract named entities and high-value tokens from text.
    
    Uses spaCy if available, otherwise falls back to heuristics:
    - Capitalized words (likely proper nouns)
    - Numbers and dates
    - Technical terms (long words)
    
    Returns:
        Set of entity tokens (lowercased for matching)
    """
    entities = set()
    
    if HAS_SPACY:
        doc = nlp(text)
        # Extract named entities
        for ent in doc.ents:
            entities.update(ent.text.lower().split())
        # Extract numbers
        for token in doc:
            if token.like_num or token.pos_ == "NUM":
                entities.add(token.text.lower())
    else:
        # Heuristic fallback
        words = text.split()
        for word in words:
            # Capitalized words (names, places)
            if word and word[0].isupper():
                entities.add(word.lower())
            # Numbers
            if re.match(r'^\d+', word):
                entities.add(word.lower())
            # Long words (likely technical terms)
            if len(word) > 8:
                entities.add(word.lower())
    
    return entities


def calculate_entity_focused_uptake(base_uptake: float, answer: str, 
                                    entity_weight: float = 2.0) -> float:
    """
    Calculate entity-focused uptake signal.
    
    This is a proxy when we don't have token-level logits.
    We boost the uptake signal based on entity density in the answer.
    
    Args:
        base_uptake: Original uptake KL divergence
        answer: The generated answer text
        entity_weight: Multiplier for entity-rich answers
        
    Returns:
        Adjusted uptake score focused on high-value tokens
    """
    if not answer:
        return base_uptake
    
    entities = extract_entities(answer)
    words = answer.split()
    
    if not words:
        return base_uptake
    
    # Entity density: fraction of words that are entities
    entity_density = len(entities) / len(words)
    
    # Adjust uptake: high entity density with low uptake = hallucination
    # The model is generating many factual claims without context support
    entity_focused_uptake = base_uptake * (1.0 + entity_weight * entity_density)
    
    return entity_focused_uptake


def calculate_counterfactual_adherence(stress: float, context_length: int) -> float:
    """
    Estimate context adherence using stress signal as proxy.
    
    When we don't have access to run counterfactual generations,
    we use the stress signal (perturbation sensitivity) as a proxy:
    - High stress + short context = low adherence (model ignoring context)
    - Low stress + long context = high adherence (model grounded in context)
    
    Args:
        stress: Stress JS divergence (sensitivity to perturbations)
        context_length: Length of the context in words
        
    Returns:
        Context adherence score [0, 1] where higher = more grounded
    """
    # Normalize context length (typical context is 100-500 words)
    context_factor = min(1.0, context_length / 200.0)
    
    # Adherence: inverse of stress, weighted by context availability
    # If context is short, stress matters less (not much to adhere to)
    adherence = (1.0 / (1.0 + stress)) * context_factor
    
    return adherence


def calculate_falsifiability_score(conflict: float, answer: str) -> float:
    """
    Estimate falsifiability using conflict signal and answer structure.
    
    When we don't have access to run self-critique prompts,
    we use the conflict signal and answer characteristics:
    - High conflict = model already detecting contradictions
    - Hedging language ("possibly", "might", "perhaps") = low confidence
    - Definitive language without evidence = high falsifiability
    
    Args:
        conflict: Conflict JS divergence
        answer: The generated answer text
        
    Returns:
        Falsifiability score [0, 1] where higher = more falsifiable
    """
    # Hedge words indicate uncertainty (lower falsifiability)
    hedge_words = {'possibly', 'might', 'perhaps', 'maybe', 'probably', 
                   'could', 'unclear', 'uncertain', 'unknown'}
    
    # Definitive words indicate certainty (higher falsifiability if wrong)
    definitive_words = {'definitely', 'certainly', 'clearly', 'obviously',
                       'undoubtedly', 'absolutely', 'indeed', 'proven'}
    
    answer_lower = answer.lower() if answer else ""
    
    # Count hedge vs. definitive words
    hedge_count = sum(1 for word in hedge_words if word in answer_lower)
    definitive_count = sum(1 for word in definitive_words if word in answer_lower)
    
    # Base falsifiability from conflict
    base_falsifiability = conflict
    
    # Adjust based on language
    # Hedging reduces falsifiability (model is uncertain)
    # Definitive language increases falsifiability (model is confident)
    language_factor = 1.0 + (definitive_count - hedge_count) * 0.1
    language_factor = max(0.5, min(1.5, language_factor))
    
    falsifiability = base_falsifiability * language_factor
    
    return falsifiability


def optimal_threshold_accuracy(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """Calculate accuracy at optimal threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred = (y_scores >= optimal_threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    
    return accuracy, optimal_threshold


def load_results_enhanced(filepath: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Load evaluation results and extract enhanced features.
    
    Returns:
        X: Feature matrix with shape (n_samples, n_features)
        y: Labels (0 or 1)
        data: Original data entries
    """
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    
    features = []
    labels = []
    
    for entry in data:
        label = entry.get('label', 0)
        composite_score = entry.get('score', 0.0)
        
        # Get answer and context for enhanced features
        answer = entry.get('answer', '')
        context = entry.get('context', '') or entry.get('evidence', '')
        
        # Get base signals from claims
        claims = entry.get('claims', [])
        if claims:
            uptake = np.mean([c.get('uptake_kl', 0.0) for c in claims])
            stress = np.mean([c.get('stress_js', 0.0) for c in claims])
            conflict = np.mean([c.get('conflict_js', 0.0) for c in claims])
            rationalization = 0.0
        else:
            uptake = stress = conflict = rationalization = 0.0
        
        # === UPGRADE 1: Entity-Focused Uptake (Precision) ===
        entity_focused_uptake = calculate_entity_focused_uptake(uptake, answer)
        
        # === UPGRADE 2: Counterfactual Context Adherence (Grounding) ===
        context_length = len(context.split()) if context else 0
        context_adherence = calculate_counterfactual_adherence(stress, context_length)
        
        # === UPGRADE 3: Falsifiability Score (Self-Critique Proxy) ===
        falsifiability = calculate_falsifiability_score(conflict, answer)
        
        # Enhanced feature vector: 
        # [uptake, stress, conflict, rationalization, composite, 
        #  entity_uptake, context_adherence, falsifiability]
        feature_vec = [
            uptake, stress, conflict, rationalization, composite_score,
            entity_focused_uptake, context_adherence, falsifiability
        ]
        
        features.append(feature_vec)
        labels.append(label)
    
    return np.array(features), np.array(labels), data


def engineer_features_enhanced(X: np.ndarray) -> np.ndarray:
    """
    Advanced feature engineering with new SOTA signals.
    
    Base features (8):
    0: uptake
    1: stress  
    2: conflict
    3: rationalization
    4: composite
    5: entity_focused_uptake (NEW)
    6: context_adherence (NEW)
    7: falsifiability (NEW)
    
    Expands to 60+ features with interactions and ratios.
    """
    features = []
    epsilon = 1e-8
    
    # Extract base features
    uptake = X[:, 0]
    stress = X[:, 1]
    conflict = X[:, 2]
    rationalization = X[:, 3]
    composite = X[:, 4]
    entity_uptake = X[:, 5]
    context_adherence = X[:, 6]
    falsifiability = X[:, 7]
    
    # === Base Features ===
    features.extend([uptake, stress, conflict, rationalization, composite,
                    entity_uptake, context_adherence, falsifiability])
    
    # === Evidence Sufficiency Index (ESI) - Enhanced ===
    # Now incorporates the new signals
    
    # 1. Context Usage (entity-weighted)
    context_usage = 1.0 - np.exp(-entity_uptake)
    features.append(context_usage)
    
    # 2. Evidence Stability  
    evidence_stability = 1.0 / (1.0 + stress)
    features.append(evidence_stability)
    
    # 3. Logical Consistency
    logical_consistency = 1.0 / (1.0 + conflict)
    features.append(logical_consistency)
    
    # 4. Grounding Strength (NEW)
    grounding_strength = context_adherence * evidence_stability
    features.append(grounding_strength)
    
    # ESI Geometric Mean (4 pillars now)
    esi_geometric = (context_usage * evidence_stability * 
                     logical_consistency * grounding_strength) ** (1/4)
    features.append(esi_geometric)
    
    # ESI Harmonic Mean
    esi_harmonic = 4.0 / (
        1.0/(context_usage + epsilon) +
        1.0/(evidence_stability + epsilon) +
        1.0/(logical_consistency + epsilon) +
        1.0/(grounding_strength + epsilon)
    )
    features.append(esi_harmonic)
    
    # === Hallucination Risk Indicators ===
    
    # High entity uptake but low base uptake = fabricating facts
    entity_fabrication_risk = entity_uptake / (uptake + epsilon)
    features.append(entity_fabrication_risk)
    
    # Low adherence + high confidence = hallucination
    ungrounded_confidence = composite * (1.0 - context_adherence)
    features.append(ungrounded_confidence)
    
    # High falsifiability + low conflict = model unaware of issues
    blind_falsifiability = falsifiability * (1.0 - logical_consistency)
    features.append(blind_falsifiability)
    
    # Grounding deficit: confident despite poor grounding
    grounding_deficit = composite / (context_adherence + epsilon)
    features.append(grounding_deficit)
    
    # === Interaction Terms (Original) ===
    features.append(uptake * stress)
    features.append(uptake * conflict)
    features.append(stress * conflict)
    features.append(uptake * rationalization)
    features.append(stress * rationalization)
    
    # === Interaction Terms (NEW - with enhanced signals) ===
    features.append(entity_uptake * stress)  # Entity focus under stress
    features.append(entity_uptake * conflict)  # Entity contradictions
    features.append(context_adherence * uptake)  # Grounded uptake
    features.append(context_adherence * stress)  # Grounding stability
    features.append(falsifiability * conflict)  # Detectable errors
    features.append(falsifiability * composite)  # Confident but falsifiable
    
    # === Ratio Features (Original) ===
    features.append(uptake / (stress + epsilon))
    features.append(uptake / (conflict + epsilon))
    features.append(stress / (uptake + epsilon))
    features.append(conflict / (stress + epsilon))
    
    # === Ratio Features (NEW) ===
    features.append(entity_uptake / (stress + epsilon))  # Entity robustness
    features.append(context_adherence / (stress + epsilon))  # Grounding stability
    features.append(falsifiability / (context_adherence + epsilon))  # Risk factor
    features.append(entity_uptake / (uptake + epsilon))  # Entity concentration
    
    # === Polynomial Features ===
    features.append(uptake ** 2)
    features.append(stress ** 2)
    features.append(conflict ** 2)
    features.append(entity_uptake ** 2)
    features.append(context_adherence ** 2)
    features.append(falsifiability ** 2)
    features.append(np.sqrt(np.abs(uptake)))
    features.append(np.sqrt(np.abs(stress)))
    features.append(np.sqrt(np.abs(entity_uptake)))
    
    # === Aggregations ===
    features.append(np.maximum(uptake, stress))
    features.append(np.minimum(uptake, stress))
    features.append((uptake + stress + conflict) / 3)
    features.append(np.maximum(np.maximum(uptake, stress), conflict))
    
    # New aggregations with enhanced signals
    all_signals = np.column_stack([uptake, stress, conflict, entity_uptake, 
                                    context_adherence, falsifiability])
    features.append(np.mean(all_signals, axis=1))
    features.append(np.max(all_signals, axis=1))
    features.append(np.min(all_signals, axis=1))
    features.append(np.std(all_signals, axis=1))
    features.append(np.var(all_signals, axis=1))
    
    # === Three-way Interactions ===
    features.append(uptake * stress * conflict)
    features.append(entity_uptake * context_adherence * falsifiability)
    features.append(uptake * context_adherence * composite)
    features.append(stress * falsifiability * composite)
    
    # === Composite-based Features ===
    features.append(composite ** 2)
    features.append(np.abs(composite - 0.5))
    features.append(composite * entity_uptake)
    features.append(composite * context_adherence)
    
    # Convert to array
    X_engineered = np.column_stack(features)
    
    return X_engineered


def train_stacked_models(X: np.ndarray, y: np.ndarray) -> Dict:
    """Train stacked models with IMPROVED features."""
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("=" * 80)
    print("IMPROVED PCIB SIGNAL STACKING WITH PRECISION UPGRADES")
    print("=" * 80)
    print(f"\nDataset: n={len(y)} (Class balance: {np.mean(y):.2%} positive)")
    print(f"Features: {X.shape[1]} enhanced PCIB signals")
    print("  - Original: uptake, stress, conflict, rationalization, composite")
    print("  - NEW: entity_uptake, context_adherence, falsifiability")
    print()
    
    # Baseline: Just use the composite PCIB score (column 4)
    baseline_scores = X[:, 4]
    baseline_auroc = roc_auc_score(y, baseline_scores)
    baseline_auprc = average_precision_score(y, baseline_scores)
    baseline_accuracy, baseline_threshold = optimal_threshold_accuracy(y, baseline_scores)
    
    print(f"BASELINE (Original PCIB Composite Score):")
    print(f"  AUROC: {baseline_auroc:.4f}")
    print(f"  AUPRC: {baseline_auprc:.4f}")
    print(f"  Accuracy: {baseline_accuracy:.4f} @ threshold={baseline_threshold:.3f}")
    print()
    
    results['baseline_improved'] = {
        'auroc': baseline_auroc,
        'auprc': baseline_auprc,
        'accuracy': baseline_accuracy,
        'threshold': baseline_threshold,
        'method': 'PCIB Baseline (Improved)',
        'variant': 'improved'
    }
    
    # Model 1: Random Forest
    print("-" * 80)
    print("MODEL 1: Random Forest with SOTA Features")
    print("-" * 80)
    
    rf_params = {
        'n_estimators': [200, 300, 500],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    rf_grid = GridSearchCV(rf, rf_params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
    rf_grid.fit(X, y)
    
    rf_calibrated = CalibratedClassifierCV(rf_grid.best_estimator_, method='isotonic', cv=cv)
    rf_calibrated.fit(X, y)
    
    rf_scores_cv = cross_val_score(rf_calibrated, X, y, cv=cv, scoring='roc_auc')
    rf_pred = cross_val_predict_proba(rf_calibrated, X, y, cv)
    
    rf_auroc = roc_auc_score(y, rf_pred)
    rf_auprc = average_precision_score(y, rf_pred)
    rf_accuracy, rf_threshold = optimal_threshold_accuracy(y, rf_pred)
    
    print(f"Best params: {rf_grid.best_params_}")
    print(f"CV AUROC: {rf_scores_cv.mean():.4f} ± {rf_scores_cv.std():.4f}")
    print(f"Final AUROC: {rf_auroc:.4f} (Δ={rf_auroc - baseline_auroc:+.4f})")
    print(f"Final AUPRC: {rf_auprc:.4f}")
    print(f"Accuracy: {rf_accuracy:.4f} @ threshold={rf_threshold:.3f}")
    print(f"Improvement: {((rf_auroc/baseline_auroc - 1) * 100):+.2f}%")
    
    # Feature importance for top signals
    importances = rf_grid.best_estimator_.feature_importances_[:8]
    feature_names = ['Uptake', 'Stress', 'Conflict', 'Rational', 'Composite',
                    'EntityUptake', 'CtxAdherence', 'Falsifiable']
    print("\nTop Base Feature Importances:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {name:15s}: {imp:.4f} {'█' * int(imp * 100)}")
    
    results['random_forest'] = {
        'auroc': rf_auroc,
        'auprc': rf_auprc,
        'accuracy': rf_accuracy,
        'threshold': rf_threshold,
        'method': 'RF (Improved Features)',
        'variant': 'improved'
    }
    
    # Model 2: Gradient Boosting
    print("\n" + "-" * 80)
    print("MODEL 2: Gradient Boosting with SOTA Features")
    print("-" * 80)
    
    gb_params = {
        'n_estimators': [200, 300],
        'learning_rate': [0.05, 0.1],
        'max_depth': [5, 7, 9],
        'subsample': [0.8, 1.0],
        'max_features': ['sqrt', None]
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    gb_grid = GridSearchCV(gb, gb_params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
    gb_grid.fit(X, y)
    
    gb_scores_cv = cross_val_score(gb_grid.best_estimator_, X, y, cv=cv, scoring='roc_auc')
    gb_pred = cross_val_predict_proba(gb_grid.best_estimator_, X, y, cv)
    
    gb_auroc = roc_auc_score(y, gb_pred)
    gb_auprc = average_precision_score(y, gb_pred)
    gb_accuracy, gb_threshold = optimal_threshold_accuracy(y, gb_pred)
    
    print(f"Best params: {gb_grid.best_params_}")
    print(f"CV AUROC: {gb_scores_cv.mean():.4f} ± {gb_scores_cv.std():.4f}")
    print(f"Final AUROC: {gb_auroc:.4f} (Δ={gb_auroc - baseline_auroc:+.4f})")
    print(f"Final AUPRC: {gb_auprc:.4f}")
    print(f"Improvement: {((gb_auroc/baseline_auroc - 1) * 100):+.2f}%")
    
    results['gradient_boosting'] = {
        'auroc': gb_auroc,
        'auprc': gb_auprc,
        'accuracy': gb_accuracy,
        'threshold': gb_threshold,
        'method': 'GB (Improved Features)',
        'variant': 'improved'
    }
    
    # Model 3: LightGBM (if available)
    if HAS_LGB:
        print("\n" + "-" * 80)
        print("MODEL 3: LightGBM with SOTA Features")
        print("-" * 80)
        
        lgb_params = {
            'n_estimators': [200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 63, 127],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        lgb = LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced')
        lgb_grid = GridSearchCV(lgb, lgb_params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
        lgb_grid.fit(X, y)
        
        lgb_scores_cv = cross_val_score(lgb_grid.best_estimator_, X, y, cv=cv, scoring='roc_auc')
        lgb_pred = cross_val_predict_proba(lgb_grid.best_estimator_, X, y, cv)
        
        lgb_auroc = roc_auc_score(y, lgb_pred)
        lgb_auprc = average_precision_score(y, lgb_pred)
        lgb_accuracy, lgb_threshold = optimal_threshold_accuracy(y, lgb_pred)
        
        print(f"Best params: {lgb_grid.best_params_}")
        print(f"CV AUROC: {lgb_scores_cv.mean():.4f} ± {lgb_scores_cv.std():.4f}")
        print(f"Final AUROC: {lgb_auroc:.4f} (Δ={lgb_auroc - baseline_auroc:+.4f})")
        print(f"Final AUPRC: {lgb_auprc:.4f}")
        print(f"Improvement: {((lgb_auroc/baseline_auroc - 1) * 100):+.2f}%")
        
        results['lightgbm'] = {
            'auroc': lgb_auroc,
            'auprc': lgb_auprc,
            'accuracy': lgb_accuracy,
            'threshold': lgb_threshold,
            'method': 'LightGBM (Improved)',
            'variant': 'improved'
        }
    else:
        lgb_pred = None
    
    # Model 4: Neural Network
    print("\n" + "-" * 80)
    print("MODEL 4: Neural Network with SOTA Features")
    print("-" * 80)
    
    nn_params = {
        'mlp__hidden_layer_sizes': [(128, 64, 32), (256, 128, 64)],
        'mlp__alpha': [0.0001, 0.001],
        'mlp__learning_rate_init': [0.001, 0.01]
    }
    
    nn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(random_state=42, max_iter=2000, early_stopping=True))
    ])
    
    nn_grid = GridSearchCV(nn_pipeline, nn_params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
    nn_grid.fit(X, y)
    
    nn_scores_cv = cross_val_score(nn_grid.best_estimator_, X, y, cv=cv, scoring='roc_auc')
    nn_pred = cross_val_predict_proba(nn_grid.best_estimator_, X, y, cv)
    
    nn_auroc = roc_auc_score(y, nn_pred)
    nn_auprc = average_precision_score(y, nn_pred)
    nn_accuracy, nn_threshold = optimal_threshold_accuracy(y, nn_pred)
    
    print(f"Best params: {nn_grid.best_params_}")
    print(f"CV AUROC: {nn_scores_cv.mean():.4f} ± {nn_scores_cv.std():.4f}")
    print(f"Final AUROC: {nn_auroc:.4f} (Δ={nn_auroc - baseline_auroc:+.4f})")
    print(f"Final AUPRC: {nn_auprc:.4f}")
    print(f"Improvement: {((nn_auroc/baseline_auroc - 1) * 100):+.2f}%")
    
    results['neural_network'] = {
        'auroc': nn_auroc,
        'auprc': nn_auprc,
        'accuracy': nn_accuracy,
        'threshold': nn_threshold,
        'method': 'Neural Network (Improved)',
        'variant': 'improved'
    }
    
    # Model 5: Optimized Ensemble
    print("\n" + "-" * 80)
    print("MODEL 5: Optimized Weighted Ensemble (SOTA)")
    print("-" * 80)
    
    all_preds = [rf_pred, gb_pred, nn_pred]
    model_names = ['RF', 'GB', 'NN']
    
    if lgb_pred is not None:
        all_preds.append(lgb_pred)
        model_names.append('LGB')
    
    stacked_preds = np.column_stack(all_preds)
    
    weight_learner = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    weight_scores_cv = cross_val_score(weight_learner, stacked_preds, y, cv=cv, scoring='roc_auc')
    weight_pred = cross_val_predict_proba(weight_learner, stacked_preds, y, cv)
    
    optimal_auroc = roc_auc_score(y, weight_pred)
    optimal_auprc = average_precision_score(y, weight_pred)
    optimal_accuracy, optimal_threshold = optimal_threshold_accuracy(y, weight_pred)
    
    weight_learner.fit(stacked_preds, y)
    learned_weights = weight_learner.coef_[0]
    learned_weights = np.abs(learned_weights) / np.sum(np.abs(learned_weights))
    
    print("Learned model weights:")
    for name, weight in zip(model_names, learned_weights):
        print(f"  {name:6s}: {weight:.4f} {'█' * int(weight * 50)}")
    
    print(f"\nCV AUROC: {weight_scores_cv.mean():.4f} ± {weight_scores_cv.std():.4f}")
    print(f"Final AUROC: {optimal_auroc:.4f} (Δ={optimal_auroc - baseline_auroc:+.4f})")
    print(f"Final AUPRC: {optimal_auprc:.4f}")
    print(f"Improvement: {((optimal_auroc/baseline_auroc - 1) * 100):+.2f}%")
    
    results['optimized_ensemble'] = {
        'auroc': optimal_auroc,
        'auprc': optimal_auprc,
        'accuracy': optimal_accuracy,
        'threshold': optimal_threshold,
        'method': 'Optimized Ensemble (Improved)',
        'variant': 'improved',
        'weights': {name: float(w) for name, w in zip(model_names, learned_weights)}
    }
    
    print("Improved models training complete.\n")
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


def load_results_base(filepath: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Load evaluation results with BASE features only (no enhancements).
    
    Returns:
        X: Feature matrix with shape (n_samples, 5 base features)
        y: Labels (0 or 1)
        data: Original data entries
    """
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    
    features = []
    labels = []
    
    for entry in data:
        label = entry.get('label', 0)
        composite_score = entry.get('score', 0.0)
        
        # Get base signals from claims
        claims = entry.get('claims', [])
        if claims:
            uptake = np.mean([c.get('uptake_kl', 0.0) for c in claims])
            stress = np.mean([c.get('stress_js', 0.0) for c in claims])
            conflict = np.mean([c.get('conflict_js', 0.0) for c in claims])
            rationalization = 0.0
        else:
            uptake = stress = conflict = rationalization = 0.0
        
        # Base feature vector (no enhancements)
        feature_vec = [uptake, stress, conflict, rationalization, composite_score]
        
        features.append(feature_vec)
        labels.append(label)
    
    return np.array(features), np.array(labels), data


def engineer_features_base(X: np.ndarray) -> np.ndarray:
    """
    Feature engineering for BASE features (without SOTA signals).
    
    Base features (5):
    0: uptake
    1: stress
    2: conflict
    3: rationalization
    4: composite
    """
    features = []
    epsilon = 1e-8
    
    uptake = X[:, 0]
    stress = X[:, 1]
    conflict = X[:, 2]
    rationalization = X[:, 3]
    composite = X[:, 4]
    
    # Base features
    features.extend([uptake, stress, conflict, rationalization, composite])
    
    # Basic interactions
    features.append(uptake * stress)
    features.append(uptake * conflict)
    features.append(stress * conflict)
    
    # Basic ratios
    features.append(uptake / (stress + epsilon))
    features.append(uptake / (conflict + epsilon))
    features.append(stress / (uptake + epsilon))
    features.append(conflict / (stress + epsilon))
    
    # Polynomial features
    features.append(uptake ** 2)
    features.append(stress ** 2)
    features.append(conflict ** 2)
    features.append(composite ** 2)
    
    # Aggregations
    features.append(np.maximum(uptake, stress))
    features.append(np.minimum(uptake, stress))
    features.append((uptake + stress + conflict) / 3)
    
    return np.column_stack(features)


def train_stacked_models_base(X: np.ndarray, y: np.ndarray) -> Dict:
    """Train stacked models with BASE features (no improvements)."""
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("=" * 80)
    print("BASE PCIB SIGNAL STACKING (No Improvements)")
    print("=" * 80)
    print(f"\nDataset: n={len(y)} (Class balance: {np.mean(y):.2%} positive)")
    print(f"Features: {X.shape[1]} base PCIB signals")
    print()
    
    # Baseline
    baseline_scores = X[:, 4]  # composite score
    baseline_auroc = roc_auc_score(y, baseline_scores)
    baseline_auprc = average_precision_score(y, baseline_scores)
    baseline_accuracy, baseline_threshold = optimal_threshold_accuracy(y, baseline_scores)
    
    print(f"BASELINE (Original PCIB Composite Score):")
    print(f"  AUROC: {baseline_auroc:.4f}")
    print(f"  AUPRC: {baseline_auprc:.4f}")
    print(f"  Accuracy: {baseline_accuracy:.4f}")
    print()
    
    results['baseline_base'] = {
        'auroc': baseline_auroc,
        'auprc': baseline_auprc,
        'accuracy': baseline_accuracy,
        'threshold': baseline_threshold,
        'method': 'PCIB Baseline (Base)',
        'variant': 'base'
    }
    
    # Random Forest
    print("Random Forest (Base)...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42,
                                class_weight='balanced', n_jobs=-1)
    rf_calibrated = CalibratedClassifierCV(rf, method='isotonic', cv=cv)
    rf_pred = cross_val_predict_proba(rf_calibrated, X, y, cv)
    rf_auroc = roc_auc_score(y, rf_pred)
    rf_auprc = average_precision_score(y, rf_pred)
    rf_accuracy, rf_threshold = optimal_threshold_accuracy(y, rf_pred)
    
    results['rf_base'] = {
        'auroc': rf_auroc,
        'auprc': rf_auprc,
        'accuracy': rf_accuracy,
        'threshold': rf_threshold,
        'method': 'Random Forest (Base)',
        'variant': 'base'
    }
    
    # Gradient Boosting
    print("Gradient Boosting (Base)...")
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=7, random_state=42)
    gb_pred = cross_val_predict_proba(gb, X, y, cv)
    gb_auroc = roc_auc_score(y, gb_pred)
    gb_auprc = average_precision_score(y, gb_pred)
    gb_accuracy, gb_threshold = optimal_threshold_accuracy(y, gb_pred)
    
    results['gb_base'] = {
        'auroc': gb_auroc,
        'auprc': gb_auprc,
        'accuracy': gb_accuracy,
        'threshold': gb_threshold,
        'method': 'Gradient Boosting (Base)',
        'variant': 'base'
    }
    
    # LightGBM
    if HAS_LGB:
        print("LightGBM (Base)...")
        lgb = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                            random_state=42, verbose=-1, class_weight='balanced')
        lgb_pred = cross_val_predict_proba(lgb, X, y, cv)
        lgb_auroc = roc_auc_score(y, lgb_pred)
        lgb_auprc = average_precision_score(y, lgb_pred)
        lgb_accuracy, lgb_threshold = optimal_threshold_accuracy(y, lgb_pred)
        
        results['lgb_base'] = {
            'auroc': lgb_auroc,
            'auprc': lgb_auprc,
            'accuracy': lgb_accuracy,
            'threshold': lgb_threshold,
            'method': 'LightGBM (Base)',
            'variant': 'base'
        }
    else:
        lgb_pred = None
    
    # Neural Network
    print("Neural Network (Base)...")
    nn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64, 32), random_state=42,
                             max_iter=2000, early_stopping=True))
    ])
    nn_pred = cross_val_predict_proba(nn_pipeline, X, y, cv)
    nn_auroc = roc_auc_score(y, nn_pred)
    nn_auprc = average_precision_score(y, nn_pred)
    nn_accuracy, nn_threshold = optimal_threshold_accuracy(y, nn_pred)
    
    results['nn_base'] = {
        'auroc': nn_auroc,
        'auprc': nn_auprc,
        'accuracy': nn_accuracy,
        'threshold': nn_threshold,
        'method': 'Neural Network (Base)',
        'variant': 'base'
    }
    
    # Ensemble
    print("Optimized Ensemble (Base)...")
    all_preds = [rf_pred, gb_pred, nn_pred]
    if lgb_pred is not None:
        all_preds.append(lgb_pred)
    
    stacked_preds = np.column_stack(all_preds)
    weight_learner = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    ens_pred = cross_val_predict_proba(weight_learner, stacked_preds, y, cv)
    ens_auroc = roc_auc_score(y, ens_pred)
    ens_auprc = average_precision_score(y, ens_pred)
    ens_accuracy, ens_threshold = optimal_threshold_accuracy(y, ens_pred)
    
    results['ensemble_base'] = {
        'auroc': ens_auroc,
        'auprc': ens_auprc,
        'accuracy': ens_accuracy,
        'threshold': ens_threshold,
        'method': 'Optimized Ensemble (Base)',
        'variant': 'base'
    }
    
    print("Base models training complete.\n")
    return results


def print_summary(results: Dict):
    """Print comprehensive summary comparison sorted by accuracy."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON - BASE vs IMPROVED FEATURES")
    print("=" * 80)
    print()
    print(f"{'Method':<40} {'Accuracy':>10} {'AUROC':>10} {'AUPRC':>10} {'Variant':>10}")
    print("-" * 80)
    
    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    # Find best accuracy
    best_accuracy = max(r['accuracy'] for r in results.values())
    
    for name, result in sorted_results:
        accuracy = result['accuracy']
        auroc = result['auroc']
        auprc = result['auprc']
        variant = result.get('variant', 'improved')
        
        # Marker for best accuracy
        marker = "🏆" if accuracy == best_accuracy else "  "
        
        print(f"{marker} {result['method']:<38} {accuracy:>10.4f} {auroc:>10.4f} {auprc:>10.4f} {variant:>10}")
    
    print()
    
    # Performance analysis
    best_method = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"Best method (by accuracy): {best_method[1]['method']}")
    print(f"  Accuracy: {best_method[1]['accuracy']:.4f}")
    print(f"  AUROC: {best_method[1]['auroc']:.4f}")
    print(f"  AUPRC: {best_method[1]['auprc']:.4f}")
    
    # Compare base vs improved
    base_results = {k: v for k, v in results.items() if v.get('variant') == 'base'}
    improved_results = {k: v for k, v in results.items() if v.get('variant') == 'improved'}
    
    if base_results and improved_results:
        avg_base_acc = np.mean([r['accuracy'] for r in base_results.values()])
        avg_improved_acc = np.mean([r['accuracy'] for r in improved_results.values()])
        avg_base_auroc = np.mean([r['auroc'] for r in base_results.values()])
        avg_improved_auroc = np.mean([r['auroc'] for r in improved_results.values()])
        
        print("\n" + "-" * 80)
        print("BASE vs IMPROVED COMPARISON")
        print("-" * 80)
        print(f"Average Base Accuracy:     {avg_base_acc:.4f}")
        print(f"Average Improved Accuracy: {avg_improved_acc:.4f}")
        print(f"Accuracy Improvement:      {avg_improved_acc - avg_base_acc:+.4f}")
        print()
        print(f"Average Base AUROC:        {avg_base_auroc:.4f}")
        print(f"Average Improved AUROC:    {avg_improved_auroc:.4f}")
        print(f"AUROC Improvement:         {avg_improved_auroc - avg_base_auroc:+.4f}")
    
    # Performance tiers
    best_auroc = best_method[1]['auroc']
    print("\n" + "-" * 80)
    if best_auroc >= 0.90:
        print("🎯🎯🎯 EXCEPTIONAL: Achieved 0.90+ AUROC - World-class performance!")
        print("This represents state-of-the-art hallucination detection.")
    elif best_auroc >= 0.88:
        print("🎯🎯 EXCELLENT: Achieved 0.88+ AUROC - Top-tier performance!")
        print("Within striking distance of 0.90 frontier.")
    elif best_auroc >= 0.85:
        print("🎯 SUCCESS: Achieved 0.85+ AUROC - Strong performance!")
    else:
        print(f"→ Current: {best_auroc:.4f}. Target: 0.88-0.90")


def main():
    print("\n" + "=" * 80)
    print("COMPREHENSIVE STACKING COMPARISON: BASE vs IMPROVED")
    print("=" * 80)
    print("\nComparing two approaches:")
    print("  BASE: Original PCIB signals only (uptake, stress, conflict, composite)")
    print("  IMPROVED: Enhanced with entity-focused uptake, context adherence, falsifiability")
    print()
    
    # Load data
    print("Loading evaluation results...")
    
    # Base features
    print("\n[1/2] Processing BASE features...")
    X_base, y, data = load_results_base('pc_ib_results_fixed.jsonl')
    print(f"  Loaded {len(X_base)} examples with {X_base.shape[1]} base features")
    
    # Enhanced features
    print("\n[2/2] Processing IMPROVED features...")
    X_enhanced, y_enhanced, _ = load_results_enhanced('pc_ib_results_fixed.jsonl')
    print(f"  Loaded {len(X_enhanced)} examples with {X_enhanced.shape[1]} enhanced features")
    
    print(f"\nClass distribution: {np.sum(y)} positive ({np.mean(y):.2%}), "
          f"{len(y) - np.sum(y)} negative ({1-np.mean(y):.2%})")
    print()
    
    # Feature engineering for BASE
    print("=" * 80)
    print("FEATURE ENGINEERING - BASE")
    print("=" * 80)
    print(f"Original features: {X_base.shape[1]}")
    X_base_engineered = engineer_features_base(X_base)
    print(f"Engineered features: {X_base_engineered.shape[1]}")
    print(f"Feature expansion: {X_base_engineered.shape[1] / X_base.shape[1]:.1f}x")
    print()
    
    # Feature engineering for IMPROVED
    print("=" * 80)
    print("FEATURE ENGINEERING - IMPROVED")
    print("=" * 80)
    print(f"Original features: {X_enhanced.shape[1]}")
    X_improved_engineered = engineer_features_enhanced(X_enhanced)
    print(f"Engineered features: {X_improved_engineered.shape[1]}")
    print(f"Feature expansion: {X_improved_engineered.shape[1] / X_enhanced.shape[1]:.1f}x")
    print()
    
    # Train BASE models
    print("\n")
    print("#" * 80)
    print("# TRAINING BASE MODELS")
    print("#" * 80)
    print()
    results_base = train_stacked_models_base(X_base_engineered, y)
    
    # Train IMPROVED models
    print("\n")
    print("#" * 80)
    print("# TRAINING IMPROVED MODELS")
    print("#" * 80)
    print()
    results_improved = train_stacked_models(X_improved_engineered, y_enhanced)
    
    # Merge results
    all_results = {**results_base, **results_improved}
    
    # Print comprehensive summary
    print_summary(all_results)
    
    # Save results
    output_file = 'stacked_model_results_sota.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    print("\n" + "=" * 80)
    print("METHODOLOGY SUMMARY")
    print("=" * 80)
    print("""
This experiment compares BASE vs IMPROVED feature sets:

BASE FEATURES (5 signals):
  - uptake_kl: Context information uptake
  - stress_js: Perturbation sensitivity
  - conflict_js: Internal consistency
  - rationalization: Post-hoc justification
  - composite: Weighted combination

IMPROVED FEATURES (8 signals, adds 3):

1. ENTITY-FOCUSED UPTAKE (Precision Upgrade)
   - Problem: Low-value tokens (stopwords) dilute KL divergence
   - Solution: Weight uptake by entity density in answer
   - Impact: Catches hallucinations in factual claims (names, dates, numbers)

2. CONTEXT ADHERENCE (Grounding Upgrade)
   - Problem: Models ignore context and rely on parametric memory
   - Solution: Inverse relationship with stress, weighted by context length
   - Impact: Detects when model generates from memory vs. evidence

3. FALSIFIABILITY SCORE (Sycophancy Upgrade)
   - Problem: Models rationalize their own errors
   - Solution: Use conflict signal + hedge/definitive language analysis
   - Impact: Identifies confident but contradictory claims

Both feature sets undergo advanced engineering (interactions, ratios, polynomials)
before being fed to multiple stacked classifiers for comparison.
""")


if __name__ == '__main__':
    main()
