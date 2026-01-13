"""
Learn optimal signal weights from labeled data instead of using manual weights.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import json
from typing import Dict, List, Tuple, Optional

class SignalWeightLearner:
    """Learn optimal weights for PCIB signals."""
    
    def __init__(self):
        self.model = LogisticRegression(
            penalty='l2',
            C=1.0,
            class_weight='balanced',  # Handle imbalanced data
            max_iter=1000
        )
        self.weights = None
        self.intercept = None
    
    def extract_features(self, result: Dict) -> np.ndarray:
        """
        Extract [uptake, stress, conflict, rationalization] features.
        
        Args:
            result: Detection result with signal values
            
        Returns:
            Feature vector [U, S, C, R]
        """
        return np.array([
            result.get('uptake', 0.0),
            result.get('stress', 0.0),
            result.get('conflict', 0.0),
            result.get('rationalization', 0.0)
        ])
    
    def train_from_json(self, json_path: str) -> Dict[str, float]:
        """
        Train on ablation study results.
        
        Args:
            json_path: Path to raw_data JSON file
            
        Returns:
            Learned weights as dict
        """
        with open(json_path) as f:
            data = json.load(f)
        
        # Extract features and labels
        X, y = [], []
        for config in data.get('all', {}).values():
            for example in config.get('examples', []):
                features = self.extract_features(example)
                label = example.get('label', 0)
                X.append(features)
                y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        if len(X) == 0:
            raise ValueError("No training examples found in JSON file")
        
        # Train with cross-validation
        try:
            scores = cross_val_score(self.model, X, y, cv=5, scoring='roc_auc')
            print(f"Cross-validation AUROC: {scores.mean():.3f} ± {scores.std():.3f}")
        except Exception as e:
            print(f"Cross-validation failed: {e}")
        
        # Train on full data
        self.model.fit(X, y)
        
        # Extract learned weights
        self.weights = self.model.coef_[0]
        self.intercept = self.model.intercept_[0]
        
        weight_dict = {
            'uptake': float(self.weights[0]),
            'stress': float(self.weights[1]),
            'conflict': float(self.weights[2]),
            'rationalization': float(self.weights[3]),
            'intercept': float(self.intercept)
        }
        
        print("Learned weights:", weight_dict)
        return weight_dict
    
    def compute_score(self, signals: Dict[str, float]) -> float:
        """Compute score using learned weights."""
        if self.weights is None:
            raise ValueError("Must train model first")
        
        features = np.array([
            signals.get('uptake', 0.0),
            signals.get('stress', 0.0),
            signals.get('conflict', 0.0),
            signals.get('rationalization', 0.0)
        ])
        
        return float(np.dot(self.weights, features) + self.intercept)
