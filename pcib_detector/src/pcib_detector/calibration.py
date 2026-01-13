"""Calibration utilities for PCIB detector."""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple


def fit_logistic_weights(
    features: np.ndarray,
    labels: np.ndarray,
    lr: float = 0.1,
    n_iter: int = 1000
) -> np.ndarray:
    """
    Fit logistic regression weights via gradient descent.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Binary labels (n_samples,)
        lr: Learning rate
        n_iter: Number of iterations
        
    Returns:
        Weight vector [bias, w1, w2, ..., wn]
    """
    n_samples, n_features = features.shape
    weights = np.zeros(n_features + 1)  # +1 for bias
    
    # Add bias column
    X = np.column_stack([np.ones(n_samples), features])
    y = labels.astype(float)
    
    for _ in range(n_iter):
        # Logistic prediction
        logits = X @ weights
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
        
        # Gradient
        grad = X.T @ (probs - y) / n_samples
        
        # Update with L2 regularization
        weights -= lr * (grad + 0.01 * weights)
    
    return weights


class PCIBCalibrator:
    """Calibrate PCIB scores to probabilities."""
    
    def __init__(self, method: str = 'platt'):
        """
        Args:
            method: 'platt' (Platt scaling) or 'isotonic' (Isotonic regression)
        """
        self.method = method
        self.calibrator = None
    
    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """
        Fit calibration model.
        
        Args:
            scores: Raw PCIB scores
            labels: Binary labels (0/1)
        """
        scores = scores.reshape(-1, 1)
        
        if self.method == 'platt':
            # Platt scaling (logistic regression)
            self.calibrator = LogisticRegression()
            self.calibrator.fit(scores, labels)
        
        elif self.method == 'isotonic':
            # Isotonic regression (non-parametric)
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(scores.ravel(), labels)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """
        Convert raw scores to calibrated probabilities.
        
        Args:
            scores: Raw PCIB scores
            
        Returns:
            Calibrated probabilities in [0, 1]
        """
        if self.calibrator is None:
            raise ValueError("Must fit calibrator first")
        
        scores = np.array(scores).reshape(-1, 1)
        
        if self.method == 'platt':
            probs = self.calibrator.predict_proba(scores)[:, 1]
        else:
            probs = self.calibrator.predict(scores.ravel())
        
        return probs
    
    def evaluate_calibration(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute calibration curve (reliability diagram).
        
        Returns:
            (predicted_probs, true_frequencies) for each bin
        """
        probs = self.calibrate(scores)
        
        # Bin predictions
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probs, bins[:-1]) - 1
        
        bin_sums = np.bincount(bin_indices, weights=labels, minlength=n_bins)
        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        
        # Avoid division by zero
        bin_counts = np.maximum(bin_counts, 1)
        
        predicted_probs = (bins[:-1] + bins[1:]) / 2
        true_frequencies = bin_sums / bin_counts
        
        return predicted_probs, true_frequencies
