"""Mathematical utilities for PC+IB detection."""

import math
from typing import Dict

LABELS = ["ENTAIL", "CONTRADICT", "UNKNOWN"]


def _clamp(x: float, lo: float = 1e-12, hi: float = 1.0) -> float:
    """Clamp value to range [lo, hi]."""
    return max(lo, min(hi, x))


def normalize_dist(d: Dict[str, float]) -> Dict[str, float]:
    """Normalize a distribution over LABELS to sum to 1."""
    out = {k: float(d.get(k, 0.0) or 0.0) for k in LABELS}
    s = sum(out.values())
    if s <= 0:
        return {k: 1.0 / len(LABELS) for k in LABELS}
    return {k: out[k] / s for k in LABELS}


def kl_cat(p: Dict[str, float], q: Dict[str, float]) -> float:
    """
    KL divergence KL(p || q) for categorical distributions.
    
    Args:
        p: Target distribution
        q: Reference distribution
        
    Returns:
        KL divergence in nats
    """
    eps = 1e-12
    out = 0.0
    for k in LABELS:
        pk = _clamp(p.get(k, 0.0), eps, 1.0)
        qk = _clamp(q.get(k, 0.0), eps, 1.0)
        out += pk * math.log(pk / qk)
    return out


def js_cat(p: Dict[str, float], q: Dict[str, float]) -> float:
    """
    Jensen-Shannon divergence (symmetric, bounded [0, ln(2)]).
    
    Args:
        p: First distribution
        q: Second distribution
        
    Returns:
        JS divergence in nats
    """
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in LABELS}
    return 0.5 * kl_cat(p, m) + 0.5 * kl_cat(q, m)


def entropy(p: Dict[str, float]) -> float:
    """
    Shannon entropy of categorical distribution.
    
    Args:
        p: Probability distribution
        
    Returns:
        Entropy in nats
    """
    eps = 1e-12
    h = 0.0
    for k in LABELS:
        pk = _clamp(p.get(k, 0.0), eps, 1.0)
        h -= pk * math.log(pk)
    return h


def sigmoid(x: float) -> float:
    """Sigmoid function with numerical stability."""
    if x < -20:
        return 0.0
    if x > 20:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float, eps: float = 1e-6) -> float:
    """
    Convert probability to log-odds.
    Amplifies differences at extremes (0.9 vs 0.95 becomes much larger difference).
    
    Args:
        p: Probability in [0, 1]
        eps: Small value for numerical stability
        
    Returns:
        Log-odds (unbounded)
    """
    p = _clamp(p, eps, 1.0 - eps)
    return math.log(p / (1.0 - p))
