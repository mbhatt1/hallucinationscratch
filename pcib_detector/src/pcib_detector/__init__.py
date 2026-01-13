"""
PCIB Detector: Predictive-Coding + Information-Bottleneck Hallucination Detection

Detects hallucinations by testing whether models actually use evidence through:
- Evidence uptake (predictive coding): Did evidence update beliefs?
- Bottleneck stress (information theory): Is judgment stable under noise?
- Conflict sensitivity: Does the model resist contradictions?
"""

from .core import PCIBDetector
from .eval import evaluate_dataset
from .types import Config, DetectionResult, ClaimResult
from .math_utils import kl_cat, js_cat, logit, sigmoid

__version__ = "0.1.0"

__all__ = [
    "PCIBDetector",
    "Config",
    "DetectionResult",
    "ClaimResult",
    "evaluate_dataset",
    "kl_cat",
    "js_cat",
    "logit",
    "sigmoid",
]
