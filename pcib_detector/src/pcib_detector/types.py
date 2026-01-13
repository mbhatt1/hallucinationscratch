"""Type definitions for PCIB detector."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class Config:
    """Configuration for PCIB detector."""
    
    provider: str = "openai"
    """LLM provider: 'openai', 'anthropic', or 'gemini'"""
    
    model: Optional[str] = None
    """Model to use (if None, uses provider default)"""
    
    temperature: float = 0.0
    """Sampling temperature"""
    
    max_claims: int = 4
    """Maximum number of claims to extract from answer"""
    
    max_chunk_chars: int = 1200
    """Maximum characters per evidence chunk"""
    
    max_chunks: int = 6
    """Maximum number of evidence chunks"""
    
    distractor_chars: int = 1500
    """Size of distractor text for stress testing"""
    
    n_ensemble: int = 1
    """Number of verification samples to average (1=no ensemble, 3-5 recommended)"""
    
    ensemble_temperature: float = 0.7
    """Temperature for ensemble diversity"""
    
    api_key: Optional[str] = None
    """API key for the provider (if None, uses env var)"""
    
    max_concurrent: int = 10
    """Maximum concurrent API requests"""
    
    # Trace validation settings
    enable_trace_validation: bool = False
    """Enable reasoning trace validation and rationalization detection"""
    
    detect_rationalization: bool = True
    """Generate backward traces to detect post-hoc rationalization"""
    
    trace_temperature: float = 0.3
    """Temperature for trace generation (higher = more diverse)"""


@dataclass
class Belief:
    """Belief distribution over ENTAIL/CONTRADICT/UNKNOWN."""
    
    dist: Dict[str, float]
    """Full distribution"""
    
    entail: float
    """P(ENTAIL)"""
    
    contradict: float
    """P(CONTRADICT)"""
    
    unknown: float
    """P(UNKNOWN)"""


@dataclass
class ClaimSignals:
    """Raw signals for a single claim."""
    
    prior: Belief
    """Prior belief (uniform baseline)"""
    
    post: Belief
    """Posterior belief given evidence"""
    
    uptake_kl: float
    """Evidence uptake: KL(post || prior)"""
    
    stress_js: float
    """Bottleneck stress: JS(post, post+distractor)"""
    
    conflict_js: float
    """Conflict sensitivity: JS(post, post+conflict)"""
    
    post_entropy: float
    """Entropy of posterior"""
    
    # Trace validation signals (optional, only if trace validation enabled)
    trace_consistency: Optional[float] = None
    """Logical consistency of reasoning trace (0-1)"""
    
    trace_support: Optional[float] = None
    """How well trace supports conclusion (0-1)"""
    
    rationalization_score: Optional[float] = None
    """Post-hoc rationalization likelihood (0-1, higher=more likely)"""
    
    trace_length: Optional[int] = None
    """Word count of reasoning trace"""
    
    def get_features(self) -> np.ndarray:
        """Extract feature vector for calibration/scoring."""
        import math
        from .math_utils import logit, _clamp
        
        eps = 1e-6
        
        # Base posterior (log-odds)
        contradict_logodds = logit(max(eps, min(1-eps, self.post.contradict)))
        entail_logodds = logit(max(eps, min(1-eps, self.post.entail)))
        unknown_logodds = logit(max(eps, min(1-eps, self.post.unknown)))
        
        # PC/IB signals (log-transformed)
        log_uptake = math.log1p(self.uptake_kl)
        log_stress = math.log1p(self.stress_js)
        log_conflict = math.log1p(self.conflict_js)
        
        # Bayes factors (clipped)
        bf_contradict = np.clip(self.bayes_factor_contradict(), -10, 10)
        bf_entail = np.clip(self.bayes_factor_entail(), -10, 10)
        
        # Interaction terms
        contradict_x_low_uptake = self.post.contradict * math.exp(-self.uptake_kl)
        entail_x_low_conflict = self.post.entail * math.exp(-self.conflict_js * 2)
        low_entail_x_low_uptake = (1 - self.post.entail) * math.exp(-self.uptake_kl)
        stress_over_conflict = self.stress_js / (self.conflict_js + 0.01)
        bf_contradict_x_low_uptake = max(0, bf_contradict) * math.exp(-self.uptake_kl)
        
        base_features = [
            contradict_logodds,
            entail_logodds,
            unknown_logodds,
            log_uptake,
            log_stress,
            log_conflict,
            bf_contradict,
            bf_entail,
            contradict_x_low_uptake,
            entail_x_low_conflict,
            low_entail_x_low_uptake,
            stress_over_conflict,
            bf_contradict_x_low_uptake,
        ]
        
        # Add trace validation features if available
        if self.trace_consistency is not None:
            # Trace features
            trace_inconsistency = 1.0 - self.trace_consistency
            trace_lack_support = 1.0 - (self.trace_support or 0.0)
            rationalization = self.rationalization_score or 0.0
            
            # Interactions with base signals
            entail_x_low_consistency = self.post.entail * trace_inconsistency
            entail_x_rationalization = self.post.entail * rationalization
            low_support_x_high_entail = trace_lack_support * self.post.entail
            
            trace_features = [
                trace_inconsistency,
                trace_lack_support,
                rationalization,
                entail_x_low_consistency,
                entail_x_rationalization,
                low_support_x_high_entail,
            ]
            
            return np.array(base_features + trace_features, dtype=np.float32)
        
        return np.array(base_features, dtype=np.float32)
    
    def bayes_factor_contradict(self) -> float:
        """Log Bayes factor for CONTRADICT."""
        import math
        prior_odds = self.prior.contradict / (1.0 - self.prior.contradict + 1e-9)
        post_odds = self.post.contradict / (1.0 - self.post.contradict + 1e-9)
        return math.log(post_odds / (prior_odds + 1e-9) + 1e-9)
    
    def bayes_factor_entail(self) -> float:
        """Log Bayes factor for ENTAIL."""
        import math
        prior_odds = self.prior.entail / (1.0 - self.prior.entail + 1e-9)
        post_odds = self.post.entail / (1.0 - self.post.entail + 1e-9)
        return math.log(post_odds / (prior_odds + 1e-9) + 1e-9)


@dataclass
class ClaimResult:
    """Detection result for a single claim."""
    
    text: str
    """The claim text"""
    
    score: float
    """Hallucination score (0-1, higher=more likely hallucination)"""
    
    signals: ClaimSignals
    """Raw PC+IB signals"""
    
    flagged: bool = False
    """Whether this claim is flagged as hallucination"""


@dataclass
class DetectionResult:
    """Detection result for an answer."""
    
    flagged: bool
    """Whether the answer contains hallucinations"""
    
    score: float
    """Overall hallucination score (0-1)"""
    
    claims: List[ClaimResult]
    """Per-claim results"""
    
    signals: Optional[Dict[str, float]] = None
    """Aggregate signals (optional)"""
    
    answer: Optional[str] = None
    """Original answer text"""
    
    evidence: Optional[str] = None
    """Evidence text used"""
