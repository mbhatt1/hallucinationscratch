"""Core PC+IB hallucination detector."""

import asyncio
import re
import ast
from typing import List, Optional
import numpy as np

from .types import Config, DetectionResult, ClaimResult, ClaimSignals, Belief
from .math_utils import kl_cat, js_cat, entropy, normalize_dist, LABELS, sigmoid
from .backends import Backend, OpenAIBackend, AnthropicBackend, GeminiBackend
from .perturbations import make_distractor, make_conflict_snippet
from .trace_validation import TraceValidator, TraceValidationConfig


# Uniform prior for baseline
UNIFORM_PRIOR = {k: 1.0 / len(LABELS) for k in LABELS}


def create_backend(config: Config) -> Backend:
    """
    Factory function to create the appropriate backend.
    
    Args:
        config: Configuration with provider and API key settings
        
    Returns:
        Backend instance
        
    Raises:
        ValueError: If provider is unsupported
    """
    provider = config.provider.lower()
    
    if provider == "openai":
        backend = OpenAIBackend(api_key=config.api_key, max_concurrent=config.max_concurrent)
    elif provider == "anthropic":
        backend = AnthropicBackend(api_key=config.api_key, max_concurrent=config.max_concurrent)
    elif provider == "gemini":
        backend = GeminiBackend(api_key=config.api_key, max_concurrent=config.max_concurrent)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Choose 'openai', 'anthropic', or 'gemini'")
    
    return backend


class PCIBDetector:
    """
    Predictive-Coding + Information-Bottleneck hallucination detector.
    
    Detects hallucinations by measuring:
    1. Evidence uptake (predictive coding): KL(post || prior)
    2. Bottleneck stress (information bottleneck): JS(post, post+distractor)
    3. Conflict sensitivity: JS(post, post+conflict)
    4. Bayes factors and interaction terms
    """
    
    def __init__(self, config: Optional[Config] = None, weights: Optional[np.ndarray] = None):
        """
        Initialize detector.
        
        Args:
            config: Configuration (uses defaults if None)
            weights: Calibrated weights for scoring (uses heuristic if None)
        """
        self.config = config or Config()
        self.backend = create_backend(self.config)
        
        # Use provider default model if not specified
        if self.config.model is None:
            self.config.model = self.backend.get_default_model()
        
        self.weights = weights
        
        # Initialize trace validator if enabled
        self.trace_validator = None
        if self.config.enable_trace_validation:
            trace_config = TraceValidationConfig(
                enabled=True,
                detect_rationalization=self.config.detect_rationalization,
                trace_temperature=self.config.trace_temperature,
            )
            self.trace_validator = TraceValidator(self.backend, trace_config, self.config.model)
    
    async def detect_hallucination(
        self,
        answer: str,
        evidence: str,
        return_details: bool = False,
        threshold: float = 0.5
    ) -> DetectionResult:
        """
        Detect hallucinations in an answer given evidence.
        
        Args:
            answer: The answer text to check
            evidence: The evidence/context to verify against
            return_details: Whether to include detailed per-claim results
            threshold: Score threshold for flagging (0-1)
            
        Returns:
            DetectionResult with flagged status and scores
        """
        # Extract claims
        claims = await self._extract_claims(answer)
        
        if not claims:
            # No factual claims found - treat as low risk
            return DetectionResult(
                flagged=False,
                score=0.0,
                claims=[],
                answer=answer if return_details else None,
                evidence=evidence if return_details else None
            )
        
        # Analyze each claim
        claim_results = []
        for claim_text in claims:
            signals = await self._compute_claim_signals(claim_text, evidence)
            score = self._score_claim(signals)
            
            claim_results.append(ClaimResult(
                text=claim_text,
                score=score,
                signals=signals,
                flagged=(score > threshold)
            ))
        
        # Aggregate: weighted mean by signal strength
        weighted_scores = []
        weights_list = []
        for cr in claim_results:
            # Weight by contradict + (1 - entail)
            signal_strength = cr.signals.post.contradict + (1.0 - cr.signals.post.entail)
            weight = max(0.1, signal_strength)
            weights_list.append(weight)
            weighted_scores.append(cr.score * weight)
        
        overall_score = sum(weighted_scores) / sum(weights_list) if weights_list else 0.0
        
        # Aggregate signals
        agg_signals = None
        if return_details and claim_results:
            agg_signals = {
                "mean_contradict": np.mean([cr.signals.post.contradict for cr in claim_results]),
                "mean_entail": np.mean([cr.signals.post.entail for cr in claim_results]),
                "mean_uptake_kl": np.mean([cr.signals.uptake_kl for cr in claim_results]),
                "mean_stress_js": np.mean([cr.signals.stress_js for cr in claim_results]),
                "mean_conflict_js": np.mean([cr.signals.conflict_js for cr in claim_results]),
                "n_flagged": sum(1 for cr in claim_results if cr.flagged),
            }
        
        return DetectionResult(
            flagged=(overall_score > threshold),
            score=overall_score,
            claims=claim_results if return_details else [],
            signals=agg_signals,
            answer=answer if return_details else None,
            evidence=evidence if return_details else None
        )
    
    async def detect_batch(
        self,
        examples: List[dict],
        return_details: bool = False,
        threshold: float = 0.5
    ) -> List[DetectionResult]:
        """
        Detect hallucinations in a batch of examples.
        
        Args:
            examples: List of {answer: str, evidence: str} dicts
            return_details: Whether to include detailed results
            threshold: Score threshold for flagging
            
        Returns:
            List of DetectionResults
        """
        tasks = [
            self.detect_hallucination(
                ex["answer"],
                ex["evidence"],
                return_details=return_details,
                threshold=threshold
            )
            for ex in examples
        ]
        return await asyncio.gather(*tasks)
    
    async def _extract_claims(self, answer: str) -> List[str]:
        """Extract atomic factual claims from answer."""
        prompt = f"""Extract atomic factual claims from the ANSWER. Return them as a JSON object with a "claims" array.

Each claim should be an object with:
- "text": the claim text (string)
- "kind": type of claim ("fact", "number", "date", "relation", "definition", or "other")

Rules:
- One predicate per claim.
- Keep qualifiers (time, scope).
- Exclude opinions and recommendations.
- If none, return {{"claims": []}}

ANSWER:
{answer.strip()}

Return format: {{"claims": [{{"text": "...", "kind": "fact"}}, ...]}}
"""
        
        schema = {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "kind": {"type": "string", "enum": ["fact", "number", "date", "relation", "definition", "other"]},
                        },
                        "required": ["text", "kind"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["claims"],
            "additionalProperties": False,
        }
        
        result = await self.backend.call_json_schema(
            model=self.config.model,
            prompt=prompt,
            schema_name="claims",
            schema=schema,
            max_output_tokens=700,
            temperature=0.0
        )
        
        claims = []
        for c in result.get("claims", []):
            # Handle both dict and string formats
            if isinstance(c, dict):
                txt = (c.get("text") or "").strip()
            elif isinstance(c, str):
                txt = c.strip()
            else:
                continue
            
            if txt:
                claims.append(txt)
        
        # De-duplicate
        seen = set()
        out = []
        for c in claims:
            k = c.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(c)
        
        return out[:self.config.max_claims]
    
    async def _verify(self, claim: str, evidence: str) -> Belief:
        """
        Verify claim against evidence using CoT reasoning.
        Returns distribution over ENTAIL/CONTRADICT/UNKNOWN.
        """
        if self.config.n_ensemble == 1:
            # Single sample
            result = await self._verify_single(claim, evidence)
            return result
        
        # Ensemble: average multiple samples
        tasks = [
            self._verify_single(claim, evidence, temperature=self.config.ensemble_temperature)
            for _ in range(self.config.n_ensemble)
        ]
        beliefs = await asyncio.gather(*tasks)
        
        # Average distributions
        avg_dist = {k: 0.0 for k in LABELS}
        for b in beliefs:
            for k in LABELS:
                avg_dist[k] += b.dist[k] / self.config.n_ensemble
        
        return Belief(
            dist=avg_dist,
            entail=avg_dist["ENTAIL"],
            contradict=avg_dist["CONTRADICT"],
            unknown=avg_dist["UNKNOWN"]
        )
    
    async def _verify_single(self, claim: str, evidence: str, temperature: Optional[float] = None) -> Belief:
        """Single verification pass with CoT reasoning."""
        prompt = f"""You are a strict factual verifier. Analyze the relationship between CLAIM and EVIDENCE through careful reasoning.

STEP 1 - CLAIM INTERPRETATION:
What specific factual assertion does the claim make? Be precise about:
- The subject and predicate
- Any quantifiers (all, some, none)
- Temporal or scope qualifiers
- Implicit assumptions

STEP 2 - EVIDENCE INTERPRETATION:
What does the evidence actually state? Identify:
- Explicit facts mentioned
- What is NOT mentioned (absence of evidence)
- Temporal and scope context
- Level of specificity

STEP 3 - ALIGNMENT ANALYSIS:
Compare claim vs evidence:
- Does evidence directly support the claim? (ENTAIL signal)
- Does evidence contradict or refute the claim? (CONTRADICT signal)
- Is evidence insufficient, ambiguous, or off-topic? (UNKNOWN signal)
- Consider both explicit statements and logical implications

STEP 4 - DISTRIBUTION:
Based on your analysis, assign probabilities to:
- ENTAIL: Evidence logically implies or strongly supports the claim
- CONTRADICT: Evidence refutes, contradicts, or is inconsistent with the claim
- UNKNOWN: Evidence is insufficient, ambiguous, or doesn't address the claim

Be decisive when evidence is clear. Use UNKNOWN primarily when evidence is genuinely insufficient or ambiguous, not as a hedge.

CLAIM:
{claim}

EVIDENCE:
{evidence.strip() if evidence else "(no evidence provided)"}

Provide your structured analysis:
"""
        
        schema = {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Step-by-step analysis"},
                "claim_interpretation": {"type": "string"},
                "evidence_interpretation": {"type": "string"},
                "alignment": {"type": "string", "enum": ["aligned", "contradicts", "insufficient", "partially_aligned", "ambiguous"]},
                "dist": {
                    "type": "object",
                    "properties": {
                        "ENTAIL": {"type": "number"},
                        "CONTRADICT": {"type": "number"},
                        "UNKNOWN": {"type": "number"},
                    },
                    "required": ["ENTAIL", "CONTRADICT", "UNKNOWN"],
                    "additionalProperties": False,
                },
                "confidence": {"type": "number"},
            },
            "required": ["reasoning", "claim_interpretation", "evidence_interpretation", "alignment", "dist", "confidence"],
            "additionalProperties": False,
        }
        
        result = await self.backend.call_json_schema(
            model=self.config.model,
            prompt=prompt,
            schema_name="verify",
            schema=schema,
            max_output_tokens=1000,
            temperature=temperature if temperature is not None else self.config.temperature
        )
        
        dist = normalize_dist(result.get("dist", {}))
        return Belief(
            dist=dist,
            entail=dist["ENTAIL"],
            contradict=dist["CONTRADICT"],
            unknown=dist["UNKNOWN"]
        )
    
    async def _compute_claim_signals(self, claim: str, evidence: str) -> ClaimSignals:
        """Compute all PC+IB signals for a claim."""
        # Prior: uniform baseline
        prior = Belief(
            dist=UNIFORM_PRIOR.copy(),
            entail=UNIFORM_PRIOR["ENTAIL"],
            contradict=UNIFORM_PRIOR["CONTRADICT"],
            unknown=UNIFORM_PRIOR["UNKNOWN"]
        )
        
        if not evidence.strip():
            # No evidence: return neutral signals
            return ClaimSignals(
                prior=prior,
                post=prior,
                uptake_kl=0.0,
                stress_js=0.0,
                conflict_js=0.0,
                post_entropy=entropy(prior.dist)
            )
        
        # 1) Posterior given evidence
        post = await self._verify(claim, evidence)
        
        # 2) Evidence uptake (predictive coding)
        uptake = kl_cat(post.dist, prior.dist)
        
        # 3) Bottleneck stress (add distractor)
        distractor = make_distractor(self.config.distractor_chars)
        stressed_evidence = evidence.strip() + "\n\n" + distractor
        post_stress = await self._verify(claim, stressed_evidence)
        stress = js_cat(post.dist, post_stress.dist)
        
        # 4) Conflict sensitivity
        conflict_text = make_conflict_snippet(claim, evidence)
        conflicted_evidence = evidence.strip() + "\n\nCONFLICT:\n" + conflict_text
        post_conflict = await self._verify(claim, conflicted_evidence)
        conflict = js_cat(post.dist, post_conflict.dist)
        
        # Base signals
        signals = ClaimSignals(
            prior=prior,
            post=post,
            uptake_kl=uptake,
            stress_js=stress,
            conflict_js=conflict,
            post_entropy=entropy(post.dist)
        )
        
        # 5) Trace validation (if enabled)
        if self.trace_validator is not None:
            trace_result = await self.trace_validator.validate_claim_with_trace(claim, evidence)
            signals.trace_consistency = trace_result.trace_consistency
            signals.trace_support = trace_result.trace_support
            signals.rationalization_score = trace_result.rationalization_score
            signals.trace_length = trace_result.trace_length
        
        return signals
    
    def _score_claim(self, signals: ClaimSignals) -> float:
        """
        Score a claim using calibrated weights or heuristic.
        
        Returns score in [0, 1] where higher = more likely hallucination.
        """
        features = signals.get_features()
        
        if self.weights is not None:
            # Use calibrated logistic model
            logit_score = self.weights[0]  # bias
            for i, f in enumerate(features):
                logit_score += self.weights[i + 1] * f
            return sigmoid(logit_score)
        
        # Heuristic scoring with trace awareness
        base_score = 3.0 * signals.post.contradict + 2.0 * (1.0 - signals.post.entail)
        
        # Add trace validation penalties if available
        if signals.trace_consistency is not None:
            # Inconsistent reasoning adds risk
            trace_inconsistency = 1.0 - signals.trace_consistency
            
            # Lack of trace support adds risk
            trace_lack_support = 1.0 - (signals.trace_support or 0.0)
            
            # Rationalization adds significant risk
            rationalization = signals.rationalization_score or 0.0
            
            # Weighted trace penalty
            trace_penalty = (
                0.8 * trace_inconsistency +  # Inconsistent reasoning
                0.6 * trace_lack_support +    # Reasoning doesn't support conclusion
                1.2 * rationalization         # Post-hoc rationalization (strongest signal)
            )
            
            # Combine base score with trace signals
            # Use geometric mean-like combination to avoid over-penalizing
            base_score = base_score + 0.5 * trace_penalty
        
        return base_score
    
    async def calibrate(self, examples: List[dict]) -> np.ndarray:
        """
        Calibrate detector weights from labeled examples.
        
        Args:
            examples: List of {answer: str, evidence: str, label: int} dicts
                     where label=1 means hallucination, label=0 means grounded
        
        Returns:
            Calibrated weights array
        """
        from .calibration import fit_logistic_weights
        
        # Extract features and labels
        features_list = []
        labels_list = []
        
        for ex in examples:
            # Process each example
            claims = await self._extract_claims(ex["answer"])
            if not claims:
                continue
            
            # Get signals for each claim
            claim_signals = []
            for claim in claims:
                signals = await self._compute_claim_signals(claim, ex["evidence"])
                claim_signals.append(signals)
            
            # Take max-scoring claim (most suspicious)
            if claim_signals:
                max_sig = max(claim_signals, key=lambda s: self._score_claim(s))
                features_list.append(max_sig.get_features())
                labels_list.append(ex["label"])
        
        features_arr = np.array(features_list)
        labels_arr = np.array(labels_list)
        
        # Fit logistic regression
        weights = fit_logistic_weights(features_arr, labels_arr)
        return weights
