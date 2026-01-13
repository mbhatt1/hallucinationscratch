"""
trace_validation.py

Reasoning trace validation and post-hoc rationalization detection for PCIB.

Implements:
1. Chain-of-Thought (CoT) trace generation
2. Trace consistency validation
3. Post-hoc rationalization detection
4. Trace-conclusion alignment checking

Theory:
- Valid reasoning should produce consistent traces regardless of generation order
- Post-hoc rationalization shows high trace divergence when conclusion is given vs derived
- Circular reasoning shows low information gain in traces
- Disconnected reasoning shows low trace-conclusion alignment
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .backends.base import Backend


@dataclass
class ReasoningTrace:
    """A reasoning trace for a claim verification."""
    
    claim: str
    evidence: str
    
    # Forward trace (claim -> reasoning -> judgment)
    forward_trace: str
    forward_judgment: Dict[str, float]
    forward_confidence: float
    
    # Backward trace (judgment given -> reasoning)
    backward_trace: Optional[str] = None
    backward_judgment: Optional[Dict[str, float]] = None
    backward_confidence: Optional[float] = None
    
    # Metrics
    trace_consistency: float = 0.0  # How consistent is the reasoning?
    trace_support: float = 0.0  # How well does trace support conclusion?
    rationalization_score: float = 0.0  # Likelihood of post-hoc rationalization
    trace_length: int = 0  # Word count of trace
    

@dataclass
class TraceValidationConfig:
    """Configuration for trace validation."""
    
    # Whether to enable trace validation
    enabled: bool = True
    
    # Whether to generate backward traces for rationalization detection
    detect_rationalization: bool = True
    
    # Temperature for trace generation (higher = more diverse)
    trace_temperature: float = 0.3
    
    # Number of trace samples for consistency checking
    n_samples: int = 2
    
    # Max tokens for trace generation
    max_trace_tokens: int = 400
    
    # Threshold for detecting rationalization (JS divergence)
    rationalization_threshold: float = 0.15


FORWARD_TRACE_PROMPT = """Verify this claim against the evidence using step-by-step reasoning.

CLAIM:
{claim}

EVIDENCE:
{evidence}

Provide your reasoning step-by-step, then conclude with one of: ENTAIL, CONTRADICT, or UNKNOWN.

Format:
REASONING:
[your step-by-step analysis]

CONCLUSION: [ENTAIL/CONTRADICT/UNKNOWN]
CONFIDENCE: [0.0-1.0]
"""


BACKWARD_TRACE_PROMPT = """Given that the claim is {conclusion} by the evidence, explain the reasoning.

CLAIM:
{claim}

EVIDENCE:
{evidence}

GIVEN CONCLUSION: {conclusion}

Provide step-by-step reasoning explaining WHY this conclusion is correct.

Format:
REASONING:
[your step-by-step explanation]

CONFIDENCE: [0.0-1.0]
"""


CONSISTENCY_CHECK_PROMPT = """Evaluate the consistency of this reasoning trace.

CLAIM:
{claim}

EVIDENCE:
{evidence}

REASONING:
{trace}

CONCLUSION: {conclusion}

Check for:
1. Logical consistency - no contradictions in reasoning
2. Evidence grounding - reasoning references evidence
3. Conclusion support - reasoning actually supports conclusion
4. Circular reasoning - not just restating claim/conclusion

Rate each aspect (0.0-1.0) and provide overall consistency score.

Output JSON with:
- logical_consistency: float
- evidence_grounding: float  
- conclusion_support: float
- circularity_penalty: float (0=circular, 1=non-circular)
- overall_consistency: float
- notes: str
"""


class TraceValidator:
    """Validates reasoning traces and detects post-hoc rationalization."""
    
    def __init__(self, backend: Backend, config: TraceValidationConfig, model: str):
        self.backend = backend
        self.config = config
        self.model = model
    
    async def validate_claim_with_trace(
        self,
        claim: str,
        evidence: str,
    ) -> ReasoningTrace:
        """
        Generate and validate reasoning trace for a claim.
        
        Returns:
            ReasoningTrace with validation metrics
        """
        # Generate forward trace (natural reasoning)
        forward_trace, forward_judgment, forward_conf = await self._generate_forward_trace(
            claim, evidence
        )
        
        trace = ReasoningTrace(
            claim=claim,
            evidence=evidence,
            forward_trace=forward_trace,
            forward_judgment=forward_judgment,
            forward_confidence=forward_conf,
            trace_length=len(forward_trace.split()),
        )
        
        # Validate trace consistency
        consistency = await self._check_trace_consistency(
            claim, evidence, forward_trace, forward_judgment
        )
        trace.trace_consistency = consistency["overall_consistency"]
        trace.trace_support = consistency["conclusion_support"]
        
        # Detect post-hoc rationalization if enabled
        if self.config.detect_rationalization:
            backward_trace, backward_judgment, backward_conf = await self._generate_backward_trace(
                claim, evidence, forward_judgment
            )
            trace.backward_trace = backward_trace
            trace.backward_judgment = backward_judgment
            trace.backward_confidence = backward_conf
            
            # Compare forward vs backward traces
            trace.rationalization_score = await self._detect_rationalization(
                forward_trace, backward_trace, forward_judgment, backward_judgment
            )
        
        return trace
    
    async def _generate_forward_trace(
        self,
        claim: str,
        evidence: str,
    ) -> tuple[str, Dict[str, float], float]:
        """Generate forward reasoning trace (claim -> reasoning -> judgment)."""
        
        prompt = FORWARD_TRACE_PROMPT.format(claim=claim, evidence=evidence)
        
        # Use backend's text generation (not structured, to allow free reasoning)
        response = await self.backend.call_text(
            prompt=prompt,
            max_tokens=self.config.max_trace_tokens,
            temperature=self.config.trace_temperature,
        )
        
        # Parse response
        reasoning = self._extract_section(response, "REASONING:")
        conclusion_str = self._extract_field(response, "CONCLUSION:")
        confidence = self._extract_confidence(response)
        
        # Convert conclusion to distribution
        judgment = self._conclusion_to_dist(conclusion_str)
        
        return reasoning, judgment, confidence
    
    async def _generate_backward_trace(
        self,
        claim: str,
        evidence: str,
        forward_judgment: Dict[str, float],
    ) -> tuple[str, Dict[str, float], float]:
        """Generate backward reasoning trace (conclusion given -> explain why)."""
        
        # Get the conclusion from forward judgment
        conclusion = max(forward_judgment, key=forward_judgment.get)
        
        prompt = BACKWARD_TRACE_PROMPT.format(
            claim=claim,
            evidence=evidence,
            conclusion=conclusion,
        )
        
        response = await self.backend.call_text(
            prompt=prompt,
            max_tokens=self.config.max_trace_tokens,
            temperature=self.config.trace_temperature,
        )
        
        reasoning = self._extract_section(response, "REASONING:")
        confidence = self._extract_confidence(response)
        
        # Backward trace uses same conclusion
        judgment = forward_judgment.copy()
        
        return reasoning, judgment, confidence
    
    async def _check_trace_consistency(
        self,
        claim: str,
        evidence: str,
        trace: str,
        judgment: Dict[str, float],
    ) -> Dict[str, float]:
        """Check reasoning trace for consistency, grounding, and support."""
        
        conclusion = max(judgment, key=judgment.get)
        
        prompt = CONSISTENCY_CHECK_PROMPT.format(
            claim=claim,
            evidence=evidence,
            trace=trace,
            conclusion=conclusion,
        )
        
        # Get structured response
        schema = {
            "type": "object",
            "properties": {
                "logical_consistency": {"type": "number"},
                "evidence_grounding": {"type": "number"},
                "conclusion_support": {"type": "number"},
                "circularity_penalty": {"type": "number"},
                "overall_consistency": {"type": "number"},
                "notes": {"type": "string"},
            },
            "required": [
                "logical_consistency",
                "evidence_grounding",
                "conclusion_support",
                "circularity_penalty",
                "overall_consistency",
                "notes",
            ],
            "additionalProperties": False,
        }
        
        response = await self.backend.call_json_schema(
            model=self.model,
            prompt=prompt,
            schema_name="consistency_check",
            schema=schema,
            max_output_tokens=300,
            temperature=0.0,
        )
        
        return response
    
    async def _detect_rationalization(
        self,
        forward_trace: str,
        backward_trace: str,
        forward_judgment: Dict[str, float],
        backward_judgment: Dict[str, float],
    ) -> float:
        """
        Detect post-hoc rationalization by comparing forward vs backward traces.
        
        High score = likely rationalization (reasoning fabricated to fit conclusion)
        Low score = consistent reasoning (traces align)
        
        Returns:
            Rationalization score (0.0 = consistent, 1.0 = likely rationalization)
        """
        # Measure trace divergence (simple: overlap in key phrases)
        forward_phrases = set(self._extract_key_phrases(forward_trace))
        backward_phrases = set(self._extract_key_phrases(backward_trace))
        
        if not forward_phrases or not backward_phrases:
            return 0.5  # Unknown
        
        # Jaccard similarity of key phrases
        intersection = forward_phrases & backward_phrases
        union = forward_phrases | backward_phrases
        similarity = len(intersection) / len(union) if union else 0.0
        
        # Divergence = 1 - similarity
        divergence = 1.0 - similarity
        
        # If divergence is high, likely rationalization
        if divergence > self.config.rationalization_threshold:
            return min(1.0, divergence * 2.0)  # Scale up
        
        return divergence
    
    def _extract_section(self, text: str, header: str) -> str:
        """Extract a section from formatted output."""
        lines = text.split("\n")
        in_section = False
        section_lines = []
        
        for line in lines:
            if header.lower() in line.lower():
                in_section = True
                continue
            if in_section:
                # Stop at next section header (all caps + colon)
                if re.match(r"^[A-Z\s]+:", line.strip()):
                    break
                section_lines.append(line)
        
        return "\n".join(section_lines).strip()
    
    def _extract_field(self, text: str, field: str) -> str:
        """Extract a single field value."""
        pattern = rf"{re.escape(field)}\s*(.+?)(?:\n|$)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score."""
        conf_str = self._extract_field(text, "CONFIDENCE:")
        try:
            # Extract first number
            match = re.search(r"(\d+\.?\d*)", conf_str)
            if match:
                val = float(match.group(1))
                # Normalize if needed (e.g., 85 -> 0.85)
                if val > 1.0:
                    val /= 100.0
                return max(0.0, min(1.0, val))
        except ValueError:
            pass
        return 0.5
    
    def _conclusion_to_dist(self, conclusion_str: str) -> Dict[str, float]:
        """Convert conclusion string to distribution."""
        conclusion = conclusion_str.upper().strip()
        
        if "ENTAIL" in conclusion:
            return {"ENTAIL": 0.9, "CONTRADICT": 0.05, "UNKNOWN": 0.05}
        elif "CONTRADICT" in conclusion:
            return {"ENTAIL": 0.05, "CONTRADICT": 0.9, "UNKNOWN": 0.05}
        elif "UNKNOWN" in conclusion:
            return {"ENTAIL": 0.05, "CONTRADICT": 0.05, "UNKNOWN": 0.9}
        else:
            # Default to unknown
            return {"ENTAIL": 0.33, "CONTRADICT": 0.33, "UNKNOWN": 0.34}
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from reasoning trace (simple: bigrams and trigrams)."""
        words = text.lower().split()
        
        # Filter stopwords
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "this", "that", "these", "those",
            "of", "to", "in", "for", "on", "at", "by", "with", "from",
        }
        words = [w for w in words if w not in stopwords and len(w) > 2]
        
        phrases = []
        
        # Bigrams
        for i in range(len(words) - 1):
            phrases.append(f"{words[i]} {words[i+1]}")
        
        # Trigrams
        for i in range(len(words) - 2):
            phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        return phrases
    
    def compute_semantic_trace_similarity(self, trace1: str, trace2: str) -> float:
        """
        Use embeddings for trace comparison instead of surface-level Jaccard.
        
        Better captures semantic consistency even with different wording.
        
        Args:
            trace1: First reasoning trace
            trace2: Second reasoning trace
            
        Returns:
            Semantic similarity score in [0, 1]
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            # Lazy load embedder
            if not hasattr(self, '_embedder'):
                self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            emb1 = self._embedder.encode([trace1])[0]
            emb2 = self._embedder.encode([trace2])[0]
            
            import numpy as np
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            # Normalize to [0, 1]
            return float((similarity + 1.0) / 2.0)
            
        except ImportError:
            # Fallback to Jaccard if sentence-transformers not available
            forward_phrases = set(self._extract_key_phrases(trace1))
            backward_phrases = set(self._extract_key_phrases(trace2))
            
            if not forward_phrases or not backward_phrases:
                return 0.5
            
            intersection = forward_phrases & backward_phrases
            union = forward_phrases | backward_phrases
            return len(intersection) / len(union) if union else 0.0
    
    async def detect_logical_inconsistencies(self, traces: List[str]) -> float:
        """
        Detect logical contradictions between traces.
        
        Args:
            traces: List of reasoning traces to check for consistency
            
        Returns:
            Inconsistency score in [0, 1], higher = more inconsistencies
        """
        if len(traces) < 2:
            return 0.0
        
        inconsistencies = []
        
        # Check pairs of traces for contradictions
        for i in range(min(len(traces), 3)):  # Limit to avoid too many calls
            for j in range(i+1, min(len(traces), 3)):
                prompt = f"""Do these two reasoning traces contradict each other?

Trace 1: {traces[i][:500]}
Trace 2: {traces[j][:500]}

Answer YES (they contradict) or NO (they're consistent):"""
                
                try:
                    response = await self.backend.call_text(
                        prompt=prompt,
                        max_tokens=10,
                        temperature=0.0
                    )
                    if 'YES' in response.strip().upper():
                        inconsistencies.append((i, j))
                except Exception:
                    # If check fails, assume consistent
                    pass
        
        # Higher score = more inconsistencies
        max_pairs = min(len(traces), 3) * (min(len(traces), 3) - 1) / 2
        return len(inconsistencies) / max_pairs if max_pairs > 0 else 0.0
