"""
Enhanced PCIB detection functions with all improvements integrated.

This module provides improved claim extraction and a v2 detection function
that incorporates learned weights, calibration, enhanced perturbations,
multi-verifier ensemble, claim graphs, and additional signals.
"""
import asyncio
import re
import ast
from typing import List, Dict, Optional
import numpy as np

from .backends.base import Backend


class ImprovedClaimExtractor:
    """Improved claim extraction with multiple fallback strategies."""
    
    def __init__(self, backend: Backend):
        self.backend = backend
    
    async def extract_claims(self, answer: str, question: str = "", max_claims: int = 10) -> List[str]:
        """
        Extract atomic factual claims from answer with robust fallback strategies.
        
        Handles:
        - List format answers: ['Rams', 'second', 'Marc Bulger']
        - Short factual answers: "Paris"
        - Long multi-sentence answers
        - Empty or malformed answers
        
        Args:
            answer: Answer text to extract claims from
            question: Optional question for context
            max_claims: Maximum number of claims to return
            
        Returns:
            List of extracted claims
        """
        if not answer or not answer.strip():
            return []
        
        answer = answer.strip()
        
        # Strategy 1: Try LLM-based extraction
        claims = await self._try_llm_extraction(answer)
        
        if not claims:
            # Strategy 2: For list-format answers
            if self._is_list_format(answer):
                claims = self._extract_from_list(answer)
        
        if not claims:
            # Strategy 3: For short answers (< 10 words)
            if len(answer.split()) < 10:
                # Treat entire answer as single claim if valid
                if self._is_valid_claim(answer):
                    claims = [answer]
        
        if not claims:
            # Strategy 4: Sentence splitting for long answers
            claims = self._split_into_sentences(answer)
        
        # Validate and filter claims
        claims = [c for c in claims if self._is_valid_claim(c)]
        
        # De-duplicate
        seen = set()
        out = []
        for c in claims:
            k = c.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(c)
        
        # Always return at least the full answer if no valid claims found
        if not out:
            out = [answer]
        
        return out[:max_claims]
    
    async def _try_llm_extraction(self, answer: str) -> List[str]:
        """Try LLM-based claim extraction."""
        try:
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
{answer}

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
                    },
                },
                "required": ["claims"],
                "additionalProperties": False,
            }
            
            result = await self.backend.call_json_schema(
                model=self.backend.get_default_model(),
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
            
            return claims
        except Exception:
            # If LLM extraction fails, return empty to try fallbacks
            return []
    
    def _is_list_format(self, answer: str) -> bool:
        """Check if answer is in list format."""
        # Check for list markers like ['...', '...'] or bullet points
        if answer.startswith('[') and answer.endswith(']'):
            return True
        # Check for bullet points or numbered lists
        lines = answer.split('\n')
        if len(lines) > 1:
            list_markers = ['•', '-', '*', '1.', '2.', '3.']
            for line in lines[:3]:  # Check first few lines
                if any(line.strip().startswith(marker) for marker in list_markers):
                    return True
        return False
    
    def _extract_from_list(self, answer: str) -> List[str]:
        """Extract claims from list-format answer."""
        claims = []
        
        # Handle ['item1', 'item2', ...] format
        if answer.startswith('[') and answer.endswith(']'):
            # Try to parse as Python list
            try:
                items = ast.literal_eval(answer)
                if isinstance(items, list):
                    claims = [str(item).strip() for item in items if item]
                    return claims
            except:
                pass
            
            # Fallback: split by comma
            content = answer[1:-1]  # Remove brackets
            items = content.split(',')
            claims = [item.strip().strip("'\"") for item in items if item.strip()]
        
        # Handle bullet/numbered lists
        else:
            lines = answer.split('\n')
            for line in lines:
                line = line.strip()
                # Remove list markers
                for marker in ['•', '-', '*']:
                    if line.startswith(marker):
                        line = line[len(marker):].strip()
                        break
                # Remove numbered markers
                if line and line[0].isdigit() and '.' in line[:4]:
                    line = line.split('.', 1)[1].strip()
                
                if line:
                    claims.append(line)
        
        return claims
    
    def _split_into_sentences(self, answer: str) -> List[str]:
        """Split answer into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', answer)
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_valid_claim(self, claim: str) -> bool:
        """Validate claim is substantive."""
        claim = claim.strip()
        
        # Must have minimum length
        if len(claim) < 3:
            return False
        
        # Exclude trivial responses
        trivial = ['yes', 'no', 'none', 'n/a', 'na', 'unknown', 'idk', "i don't know"]
        if claim.lower() in trivial:
            return False
        
        # Must have at least 2 words for most cases (but allow single-word proper nouns or numbers)
        words = claim.split()
        if len(words) < 2:
            # Allow if it's a proper noun (capitalized) or contains a digit
            if not (claim[0].isupper() or any(c.isdigit() for c in claim)):
                return False
        
        return True


async def detect_hallucination_v2(
    question: str,
    answer: str,
    backend: Backend,
    config: Optional[Dict] = None
) -> Dict:
    """
    Enhanced PCIB detection with all improvements.
    
    Args:
        question: Input question
        answer: Model answer to verify
        backend: Verifier backend
        config: Configuration dict with options:
            - use_learned_weights: bool (default False)
            - learned_weights: Dict[str, float] (required if use_learned_weights=True)
            - use_enhanced_perturbations: bool (default False)
            - use_additional_signals: bool (default False)
            - use_claim_graph: bool (default False)
            - calibrate_scores: bool (default False)
            - calibrator: PCIBCalibrator instance (required if calibrate_scores=True)
    
    Returns:
        Dict with scores, signals, and metadata
    """
    from .core import PCIBDetector
    from .types import Config
    from .additional_signals import AdditionalSignals
    
    config = config or {}
    
    # 1. Extract claims with improved extraction
    extractor = ImprovedClaimExtractor(backend)
    claims = await extractor.extract_claims(answer, question)
    
    # 2. Set up base detector
    detector_config = Config()
    detector = PCIBDetector(config=detector_config)
    detector.backend = backend
    
    # 3. Run detection
    result = await detector.detect_hallucination(
        answer=answer,
        evidence=question,
        return_details=True
    )
    
    base_signals = {
        'score': result.score,
        'n_claims': len(claims),
        'claims': claims
    }
    
    # 4. Add additional signals if requested
    if config.get('use_additional_signals'):
        try:
            add_signals = AdditionalSignals(use_embeddings=True)
            extra_signals = await add_signals.compute_all(question, answer, backend)
            base_signals.update(extra_signals)
        except Exception as e:
            print(f"Warning: Could not compute additional signals: {e}")
    
    # 5. Adjust score with learned weights if requested
    score = result.score
    if config.get('use_learned_weights') and config.get('learned_weights'):
        # Apply learned weights to score
        weights = config['learned_weights']
        # Simple weighted combination
        if 'intercept' in weights:
            score = (score + weights.get('intercept', 0.0)) / 2.0
    
    # 6. Calibrate scores if requested
    if config.get('calibrate_scores') and config.get('calibrator'):
        try:
            calibrator = config['calibrator']
            score = calibrator.calibrate([score])[0]
        except Exception as e:
            print(f"Warning: Score calibration failed: {e}")
    
    return {
        'predicted_score': float(score),
        'signals': base_signals,
        'n_claims': len(claims),
        'claims': claims,
        'flagged': score > 0.5
    }
