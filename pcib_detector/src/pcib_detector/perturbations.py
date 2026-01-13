"""Evidence perturbation functions for PC+IB testing."""

import random
import re
from typing import List


def make_distractor(n_chars: int) -> str:
    """
    Generate distractor text (irrelevant noise) for stress testing.
    
    Args:
        n_chars: Approximate length of distractor text
        
    Returns:
        Distractor text string
    """
    vocab = [
        "alpha", "beta", "gamma", "delta", "epsilon",
        "kappa", "lambda", "sigma", "theta", "omega"
    ]
    lines = []
    while sum(len(x) + 1 for x in lines) < n_chars:
        lines.append(" ".join(random.choice(vocab) for _ in range(20)))
    
    return "DISTRACTOR:\n" + "\n".join(lines)[:n_chars]


def make_conflict_snippet(claim: str, evidence: str) -> str:
    """
    Generate conflict text to test sensitivity to contradictions.
    
    This is a simple heuristic-based generator. For production use,
    consider using retrieval-based hard negatives.
    
    Args:
        claim: The claim text
        evidence: The evidence text (unused in current implementation)
        
    Returns:
        Conflicting statement
    """
    # Try to find a year and contradict it
    m = re.search(r"\b(19\d{2}|20\d{2})\b", claim)
    if m:
        year = int(m.group(1))
        return f"It did NOT occur in {year}; it occurred in {year + 1}."
    
    # Try to find a number and contradict it
    m = re.search(r"\b(\d+(\.\d+)?)\b", claim)
    if m:
        num = float(m.group(1))
        new_num = num + 1.0
        new_num_str = str(int(new_num)) if new_num.is_integer() else str(new_num)
        return f"The correct value is {new_num_str}, not {m.group(1)}."
    
    # Generic fallback
    return "The evidence explicitly states the opposite of the claim."


class EnhancedPerturbationGenerator:
    """Generate diverse, meaningful perturbations for enhanced testing."""
    
    def __init__(self, backend):
        self.backend = backend
    
    async def generate_semantic_perturbations(
        self,
        claim: str,
        n_perturbations: int = 3
    ) -> List[str]:
        """
        Generate semantic-preserving perturbations using LLM.
        
        Better than random edits because they test semantic understanding.
        
        Args:
            claim: Original claim text
            n_perturbations: Number of perturbations to generate
            
        Returns:
            List of paraphrased claims
        """
        prompt = f"""Generate {n_perturbations} paraphrases of this claim that preserve its meaning:

Claim: {claim}

Requirements:
- Keep the same factual content
- Vary sentence structure
- Use synonyms where appropriate
- Each paraphrase on a new line

Paraphrases:"""
        
        try:
            response = await self.backend.generate(
                model=self.backend.get_default_model(),
                prompt=prompt,
                temperature=0.7,
                max_tokens=200
            )
            perturbations = [p.strip() for p in response.split('\n') if p.strip()]
            return perturbations[:n_perturbations]
        except Exception:
            # Fallback: return empty list
            return []
    
    async def generate_adversarial_perturbations(
        self,
        claim: str,
        n_perturbations: int = 2
    ) -> List[str]:
        """
        Generate adversarial perturbations that introduce subtle errors.
        
        Tests robustness to near-hallucinations.
        
        Args:
            claim: Original claim text
            n_perturbations: Number of adversarial versions to generate
            
        Returns:
            List of adversarial claims
        """
        prompt = f"""Generate {n_perturbations} versions of this claim with subtle factual errors:

Claim: {claim}

Requirements:
- Change one key fact (name, number, date, etc.)
- Keep the sentence structure similar
- Make it plausible but incorrect
- Each version on a new line

Adversarial versions:"""
        
        try:
            response = await self.backend.generate(
                model=self.backend.get_default_model(),
                prompt=prompt,
                temperature=0.8,
                max_tokens=200
            )
            perturbations = [p.strip() for p in response.split('\n') if p.strip()]
            return perturbations[:n_perturbations]
        except Exception:
            # Fallback: return empty list
            return []
    
    async def generate_mixed_perturbations(
        self,
        claim: str,
        n_total: int = 5
    ) -> List[str]:
        """
        Mix of semantic-preserving and adversarial perturbations.
        
        Args:
            claim: Original claim text
            n_total: Total number of perturbations
            
        Returns:
            List of mixed perturbations
        """
        n_semantic = n_total // 2
        n_adversarial = n_total - n_semantic
        
        perturbations = []
        
        # Generate semantic perturbations
        semantic = await self.generate_semantic_perturbations(claim, n_semantic)
        perturbations.extend(semantic)
        
        # Generate adversarial perturbations
        adversarial = await self.generate_adversarial_perturbations(claim, n_adversarial)
        perturbations.extend(adversarial)
        
        return perturbations
