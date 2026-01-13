"""
Multi-verifier ensemble for diverse predictions.

Instead of running same verifier multiple times (adds noise),
use different verifier models for true diversity.
"""
import asyncio
from typing import List, Dict, Optional
import numpy as np


class MultiVerifierEnsemble:
    """Ensemble multiple verifier models."""
    
    def __init__(self, verifier_configs: List[Dict]):
        """
        Args:
            verifier_configs: List of dicts with 'backend', 'model', 'weight'
            
        Example:
            [
                {'backend': 'openai', 'model': 'gpt-4o-mini', 'weight': 0.4},
                {'backend': 'anthropic', 'model': 'claude-3-haiku', 'weight': 0.3},
                {'backend': 'gemini', 'model': 'gemini-1.5-flash', 'weight': 0.3}
            ]
        """
        self.verifier_configs = verifier_configs
        self.backends = {}
        
        # Initialize backends
        for config in verifier_configs:
            backend_type = config['backend']
            if backend_type not in self.backends:
                self.backends[backend_type] = self._create_backend(backend_type, config['model'])
    
    def _create_backend(self, backend_type: str, model: str):
        """Create backend instance."""
        if backend_type == 'openai':
            from pcib_detector.backends.openai_backend import OpenAIBackend
            return OpenAIBackend()
        elif backend_type == 'anthropic':
            from pcib_detector.backends.anthropic_backend import AnthropicBackend
            return AnthropicBackend()
        elif backend_type == 'gemini':
            from pcib_detector.backends.gemini_backend import GeminiBackend
            return GeminiBackend()
        else:
            raise ValueError(f"Unknown backend: {backend_type}")
    
    async def compute_ensemble_score(
        self, 
        question: str, 
        answer: str,
        aggregation: str = 'weighted_average'
    ) -> Dict:
        """
        Compute PCIB score using multiple verifiers.
        
        Args:
            question: Input question
            answer: Model answer to verify
            aggregation: 'weighted_average', 'max', 'voting'
            
        Returns:
            Dict with ensemble_score and individual_scores
        """
        from pcib_detector.core import PCIBDetector
        from pcib_detector.types import Config
        
        individual_results = []
        
        # Run detection with each verifier
        tasks = []
        for config in self.verifier_configs:
            tasks.append(self._run_single_verifier(question, answer, config))
        
        individual_results = await asyncio.gather(*tasks)
        
        # Aggregate scores
        if aggregation == 'weighted_average':
            weights = [r['weight'] for r in individual_results]
            scores = [r['score'] for r in individual_results]
            ensemble_score = np.average(scores, weights=weights)
        
        elif aggregation == 'max':
            ensemble_score = max(r['score'] for r in individual_results)
        
        elif aggregation == 'voting':
            # Vote on hallucination (threshold=0.5)
            votes = [int(r['score'] > 0.5) for r in individual_results]
            ensemble_score = float(np.mean(votes))
        
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        return {
            'ensemble_score': ensemble_score,
            'individual_results': individual_results,
            'aggregation': aggregation
        }
    
    async def _run_single_verifier(self, question: str, answer: str, config: Dict) -> Dict:
        """Run detection with a single verifier."""
        from pcib_detector.core import PCIBDetector
        from pcib_detector.types import Config
        
        # Create detector with this backend
        backend = self.backends[config['backend']]
        
        # Create config
        detector_config = Config(
            provider=config['backend'],
            model=config['model']
        )
        
        detector = PCIBDetector(config=detector_config)
        
        # Run detection
        result = await detector.detect_hallucination(
            answer=answer,
            evidence=question,
            return_details=False
        )
        
        return {
            'verifier': f"{config['backend']}-{config['model']}",
            'score': result.score,
            'weight': config['weight'],
            'flagged': result.flagged
        }
