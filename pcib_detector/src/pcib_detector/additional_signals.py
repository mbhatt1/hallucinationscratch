"""
Additional grounding signals beyond U, S, C, R.
"""
import asyncio
import numpy as np
from typing import Dict, Optional, List
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class AdditionalSignals:
    """Compute supplementary hallucination signals."""
    
    def __init__(self, use_embeddings: bool = True):
        """
        Args:
            use_embeddings: Whether to use sentence embeddings (requires sentence-transformers)
        """
        self.embedder = None
        if use_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Load lightweight embedding model
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Warning: Could not load sentence transformer: {e}")
                self.embedder = None
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between embeddings.
        
        Low similarity between question and answer may indicate hallucination.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score in [0, 1], or 0.5 if embeddings unavailable
        """
        if self.embedder is None:
            # Fallback: simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.5
            overlap = len(words1 & words2)
            total = len(words1 | words2)
            return overlap / total if total > 0 else 0.0
        
        try:
            emb1 = self.embedder.encode([text1])[0]
            emb2 = self.embedder.encode([text2])[0]
            
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            # Normalize to [0, 1]
            return float((similarity + 1.0) / 2.0)
        except Exception:
            return 0.5  # Neutral score on error
    
    async def entity_consistency(
        self, 
        question: str, 
        answer: str, 
        backend
    ) -> float:
        """
        Check if entities in answer are consistent with question.
        
        Returns score in [0, 1], higher = more consistent.
        
        Args:
            question: Question text
            answer: Answer text
            backend: Backend for LLM calls
            
        Returns:
            Consistency score
        """
        # Extract entities from question and answer
        q_entities = await self._extract_entities(question, backend)
        a_entities = await self._extract_entities(answer, backend)
        
        if not a_entities:
            return 1.0  # No entities to check
        
        # Check each answer entity
        consistent_count = 0
        for entity in a_entities:
            if await self._is_consistent_with(entity, q_entities, question, backend):
                consistent_count += 1
        
        return consistent_count / len(a_entities)
    
    async def _extract_entities(self, text: str, backend) -> List[str]:
        """Extract named entities."""
        prompt = f"""Extract all named entities (people, places, organizations, dates, numbers) from this text:

Text: {text}

List only the entities, one per line:"""
        
        try:
            response = await backend.generate(
                model=backend.get_default_model(),
                prompt=prompt,
                temperature=0.0,
                max_tokens=150
            )
            entities = [e.strip() for e in response.split('\n') if e.strip()]
            return entities[:20]  # Limit to avoid excessive processing
        except Exception:
            return []
    
    async def _is_consistent_with(
        self, 
        entity: str, 
        question_entities: List[str], 
        question: str, 
        backend
    ) -> bool:
        """Check if answer entity is consistent with question."""
        # If entity appears in question, it's consistent
        if entity.lower() in question.lower():
            return True
        
        # Check if entity is related to question entities
        for q_entity in question_entities:
            if await self._are_related(entity, q_entity, backend):
                return True
        
        return False
    
    async def _are_related(self, entity1: str, entity2: str, backend) -> bool:
        """Check if two entities are semantically related."""
        # Use semantic similarity
        similarity = self.semantic_similarity(entity1, entity2)
        return similarity > 0.7
    
    def answer_specificity(self, answer: str) -> float:
        """
        Measure answer specificity.
        
        Vague answers may indicate hallucination or uncertainty.
        Returns score in [0, 1], higher = more specific.
        
        Args:
            answer: Answer text
            
        Returns:
            Specificity score
        """
        # Check for hedging phrases
        hedges = [
            'maybe', 'perhaps', 'possibly', 'might', 'could',
            'i think', 'i believe', 'probably', 'generally', 
            'usually', 'typically', 'often', 'sometimes'
        ]
        
        answer_lower = answer.lower()
        hedge_count = sum(1 for hedge in hedges if hedge in answer_lower)
        
        # More hedges = less specific
        hedge_penalty = min(hedge_count * 0.2, 1.0)
        
        # Check for specific details (numbers, names, dates)
        has_numbers = any(char.isdigit() for char in answer)
        has_proper_nouns = any(word[0].isupper() for word in answer.split() if len(word) > 1)
        
        specificity_bonus = 0.3 * has_numbers + 0.3 * has_proper_nouns
        
        # Base specificity
        base_specificity = 0.5
        
        final_score = base_specificity + specificity_bonus - hedge_penalty
        return max(0.0, min(1.0, final_score))
    
    async def compute_all(
        self, 
        question: str, 
        answer: str, 
        backend
    ) -> Dict[str, float]:
        """
        Compute all additional signals.
        
        Args:
            question: Question text
            answer: Answer text
            backend: Backend for LLM calls
            
        Returns:
            Dict with all signal scores
        """
        # Run entity consistency asynchronously
        entity_task = asyncio.create_task(
            self.entity_consistency(question, answer, backend)
        )
        
        # Compute synchronous signals
        signals = {
            'semantic_similarity': self.semantic_similarity(question, answer),
            'answer_specificity': self.answer_specificity(answer)
        }
        
        # Wait for async signal
        try:
            signals['entity_consistency'] = await entity_task
        except Exception as e:
            print(f"Warning: Entity consistency check failed: {e}")
            signals['entity_consistency'] = 0.5  # Neutral score
        
        return signals
