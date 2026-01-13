"""
Graph-based claim aggregation for better multi-claim handling.
"""
import asyncio
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple


class ClaimDependencyGraph:
    """Build and analyze claim dependency graphs."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
    
    async def build_graph(
        self, 
        claims: List[str], 
        question: str,
        backend
    ) -> nx.DiGraph:
        """
        Build claim dependency graph.
        
        Edges represent logical dependencies or entailments.
        
        Args:
            claims: List of claim strings
            question: Original question text
            backend: Backend for LLM calls
            
        Returns:
            NetworkX DiGraph with claims as nodes
        """
        self.graph = nx.DiGraph()
        
        if not claims:
            return self.graph
        
        # Add claim nodes
        for i, claim in enumerate(claims):
            self.graph.add_node(i, text=claim, score=0.0)
        
        # Add question node
        self.graph.add_node('question', text=question, score=0.0)
        
        # Detect dependencies (run in parallel)
        tasks = []
        
        # Check if each claim depends on question
        for i, claim_i in enumerate(claims):
            tasks.append(self._check_dependency(claim_i, question, 'question', i, 1.0, backend))
        
        # Check if claims depend on other claims
        for i, claim_i in enumerate(claims):
            for j, claim_j in enumerate(claims):
                if i != j:
                    tasks.append(self._check_dependency(claim_i, claim_j, j, i, 0.5, backend))
        
        # Execute all checks in parallel
        results = await asyncio.gather(*tasks)
        
        # Add edges based on results
        for result in results:
            if result and result['depends']:
                self.graph.add_edge(result['from'], result['to'], weight=result['weight'])
        
        return self.graph
    
    async def _check_dependency(
        self, 
        claim: str, 
        context: str, 
        from_node, 
        to_node, 
        weight: float,
        backend
    ) -> Dict:
        """Check if claim logically depends on context."""
        depends = await self._depends_on(claim, context, backend)
        return {
            'from': from_node,
            'to': to_node,
            'weight': weight,
            'depends': depends
        }
    
    async def _depends_on(self, claim: str, context: str, backend) -> bool:
        """Check if claim logically depends on context."""
        prompt = f"""Does the following claim logically depend on or follow from the context?

Context: {context}
Claim: {claim}

Answer YES or NO:"""
        
        try:
            response = await backend.generate(
                model=backend.get_default_model(),
                prompt=prompt,
                temperature=0.0,
                max_tokens=10
            )
            return 'YES' in response.strip().upper()
        except Exception:
            # If check fails, assume no dependency
            return False
    
    def compute_importance_weights(self) -> Dict[int, float]:
        """
        Compute importance weight for each claim using PageRank.
        
        Claims that many other claims depend on get higher weight.
        
        Returns:
            Dict mapping claim_id -> importance weight
        """
        if len(self.graph) == 0:
            return {}
        
        # Remove question node for PageRank (only rank claims)
        graph_copy = self.graph.copy()
        if 'question' in graph_copy:
            graph_copy.remove_node('question')
        
        if len(graph_copy) == 0:
            return {0: 1.0}
        
        # Compute PageRank
        try:
            pagerank = nx.pagerank(graph_copy, weight='weight')
        except:
            # If PageRank fails, use uniform weights
            n = len(graph_copy)
            pagerank = {node: 1.0/n for node in graph_copy.nodes()}
        
        return pagerank
    
    def aggregate_scores(
        self, 
        claim_scores: Dict[int, float],
        method: str = 'weighted'
    ) -> float:
        """
        Aggregate individual claim scores into final score.
        
        Args:
            claim_scores: Dict mapping claim_id -> hallucination_score
            method: 'weighted' (PageRank), 'max', 'average'
            
        Returns:
            Aggregated hallucination score
        """
        if not claim_scores:
            return 0.0
        
        if method == 'weighted':
            weights = self.compute_importance_weights()
            
            if not weights:
                # Fallback to average if weights empty
                return np.mean(list(claim_scores.values()))
            
            # Normalize weights
            total_weight = sum(weights.get(i, 0.0) for i in claim_scores.keys())
            if total_weight == 0:
                return np.mean(list(claim_scores.values()))
            
            # Weighted average
            score = sum(
                claim_scores.get(i, 0.0) * weights.get(i, 0.0) 
                for i in claim_scores.keys()
            ) / total_weight
            return score
        
        elif method == 'max':
            return max(claim_scores.values())
        
        elif method == 'average':
            return np.mean(list(claim_scores.values()))
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def visualize_graph(self, filename: str = 'claim_graph.png'):
        """
        Visualize the claim dependency graph.
        
        Args:
            filename: Output filename for graph visualization
        """
        try:
            import matplotlib.pyplot as plt
            
            pos = nx.spring_layout(self.graph)
            
            # Draw nodes
            nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', node_size=500)
            
            # Draw edges
            nx.draw_networkx_edges(self.graph, pos, edge_color='gray', arrows=True)
            
            # Draw labels
            labels = {node: f"{node}\n{self.graph.nodes[node].get('text', '')[:30]}..." 
                     for node in self.graph.nodes()}
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
            
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            
        except ImportError:
            print("matplotlib not available for visualization")
