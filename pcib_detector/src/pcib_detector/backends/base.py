"""Base backend interface for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class Backend(ABC):
    """Abstract base class for LLM provider backends."""
    
    @abstractmethod
    async def call_json_schema(
        self,
        *,
        model: str,
        prompt: str,
        schema_name: str,
        schema: Dict[str, Any],
        max_output_tokens: int = 600,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Call the LLM with structured output (JSON schema).
        
        Args:
            model: Model identifier
            prompt: Input prompt
            schema_name: Name for the schema
            schema: JSON schema definition
            max_output_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Parsed JSON object matching the schema
        """
        pass
    
    @abstractmethod
    async def call_text(
        self,
        *,
        prompt: str,
        max_tokens: int = 600,
        temperature: float = 0.0,
    ) -> str:
        """
        Call the LLM for plain text generation (no structured output).
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this backend."""
        pass
    
    @abstractmethod
    def validate_model(self, model: str) -> bool:
        """Check if the model is supported by this backend."""
        pass
