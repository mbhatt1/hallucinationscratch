"""OpenAI API backend for PCIB detector."""

import asyncio
import json
import os
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

from .base import Backend


class OpenAIBackend(Backend):
    """Async OpenAI API wrapper with rate limiting."""
    
    SUPPORTED_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
    ]
    
    def __init__(self, api_key: Optional[str] = None, max_concurrent: int = 10):
        """
        Initialize backend.
        
        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
            max_concurrent: Maximum concurrent API requests
        """
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    def get_default_model(self) -> str:
        """Get the default model for OpenAI."""
        return "gpt-4o-mini"
    
    def validate_model(self, model: str) -> bool:
        """Check if the model is supported."""
        return any(model.startswith(supported) for supported in self.SUPPORTED_MODELS)
    
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
        Call OpenAI API with JSON schema structured output.
        
        Args:
            model: Model name
            prompt: User prompt
            schema_name: Schema name
            schema: JSON schema for structured output
            max_output_tokens: Max tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Parsed JSON response
        """
        async with self.semaphore:
            # Use chat completions with JSON mode
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                # max_tokens=max_output_tokens,
                # temperature=temperature,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
    
    async def call_text(
        self,
        *,
        prompt: str,
        max_tokens: int = 600,
        temperature: float = 0.0,
    ) -> str:
        """
        Call OpenAI API for plain text generation.
        
        Args:
            prompt: User prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        async with self.semaphore:
            # Use default model for text generation
            model = self.get_default_model()
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                # max_tokens=max_tokens,
                # temperature=temperature,
            )
            
            return response.choices[0].message.content or ""
