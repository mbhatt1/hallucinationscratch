"""OpenAI API backend for PCIB detector."""

import asyncio
import json
import os
from typing import Any, Dict, Optional

from openai import AsyncOpenAI


class OpenAIBackend:
    """Async OpenAI API wrapper with rate limiting."""
    
    def __init__(self, api_key: Optional[str] = None, max_concurrent: int = 10):
        """
        Initialize backend.
        
        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
            max_concurrent: Maximum concurrent API requests
        """
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def call_json_schema(
        self,
        *,
        prompt: str,
        schema: Dict[str, Any],
        model: str,
        name: str,
        max_output_tokens: int = 600,
        temperature: float = 0.0,
        instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call OpenAI API with JSON schema structured output.
        
        Args:
            prompt: User prompt
            schema: JSON schema for structured output
            model: Model name
            name: Schema name
            max_output_tokens: Max tokens to generate
            temperature: Sampling temperature
            instructions: System instructions (optional)
            
        Returns:
            Parsed JSON response
        """
        async with self.semaphore:
            messages = []
            if instructions:
                messages.append({"role": "system", "content": instructions})
            messages.append({"role": "user", "content": prompt})
            
            response = await self.client.responses.create(
                model=model,
                input=messages,
                max_output_tokens=max_output_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": name,
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
            
            # Extract output text
            output_text = getattr(response, "output_text", "") or ""
            return json.loads(output_text)
