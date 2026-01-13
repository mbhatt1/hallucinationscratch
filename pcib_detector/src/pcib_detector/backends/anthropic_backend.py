"""Anthropic API backend for PCIB detector."""

import asyncio
import json
import os
from typing import Any, Dict, Optional

try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .base import Backend


class AnthropicBackend(Backend):
    """Async Anthropic API wrapper with rate limiting."""
    
    SUPPORTED_MODELS = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]
    
    def __init__(self, api_key: Optional[str] = None, max_concurrent: int = 10):
        """
        Initialize backend.
        
        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if None)
            max_concurrent: Maximum concurrent API requests
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )
        
        self.client = AsyncAnthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    def get_default_model(self) -> str:
        """Get the default model for Anthropic."""
        return "claude-3-5-sonnet-20241022"
    
    def validate_model(self, model: str) -> bool:
        """Check if the model is supported."""
        return model in self.SUPPORTED_MODELS
    
    def _schema_to_prompt(self, schema: Dict[str, Any]) -> str:
        """Convert JSON schema to prompt instructions."""
        lines = ["Return a JSON object with the following structure:"]
        
        if "properties" in schema:
            lines.append("\n{")
            for key, spec in schema["properties"].items():
                type_str = spec.get("type", "any")
                desc = spec.get("description", "")
                if desc:
                    lines.append(f'  "{key}": {type_str}  // {desc}')
                else:
                    lines.append(f'  "{key}": {type_str}')
            lines.append("}")
        
        required = schema.get("required", [])
        if required:
            lines.append(f"\nRequired fields: {', '.join(required)}")
        
        return "\n".join(lines)
    
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
        Call Anthropic API with JSON schema output.
        
        Args:
            model: Model name
            prompt: User prompt
            schema_name: Schema name (for documentation)
            schema: JSON schema for structured output
            max_output_tokens: Max tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Parsed JSON response
        """
        async with self.semaphore:
            # Add schema instructions to prompt
            schema_instructions = self._schema_to_prompt(schema)
            full_prompt = f"{prompt}\n\n{schema_instructions}\n\nReturn only valid JSON, no other text."
            
            response = await self.client.messages.create(
                model=model,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": full_prompt}]
            )
            
            # Extract text content
            content = response.content[0].text if response.content else ""
            
            # Try to extract JSON from response
            # Sometimes models wrap JSON in markdown code blocks
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                # Try to find JSON object in response
                start = content.find("{")
                end = content.rfind("}") + 1
                if start != -1 and end > start:
                    try:
                        return json.loads(content[start:end])
                    except:
                        pass
                raise ValueError(f"Failed to parse JSON from response: {content[:200]}") from e
    
    async def call_text(
        self,
        *,
        prompt: str,
        max_tokens: int = 600,
        temperature: float = 0.0,
    ) -> str:
        """
        Call Anthropic API for plain text generation.
        
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
            
            response = await self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract text content
            return response.content[0].text if response.content else ""
