"""Google Gemini API backend for PCIB detector."""

import asyncio
import json
import os
from typing import Any, Dict, Optional

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .base import Backend


class GeminiBackend(Backend):
    """Async Google Gemini API wrapper with rate limiting."""
    
    SUPPORTED_MODELS = [
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.0-pro",
    ]
    
    def __init__(self, api_key: Optional[str] = None, max_concurrent: int = 10):
        """
        Initialize backend.
        
        Args:
            api_key: Google API key (uses GOOGLE_API_KEY env var if None)
            max_concurrent: Maximum concurrent API requests
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai package not installed. Install with: pip install google-generativeai"
            )
        
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable must be set")
        
        genai.configure(api_key=api_key)
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    def get_default_model(self) -> str:
        """Get the default model for Gemini."""
        return "gemini-2.0-flash-exp"
    
    def validate_model(self, model: str) -> bool:
        """Check if the model is supported."""
        return any(model.startswith(supported) for supported in self.SUPPORTED_MODELS)
    
    def _schema_to_prompt(self, schema: Dict[str, Any]) -> str:
        """Convert JSON schema to prompt instructions."""
        lines = ["Return a valid JSON object with this exact structure:"]
        
        if "properties" in schema:
            lines.append("\n{")
            for key, spec in schema["properties"].items():
                type_str = spec.get("type", "any")
                desc = spec.get("description", "")
                
                # Handle nested objects
                if type_str == "object" and "properties" in spec:
                    lines.append(f'  "{key}": {{')
                    for subkey, subspec in spec["properties"].items():
                        subtype = subspec.get("type", "any")
                        lines.append(f'    "{subkey}": {subtype},')
                    lines.append("  },")
                elif type_str == "array":
                    items = spec.get("items", {})
                    item_type = items.get("type", "object")
                    lines.append(f'  "{key}": [{item_type}],')
                else:
                    if desc:
                        lines.append(f'  "{key}": {type_str},  // {desc}')
                    else:
                        lines.append(f'  "{key}": {type_str},')
            lines.append("}")
        
        required = schema.get("required", [])
        if required:
            lines.append(f"\nRequired fields: {', '.join(required)}")
        
        lines.append("\nIMPORTANT: Return ONLY the JSON object, no markdown, no explanation, no code blocks.")
        
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
        Call Gemini API with JSON schema output.
        
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
            full_prompt = f"{prompt}\n\n{schema_instructions}"
            
            # Configure model
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "response_mime_type": "application/json",
            }
            
            model_obj = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config
            )
            
            # Generate response (sync call, run in executor)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model_obj.generate_content(full_prompt)
            )
            
            # Extract text content
            content = response.text.strip()
            
            # Try to extract JSON from response
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
                raise ValueError(f"Failed to parse JSON from Gemini response: {content[:200]}") from e
    
    async def call_text(
        self,
        *,
        prompt: str,
        max_tokens: int = 600,
        temperature: float = 0.0,
    ) -> str:
        """
        Call Gemini API for plain text generation.
        
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
            
            # Configure model
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            
            model_obj = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config
            )
            
            # Generate response (sync call, run in executor)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model_obj.generate_content(prompt)
            )
            
            return response.text.strip()
