# app/ai_engine.py

from abc import ABC, abstractmethod
import os
import json
# Import real libraries
from openai import OpenAI
from anthropic import Anthropic
import google.genai as genai
from google.genai import types

class BaseAIProvider(ABC):
    """Abstract Base Class defining the interface for all AI providers."""

    @abstractmethod
    def extract_structured_data(self, text: str) -> dict:
        """Extracts skills and summary from text using the provider's LLM."""
        pass

    @abstractmethod
    def generate_embedding(self, text: str) -> list:
        """Generates a vector embedding for the given text."""
        pass

# This is our shared prompt template to ensure consistency across all providers
SYSTEM_PROMPT = """
You are an expert recruitment assistant. Your task is to analyze a job description and extract key information for a high-precision matching system.
Return ONLY a valid JSON object with the following keys:
- "skills": A flat list of specific technical skills, tools, and hard competencies mentioned.
- "requirements": A list of key responsibilities or qualitative requirements (e.g., "leadership", "customer-facing").
- "summary": A concise, professional summary of the role (2-3 sentences) that captures the essence of the position.

Do not include any conversational text, markdown formatting (like ```json), or explanations.
"""

class OpenAIProvider(BaseAIProvider):
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def extract_structured_data(self, text: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o", # or gpt-3.5-turbo
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"} # Ensures valid JSON
            )
            content = response.choices[0].message.content
            if content is None:
                return {"skills": [], "summary": ""}
            return json.loads(content)
        except Exception as e:
            print(f"[OpenAI Error] {e}")
            return {"skills": [], "summary": ""}

    def generate_embedding(self, text: str) -> list:
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"[OpenAI Embedding Error] {e}")
            return []

class ClaudeProvider(BaseAIProvider):
    def __init__(self, api_key=None):
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def extract_structured_data(self, text: str) -> dict:
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text}]
            )
            # Claude may return content as a list of block objects, so extract text safely.
            content = None
            if hasattr(message, "content") and message.content:
                first_block = message.content[0]
                if isinstance(first_block, dict):
                    content = first_block.get("text") or first_block.get("value") or first_block.get("content")
                else:
                    content = (
                        getattr(first_block, "text", None)
                        or getattr(first_block, "value", None)
                        or getattr(first_block, "content", None)
                    )
                if isinstance(content, list):
                    content = "".join(
                        str(getattr(block, "text", None) or getattr(block, "value", None) or getattr(block, "content", ""))
                        if not isinstance(block, dict)
                        else str(block.get("text") or block.get("value") or block.get("content", ""))
                        for block in content
                    )
            if content is None:
                return {"skills": [], "summary": ""}
            return json.loads(content)
        except Exception as e:
            print(f"[Claude Error] {e}")
            return {"skills": [], "summary": ""}

    def generate_embedding(self, text: str) -> list:
        # Note: Anthropic does not have a direct embedding endpoint like OpenAI.
        # Usually, you'd use an OpenAI model or a local model for embeddings 
        # even if using Claude for extraction. For now, we'll return empty or handle via AIEngine logic.
        print("[Claude] Embedding not natively supported via Anthropic API. Use another provider.")
        return []

class GeminiProvider(BaseAIProvider):
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = 'gemini-3-flash'

    def extract_structured_data(self, text: str, temp: float) -> dict:
        try:
            # We instruct Gemini to return JSON
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=f"{SYSTEM_PROMPT}\n\nText: {text}",
                config=types.GenerateContentConfig(
                    temperature=0.1
                )
            )
            content = response.text
            if content is None:
                return {"skills": [], "summary": ""}
            return json.loads(content)
        except Exception as e:
            print(f"[Gemini Error] {e}")
            return {"skills": [], "summary": ""}

    def generate_embedding(self, text: str) -> list:
        try:
            result = self.client.models.embed_content(model="text-embedding-004", contents=text)
            embeddings = result.embeddings or []
            return [emb.values for emb in embeddings if emb is not None and getattr(emb, 'values', None) is not None]
        except Exception as e:
            print(f"[Gemini Embedding Error] {e}")
            return []

class OpenRouterProvider(BaseAIProvider):
    """Acts as a proxy to various models via OpenRouter."""
    def __init__(self, api_key=None):
        # OpenRouter is OpenAI-compatible
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.getenv("OPENROUTER_API_KEY")
        )

    def extract_structured_data(self, text: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model="anthropic/claude-3.5-sonnet", # Example model via OpenRouter
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            if content is None:
                return {"skills": [], "summary": ""}
            return json.loads(content)
        except Exception as e:
            print(f"[OpenRouter Error] {e}")
            return {"skills": [], "summary": ""}

    def generate_embedding(self, text: str) -> list:
        # OpenRouter typically routes to OpenAI-compatible embedding endpoints
        try:
            response = self.client.embeddings.create(
                input=text,
                model="openai/text-embedding-3-small" 
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"[OpenRouter Embedding Error] {e}")
            return []

class LMStudioProvider(BaseAIProvider):
    """Local provider using LM Studio's OpenAI-compatible local server."""
    def __init__(self, base_url="http://localhost", port='1234', api_key="lm-studio"):
        self.client = OpenAI(base_url=f'{base_url}:{port}/v1', api_key=api_key)
    def extract_structured_data(self, text: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=os.getenv('EXTRACTION_MODEL', 'local-model'), # LM Studio usually ignores this and uses the loaded model
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            if content is None:
                return {"skills": [], "summary": ""}
            return json.loads(content)
        except Exception as e:
            print(f"[LM Studio Error] {e}")
            return {"skills": [], "summary": ""}

    def generate_embedding(self, text: str) -> list:
        try:
            response = self.client.embeddings.create(
                input=text,
                model="local-model"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"[LM Studio Embedding Error] {e}")
            return []

class AIEngine:
    """The main controller that manages multiple AI providers."""
    def __init__(self, default_provider_name: str):
        self.default_provider_name = default_provider_name.lower()
        self._providers = {}
        # Initialize the default provider immediately
        self._get_provider(self.default_provider_name)

    def _get_provider(self, name: str) -> BaseAIProvider:
        name = name.lower()
        if name in self._providers:
            return self._providers[name]

        providers = {
            "chatgpt": OpenAIProvider,
            "claude": ClaudeProvider,
            "gemini": GeminiProvider,
            "openrouter": OpenRouterProvider,
            "lm_studio": LMStudioProvider(base_url=os.getenv('LMS_URL', 'http://localhost'),port=os.getenv('LMS_PORT', '1234'),api_key=os.getenv('LMS_API_KEY', 'lm-studio'))
        }
        
        provider_class = providers.get(name)
        if not provider_class:
            raise ValueError(f"Unknown provider: {name}. Choose from {list(providers.keys())}")
        
        instance = provider_class()
        self._providers[name] = instance
        return instance

    def extract(self, text: str, provider_name: str | None = None) -> dict:
        """Uses the specified provider (or default) to extract data."""
        target = provider_name if provider_name else self.default_provider_name
        provider = self._get_provider(target)
        return provider.extract_structured_data(text)

    def embed(self, text: str, provider_name: str | None = None) -> list:
        """Uses the specified provider (or default) to generate embeddings."""
        target = provider_name if provider_name else self.default_provider_name
        provider = self._get_provider(target)
        return provider.generate_embedding(text)