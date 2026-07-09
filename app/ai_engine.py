# app/ai_engine.py

from abc import ABC, abstractmethod
import os
import json
# Import real libraries
from openai import OpenAI
from anthropic import Anthropic
import google.genai as genai
from google.genai import types

# LLM Usage Tracking
from app.llm_usage_tracker import usage_tracker

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

    def load_model(self) -> None:
        """
        Load the LLM model into memory.
        Default no-op — override in providers that support dynamic loading/unloading.
        """
        pass

    def unload_model(self) -> None:
        """
        Unload / release the LLM model from memory.
        Default no-op — override in providers that support dynamic loading/unloading.
        """
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
        self._provider_name = "openai"


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
            usage_tracker.record_from_response(
                provider=self._provider_name, model="gpt-4o",
                operation="extraction", response=response,
                context=f"extract_structured_data ({len(text)} chars)"
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
            usage_tracker.record_from_response(
                provider=self._provider_name, model="text-embedding-3-small",
                operation="embedding", response=response,
                context=f"generate_embedding ({len(text)} chars)"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"[OpenAI Embedding Error] {e}")
            return []

class ClaudeProvider(BaseAIProvider):
    def __init__(self, api_key=None):
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self._provider_name = "claude"
        self._model = "claude-3-5-sonnet-20240620"


    def extract_structured_data(self, text: str) -> dict:
        try:
            message = self.client.messages.create(
                model=self._model,
                max_tokens=1000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text}]
            )
            usage_tracker.record_from_response(
                provider=self._provider_name, model=self._model,
                operation="extraction", response=message,
                context=f"extract_structured_data ({len(text)} chars)"
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
        self._provider_name = "gemini"
        self.model_name = 'gemini-2.0-flash-exp'

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
            usage_tracker.record_from_response(
                provider=self._provider_name, model=self.model_name,
                operation="extraction", response=response,
                context=f"extract_structured_data ({len(text)} chars)"
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
            usage_tracker.record(
                provider=self._provider_name, model="text-embedding-004",
                operation="embedding",
                context=f"generate_embedding ({len(text)} chars)"
            )
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
        self._provider_name = "openrouter"
        self._extraction_model = "anthropic/claude-3.5-sonnet"
        self._embedding_model = "openai/text-embedding-3-small"


    def extract_structured_data(self, text: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=self._extraction_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"}
            )
            usage_tracker.record_from_response(
                provider=self._provider_name, model=self._extraction_model,
                operation="extraction", response=response,
                context=f"extract_structured_data ({len(text)} chars)"
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
                model=self._embedding_model
            )
            usage_tracker.record_from_response(
                provider=self._provider_name, model=self._embedding_model,
                operation="embedding", response=response,
                context=f"generate_embedding ({len(text)} chars)"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"[OpenRouter Embedding Error] {e}")
            return []

class LMStudioProvider(BaseAIProvider):
    """Local provider using LM Studio's OpenAI-compatible local server."""
    def __init__(self, base_url="http://localhost", port='1234', api_key="lm-studio", extraction_model="local-model", embeddings_model="local-model"):
        self.base_url = base_url
        self.port = port
        self._api_base = f'{base_url}:{port}/v1'
        self.client = OpenAI(base_url=self._api_base, api_key=api_key)
        self._provider_name = "lm_studio"
        self.extraction_model = extraction_model
        self.embeddings_model = embeddings_model
        self._model_loaded = False
        # Import requests here so the module can be used without requests installed
        import requests as _req
        self._http = _req

    def _get_loaded_model_ids(self) -> list:
        urls = [
            f"{self.base_url}:{self.port}/api/v1/models",
            f"{self._api_base}/models"
        ]
        for url in urls:
            try:
                resp = self._http.get(url, timeout=5)
                if resp.status_code == 200:
                    models = resp.json().get("data", [])
                    return [m.get("id") for m in models if m.get("id")]
            except Exception:
                pass
        return []

    def _load_single_model(self, model_name: str) -> None:
        if not model_name:
            return

        # ── Clear existing models from memory first ──
        loaded_ids = self._get_loaded_model_ids()
        if loaded_ids:
            if len(loaded_ids) == 1 and loaded_ids[0] == model_name:
                print(f"[LM Studio] Model '{model_name}' is already loaded and is the only model. Skipping clear/load.")
                return

            print(f"[LM Studio] Clearing existing model(s) from memory: {loaded_ids}")
            for m_id in loaded_ids:
                self._unload_single_model(m_id)

            # Wait for all models to unload
            import time
            poll_interval = 2
            deadline = time.time() + 45
            unloaded_all = False
            while time.time() < deadline:
                current_loaded = self._get_loaded_model_ids()
                if not current_loaded:
                    print("[LM Studio] All models successfully unloaded.")
                    unloaded_all = True
                    break
                time.sleep(poll_interval)
            
            if not unloaded_all:
                print("[LM Studio] Warning: Some models did not unload within 45 seconds. Attempting to proceed anyway.")

        print(f"[LM Studio] Loading model '{model_name}' ...")
        urls = [
            f"{self.base_url}:{self.port}/api/v1/models/load",
            f"{self._api_base}/models/load"
        ]
        success = False
        last_err = None
        for url in urls:
            try:
                resp = self._http.post(
                    url,
                    json={
                        "model": model_name
                    },
                    timeout=30,
                )
                if resp.status_code == 200:
                    print(f"[LM Studio] Model '{model_name}' load request accepted via {url}.")
                    success = True
                    break
                else:
                    last_err = f"HTTP {resp.status_code}: {resp.text}"
            except Exception as e:
                last_err = str(e)
        if not success:
            print(f"[LM Studio] Error/Warning sending load request for '{model_name}': {last_err}")

    def _unload_single_model(self, model_name: str) -> None:
        if not model_name:
            return
        print(f"[LM Studio] Unloading model '{model_name}' ...")
        urls = [
            f"{self.base_url}:{self.port}/api/v1/models/unload",
            f"{self._api_base}/models/unload"
        ]
        success = False
        last_err = None
        for url in urls:
            try:
                resp = self._http.post(
                    url,
                    json={
                        "instance_id": model_name
                    },
                    timeout=30,
                )
                if resp.status_code == 200:
                    print(f"[LM Studio] Model '{model_name}' unload request accepted via {url}.")
                    success = True
                    break
                else:
                    last_err = f"HTTP {resp.status_code}: {resp.text}"
            except Exception as e:
                last_err = str(e)
        if not success:
            print(f"[LM Studio] Error/Warning sending unload request for '{model_name}': {last_err}")

    def load_model(self) -> None:
        """
        Load the extraction model into LM Studio via its HTTP API.
        """
        self._load_single_model(self.extraction_model)
        self._model_loaded = True

    def unload_model(self) -> None:
        """
        Unload the extraction model from LM Studio memory via its HTTP API.
        """
        self._unload_single_model(self.extraction_model)
        self._model_loaded = False

    def wait_for_model_loaded(self, timeout: int = 360, poll_interval: int = 5) -> bool:
        """
        Poll LM Studio's models endpoint until the extraction model appears
        as loaded, or until *timeout* seconds have elapsed.
        """
        import time
        model_name = self.extraction_model
        deadline = time.time() + timeout
        while time.time() < deadline:
            loaded_ids = self._get_loaded_model_ids()
            if model_name in loaded_ids:
                print(f"[LM Studio] Model '{model_name}' is now loaded and ready.")
                self._model_loaded = True
                return True
            time.sleep(poll_interval)
        print(f"[LM Studio] Timeout waiting for model '{model_name}' after {timeout}s.")
        return False

    def load_model_by_name(self, model_name: str) -> None:
        self._load_single_model(model_name)

    def unload_model_by_name(self, model_name: str) -> None:
        self._unload_single_model(model_name)

    def wait_for_model_loaded_by_name(self, model_name: str, timeout: int = 180, poll_interval: int = 5) -> bool:
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            loaded_ids = self._get_loaded_model_ids()
            if model_name in loaded_ids:
                print(f"[LM Studio] Model '{model_name}' is loaded and ready.")
                return True
            time.sleep(poll_interval)
        print(f"[LM Studio] Timeout waiting for model '{model_name}' after {timeout}s.")
        return False

    def wait_for_model_unloaded_by_name(self, model_name: str, timeout: int = 180, poll_interval: int = 5) -> bool:
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            loaded_ids = self._get_loaded_model_ids()
            if model_name not in loaded_ids:
                print(f"[LM Studio] Model '{model_name}' has been successfully unloaded.")
                return True
            print(f"[LM Studio] Waiting for model '{model_name}' to unload ...")
            time.sleep(poll_interval)
        print(f"[LM Studio] Timeout waiting for model '{model_name}' to unload after {timeout}s.")
        return False



    def extract_structured_data(self, text: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=self.extraction_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ]
            )
            usage_tracker.record_from_response(
                provider=self._provider_name, model=self.extraction_model,
                operation="extraction", response=response,
                context=f"extract_structured_data ({len(text)} chars)"
            )
            content = response.choices[0].message.content
            if content is None:
                return {"skills": [], "summary": ""}
            return json.loads(content)
        except Exception as e:
            print(f"[LM Studio Error] {e}")
            return {"skills": [], "summary": ""}

    def generate_embedding(self, text: str) -> list:
        """
        Generate an embedding via LM Studio's /v1/embeddings endpoint.
        Uses raw HTTP requests so we can control the payload format regardless
        of which model is currently loaded in the server.
        """
        url = f"{self.base_url}:{self.port}/v1/embeddings"
        if self.embeddings_model:
            model_name = self.embeddings_model
        else:
            model_name = "local-model"
        payload = {
            "model": model_name,
            "input": text
        }
        try:
            resp = self._http.post(url, json=payload, timeout=120)
            if resp.status_code != 200:
                print(f"[LM Studio Embedding Error] HTTP {resp.status_code}: {resp.text[:500]}")
                return []
            data = resp.json()
            embedding = data["data"][0]["embedding"]
            # record usage via the response dictionary to extract actual tokens if present
            usage_tracker.record_from_response(
                provider=self._provider_name, model=model_name,
                operation="embedding", response=data,
                context=f"generate_embedding ({len(text)} chars)"
            )
            return embedding
        except Exception as e:
            print(f"[LM Studio Embedding Error] {e}")
            return []

class AIEngine:
    """The main controller that manages multiple AI providers."""
    def __init__(self, default_provider_name: str, extraction_model: str = "local-model", embeddings_model: str = "local-model"):
        self.default_provider_name = default_provider_name.lower()
        self.extraction_model = extraction_model
        self.embeddings_model = embeddings_model
        self._providers = {}
        # Initialize the default provider immediately
        self._get_provider(self.default_provider_name)

    def _get_provider(self, name: str) -> BaseAIProvider:
        name = name.lower()
        if name in self._providers:
            return self._providers[name]

        provider_map = {
            "chatgpt": OpenAIProvider,
            "claude": ClaudeProvider,
            "gemini": GeminiProvider,
            "openrouter": OpenRouterProvider,
            "lm_studio": LMStudioProvider,
        }
        
        provider_class = provider_map.get(name)
        if not provider_class:
            raise ValueError(f"Unknown provider: {name}. Choose from {list(provider_map.keys())}")
        
        # Handle LM Studio's custom constructor arguments
        if name == "lm_studio":
            instance = provider_class(
                base_url=os.getenv('LMS_URL', 'http://localhost'),
                port=os.getenv('LMS_PORT', '1234'),
                api_key=os.getenv('LMS_API_KEY', 'lm-studio'),
                extraction_model=self.extraction_model,
                embeddings_model=self.embeddings_model
            )
        else:
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

    def load_model(self, provider_name: str | None = None, model_name: str | None = None) -> None:
        """
        Instruct the provider to load its model into memory.
        For LM Studio this sends a load request to the local server.
        Other providers treat this as a no-op.
        """
        target = provider_name if provider_name else self.default_provider_name
        provider = self._get_provider(target)
        if isinstance(provider, LMStudioProvider) and model_name:
            provider.load_model_by_name(model_name)
        else:
            provider.load_model()

    def unload_model(self, provider_name: str | None = None, model_name: str | None = None) -> None:
        """
        Instruct the provider to unload its model from memory.
        For LM Studio this sends an unload request to the local server.
        Other providers treat this as a no-op.
        """
        target = provider_name if provider_name else self.default_provider_name
        provider = self._get_provider(target)
        if isinstance(provider, LMStudioProvider) and model_name:
            provider.unload_model_by_name(model_name)
        else:
            provider.unload_model()

    def wait_for_model_loaded(self, provider_name: str | None = None,
                               timeout: int = 180, poll_interval: int = 5,
                               model_name: str | None = None) -> bool:
        """
        Wait for the provider's model to become available / ready.
        For LM Studio this polls the /v1/models endpoint.

        Args:
            provider_name: Provider to check. Defaults to the engine's default.
            timeout: Maximum seconds to wait.
            poll_interval: Seconds between polls.
            model_name: Optional specific model to wait for.

        Returns:
            True if model became available, False otherwise.
        """
        target = provider_name if provider_name else self.default_provider_name
        provider = self._get_provider(target)
        # Only LMStudioProvider has this method — others always return True
        if isinstance(provider, LMStudioProvider):
            if model_name:
                return provider.wait_for_model_loaded_by_name(model_name, timeout=timeout, poll_interval=poll_interval)
            return provider.wait_for_model_loaded(timeout=timeout, poll_interval=poll_interval)
        return True

    def wait_for_model_unloaded(self, provider_name: str | None = None,
                                 timeout: int = 180, poll_interval: int = 5,
                                 model_name: str | None = None) -> bool:
        """
        Wait for the provider's model to become unloaded / released.
        For LM Studio this polls the /v1/models endpoint.
        """
        target = provider_name if provider_name else self.default_provider_name
        provider = self._get_provider(target)
        if isinstance(provider, LMStudioProvider):
            target_model = model_name or provider.extraction_model
            if target_model:
                return provider.wait_for_model_unloaded_by_name(target_model, timeout=timeout, poll_interval=poll_interval)
        return True
