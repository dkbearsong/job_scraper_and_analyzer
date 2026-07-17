# app/ai_limiter.py

import os
import yaml
import asyncio
import time
import sys
from functools import partial
from typing import Dict, Any

async def run_in_thread(func, *args, **kwargs):
    """Runs a synchronous function in an executor thread to prevent event loop blocking."""
    if sys.version_info >= (3, 9):
        return await asyncio.to_thread(func, *args, **kwargs)
    else:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(func, *args, **kwargs))

class AILimiter:
    """
    Manages rate limiting (Requests Per Minute) and concurrency (Semaphore)
    for AI calls, adjusting dynamically based on .env tier flags,
    concurrency mode, and user preference overrides.
    """
    PROVIDER_LIMITS = {
        "gemini": {
            "free": {"rpm": 15, "concurrency": 2},
            "paid": {"rpm": 300, "concurrency": 10}
        },
        "openai": {
            "free": {"rpm": 3, "concurrency": 1},
            "paid": {"rpm": 200, "concurrency": 10}
        },
        "openrouter": {
            "free": {"rpm": 10, "concurrency": 2},
            "paid": {"rpm": 120, "concurrency": 8}
        },
        "grok": {
            "free": {"rpm": 10, "concurrency": 2},
            "paid": {"rpm": 120, "concurrency": 8}
        },
        "groq": {
            "free": {"rpm": 30, "concurrency": 2},
            "paid": {"rpm": 1000, "concurrency": 10}
        },
        "nvidia": {
            "free": {"rpm": 15, "concurrency": 2},
            "paid": {"rpm": 120, "concurrency": 8}
        },
        "cohere": {
            "free": {"rpm": 10, "concurrency": 2},
            "paid": {"rpm": 100, "concurrency": 8}
        },
        "huggingface": {
            "free": {"rpm": 10, "concurrency": 2},
            "paid": {"rpm": 120, "concurrency": 8}
        },
        "lm_studio": {
            "free": {"rpm": 60, "concurrency": 1},  # Defaults to 1 (synchronous)
            "paid": {"rpm": 120, "concurrency": 5}
        },
        "ollama": {
            "free": {"rpm": 60, "concurrency": 1},  # Defaults to 1 (synchronous)
            "paid": {"rpm": 120, "concurrency": 4}
        }
    }

    def __init__(self, stage_name: str, provider_name: str):
        self.stage_name = stage_name.lower().replace(" ", "_")
        self.provider_name = provider_name.lower()
        
        # Load user configuration
        user_config = self._load_user_config()
        
        # Determine mode (concurrent vs synchronous)
        concurrency_mode = os.getenv("CONCURRENCY_MODE", "concurrent").strip().lower()
        is_synchronous = (concurrency_mode == "synchronous")
        
        # Determine tier from env
        # e.g., STAGE_2_EXTRACTION_TIER, STAGE_2_EMBEDDING_TIER, STAGE_6_TIER, STAGE_7_TIER
        tier_env_var = f"{self.stage_name.upper()}_TIER"
        tier = os.getenv(tier_env_var, "free").strip().lower()
        if tier not in ("free", "paid"):
            tier = "free"
        
        # Load default limits for the provider
        provider_limits = self.PROVIDER_LIMITS.get(self.provider_name, {
            "free": {"rpm": 10, "concurrency": 2},
            "paid": {"rpm": 100, "concurrency": 8}
        })
        
        limits = provider_limits.get(tier, provider_limits["free"])
        
        rpm = limits.get("rpm", 10)
        concurrency = limits.get("concurrency", 2)
        
        # LM Studio and Ollama option override in user_preferences.yaml
        if self.provider_name in ("lm_studio", "ollama"):
            pref_key = f"{self.provider_name}_limit"
            if pref_key in user_config:
                val = user_config[pref_key]
                if isinstance(val, dict):
                    rpm = val.get("rpm", rpm)
                    concurrency = val.get("concurrency", concurrency)
                elif isinstance(val, (int, float)):
                    concurrency = int(val)
                    rpm = 9999  # unlimited RPM if only concurrency integer specified
            else:
                # No override found; keep concurrency at default 1
                concurrency = 1
                    
        # Apply synchronous override if set
        if is_synchronous:
            concurrency = 1
            
        self.rpm = rpm
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        self.delay = 60.0 / rpm if rpm > 0 else 0.0
        self.last_request_time = 0.0
        self.lock = asyncio.Lock()
        
        # Token Bucket (Tokens Per Second - TPS) limiting
        tps = limits.get("tps", 0)
        self.tps = tps
        self.tokens = float(tps)
        self.last_leak_time = time.time()
        
        print(f"[RateLimiter] Initialized for stage '{self.stage_name}' with provider '{self.provider_name}' (tier: {tier}, concurrency: {self.concurrency}, rpm: {self.rpm}, tps: {self.tps})")

    def _load_user_config(self) -> dict:
        prefs_path = os.getenv("USER_PREFERENCES_YAML", "user_preferences.yaml")
        if os.path.exists(prefs_path):
            try:
                with open(prefs_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Failed to load user preferences: {e}")
        return {}

    async def wait(self, estimated_tokens: int = 0):
        """Enforces the rate limit (RPM and TPS/Token Bucket) by waiting if needed."""
        async with self.lock:
            # 1. Enforce token rate limit (TPS)
            if self.tps > 0 and estimated_tokens > 0:
                now = time.time()
                elapsed = now - self.last_leak_time
                self.tokens = min(self.tps, self.tokens + elapsed * self.tps)
                self.last_leak_time = now

                if self.tokens < estimated_tokens:
                    needed_tokens = estimated_tokens - self.tokens
                    wait_time = needed_tokens / self.tps
                    wait_time = min(wait_time, 60.0)
                    print(f"[RateLimiter] TPS Limit reached for {self.provider_name}. Throttling request for {wait_time:.2f}s (Needed: {estimated_tokens}, Available: {self.tokens:.1f})")
                    await asyncio.sleep(wait_time)
                    # Refill again after sleeping
                    now = time.time()
                    elapsed = now - self.last_leak_time
                    self.tokens = min(self.tps, self.tokens + elapsed * self.tps)
                    self.last_leak_time = now

                self.tokens = max(0.0, self.tokens - estimated_tokens)

            # 2. Enforce RPM limit
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.delay:
                sleep_time = self.delay - elapsed
                await asyncio.sleep(sleep_time)
            self.last_request_time = time.time()
