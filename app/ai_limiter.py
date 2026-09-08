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
            "free": {"rpm": 20, "concurrency": 5},
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
            "free": {"rpm": 5, "concurrency": 1},
            "paid": {"rpm": 100, "concurrency": 8}
        },
        "huggingface": {
            "free": {"rpm": 10, "concurrency": 2},
            "paid": {"rpm": 120, "concurrency": 8}
        },
        "siliconflow": {
            "free": {"rpm": 100, "concurrency": 4},
            "paid": {"rpm": 1000, "concurrency": 20}
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
        provider_defaults = self.PROVIDER_LIMITS.get(self.provider_name, {
            "free": {"rpm": 10, "concurrency": 2},
            "paid": {"rpm": 100, "concurrency": 8}
        })
        
        tier_limits = dict(provider_defaults.get(tier, provider_defaults.get("free", {})))
        
        def _apply_override(target: dict, source: Any):
            if not isinstance(source, dict):
                if isinstance(source, (int, float)):
                    target["concurrency"] = int(source)
                    target["rpm"] = 9999
                    target["requests_per_minute"] = 9999
                return

            if self.provider_name in source and isinstance(source[self.provider_name], dict):
                _apply_override(target, source[self.provider_name])
                return

            if tier in source and isinstance(source[tier], dict):
                _apply_override(target, source[tier])
                return

            for k, v in source.items():
                if k not in ("free", "paid", self.provider_name) and not isinstance(v, dict):
                    target[k] = v

        # 1. Check provider-level overrides in user_config
        for root_key in ("stage_rate_limits", "rate_limits", "ai_rate_limits", "ai_limits"):
            if root_key in user_config and isinstance(user_config[root_key], dict):
                if self.provider_name in user_config[root_key]:
                    _apply_override(tier_limits, user_config[root_key][self.provider_name])

        pref_key = f"{self.provider_name}_limit"
        if pref_key in user_config:
            _apply_override(tier_limits, user_config[pref_key])

        # 2. Check stage-level overrides in user_config (takes precedence over provider-level)
        for root_key in ("stage_rate_limits", "rate_limits", "ai_rate_limits", "ai_limits"):
            if root_key in user_config and isinstance(user_config[root_key], dict):
                if self.stage_name in user_config[root_key]:
                    _apply_override(tier_limits, user_config[root_key][self.stage_name])

        stage_pref_key = f"{self.stage_name}_limit"
        if stage_pref_key in user_config:
            _apply_override(tier_limits, user_config[stage_pref_key])

        if self.stage_name in user_config and isinstance(user_config[self.stage_name], dict):
            _apply_override(tier_limits, user_config[self.stage_name])

        # 3. Environment variable overrides (highest priority)
        env_rpm = os.getenv(f"{self.stage_name.upper()}_REQUESTS_PER_MINUTE") or os.getenv(f"{self.stage_name.upper()}_RPM")
        if env_rpm is not None:
            try:
                tier_limits["requests_per_minute"] = float(env_rpm)
            except ValueError:
                pass
                
        env_tpm = os.getenv(f"{self.stage_name.upper()}_TOKENS_PER_MINUTE") or os.getenv(f"{self.stage_name.upper()}_TPM")
        if env_tpm is not None:
            try:
                tier_limits["tokens_per_minute"] = float(env_tpm)
            except ValueError:
                pass

        env_conc = os.getenv(f"{self.stage_name.upper()}_CONCURRENCY")
        if env_conc is not None:
            try:
                tier_limits["concurrency"] = int(env_conc)
            except ValueError:
                pass

        # Resolve Requests Per Minute (RPM)
        if "requests_per_minute" in tier_limits and tier_limits["requests_per_minute"] is not None:
            rpm = float(tier_limits["requests_per_minute"])
        elif "rpm" in tier_limits and tier_limits["rpm"] is not None:
            rpm = float(tier_limits["rpm"])
        else:
            rpm = 10.0

        # Resolve Tokens Per Minute (TPM) / Tokens Per Second (TPS)
        tpm = 0.0
        tps = 0.0
        if "tokens_per_minute" in tier_limits and tier_limits["tokens_per_minute"] is not None:
            tpm = float(tier_limits["tokens_per_minute"])
            tps = tpm / 60.0 if tpm > 0 else 0.0
        elif "tpm" in tier_limits and tier_limits["tpm"] is not None:
            tpm = float(tier_limits["tpm"])
            tps = tpm / 60.0 if tpm > 0 else 0.0
        elif "tps" in tier_limits and tier_limits["tps"] is not None:
            tps = float(tier_limits["tps"])
            tpm = tps * 60.0

        # Resolve Concurrency
        concurrency = tier_limits.get("concurrency")
        if concurrency is None:
            if self.provider_name in ("lm_studio", "ollama"):
                concurrency = 1
            else:
                concurrency = 2
        else:
            concurrency = int(concurrency)

        # Apply synchronous mode override
        if is_synchronous:
            concurrency = 1

        self.rpm = rpm
        self.tpm = tpm
        self.tps = tps
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        self.delay = 60.0 / rpm if rpm > 0 else 0.0
        self._original_delay = self.delay
        self._throttled = False
        self.last_request_time = 0.0
        self.lock = asyncio.Lock()

        # Token Bucket (Tokens Per Second / Tokens Per Minute) limiting
        self.max_tokens = max(self.tps, self.tpm) if self.tpm > 0 else self.tps
        self.tokens = float(self.max_tokens)
        self.last_leak_time = time.time()

        tpm_str = f"{self.tpm:.0f}" if self.tpm > 0 else "unlimited"
        print(f"[RateLimiter] Initialized for stage '{self.stage_name}' with provider '{self.provider_name}' (tier: {tier}, concurrency: {self.concurrency}, rpm: {self.rpm}, tpm: {tpm_str}, tps: {self.tps:.2f})")

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
        """Enforces the rate limit (RPM and TPS/Token Bucket) based strictly on configured tier parameters."""
        sleep_time = 0.0
        
        async with self.lock:
            # 1. Enforce token rate limit (TPS / TPM)
            if self.tps > 0 and estimated_tokens > 0:
                now = time.time()
                elapsed = now - self.last_leak_time
                max_cap = getattr(self, "max_tokens", self.tps)
                self.tokens = min(max_cap, self.tokens + elapsed * self.tps)
                self.last_leak_time = now

                if self.tokens < estimated_tokens:
                    needed_tokens = estimated_tokens - self.tokens
                    wait_time = needed_tokens / self.tps
                    sleep_time = max(sleep_time, min(wait_time, 60.0))

                self.tokens = max(0.0, self.tokens - estimated_tokens)

            # 2. Enforce RPM limit based strictly on tier parameters
            now = time.time()
            if self.delay > 0:
                if self.last_request_time < now:
                    self.last_request_time = now
                else:
                    sleep_time = max(sleep_time, self.last_request_time - now)
                
                # Advance last_request_time by self.delay for this request slot
                self.last_request_time += self.delay
            else:
                self.last_request_time = now

        # Sleep outside the lock to prevent blocking/starving other tasks that are computing their rates
        if sleep_time > 0:
            if self.tps > 0 and estimated_tokens > 0 and sleep_time > 1.0:
                tpm_str = f" (TPM: {self.tpm:.0f})" if getattr(self, "tpm", 0) > 0 else ""
                print(f"[RateLimiter] [{self.stage_name}/{self.provider_name}] Token limit reached{tpm_str}. Pacing request for {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)

    async def execute(self, func, *args, est_tokens: int = 0, use_semaphore: bool = True, **kwargs):
        """
        Executes a function with rate limiting based on tier parameters,
        with isolated exponential backoff that applies ONLY to the failing call.
        """
        from app.ai_engine import is_rate_limit_exception, is_transient_ai_exception, RateLimitError
        import random

        initial_backoff = float(os.getenv("AI_INITIAL_BACKOFF", "2.0"))
        max_backoff = float(os.getenv("AI_MAX_BACKOFF", "60.0"))
        current_backoff = initial_backoff
        rate_retries = 0
        max_rate_retries = int(os.getenv("AI_MAX_RATE_RETRIES", "6"))

        general_retries = 0
        max_general_retries = int(os.getenv("AI_MAX_RETRIES", "2"))
        retry_delay = float(os.getenv("AI_RETRY_DELAY", "5.0"))

        self.last_input_tokens = est_tokens
        while True:
            # 1. Wait for our rate limiter window (paced at steady-state tier rate)
            await self.wait(est_tokens)

            try:
                # 2. Acquire semaphore and run the task
                t0 = time.time()
                if use_semaphore:
                    async with self.semaphore:
                        if asyncio.iscoroutinefunction(func):
                            res = await func(*args, **kwargs)
                        else:
                            from app.ai_limiter import run_in_thread
                            res = await run_in_thread(func, *args, **kwargs)
                else:
                    if asyncio.iscoroutinefunction(func):
                        res = await func(*args, **kwargs)
                    else:
                        from app.ai_limiter import run_in_thread
                        res = await run_in_thread(func, *args, **kwargs)
                
                # Success: call finishes cleanly without mutating global delay
                return res

            except Exception as e:
                is_rate_limit = isinstance(e, RateLimitError) or is_rate_limit_exception(e)
                is_transient = is_rate_limit or is_transient_ai_exception(e)

                if is_transient:
                    rate_retries += 1
                    err_type = "Rate limit (429)" if is_rate_limit else "Transient server overload (503/server error)"
                    if rate_retries > max_rate_retries:
                        print(f"\n[RateLimiter] [{self.stage_name}/{self.provider_name}] Maximum retries ({max_rate_retries}) reached for {err_type} (Input tokens: {est_tokens}). Propagating error: {e}")
                        raise

                    # Parse suggested retry delay if provided by the SDK (e.g. Gemini RetryInfo '5s' or 'retry in 5.08s')
                    suggested_delay = None
                    import re
                    retry_match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(e), re.IGNORECASE) or re.search(r"retryDelay': '(\d+)s'", str(e))
                    if retry_match:
                        try:
                            suggested_delay = float(retry_match.group(1)) + 1.0
                        except ValueError:
                            pass

                    # Exponential backoff applied ONLY to this specific failing call
                    sleep_time = suggested_delay if suggested_delay else (current_backoff + (random.random() * 0.5 * current_backoff))
                    print(f"\n[RateLimiter] [{self.stage_name}/{self.provider_name}] {err_type}: {e} (Input tokens: {est_tokens}). Backoff retry {rate_retries}/{max_rate_retries} for this call in {sleep_time:.2f}s...")
                    await asyncio.sleep(sleep_time)
                    current_backoff = min(max_backoff, current_backoff * 2.0)
                else:
                    general_retries += 1
                    if general_retries > max_general_retries:
                        print(f"\n[RateLimiter] [{self.stage_name}/{self.provider_name}] Maximum retries ({max_general_retries}) reached for error: {e} (Input tokens: {est_tokens}). Propagating error.")
                        raise
                    print(f"\n[RateLimiter] [{self.stage_name}/{self.provider_name}] AI processing error: {e} (Input tokens: {est_tokens}). Retrying attempt {general_retries}/{max_general_retries} in {retry_delay:.1f}s...")
                    await asyncio.sleep(retry_delay)



class ProgressTracker:
    def __init__(self, total: int, prefix: str = "Processing"):
        self.total = total
        self.completed = 0
        self.prefix = prefix
        self.animation_task = None
        self._lock = asyncio.Lock()

    async def start(self):
        if self.total > 0:
            self.animation_task = asyncio.create_task(self._animate())
        else:
            self._print_progress(0, 3)
            import sys
            sys.stdout.write("\n")
            sys.stdout.flush()

    async def increment(self):
        if self.total <= 0:
            return
        async with self._lock:
            self.completed = min(self.total, self.completed + 1)

    async def _animate(self):
        dot_count = 0
        try:
            while self.completed < self.total:
                dot_count = (dot_count + 1) % 4
                self._print_progress(self.completed, dot_count)
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
        finally:
            self._print_progress(self.completed, 3)
            import sys
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _print_progress(self, completed: int, dot_count: int):
        import sys
        dots = "." * dot_count + " " * (3 - dot_count)
        sys.stdout.write(f"\r{self.prefix}: {completed}/{self.total} jobs{dots}")
        sys.stdout.flush()

    def stop(self):
        if self.animation_task and not self.animation_task.done():
            self.animation_task.cancel()
