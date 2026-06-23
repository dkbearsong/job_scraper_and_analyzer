"""
LLM Usage Tracker

Tracks all LLM API calls made throughout the pipeline, recording:
- The provider and model used
- The operation type (extraction, embedding, classification, reranking)
- Input and output token counts
- A cumulative summary of all usage

This is a module-level singleton — import and use `usage_tracker` anywhere.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import datetime


@dataclass
class LLMUsageRecord:
    """A single LLM API call record."""
    provider: str
    model: str
    operation: str          # 'extraction', 'embedding', 'classification', 'reranking'
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    timestamp: str = ""
    context: str = ""       # optional description (e.g. job title, document name)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().isoformat()


class LLMUsageTracker:
    """
    Singleton tracker that accumulates all LLM usage records.
    Provides a summary at the end of the pipeline.
    """

    _instance: Optional["LLMUsageTracker"] = None

    def __new__(cls) -> "LLMUsageTracker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._records = []
            cls._instance._enabled = True
        return cls._instance

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def reset(self) -> None:
        """Clear all records (useful for testing)."""
        self._records = []

    def record(self,
               provider: str,
               model: str,
               operation: str,
               input_tokens: int = 0,
               output_tokens: int = 0,
               total_tokens: int = 0,
               context: str = "") -> None:
        """Record a single LLM API call."""
        if not self._enabled:
            return
        record = LLMUsageRecord(
            provider=provider,
            model=model,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens or (input_tokens + output_tokens),
            context=context,
        )
        self._records.append(record)

    def record_from_response(self,
                              provider: str,
                              model: str,
                              operation: str,
                              response: Any,
                              context: str = "") -> None:
        """
        Record usage by extracting token counts from a provider's API response.
        The extraction logic handles OpenAI, Anthropic, Gemini, and other
        OpenAI-compatible response formats.
        """
        input_tokens, output_tokens, total_tokens = 0, 0, 0

        if response is None:
            self.record(provider, model, operation, context=context)
            return

        # Determine which provider format this response uses by checking
        # the attributes available on the usage object.
        usage_obj = getattr(response, "usage", None)
        usage_meta = getattr(response, "usage_metadata", None)

        try:
            if usage_meta is not None:
                # ── Gemini (google.genai) ──
                input_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
                output_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0
                total_tokens = getattr(usage_meta, "total_token_count", 0) or 0
            elif usage_obj is not None:
                # Check for Anthropic-style fields first
                inp = getattr(usage_obj, "input_tokens", None)
                if inp is not None:
                    # ── Anthropic ──
                    input_tokens = inp or 0
                    output_tokens = getattr(usage_obj, "output_tokens", 0) or 0
                    total_tokens = input_tokens + output_tokens
                else:
                    # ── OpenAI / OpenAI-compatible (LM Studio, OpenRouter, Ollama) ──
                    input_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
                    output_tokens = getattr(usage_obj, "completion_tokens", 0) or 0
                    total_tokens = getattr(usage_obj, "total_tokens", 0) or 0
        except Exception:
            pass

        self.record(
            provider=provider,
            model=model,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens or (input_tokens + output_tokens),
            context=context,
        )

    @property
    def all_records(self) -> list[LLMUsageRecord]:
        """Return all accumulated records."""
        return list(self._records)

    @property
    def total_calls(self) -> int:
        return len(self._records)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self._records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self._records)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self._records)

    def summary_by_provider(self) -> dict:
        """
        Return a dict keyed by provider name with aggregated stats.
        """
        by_provider: dict = {}
        for r in self._records:
            key = f"{r.provider}/{r.model}"
            if key not in by_provider:
                by_provider[key] = {
                    "provider": r.provider,
                    "model": r.model,
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "operations": set(),
                }
            by_provider[key]["calls"] += 1
            by_provider[key]["input_tokens"] += r.input_tokens
            by_provider[key]["output_tokens"] += r.output_tokens
            by_provider[key]["total_tokens"] += r.total_tokens
            by_provider[key]["operations"].add(r.operation)

        # Convert sets to sorted lists for JSON-serializable output
        for v in by_provider.values():
            v["operations"] = sorted(v["operations"])

        return by_provider

    def summary_by_operation(self) -> dict:
        """Return a dict keyed by operation type with aggregated stats."""
        by_op: dict = {}
        for r in self._records:
            if r.operation not in by_op:
                by_op[r.operation] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }
            by_op[r.operation]["calls"] += 1
            by_op[r.operation]["input_tokens"] += r.input_tokens
            by_op[r.operation]["output_tokens"] += r.output_tokens
            by_op[r.operation]["total_tokens"] += r.total_tokens
        return by_op

    def full_summary(self) -> dict:
        """Return a complete summary dict suitable for logging / display."""
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "by_provider": self.summary_by_provider(),
            "by_operation": self.summary_by_operation(),
        }

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        summary = self.full_summary()
        print("\n" + "=" * 60)
        print("LLM USAGE TRACKING SUMMARY")
        print("=" * 60)
        print(f"Total API Calls:    {summary['total_calls']}")
        print(f"Total Input Tokens: {summary['total_input_tokens']:,}")
        print(f"Total Output Tokens:{summary['total_output_tokens']:,}")
        print(f"Total Tokens:       {summary['total_tokens']:,}")
        print("-" * 60)

        if summary["by_provider"]:
            print("By Provider / Model:")
            for key, data in sorted(summary["by_provider"].items()):
                print(f"  {key}:")
                print(f"    Calls:     {data['calls']}")
                print(f"    Input:     {data['input_tokens']:,} tokens")
                print(f"    Output:    {data['output_tokens']:,} tokens")
                print(f"    Total:     {data['total_tokens']:,} tokens")
                print(f"    Ops:       {', '.join(data['operations'])}")
            print("-" * 60)

        if summary["by_operation"]:
            print("By Operation:")
            for op, data in sorted(summary["by_operation"].items()):
                print(f"  {op}:")
                print(f"    Calls:     {data['calls']}")
                print(f"    Input:     {data['input_tokens']:,} tokens")
                print(f"    Output:    {data['output_tokens']:,} tokens")
                print(f"    Total:     {data['total_tokens']:,} tokens")

        print("=" * 60)


# Module-level singleton
usage_tracker = LLMUsageTracker()