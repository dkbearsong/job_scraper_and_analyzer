"""
Data Format Validation Layer for Scraper Adapters.

Validates that job data returned by adapters conforms to the standardized
schema, logs non-compliant data with actionable diagnostics, and auto-fills
defaults for missing recommended fields.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.scrapers import (
    RECOMMENDED_DEFAULTS,
    RECOMMENDED_FIELDS,
    REQUIRED_FIELDS,
)


# ─────────────────────────────────────────────────────────
# Validation Result
# ─────────────────────────────────────────────────────────

@dataclass
class JobValidationResult:
    """Result of validating a single job dict."""
    is_valid: bool = True
    missing_required: List[str] = field(default_factory=list)
    missing_recommended: List[str] = field(default_factory=list)
    empty_values: List[str] = field(default_factory=list)
    type_mismatches: List[str] = field(default_factory=list)
    auto_filled: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.is_valid


@dataclass
class AdapterValidationSummary:
    """Aggregate validation stats for a single adapter."""
    adapter_name: str = ""
    total_received: int = 0
    valid_count: int = 0
    rejected_count: int = 0
    auto_filled_count: int = 0

    def log_summary(self, logger: logging.Logger) -> None:
        logger.info(
            f"[SCRAPER SUMMARY] {self.adapter_name}: "
            f"{self.valid_count} valid, {self.rejected_count} rejected, "
            f"{self.auto_filled_count} auto-filled defaults"
        )


# ─────────────────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────────────────

class ScrapedDataValidator:
    """
    Validates job dicts returned by scraper adapters.

    - Required fields ('title', 'company') must be present and non-empty strings.
    - Recommended fields are auto-filled with defaults if missing.
    - Non-compliant jobs are logged with full context and excluded.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logger = logging.getLogger("scraper.validator")
        self._summaries: Dict[str, AdapterValidationSummary] = {}

    def validate_job(
        self, job: Any, adapter_name: str = "unknown"
    ) -> Tuple[Any, JobValidationResult]:
        """
        Validate a single job dict and auto-fill recommended defaults.

        Args:
            job: Raw job dict from an adapter.
            adapter_name: Name of the adapter that produced this job (for logging).

        Returns:
            Tuple of (cleaned_job, validation_result).
            If the job fails required validation, cleaned_job is the original dict
            but the caller should discard it.
        """
        result = JobValidationResult(is_valid=True)

        if not isinstance(job, dict):
            result.is_valid = False
            result.type_mismatches.append(
                f"Expected dict, got {type(job).__name__}"
            )
            self._log_non_compliant(job, adapter_name, result, "Job is not a dict")
            return job, result

        # ── Check required fields ──
        for field_name in REQUIRED_FIELDS:
            value = job.get(field_name)
            if value is None:
                result.missing_required.append(field_name)
            elif isinstance(value, str) and value.strip() == "":
                result.empty_values.append(field_name)
            elif not isinstance(value, str):
                result.type_mismatches.append(
                    f"'{field_name}' should be str, got {type(value).__name__}"
                )

        if result.missing_required or result.empty_values or result.type_mismatches:
            result.is_valid = False
            self._log_non_compliant(job, adapter_name, result)
            return job, result

        # ── Check recommended fields and auto-fill ──
        for field_name in RECOMMENDED_FIELDS:
            value = job.get(field_name)
            if value is None:
                result.missing_recommended.append(field_name)
                default = RECOMMENDED_DEFAULTS.get(field_name)
                job[field_name] = default
                result.auto_filled.append(field_name)
            elif isinstance(value, str) and value.strip() == "" and field_name not in ("url", "description"):
                # Empty string for non-optional recommended fields — fill default
                default = RECOMMENDED_DEFAULTS.get(field_name)
                if default is not None and default != "":
                    result.auto_filled.append(field_name)
                    job[field_name] = default

        # ── Normalize link/url relationship ──
        if not job.get("url") and job.get("link"):
            job["url"] = job["link"]
        elif not job.get("link") and job.get("url"):
            job["link"] = job["url"]

        return job, result

    def validate_and_filter(
        self, jobs: List[Dict[str, Any]], adapter_name: str = "unknown"
    ) -> List[Dict[str, Any]]:
        """
        Validate a list of jobs from an adapter, filter out non-compliant ones,
        and auto-fill defaults on the rest.

        Args:
            jobs: List of raw job dicts from a single adapter.
            adapter_name: Name of the adapter (for logging).

        Returns:
            List of valid, cleaned job dicts.
        """
        summary = AdapterValidationSummary(
            adapter_name=adapter_name,
            total_received=len(jobs),
        )

        valid_jobs: List[Dict[str, Any]] = []

        for i, job in enumerate(jobs):
            cleaned, result = self.validate_job(job, adapter_name)

            if result.is_valid:
                valid_jobs.append(cleaned)
                summary.valid_count += 1
                if result.auto_filled:
                    summary.auto_filled_count += 1
            else:
                summary.rejected_count += 1

        # Log summary
        summary.log_summary(self.logger)
        self._summaries[adapter_name] = summary

        return valid_jobs

    def validate_all(
        self, jobs_by_adapter: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Validate jobs from multiple adapters.

        Args:
            jobs_by_adapter: Dict mapping adapter_name -> list of job dicts.

        Returns:
            Combined list of all valid, cleaned jobs across all adapters.
        """
        all_valid: List[Dict[str, Any]] = []

        for adapter_name, jobs in jobs_by_adapter.items():
            valid = self.validate_and_filter(jobs, adapter_name)
            all_valid.extend(valid)

        # Log overall summary
        total_valid = sum(s.valid_count for s in self._summaries.values())
        total_rejected = sum(s.rejected_count for s in self._summaries.values())
        self.logger.info(
            f"[SCRAPER VALIDATION TOTAL] {total_valid} valid, "
            f"{total_rejected} rejected across {len(self._summaries)} adapter(s)"
        )

        return all_valid

    def get_summaries(self) -> Dict[str, AdapterValidationSummary]:
        """Return per-adapter validation summaries."""
        return dict(self._summaries)

    # ── Private helpers ──

    def _log_non_compliant(
        self,
        job: Dict[str, Any],
        adapter_name: str,
        result: JobValidationResult,
        extra_message: str = "",
    ) -> None:
        """Log detailed information about a non-compliant job."""
        issues = []

        if extra_message:
            issues.append(extra_message)

        for f in result.missing_required:
            issues.append(f"Missing required field: '{f}'")
        for f in result.empty_values:
            issues.append(f"Empty value for required field: '{f}'")
        for f in result.type_mismatches:
            issues.append(f"Type mismatch: {f}")
        for f in result.missing_recommended:
            issues.append(
                f"Missing recommended field: '{f}' "
                f"(defaulting to '{RECOMMENDED_DEFAULTS.get(f, '')}')"
            )

        # Truncate raw data for readability
        try:
            raw_str = json.dumps(job, default=str, ensure_ascii=False)
            if len(raw_str) > 500:
                raw_str = raw_str[:500] + "... [truncated]"
        except Exception:
            raw_str = str(job)[:500]

        issues_text = "\n    - ".join(issues)

        self.logger.warning(
            f"[SCRAPER VALIDATION] Adapter: \"{adapter_name}\" produced non-compliant job:\n"
            f"  Raw data: {raw_str}\n"
            f"  Issues:\n"
            f"    - {issues_text}\n"
            f"  Suggestion: Ensure the scraper filters out jobs with no title "
            f"before returning results."
        )

        if self.verbose:
            print(
                f"[SCRAPER VALIDATION] Adapter: \"{adapter_name}\" — rejected job "
                f"({', '.join(issues[:3])})"
            )