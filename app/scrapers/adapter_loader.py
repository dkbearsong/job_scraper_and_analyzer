"""
Adapter Loader: Reads scrapers_config.yaml, dynamically imports adapters,
configures them, and orchestrates execution.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
from typing import Any, Dict, List, Optional

import yaml

from app.scrapers import ScraperAdapter
from app.scrapers.validator import ScrapedDataValidator

logger = logging.getLogger("scraper.loader")


# ─────────────────────────────────────────────────────────
# Built-in adapter registry (short name -> full class path)
# ─────────────────────────────────────────────────────────

BUILTIN_ADAPTERS: Dict[str, str] = {
    "JobSpyAdapter": "app.scrapers.jobspy_adapter.JobSpyAdapter",
    "MicroserviceAdapter": "app.scrapers.microservice_adapter.MicroserviceAdapter",
    "HttpAdapter": "app.scrapers.http_adapter.HttpAdapter",
}


# ─────────────────────────────────────────────────────────
# Adapter Loader
# ─────────────────────────────────────────────────────────

class AdapterLoader:
    """
    Loads, configures, and runs scraper adapters based on YAML configuration.

    Config format (scrapers_config.yaml):

        scrapers:
          - name: my_source
            adapter: app.scrapers.JobSpyAdapter   # dotted path or built-in short name
            enabled: true
            config:
              key: value
    """

    def __init__(
        self,
        config_path: str = "scrapers_config.yaml",
        verbose: bool = False,
    ):
        self.config_path = config_path
        self.verbose = verbose
        self._adapters: List[ScraperAdapter] = []
        self._config: Dict[str, Any] = {}

    @property
    def adapters(self) -> List[ScraperAdapter]:
        """Return list of loaded adapter instances."""
        return list(self._adapters)

    def load_config(self) -> Dict[str, Any]:
        """
        Load and return the scrapers YAML config.

        Returns:
            Parsed config dict, or empty dict if file not found.
        """
        if not os.path.exists(self.config_path):
            logger.warning(
                f"Scrapers config not found at {self.config_path}. "
                "No adapters will be loaded."
            )
            return {}

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f) or {}

        self._config = config
        return config

    def load_adapters(self) -> List[ScraperAdapter]:
        """
        Parse the config and instantiate all enabled adapters.

        Returns:
            List of configured ScraperAdapter instances.
        """
        if not self._config:
            self.load_config()

        scraper_configs = self._config.get("scrapers", [])
        if not scraper_configs:
            logger.info("No scrapers defined in config.")
            return []

        adapters: List[ScraperAdapter] = []

        for entry in scraper_configs:
            name = entry.get("name", "unnamed")
            adapter_path = entry.get("adapter", "")
            enabled = entry.get("enabled", True)
            config = entry.get("config", {})

            if not enabled:
                if self.verbose:
                    logger.info(f"Adapter '{name}' is disabled, skipping.")
                continue

            if not adapter_path:
                logger.warning(f"Adapter '{name}' has no 'adapter' path defined, skipping.")
                continue

            adapter = self._import_adapter(adapter_path, name)
            if adapter is None:
                continue

            # Resolve environment variable references in config
            resolved_config = self._resolve_env_vars(config)
            # Inject the adapter name so get_name() can access it
            resolved_config["name"] = name

            try:
                adapter.configure(resolved_config)
                adapters.append(adapter)
                logger.info(f"Loaded adapter: '{name}' ({adapter_path})")
            except Exception as e:
                logger.error(f"Failed to configure adapter '{name}': {e}")

        self._adapters = adapters
        return adapters

    async def run_all(
        self,
        dp: Any = None,
        validator: Optional[ScrapedDataValidator] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run all loaded adapters, validate results, and return combined job list.

        If no adapters are loaded, calls load_adapters() first.

        Args:
            dp: Optional DataPuller instance (passed to adapters that need it).
            validator: Optional pre-configured validator. If None, a new one is created.

        Returns:
            List of validated job dicts from all adapters.
        """
        if not self._adapters:
            self.load_adapters()

        if not self._adapters:
            logger.info("No adapters to run.")
            return []

        if validator is None:
            validator = ScrapedDataValidator(verbose=self.verbose)

        # Run all adapters and collect results by adapter name
        jobs_by_adapter: Dict[str, List[Dict[str, Any]]] = {}

        for adapter in self._adapters:
            adapter_name = adapter.get_name()
            try:
                logger.info(f"Running adapter: '{adapter_name}'")
                raw_jobs = await adapter.scrape()
                jobs_by_adapter[adapter_name] = raw_jobs or []
                logger.info(
                    f"Adapter '{adapter_name}' returned {len(raw_jobs or [])} jobs."
                )
            except Exception as e:
                logger.error(f"Adapter '{adapter_name}' failed: {e}")
                jobs_by_adapter[adapter_name] = []

        # Validate all results
        validated_jobs = validator.validate_all(jobs_by_adapter)

        return validated_jobs

    # ── Private helpers ──

    def _import_adapter(
        self, adapter_path: str, name: str
    ) -> Optional[ScraperAdapter]:
        """
        Dynamically import and instantiate an adapter class.

        Args:
            adapter_path: Dotted module.class path (e.g. "app.scrapers.JobSpyAdapter")
                          or a built-in short name.
            name: Display name for logging.

        Returns:
            Instantiated adapter, or None if import failed.
        """
        # Resolve built-in short names
        full_path = BUILTIN_ADAPTERS.get(adapter_path, adapter_path)

        try:
            # Split into module path and class name
            module_path, class_name = full_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            adapter_class = getattr(module, class_name)

            if not issubclass(adapter_class, ScraperAdapter):
                logger.error(
                    f"Adapter '{name}' ({full_path}) does not subclass ScraperAdapter."
                )
                return None

            return adapter_class()

        except ImportError as e:
            logger.error(
                f"Could not import adapter '{name}' from '{full_path}': {e}\n"
                f"  Make sure the module path is correct and the module is installed."
            )
            return None
        except AttributeError as e:
            logger.error(
                f"Class '{class_name}' not found in module '{module_path}': {e}"
            )
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading adapter '{name}': {e}")
            return None

    @staticmethod
    def _resolve_env_var_str(value: str) -> str:
        """Resolve ${VAR} references in a single string value."""
        import re
        def _replace_env(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0)) or match.group(0)
        return re.sub(r"\$\{(\w+)\}", _replace_env, value)

    @staticmethod
    def _resolve_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively resolve ${ENV_VAR} references in config values.

        Supports:
            - "${VAR_NAME}" → os.environ.get("VAR_NAME", "")
            - "prefix_${VAR_NAME}_suffix" → "prefix VALUE suffix"
        """
        if not isinstance(config, dict):
            return config

        resolved = {}
        for key, value in config.items():
            if isinstance(value, str) and "${" in value:
                # Replace all ${VAR} patterns
                import re
                def _replace_env(match):
                    var_name = match.group(1)
                    return os.environ.get(var_name, match.group(0)) or match.group(0)
                resolved[key] = re.sub(r"\$\{(\w+)\}", _replace_env, value)
            elif isinstance(value, dict):
                resolved[key] = AdapterLoader._resolve_env_vars(value)
            elif isinstance(value, list):
                resolved[key] = [
                    AdapterLoader._resolve_env_vars(item) if isinstance(item, dict)
                    else (AdapterLoader._resolve_env_var_str(item) if isinstance(item, str) and "${" in item else item)
                    for item in value
                ]
            else:
                resolved[key] = value

        return resolved