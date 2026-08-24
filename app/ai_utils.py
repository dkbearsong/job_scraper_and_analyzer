import os
import json
from app.ai_engine import AIEngine
from app.config_utils import _load_user_config
from app.text_engine import TextProcessor

def _provider_is_lm_studio(ai: AIEngine, provider_name: str | None = None) -> bool:
    """
    Check whether the given (or default) provider is LM Studio.

    Args:
        ai: The AIEngine instance.
        provider_name: Optional override; if None uses the engine's default.

    Returns:
        True if the resolved provider is LM Studio.
    """
    target = (provider_name or ai.default_provider_name).lower()
    return target == "lm_studio"


def _load_model_if_lm_studio(ai: AIEngine, provider_name: str | None = None,
                              stage_label: str = "", model_name: str | None = None) -> None:
    """
    If the resolved provider is LM Studio, send a model load request and wait for
    the model to become ready.  Does nothing for other providers.

    Args:
        ai: The AIEngine instance.
        provider_name: Optional provider override.
        stage_label: Human-readable label for log messages.
        model_name: Optional model override.
    """
    if not _provider_is_lm_studio(ai, provider_name):
        return
    tag = f" [{stage_label}]" if stage_label else ""
    print(f"[Model Mgmt]{tag} Loading LM Studio model ...")
    ai.load_model(provider_name=provider_name, model_name=model_name)
    loaded = ai.wait_for_model_loaded(provider_name=provider_name, model_name=model_name)
    if not loaded:
        print(f"[Model Mgmt]{tag} WARNING: model may not be fully loaded yet.")


def _unload_model_if_lm_studio(ai: AIEngine, provider_name: str | None = None,
                                stage_label: str = "", model_name: str | None = None) -> None:
    """
    If the resolved provider is LM Studio, send a model unload request and wait for
    unloading to complete.  Does nothing for other providers.

    Args:
        ai: The AIEngine instance.
        provider_name: Optional provider override.
        stage_label: Human-readable label for log messages.
        model_name: Optional model override.
    """
    if not _provider_is_lm_studio(ai, provider_name):
        return
    tag = f" [{stage_label}]" if stage_label else ""
    print(f"[Model Mgmt]{tag} Unloading LM Studio model ...")
    ai.unload_model(provider_name=provider_name, model_name=model_name)
    unloaded = ai.wait_for_model_unloaded(provider_name=provider_name, model_name=model_name)
    if not unloaded:
        print(f"[Model Mgmt]{tag} WARNING: model may not be fully unloaded yet.")


def call_llm_for_extraction(ai: AIEngine, text: str, provider_name: str | None = None) -> dict:
    """Uses the provided AI engine to extract structured data."""
    if provider_name is None:
        return ai.extract(text)
    return ai.extract(text, provider_name=provider_name)


def generate_embeddings(ai: AIEngine, text: str, provider_name: str | None = None) -> list:
    """Uses the provided AI engine to generate a vector embedding."""
    if provider_name is None:
        return ai.embed(text)
    return ai.embed(text, provider_name=provider_name)


def generate_embeddings_batch(ai: AIEngine, texts: list, provider_name: str | None = None) -> list:
    """Uses the provided AI engine to generate vector embeddings in batch."""
    if provider_name is None:
        return ai.embed_batch(texts)
    return ai.embed_batch(texts, provider_name=provider_name)


def extract_and_cache_profile(ai: AIEngine, source_path: str, raw_text: str, cache_path: str) -> dict:
    """
    Extracts structured profile data (requirements, responsibilities, summary) from a resume
    or user profile using LLM extraction, with file-modification caching.
    """
    empty_result = {"requirements": [], "responsibilities": [], "summary": ""}

    if not os.path.exists(source_path):
        print(f"Warning: profile source not found at {source_path}")
        return empty_result

    source_mtime = os.path.getmtime(source_path)

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cache = json.load(f)
            cache_time = cache.get('timestamp', 0)
            cached_data = cache.get('data', {})
            has_content = bool(cached_data.get('requirements') or cached_data.get('responsibilities'))
            if cache_time >= source_mtime and has_content:
                print(f"Using cached profile from {cache_path}")
                return cached_data
            elif not has_content:
                print(f"Cached profile at {cache_path} is empty. Invalidating cache and re-extracting...")
        except Exception as e:
            print(f"Warning: failed to read profile cache {cache_path}: {e}")

    if not raw_text:
        print(f"Warning: empty text in {source_path}")
        return empty_result

    # Load extraction LLM from user_preferences.yaml
    _user_config = _load_user_config()
    _extraction_llm = _user_config.get("extraction_llm", os.getenv("EXTRACTION_LLM", "lm_studio"))

    print(f"Extracting profile data from {source_path}...")
    ai_data = call_llm_for_extraction(ai, raw_text, provider_name=_extraction_llm)
    
    result = empty_result
    if isinstance(ai_data, dict):
        result = {
            "requirements": ai_data.get('requirements', []),
            "responsibilities": ai_data.get('responsibilities', []),
            "summary": ai_data.get('summary', ""),
        }
    elif isinstance(ai_data, str):
        try:
            parsed = json.loads(ai_data)
            if isinstance(parsed, dict):
                result = {
                    "requirements": parsed.get('requirements', []),
                    "responsibilities": parsed.get('responsibilities', []),
                    "summary": parsed.get('summary', ""),
                }
        except Exception:
            pass

    has_extracted = bool(result.get('requirements') or result.get('responsibilities') or result.get('summary'))
    if not has_extracted and raw_text:
        print(f"LLM extraction produced empty results for {source_path}. Attempting text parsing fallback...")
        tp = TextProcessor()
        reqs_raw = tp.get_section_content(raw_text, "Requirements") or tp.get_section_content(raw_text, "Skills")
        resps_raw = tp.get_section_content(raw_text, "Responsibilities") or tp.get_section_content(raw_text, "Achievements") or tp.get_section_content(raw_text, "Experience")
        summary_raw = tp.get_section_content(raw_text, "Summary") or tp.get_section_content(raw_text, "Objective")
        
        reqs = tp.clean_list_from_text(reqs_raw) if reqs_raw else []
        resps = tp.clean_list_from_text(resps_raw) if resps_raw else []
        summary = summary_raw.strip() if summary_raw else ""
        
        if not summary and raw_text:
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            summary = " ".join(lines[:5]) if lines else ""

        result = {
            "requirements": reqs,
            "responsibilities": resps,
            "summary": summary
        }

    has_final_content = bool(result.get('requirements') or result.get('responsibilities') or result.get('summary'))
    if has_final_content:
        try:
            cache_dir = os.path.dirname(cache_path)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump({"timestamp": source_mtime, "data": result}, f, indent=2)
            print(f"Cached profile data to {cache_path}")
        except Exception as e:
            print(f"Warning: failed to write profile cache {cache_path}: {e}")
    else:
        print(f"Warning: Extracted profile data for {source_path} is empty. Skipping cache write to avoid corrupting cache file.")

    return result


