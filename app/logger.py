import os
import logging

def error_logger_crash(error_msg):
    print(error_msg)
    logging.error(error_msg)
    raise ValueError(error_msg)


def error_logger_continue(error_msg):
    print(error_msg)
    logging.error(error_msg)
    return


def _setup_pipeline_stats_logger() -> logging.Logger:
    """Configure and return a logger that writes pipeline stats to ``logs/pipeline_stats.log``."""
    logger = logging.getLogger("pipeline_stats")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on re-initialization
    if logger.handlers:
        return logger

    os.makedirs("logs", exist_ok=True)
    handler = logging.FileHandler(os.path.join("logs", "pipeline_stats.log"), mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagating to the root logger (which goes to app_error.log)
    logger.propagate = False
    return logger

PIPELINE_STATS_LOGGER = _setup_pipeline_stats_logger()

def _log_pipeline_stats(
    stage_label: str,
    job_count: int,
    *,
    skipped_count: int = 0,
    source_hint: str = "",
    extra: dict | None = None,
) -> None:
    """
    Write a structured stats line to ``pipeline_stats.log``.

    Format:
        STAGE <label> | jobs=<N> | skipped=<M> | source=<hint> | extra=<json>

    Args:
        stage_label: Human-readable stage name (e.g. "1: Scrape").
        job_count: Number of jobs entering/exiting this stage.
        skipped_count: Number of jobs skipped (if applicable).
        source_hint: Short description of the data source (e.g. "hiring_cafe").
        extra: Optional dict of additional key=value pairs to log.
    """
    parts = [f"STAGE {stage_label}", f"jobs={job_count}"]
    if skipped_count:
        parts.append(f"skipped={skipped_count}")
    if source_hint:
        parts.append(f"source={source_hint}")
    if extra:
        for k, v in extra.items():
            parts.append(f"{k}={v}")
    message = " | ".join(parts)
    PIPELINE_STATS_LOGGER.info(message)


