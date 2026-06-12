"""
Textual TUI for the Job Scraper and Analyzer Pipeline.

Inspired by Frogmouth's layout:
  - Status bar with app title and pipeline status
  - Sidebar navigation with pipeline stages and controls
  - Main content area (tabbed: Pipeline | Jobs | Details | Logs)
  - Footer with keyboard shortcuts
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Label,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)

load_dotenv()

# ─────────────────────────────────────────────────────────
# Pipeline Monitor
# ─────────────────────────────────────────────────────────


class PipelineMonitor:
    """Tracks pipeline execution state across the application."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.stages: Dict[str, Any] = {
            "setup": {"status": "pending", "start": None, "end": None},
            "scrape": {"status": "pending", "start": None, "end": None},
            "embed_extract": {"status": "pending", "start": None, "end": None},
            "rule_filter": {"status": "pending", "start": None, "end": None},
            "archetype": {"status": "pending", "start": None, "end": None},
            "vector_score": {"status": "pending", "start": None, "end": None},
            "cheap_llm": {"status": "pending", "start": None, "end": None},
            "strong_llm": {"status": "pending", "start": None, "end": None},
            "final_queue": {"status": "pending", "start": None, "end": None},
        }
        self.job_counts: Dict[str, int] = {
            "scraped": 0,
            "embedded": 0,
            "active": 0,
            "filtered": 0,
            "shortlisted": 0,
            "deep_analyzed": 0,
            "final_queue": 0,
        }
        self.pipeline_status: str = "idle"
        self.current_stage: str = ""
        self.error_message: str = ""
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None


monitor = PipelineMonitor()


# ─────────────────────────────────────────────────────────
# Custom Widgets
# ─────────────────────────────────────────────────────────


class PipelineStatusBar(Static):
    """Top status bar showing pipeline state and job counts."""

    def refresh_status(self) -> None:
        state = monitor
        status_colors = {
            "idle": "grey",
            "running": "bold yellow",
            "complete": "bold green",
            "error": "bold red",
        }
        color = status_colors.get(state.pipeline_status, "grey")
        elapsed = ""
        if state.start_time:
            elapsed_secs = int(time.time() - state.start_time)
            elapsed = f" | Elapsed: {elapsed_secs}s"
        stage_info = ""
        if state.current_stage:
            stage_info = f" | Stage: {state.current_stage}"
        total = sum(state.job_counts.values())
        self.update(
            f"[bold]Job Scraper & Analyzer[/]"
            f" | Pipeline: [{color}]{state.pipeline_status.upper()}[/]{stage_info}"
            f" | Jobs: {total}{elapsed}"
        )


class SidebarStatus(Vertical):
    """Left sidebar showing pipeline stage status and controls."""

    DEFAULT_CSS = """
    SidebarStatus {
        width: 28;
        height: 100%;
        border: solid $primary;
        padding: 0 1;
        background: $surface;
    }

    SidebarStatus Label.section-title {
        text-style: bold;
        color: $accent;
        margin-top: 1;
        margin-bottom: 0;
    }

    SidebarStatus Static.stage-status {
        margin-left: 1;
        height: 1;
    }

    SidebarStatus Button {
        width: 100%;
        margin-top: 0;
    }

    SidebarStatus .job-count {
        margin-left: 1;
        color: $text-muted;
    }

    SidebarStatus .controls {
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Pipeline Stages", classes="section-title")
        yield Static("  ○ Setup", id="stage-setup", classes="stage-status")
        yield Static("  ○ Scrape", id="stage-scrape", classes="stage-status")
        yield Static("  ○ Embed + Extract", id="stage-embed_extract", classes="stage-status")
        yield Static("  ○ Rule Filter", id="stage-rule_filter", classes="stage-status")
        yield Static("  ○ Archetype", id="stage-archetype", classes="stage-status")
        yield Static("  ○ Vector Score", id="stage-vector_score", classes="stage-status")
        yield Static("  ○ Cheap LLM", id="stage-cheap_llm", classes="stage-status")
        yield Static("  ○ Strong LLM", id="stage-strong_llm", classes="stage-status")
        yield Static("  ○ Final Queue", id="stage-final_queue", classes="stage-status")

        yield Label("Job Counts", classes="section-title")
        yield Static("Scraped: 0", id="count-scraped", classes="job-count")
        yield Static("Embedded: 0", id="count-embedded", classes="job-count")
        yield Static("Active: 0", id="count-active", classes="job-count")
        yield Static("Filtered: 0", id="count-filtered", classes="job-count")
        yield Static("Shortlisted: 0", id="count-shortlisted", classes="job-count")
        yield Static("Deep Analyzed: 0", id="count-deep_analyzed", classes="job-count")
        yield Static("Final Queue: 0", id="count-final_queue", classes="job-count")

        yield Label("Controls", classes="section-title controls")
        yield Button("▶ Run Pipeline", id="btn-run", variant="primary")
        yield Button("⏹ Stop", id="btn-stop", variant="error", disabled=True)
        yield Button("↻ Reset", id="btn-reset", variant="default")

    def refresh_sidebar(self) -> None:
        """Refresh sidebar display from monitor state."""
        for stage_name, stage_data in monitor.stages.items():
            widget_id = f"stage-{stage_name}"
            widget = self.query_one(f"#{widget_id}", Static)
            status = stage_data["status"]
            symbols = {"pending": "○", "running": "◉", "complete": "●", "error": "⊗"}
            colors = {"pending": "grey", "running": "yellow", "complete": "green", "error": "red"}
            sym = symbols.get(status, "○")
            col = colors.get(status, "grey")
            widget.update(f"  [{col}]{sym}[/] {stage_name.replace('_', ' ').title()}")

        for count_name, count_val in monitor.job_counts.items():
            widget_id = f"count-{count_name}"
            widget = self.query_one(f"#{widget_id}", Static)
            widget.update(f"{count_name.replace('_', ' ').title()}: {count_val}")


class PipelineLog(RichLog):
    """Log widget for pipeline output with timestamped, color-coded messages."""

    DEFAULT_CSS = """
    PipelineLog {
        height: 100%;
        border: solid $primary;
    }
    """

    def write_log(self, message: str, level: str = "info") -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {"info": "white", "warning": "yellow", "error": "red", "success": "green"}
        color = colors.get(level, "white")
        self.write(f"[{timestamp}] [{color}]{message}[/]")


# ─────────────────────────────────────────────────────────
# Main Screen
# ─────────────────────────────────────────────────────────


class PipelineScreen(Screen[None]):
    """Main application screen with sidebar, content tabs, and footer."""

    DEFAULT_CSS = """
    PipelineScreen {
        layout: vertical;
    }

    PipelineScreen > Horizontal {
        height: 1fr;
    }

    .content-area {
        height: 100%;
        width: 1fr;
        padding: 0 1;
    }

    .content-area TabbedContent {
        height: 100%;
    }

    .content-area TabbedContent TabPane {
        padding: 1;
    }

    #pipeline-overview {
        height: 100%;
    }

    #pipeline-overview > Vertical {
        height: 100%;
    }

    DataTable {
        height: 100%;
        border: solid $primary;
    }

    #job-detail {
        height: 100%;
    }

    #job-detail > Vertical {
        height: auto;
    }

    #status-bar {
        height: 1;
        background: $surface;
        padding: 0 1;
    }

    #log-view {
        height: 100%;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "app.quit", "Quit", show=True),
        Binding("q", "app.quit", "", show=False),
        Binding("escape", "app.quit", "", show=False),
        Binding("ctrl+p", "focus_pipeline", "Pipeline", show=True),
        Binding("ctrl+j", "focus_jobs", "Jobs", show=True),
        Binding("ctrl+d", "focus_detail", "Detail", show=True),
        Binding("ctrl+l", "focus_logs", "Logs", show=True),
        Binding("f5", "run_pipeline", "Run", show=True),
        Binding("f6", "stop_pipeline", "Stop", show=True),
        Binding("f7", "reset_pipeline", "Reset", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._running_task: Optional[asyncio.Task] = None
        self._jobs_data: List[Dict] = []

    def compose(self) -> ComposeResult:
        yield PipelineStatusBar(id="status-bar")
        with Horizontal():
            yield SidebarStatus()
            with Vertical(classes="content-area"):
                with TabbedContent(initial="pipeline"):
                    with TabPane("Pipeline", id="pipeline"):
                        with Vertical(id="pipeline-overview"):
                            yield Label("[bold]Pipeline Overview[/]", id="overview-title")
                            yield PipelineLog(id="log-view", max_lines=500, highlight=True)
                    with TabPane("Jobs", id="jobs"):
                        yield DataTable(id="jobs-table", zebra_stripes=True)
                    with TabPane("Detail", id="detail"):
                        with Vertical(id="job-detail"):
                            yield Label("Select a job from the Jobs tab to see details.", id="detail-placeholder")
                    with TabPane("Logs", id="logs"):
                        yield PipelineLog(id="full-log", max_lines=1000, highlight=True)
        yield Footer()

    def on_mount(self) -> None:
        """Set up the screen after DOM is ready."""
        table = self.query_one("#jobs-table", DataTable)
        table.add_columns("ID", "Title", "Company", "Score", "Fit", "Priority", "Status")
        self._log_message("Pipeline TUI initialized. Press F5 to run the pipeline.", "info")

    # ────── Actions ──────

    def action_focus_pipeline(self) -> None:
        self.query_one(TabbedContent).active = "pipeline"

    def action_focus_jobs(self) -> None:
        self.query_one(TabbedContent).active = "jobs"

    def action_focus_detail(self) -> None:
        self.query_one(TabbedContent).active = "detail"

    def action_focus_logs(self) -> None:
        self.query_one(TabbedContent).active = "logs"

    def action_run_pipeline(self) -> None:
        self._run_pipeline()

    def action_stop_pipeline(self) -> None:
        if self._running_task and not self._running_task.done():
            self._running_task.cancel()
            monitor.pipeline_status = "idle"
            self._log_message("Pipeline stopped by user.", "warning")
            self._refresh_all_ui()
            self._set_buttons(enabled=True)

    def action_reset_pipeline(self) -> None:
        if self._running_task and not self._running_task.done():
            self._running_task.cancel()
        monitor.reset()
        self._jobs_data = []
        self.query_one("#jobs-table", DataTable).clear()
        detail = self.query_one("#job-detail", Vertical)
        detail.remove_children()
        detail.mount(Label("Select a job from the Jobs tab to see details.", id="detail-placeholder"))
        self._log_message("Pipeline reset.", "info")
        self._refresh_all_ui()
        self._set_buttons(enabled=True)

    # ────── Button Handlers ──────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-run":
            self._run_pipeline()
        elif event.button.id == "btn-stop":
            self.action_stop_pipeline()
        elif event.button.id == "btn-reset":
            self.action_reset_pipeline()

    # ────── Internal Pipeline Runner ──────

    @work(thread=False)
    async def _run_pipeline(self) -> None:
        if monitor.pipeline_status == "running":
            self._log_message("Pipeline already running.", "warning")
            return

        monitor.reset()
        monitor.pipeline_status = "running"
        monitor.start_time = time.time()
        self._set_buttons(enabled=False)
        self._log_message("=" * 50, "info")
        self._log_message("PIPELINE STARTED", "success")
        self._log_message("=" * 50, "info")
        self._refresh_all_ui()

        try:
            result = await self._execute_pipeline()
            monitor.pipeline_status = "complete"
            monitor.end_time = time.time()
            duration = int(monitor.end_time - monitor.start_time)
            self._log_message("=" * 50, "info")
            self._log_message(f"PIPELINE COMPLETE ({duration}s)", "success")
            self._log_message(f"Total jobs processed: {monitor.job_counts['scraped']}", "info")
            self._log_message(f"Final queue: {monitor.job_counts['final_queue']} jobs", "info")
            self._log_message("=" * 50, "info")
            self._jobs_data = result if isinstance(result, list) else []
            self._populate_jobs_table()
        except asyncio.CancelledError:
            monitor.pipeline_status = "idle"
            self._log_message("Pipeline cancelled.", "warning")
        except Exception as exc:
            monitor.pipeline_status = "error"
            monitor.error_message = str(exc)
            self._log_message(f"Pipeline error: {exc}", "error")
            import traceback
            self._log_message(traceback.format_exc(), "error")
        finally:
            self._set_buttons(enabled=True)
            self._refresh_all_ui()

    async def _execute_pipeline(self) -> List[Dict]:
        """Execute pipeline stages sequentially, updating UI after each."""
        # pylint: disable=import-outside-toplevel
        from main import (
            pipeline_stage_setup,
            pipeline_stage_scrape,
            pipeline_stage_embed_and_extract,
            pipeline_stage_rule_filter,
            pipeline_stage_archetype_integration,
            pipeline_stage_vector_scoring,
            pipeline_stage_cheap_llm,
            pipeline_stage_strong_llm,
            pipeline_stage_final_queue,
        )

        # Stage 0: Setup
        self._activate_stage("setup", "Setup and profile extraction...")
        setup_data = await pipeline_stage_setup(verbose=False)
        self._complete_stage("setup", "Setup complete.")

        # Stage 1: Scrape
        self._activate_stage("scrape", "Scraping jobs...")
        processed_job_pool = await pipeline_stage_scrape(
            setup_data=setup_data, skip_db=True, verbose=False,
        )
        monitor.job_counts["scraped"] = len(processed_job_pool)
        self._complete_stage("scrape", f"Scraped {len(processed_job_pool)} jobs.")

        # Stage 2: Embed + Extract
        self._activate_stage("embed_extract", "Embedding & LLM extraction...")
        processed_job_pool = await pipeline_stage_embed_and_extract(
            jobs=processed_job_pool,
            ai_engine=setup_data["ai"],
            text_processor=setup_data["tp"],
            dp=setup_data["dp"],
            skip_db=True,
            verbose=False,
        )
        monitor.job_counts["embedded"] = len(processed_job_pool)
        self._complete_stage("embed_extract", f"Embedded {len(processed_job_pool)} jobs.")

        # Stage 3: Rule Filter
        self._activate_stage("rule_filter", "Rule filtering...")
        processed_job_pool = await pipeline_stage_rule_filter(
            jobs=processed_job_pool,
            user_preferences=setup_data.get("user_preferences", {}),
            dp=setup_data["dp"],
            skip_db=True,
        )
        active_count = sum(1 for j in processed_job_pool if not j.get("skip"))
        monitor.job_counts["active"] = active_count
        self._complete_stage("rule_filter", f"{active_count} jobs active after filtering.")

        # Stage 4: Archetype
        self._activate_stage("archetype", "Archetype engine integration...")
        active_jobs, archetype_manager = await pipeline_stage_archetype_integration(
            jobs=processed_job_pool,
            ai_engine=setup_data["ai"],
            dp=setup_data["dp"],
            setup_data=setup_data,
            skip_db=True,
        )
        self._complete_stage("archetype", "Archetypes loaded.")

        # Stage 5: Vector Scoring
        self._activate_stage("vector_score", "Vector scoring...")
        filtered_job_pool = await pipeline_stage_vector_scoring(
            jobs=active_jobs,
            archetype_manager=archetype_manager,
            dp=setup_data["dp"],
            skip_db=True,
        )
        monitor.job_counts["filtered"] = len(filtered_job_pool)
        self._complete_stage("vector_score", f"{len(filtered_job_pool)} jobs passed vector scoring.")

        # Stage 6: Cheap LLM
        self._activate_stage("cheap_llm", "Cheap LLM classification...")
        shortlisted_jobs = await pipeline_stage_cheap_llm(
            jobs=filtered_job_pool,
            setup_data=setup_data,
            dp=setup_data["dp"],
            skip_db=True,
        )
        monitor.job_counts["shortlisted"] = len(shortlisted_jobs)
        self._complete_stage("cheap_llm", f"{len(shortlisted_jobs)} jobs shortlisted.")

        # Stage 7: Strong LLM
        self._activate_stage("strong_llm", "Strong LLM reranking...")
        deeply_analyzed_jobs = await pipeline_stage_strong_llm(
            jobs=shortlisted_jobs,
            setup_data=setup_data,
            dp=setup_data["dp"],
            skip_db=True,
        )
        monitor.job_counts["deep_analyzed"] = len(deeply_analyzed_jobs)
        self._complete_stage("strong_llm", f"{len(deeply_analyzed_jobs)} jobs deeply analyzed.")

        # Stage 8: Final Queue
        self._activate_stage("final_queue", "Final application queue...")
        final_queue = await pipeline_stage_final_queue(
            jobs=deeply_analyzed_jobs,
            dp=setup_data["dp"],
            skip_db=True,
        )
        monitor.job_counts["final_queue"] = len(final_queue)
        self._complete_stage("final_queue", f"{len(final_queue)} jobs in final queue.")

        return final_queue

    # ────── UI Helpers ──────

    def _activate_stage(self, stage: str, log_msg: str) -> None:
        """Mark a stage as running and log it."""
        if stage in monitor.stages:
            monitor.stages[stage]["status"] = "running"
            monitor.stages[stage]["start"] = time.time()
        monitor.current_stage = stage
        self._log_message(log_msg, "info")
        self._refresh_all_ui()

    def _complete_stage(self, stage: str, log_msg: str) -> None:
        """Mark a stage as complete and log success."""
        if stage in monitor.stages:
            monitor.stages[stage]["status"] = "complete"
            monitor.stages[stage]["end"] = time.time()
        self._log_message(f"✓ {log_msg}", "success")
        self._refresh_all_ui()

    def _set_buttons(self, enabled: bool) -> None:
        """Enable or disable control buttons."""
        try:
            self.query_one("#btn-run", Button).disabled = not enabled
            self.query_one("#btn-stop", Button).disabled = enabled
            self.query_one("#btn-reset", Button).disabled = not enabled
        except Exception:
            pass

    def _refresh_all_ui(self) -> None:
        """Refresh all UI components from current monitor state."""
        try:
            self.query_one(PipelineStatusBar).refresh_status()
            self.query_one(SidebarStatus).refresh_sidebar()
        except Exception:
            pass

    def _log_message(self, message: str, level: str = "info") -> None:
        """Write to both log widgets."""
        try:
            self.query_one("#log-view", PipelineLog).write_log(message, level)
            self.query_one("#full-log", PipelineLog).write_log(message, level)
        except Exception:
            pass

    def _populate_jobs_table(self) -> None:
        """Fill the jobs table with pipeline results."""
        table = self.query_one("#jobs-table", DataTable)
        table.clear()

        if not self._jobs_data:
            table.add_row("No jobs in final queue.")
            return

        for i, job in enumerate(self._jobs_data):
            features = job.get("features", {})
            cheap_result = job.get("cheap_llm_result", {})
            job_id = str(job.get("metadata", {}).get("job_id", i))
            title = features.get("title", "Unknown")
            company = features.get("company", "N/A")
            score = str(job.get("final_score", job.get("semantic_score_percent", 0)))
            fit = str(cheap_result.get("fit_score", ""))
            priority = job.get("priority", "N/A")
            status = "✓" if job.get("apply_recommendation") == "apply" else "○"
            table.add_row(job_id, title, company, score, fit, priority, status)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Show job detail when a row is selected."""
        if not self._jobs_data:
            return
        if event.row_key.value is None:
            return
        try:
            row_index = int(event.row_key.value)
            if 0 <= row_index < len(self._jobs_data):
                self._show_job_detail(self._jobs_data[row_index])
        except (ValueError, IndexError):
            pass

    def _show_job_detail(self, job: Dict[str, Any]) -> None:
        """Render job details in the detail tab."""
        detail_container = self.query_one("#job-detail", Vertical)
        detail_container.remove_children()

        features = job.get("features", {})
        cheap = job.get("cheap_llm_result", {})
        strong = job.get("strong_llm_result", {})
        metadata = job.get("metadata", {})

        sections: List[tuple[str, List[tuple[str, str]]]] = [
            ("Job Details", [
                ("Title", features.get("title", "N/A")),
                ("Job ID", str(metadata.get("job_id", "N/A"))),
                ("Company", features.get("company", "N/A")),
                ("Source", metadata.get("source", "N/A")),
            ]),
            ("Compensation & Logistics", [
                ("Pay", features.get("pay", "N/A")),
                ("Seniority", features.get("seniority", "N/A")),
                ("Work Type", features.get("work_type", "N/A")),
                ("Timezone", features.get("timezone", "N/A")),
            ]),
            ("Skills & Requirements", [
                ("Skills", ", ".join(features.get("skills", [])) or "N/A"),
                ("Requirements", ", ".join(features.get("requirements", [])) or "N/A"),
            ]),
            ("Scoring", [
                ("Semantic Score", f"{job.get('semantic_score_percent', 'N/A')}%"),
                ("Best Archetype", job.get("best_archetype", "N/A")),
                ("Cheap LLM Fit", f"{cheap.get('fit_score', 'N/A')}"),
                ("Cheap LLM Decision", cheap.get("decision", "N/A")),
            ]),
            ("Strong LLM Analysis", [
                ("Final Score", str(strong.get("final_score", "N/A"))),
                ("Decision", strong.get("decision", "N/A")),
                ("Confidence", str(strong.get("confidence", "N/A"))),
            ]),
            ("Final Queue", [
                ("Priority", job.get("priority", "N/A")),
                ("Final Score", str(job.get("final_score", "N/A"))),
                ("Apply Recommendation", job.get("apply_recommendation", "N/A")),
            ]),
        ]

        for section_title, fields in sections:
            detail_container.mount(Label(f"\n[bold]{section_title}[/]"))
            detail_container.mount(Static("─" * 40))
            for key, value in fields:
                detail_container.mount(Label(f"  [bold]{key}:[/] {value}"))

        summary = features.get("summary", "")
        if summary:
            detail_container.mount(Label("\n[bold]Summary[/]"))
            detail_container.mount(Static("─" * 40))
            detail_container.mount(Static(f"  {summary}"))

        strengths = cheap.get("strengths", [])
        concerns = cheap.get("concerns", [])
        if strengths:
            detail_container.mount(Label("\n[bold green]Strengths[/]"))
            for s in strengths:
                detail_container.mount(Static(f"  ✓ {s}"))
        if concerns:
            detail_container.mount(Label("\n[bold red]Concerns[/]"))
            for c in concerns:
                detail_container.mount(Static(f"  ✗ {c}"))


# ─────────────────────────────────────────────────────────
# TUI Application
# ─────────────────────────────────────────────────────────


class JobScraperTUI(App[None]):
    """Textual TUI for the Job Scraper and Analyzer Pipeline."""

    TITLE = "Job Scraper & Analyzer"
    SUB_TITLE = "Pipeline Management Interface"

    CSS = """
    Screen {
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.dark = True

    def on_mount(self) -> None:
        self.push_screen(PipelineScreen())

    def action_quit(self) -> None:
        self.exit()


def run_tui() -> None:
    """Entry point to run the TUI."""
    app = JobScraperTUI()
    app.run()


if __name__ == "__main__":
    run_tui()