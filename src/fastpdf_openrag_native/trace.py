from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import TraceEvent


class TraceRecorder:
    def __init__(self, trace_dir: Path):
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.trace_dir / "events.jsonl"
        self.summary_path = self.trace_dir / "summary.json"
        self._events: list[TraceEvent] = []

    def record(
        self,
        *,
        stage: str,
        service: str,
        action: str,
        status: str = "ok",
        request: dict[str, Any] | None = None,
        response: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        output_files: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            stage=stage,
            service=service,
            action=action,
            status=status,
            request=request or {},
            response=response or {},
            metrics=metrics or {},
            output_files=output_files or [],
            notes=notes or [],
        )
        self._events.append(event)
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json())
            handle.write("\n")
        return event

    def write_summary(self, payload: dict[str, Any]) -> Path:
        self.summary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return self.summary_path

    def load_events(self) -> list[TraceEvent]:
        if not self.trace_path.exists():
            return []
        rows: list[TraceEvent] = []
        for line in self.trace_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(TraceEvent.model_validate_json(line))
        return rows
