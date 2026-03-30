from __future__ import annotations

import asyncio
import html
import json
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .citations import ensure_summary_citations
from .fastpdf_loader import load_run_from_mongo, load_run_json, materialize_summary_payload
from .langflow import LangflowGateway
from .openrag import OpenRAGGateway
from .opensearch import OpenSearchInspector
from .pdf_workflow import run_pdf_pipeline
from .settings import fresh_settings
from .summarizer import load_manifest, load_scopes, resolve_scope_pages, resolve_scope_retrieval_sources, summarize_scope

app = FastAPI(title="fastpdf-openrag-native", version="0.2.0")
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
OUTPUT_ROOT = REPO_ROOT / "outputs"

DATA_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/debug-data", StaticFiles(directory=DATA_ROOT.as_posix()), name="debug-data")
app.mount("/debug-outputs", StaticFiles(directory=OUTPUT_ROOT.as_posix()), name="debug-outputs")

_jobs: dict[str, dict[str, Any]] = {}


class MaterializeRequest(BaseModel):
    input_path: str | None = None
    run_id: str | None = None
    output_dir: str | None = None
    include_non_survivors: bool = False
    mongo_uri: str | None = None
    mongo_database: str | None = None
    mongo_collection: str = "runs"


class IngestRequest(BaseModel):
    manifest_path: str
    apply_recommended_settings: bool = False


class ScopeRequest(BaseModel):
    manifest_path: str
    scope_path: str
    apply_recommended_settings: bool = False


class FlowUpgradeRequest(BaseModel):
    output_dir: str | None = None
    patch_live: bool = True


def _job_payload(job_id: str) -> dict[str, Any]:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="job not found")
    return _jobs[job_id]


def _set_job(job_id: str, **fields: Any) -> None:
    payload = _jobs.setdefault(job_id, {"job_id": job_id, "status": "queued"})
    payload.update(fields)


def _settings():
    return fresh_settings()


def _normalize_repo_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _debug_url_for_path(value: str | None) -> str | None:
    path = _normalize_repo_path(value)
    if not path:
        return None
    try:
        relative = path.relative_to(DATA_ROOT)
        return f"/debug-data/{quote(relative.as_posix(), safe='/')}"
    except ValueError:
        pass
    try:
        relative = path.relative_to(OUTPUT_ROOT)
        return f"/debug-outputs/{quote(relative.as_posix(), safe='/')}"
    except ValueError:
        return None


def _job_debug_artifacts(payload: dict[str, Any]) -> dict[str, dict[str, str]]:
    extraction_dir = payload.get("extraction_dir")
    trace_dir = payload.get("trace_dir")
    artifacts_dir = None
    pages_dir = None
    if extraction_dir:
        artifacts_dir = str(Path(extraction_dir) / "artifacts")
        pages_dir = str(Path(extraction_dir) / "pages")

    candidates = {
        "manifest": payload.get("manifest_path"),
        "trace_events": payload.get("trace_path"),
        "trace_summary": payload.get("trace_summary_path"),
        "pipeline_summary": payload.get("summary_path"),
        "extraction_dir": extraction_dir,
        "pages_dir": pages_dir,
        "artifacts_dir": artifacts_dir,
        "trace_dir": trace_dir,
        "chunk_dump_dir": payload.get("chunk_dump_dir"),
        "citation_index": payload.get("citation_index_path"),
        "resolved_citations": payload.get("resolved_citations_path"),
        "source_pdf_copy": payload.get("source_pdf_copy_path"),
    }

    artifact_links: dict[str, dict[str, str]] = {}
    for label, raw_path in candidates.items():
        if not raw_path:
            continue
        path = _normalize_repo_path(raw_path)
        if not path:
            continue
        artifact_links[label] = {"path": path.as_posix()}
        debug_url = _debug_url_for_path(raw_path)
        if debug_url:
            artifact_links[label]["url"] = debug_url
    return artifact_links


async def _run_job(
    *,
    job_id: str,
    pdf_path: Path,
    credentials_path: Path | None,
    question: str | None,
    max_pages: int | None,
) -> None:
    settings = _settings()
    _set_job(
        job_id,
        status="running",
        pdf_path=pdf_path.as_posix(),
        credentials_path=credentials_path.as_posix() if credentials_path else None,
        question=question,
        max_pages=max_pages,
    )
    try:
        result = await run_pdf_pipeline(
            pdf_path=pdf_path,
            credentials_path=credentials_path,
            settings=settings,
            question=question,
            max_pages=max_pages,
            apply_recommended_settings=True,
        )
    except Exception as exc:
        _set_job(
            job_id,
            status="failed",
            error=str(exc),
            error_traceback=traceback.format_exc(),
        )
        return

    _set_job(
        job_id,
        status="completed_with_errors" if result.summary_error else "completed",
        run_id=result.run_id,
        result=result.model_dump(mode="json"),
        extraction_dir=result.extraction_dir,
        trace_dir=result.trace_dir,
        trace_path=result.trace_path,
        trace_summary_path=result.trace_summary_path,
        chunk_dump_dir=result.chunk_dump_dir,
        manifest_path=result.manifest_path,
        summary_path=result.summary_path,
        citation_index_path=result.citation_index_path,
        resolved_citations_path=result.resolved_citations_path,
        source_pdf_copy_path=result.source_pdf_copy_path,
        summary_error=result.summary_error,
        debug_artifacts=_job_debug_artifacts(result.model_dump(mode="json")),
    )


def _render_home() -> str:
    jobs = sorted(_jobs.values(), key=lambda row: row.get("job_id", ""), reverse=True)
    job_rows = "\n".join(
        (
            "<tr>"
            f"<td><a href=\"/ui/runs/{html.escape(row['job_id'])}/summary\">{html.escape(row['job_id'])}</a><br /><small><a href=\"/ui/runs/{html.escape(row['job_id'])}/trace\">Trace</a></small></td>"
            f"<td>{html.escape(str(row.get('status', 'unknown')))}</td>"
            f"<td>{html.escape(str(row.get('pdf_path', '')))}</td>"
            f"<td>{html.escape(str(row.get('run_id', '')))}</td>"
            "</tr>"
        )
        for row in jobs[:20]
    ) or "<tr><td colspan='4'>No runs yet</td></tr>"

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>fastpdf-openrag-native</title>
  <style>
    :root {{ color-scheme: light; }}
    body {{ font-family: 'IBM Plex Sans', sans-serif; margin: 24px; background: linear-gradient(180deg, #f6f3ed, #fff); color: #1a1a1a; }}
    .panel {{ background: #fff; border: 1px solid #ddd4c6; border-radius: 16px; padding: 20px; margin-bottom: 20px; box-shadow: 0 10px 30px rgba(56, 43, 24, 0.08); }}
    input, textarea {{ width: 100%; margin-top: 6px; margin-bottom: 12px; padding: 10px; border: 1px solid #cfc2ad; border-radius: 8px; font: inherit; }}
    button {{ background: #1f5f5b; color: white; border: 0; border-radius: 999px; padding: 10px 18px; font: inherit; cursor: pointer; }}
    table {{ width: 100%; border-collapse: collapse; }}
    td, th {{ border-bottom: 1px solid #eee3d4; padding: 8px; text-align: left; vertical-align: top; }}
    pre {{ white-space: pre-wrap; background: #f7f3eb; padding: 12px; border-radius: 8px; overflow-x: auto; }}
  </style>
</head>
<body>
  <div class="panel">
    <h1>fastpdf-openrag-native</h1>
    <p>Upload or point to a PDF, run Google Vision OCR, ingest the generated HTML into OpenRAG, inspect chunk/debug traces, and summarize every page through OpenRAG retrieval.</p>
    <form id="run-form">
      <label>PDF path</label>
      <input name="pdf_path" value="/home/divyesh-nandlal-vishwakarma/Downloads/chinchin/merged_notes.pdf" />
      <label>PDF upload</label>
      <input type="file" name="upload" accept="application/pdf" />
      <label>Google Vision credentials path</label>
      <input name="credentials_path" value="/home/divyesh-nandlal-vishwakarma/Downloads/ocr-neuralit-4e01e06ccf84(2).json" />
      <label>Summary objective</label>
      <textarea name="question">Summarize every page independently, then provide a grounded overall summary and chronology.</textarea>
      <label>Max pages (optional)</label>
      <input name="max_pages" type="number" min="1" />
      <button type="submit">Run Pipeline</button>
    </form>
  </div>

  <div class="panel">
    <h2>Live status</h2>
    <pre id="status-box">No active run.</pre>
  </div>

  <div class="panel">
    <h2>Recent runs</h2>
    <table>
      <thead><tr><th>Job</th><th>Status</th><th>PDF</th><th>Run ID</th></tr></thead>
      <tbody>{job_rows}</tbody>
    </table>
  </div>

  <script>
    let currentJobId = null;

    async function pollJob(jobId) {{
      currentJobId = jobId;
      while (currentJobId === jobId) {{
        const response = await fetch(`/ui/runs/${{jobId}}`);
        const payload = await response.json();
        document.getElementById('status-box').textContent = JSON.stringify(payload, null, 2);
        if (
          payload.status === 'completed' ||
          payload.status === 'completed_with_errors' ||
          payload.status === 'failed'
        ) {{
          break;
        }}
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }}
    }}

    document.getElementById('run-form').addEventListener('submit', async (event) => {{
      event.preventDefault();
      const form = new FormData(event.target);
      const response = await fetch('/ui/runs', {{ method: 'POST', body: form }});
      const payload = await response.json();
      document.getElementById('status-box').textContent = JSON.stringify(payload, null, 2);
      if (payload.job_id) {{
        pollJob(payload.job_id);
      }}
    }});
  </script>
</body>
</html>
""".strip()


def _render_trace_page(job_id: str, payload: dict[str, Any]) -> str:
    trace_summary_path = _normalize_repo_path(payload.get("trace_summary_path"))
    events_path = _normalize_repo_path(payload.get("trace_path"))
    pipeline_summary_path = _normalize_repo_path(payload.get("summary_path"))
    summary_text = (
        trace_summary_path.read_text(encoding="utf-8")
        if trace_summary_path and trace_summary_path.exists()
        else "{}"
    )
    events_text = events_path.read_text(encoding="utf-8") if events_path and events_path.exists() else ""
    pipeline_summary_text = (
        pipeline_summary_path.read_text(encoding="utf-8")
        if pipeline_summary_path and pipeline_summary_path.exists()
        else ""
    )
    artifact_rows = "\n".join(
        (
            "<li>"
            f"{html.escape(label)}: "
            + (
                f"<a href=\"{html.escape(row['url'])}\">{html.escape(row['path'])}</a>"
                if row.get("url")
                else html.escape(row["path"])
            )
            + "</li>"
        )
        for label, row in payload.get("debug_artifacts", {}).items()
    ) or "<li>No stored artifact links yet</li>"
    error_blocks: list[str] = []
    if payload.get("summary_error"):
        error_blocks.append("<h3>Summary error</h3>")
        error_blocks.append(f"<pre>{html.escape(str(payload.get('summary_error')))}</pre>")
    if payload.get("error"):
        error_blocks.append("<h3>Pipeline failure</h3>")
        error_blocks.append(f"<pre>{html.escape(str(payload.get('error')))}</pre>")
        if payload.get("error_traceback"):
            error_blocks.append(f"<pre>{html.escape(str(payload.get('error_traceback', '')))}</pre>")
    error_html = ""
    if error_blocks:
        error_html = "<section><h2>Errors</h2>" + "".join(error_blocks) + "</section>"
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(job_id)}</title>
  <style>
    body {{ font-family: 'IBM Plex Sans', sans-serif; margin: 24px; background: #faf7f1; color: #1a1a1a; }}
    section {{ background: #fff; border: 1px solid #ddd4c6; border-radius: 16px; padding: 20px; margin-bottom: 20px; }}
    pre {{ white-space: pre-wrap; background: #f7f3eb; padding: 12px; border-radius: 8px; overflow-x: auto; }}
    a {{ color: #0d5c63; }}
  </style>
</head>
<body>
  <section>
    <h1>{html.escape(job_id)}</h1>
    <p>Status: {html.escape(str(payload.get('status', 'unknown')))}</p>
    <p><a href="/ui/runs/{html.escape(job_id)}">JSON status</a></p>
    <p><a href="/ui/runs/{html.escape(job_id)}/summary">Summary Results</a></p>
    <p><a href="/">Back</a></p>
  </section>
  <section>
    <h2>Artifacts</h2>
    <ul>
      {artifact_rows}
    </ul>
  </section>
  <section>
    <h2>Trace summary</h2>
    <pre>{html.escape(summary_text)}</pre>
  </section>
  <section>
    <h2>Pipeline summary output</h2>
    <pre>{html.escape(pipeline_summary_text or '{}')}</pre>
  </section>
  <section>
    <h2>Trace events</h2>
    <pre>{html.escape(events_text)}</pre>
  </section>
  {error_html}
</body>
</html>
""".strip()



def _dedupe_display_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...], int | None, str | None]] = set()
    for item in items:
        text_key = " ".join(str(item.get("text") or "").split()).strip().lower()
        citation_key = tuple(str(value) for value in item.get("citation_ids", []))
        key = (text_key, citation_key, item.get("page"), item.get("pdf_id"))
        if not text_key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped



def _merge_sections_for_display(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not sections:
        return []

    merged: list[dict[str, Any]] = []
    page_groups: dict[tuple[str, int], dict[str, Any]] = {}

    for section in sections:
        kind = str(section.get("kind") or "")
        debug_only = bool(section.get("debug_only"))
        pdf_id = section.get("pdf_id")
        page = section.get("page")
        if (
            not debug_only
            and kind in {"page_summary", "key_facts"}
            and isinstance(pdf_id, str)
            and isinstance(page, int)
        ):
            key = (pdf_id, page)
            group = page_groups.get(key)
            if group is None:
                group = {
                    "section_id": f"page-{page}-sequence",
                    "title": f"{pdf_id} · Page {page}",
                    "kind": "page_sequence",
                    "debug_only": False,
                    "pdf_id": pdf_id,
                    "page": page,
                    "items": [],
                }
                page_groups[key] = group
                merged.append(group)
            group["items"].extend(list(section.get("items") or []))
            continue
        merged.append(section)

    for section in merged:
        if isinstance(section.get("items"), list):
            section["items"] = _dedupe_display_items(list(section.get("items") or []))

    supported_sections = [section for section in merged if section.get("kind") == "supported_summary"]
    page_sections = [section for section in merged if section.get("kind") == "page_sequence"]
    other_sections = [
        section
        for section in merged
        if section.get("kind") not in {"supported_summary", "chronology", "page_sequence"}
    ]
    page_sections.sort(key=lambda section: (str(section.get("pdf_id") or ""), int(section.get("page") or 0)))

    detailed_items: list[dict[str, Any]] = []
    for section in supported_sections:
        detailed_items.extend(list(section.get("items") or []))
    for section in page_sections:
        page_title = str(section.get("title") or "Page")
        for item in list(section.get("items") or []):
            item_copy = dict(item)
            text = str(item_copy.get("text") or "").strip()
            if text:
                item_copy["text"] = f"{page_title}: {text}"
            item_copy.setdefault("pdf_id", section.get("pdf_id"))
            item_copy.setdefault("page", section.get("page"))
            detailed_items.append(item_copy)
    detailed_items = _dedupe_display_items(detailed_items)

    detailed_sections: list[dict[str, Any]] = []
    if detailed_items:
        detailed_sections.append(
            {
                "section_id": "detailed-summary",
                "title": "Detailed Summary",
                "kind": "detailed_summary",
                "debug_only": False,
                "items": detailed_items,
            }
        )

    return [*detailed_sections, *other_sections]


def _build_summary_view_model(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    summary_path = _normalize_repo_path(payload.get("summary_path"))
    manifest_path = _normalize_repo_path(payload.get("manifest_path"))
    if summary_path is None or manifest_path is None:
        raise HTTPException(status_code=409, detail="summary artifacts are not available for this job yet")
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail=f"summary not found: {summary_path}")
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"manifest not found: {manifest_path}")

    source_pdf = _normalize_repo_path(payload.get("pdf_path"))
    summary, citation_index_path, resolved_citations_path, source_pdf_copy_path = ensure_summary_citations(
        summary_path=summary_path,
        manifest_path=manifest_path,
        source_pdf=source_pdf,
    )

    payload["citation_index_path"] = citation_index_path.as_posix()
    payload["resolved_citations_path"] = resolved_citations_path.as_posix()
    if source_pdf_copy_path:
        payload["source_pdf_copy_path"] = source_pdf_copy_path.as_posix()
    payload["debug_artifacts"] = _job_debug_artifacts(payload)

    citation_lookup = {entry.id: entry for entry in summary.citation_index}
    citation_index = []
    for entry in summary.citation_index:
        row = entry.model_dump(mode="json")
        row["page_image_url"] = _debug_url_for_path(row.get("page_image_path"))
        row["source_pdf_url"] = _debug_url_for_path(row.get("source_pdf_path"))
        citation_index.append(row)

    resolved = summary.resolved_citations.model_dump(mode="json") if summary.resolved_citations else {"sections": [], "source_pages": []}
    for section in resolved.get("sections", []):
        for item in section.get("items", []):
            item["citation_numbers"] = [
                citation_lookup[citation_id].number
                for citation_id in item.get("citation_ids", [])
                if citation_id in citation_lookup
            ]
    for page in resolved.get("source_pages", []):
        page["image_url"] = _debug_url_for_path(page.get("image_path"))
        page["html_url"] = _debug_url_for_path(page.get("html_path"))
        page["source_pdf_url"] = _debug_url_for_path(page.get("source_pdf_path"))

    return {
        "job": {
            "job_id": job_id,
            "status": payload.get("status"),
            "run_id": payload.get("run_id"),
            "pdf_path": payload.get("pdf_path"),
            "question": payload.get("question"),
            "max_pages": payload.get("max_pages"),
            "summary_error": payload.get("summary_error"),
        },
        "title": summary.draft_title or summary.scope.title,
        "scope": summary.scope.model_dump(mode="json"),
        "summary": summary.model_dump(mode="json"),
        "citation_index": citation_index,
        "sections": _merge_sections_for_display(list(resolved.get("sections", []))),
        "source_pages": resolved.get("source_pages", []),
        "resolved_debug": resolved.get("debug", {}),
        "artifacts": payload.get("debug_artifacts", {}),
        "urls": {
            "status": f"/ui/runs/{quote(job_id)}",
            "trace": f"/ui/runs/{quote(job_id)}/trace",
            "summary_json": _debug_url_for_path(summary_path.as_posix()),
            "citation_index": _debug_url_for_path(citation_index_path.as_posix()),
            "resolved_citations": _debug_url_for_path(resolved_citations_path.as_posix()),
            "source_pdf": _debug_url_for_path(source_pdf_copy_path.as_posix()) if source_pdf_copy_path else None,
        },
    }


def _render_summary_page(job_id: str) -> str:
    html_page = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Summary Results</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f3efe7;
      --panel: rgba(255, 252, 247, 0.94);
      --panel-strong: #fffdf8;
      --border: #d7ccb7;
      --ink: #1f1a14;
      --muted: #625644;
      --accent: #1c6a63;
      --accent-soft: rgba(28, 106, 99, 0.1);
      --accent-strong: #0e4c47;
      --warn: #b7682d;
      --danger: #9b3c2f;
      --shadow: 0 18px 50px rgba(58, 44, 24, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(28, 106, 99, 0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(183, 104, 45, 0.12), transparent 24%),
        linear-gradient(180deg, #f8f3ec 0%, var(--bg) 100%);
      min-height: 100vh;
    }
    a { color: var(--accent-strong); }
    .page {
      width: min(1600px, calc(100vw - 32px));
      margin: 16px auto 32px;
      display: grid;
      gap: 16px;
    }
    .hero {
      display: grid;
      gap: 12px;
      padding: 18px 20px;
      border: 1px solid var(--border);
      border-radius: 22px;
      background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(246, 240, 229, 0.96));
      box-shadow: var(--shadow);
    }
    .hero-top {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }
    .title-wrap h1 {
      margin: 0;
      font-size: clamp(1.35rem, 3vw, 2rem);
      line-height: 1.1;
    }
    .title-wrap p {
      margin: 8px 0 0;
      color: var(--muted);
      max-width: 78ch;
    }
    .chip-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    .chip,
    .link-chip,
    .toggle-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      border: 1px solid var(--border);
      padding: 8px 14px;
      background: #fff;
      color: var(--ink);
      text-decoration: none;
      font: inherit;
    }
    .toggle-chip {
      cursor: pointer;
      background: var(--accent-soft);
      border-color: rgba(28, 106, 99, 0.25);
    }
    .toggle-chip[data-active="true"] {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }
    .summary-shell {
      display: grid;
      grid-template-columns: minmax(360px, 0.95fr) minmax(420px, 1.05fr);
      gap: 16px;
      align-items: start;
    }
    .panel {
      border: 1px solid var(--border);
      border-radius: 22px;
      background: var(--panel);
      box-shadow: var(--shadow);
      overflow: hidden;
      min-height: 72vh;
    }
    .panel-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 16px 18px;
      border-bottom: 1px solid rgba(215, 204, 183, 0.8);
      background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(247, 241, 232, 0.95));
    }
    .panel-head h2,
    .panel-head h3 {
      margin: 0;
      font-size: 1rem;
    }
    .summary-pane {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
    }
    .summary-scroll,
    .viewer-scroll,
    .artifact-scroll {
      overflow: auto;
      min-height: 0;
    }
    .summary-scroll {
      padding: 14px 16px 18px;
      display: grid;
      gap: 14px;
    }
    .summary-section {
      border: 1px solid rgba(215, 204, 183, 0.75);
      border-radius: 18px;
      background: var(--panel-strong);
      padding: 14px;
      display: grid;
      gap: 10px;
    }
    .summary-section h3 {
      margin: 0;
      font-size: 0.97rem;
      letter-spacing: 0.01em;
    }
    .summary-section[data-debug="true"] {
      border-style: dashed;
      background: rgba(245, 240, 231, 0.95);
    }
    .sentence-list {
      display: grid;
      gap: 8px;
    }
    .sentence-card {
      width: 100%;
      text-align: left;
      border: 1px solid rgba(205, 194, 175, 0.95);
      border-radius: 16px;
      background: #fff;
      padding: 14px 15px;
      color: inherit;
      cursor: pointer;
      display: grid;
      gap: 12px;
      transition: transform 120ms ease, border-color 120ms ease, box-shadow 120ms ease;
    }
    .sentence-card:hover {
      transform: translateY(-1px);
      border-color: rgba(28, 106, 99, 0.45);
      box-shadow: 0 10px 24px rgba(35, 82, 77, 0.12);
    }
    .sentence-card[data-active="true"] {
      border-color: rgba(28, 106, 99, 0.85);
      box-shadow: 0 0 0 2px rgba(28, 106, 99, 0.13), 0 12px 30px rgba(35, 82, 77, 0.16);
    }
    .sentence-card[data-disabled="true"] {
      cursor: default;
      opacity: 0.88;
    }
    .sentence-text {
      margin: 0;
      line-height: 1.64;
      font-size: 0.97rem;
    }
    .sentence-copy {
      display: inline;
    }
    .sentence-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .marker {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 20px;
      height: 20px;
      padding: 0 6px;
      margin-left: 6px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent-strong);
      border: 1px solid rgba(28, 106, 99, 0.22);
      font-weight: 800;
      font-size: 0.72rem;
      line-height: 1;
      vertical-align: super;
      transform: translateY(-0.25em);
      box-shadow: 0 4px 10px rgba(35, 82, 77, 0.08);
    }
    .marker[data-active="true"] {
      background: var(--accent-strong);
      color: #fff;
      border-color: rgba(28, 106, 99, 0.88);
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 0.8rem;
      border: 1px solid rgba(215, 204, 183, 0.95);
      background: rgba(247, 243, 235, 0.95);
      color: var(--muted);
    }
    .badge.warn { color: var(--warn); border-color: rgba(183, 104, 45, 0.24); background: rgba(248, 231, 216, 0.88); }
    .badge.err { color: var(--danger); border-color: rgba(155, 60, 47, 0.24); background: rgba(251, 233, 229, 0.9); }
    .viewer-pane {
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr) auto;
    }
    .viewer-meta {
      padding: 16px 18px 12px;
      display: grid;
      gap: 10px;
      border-bottom: 1px solid rgba(215, 204, 183, 0.8);
    }
    .viewer-meta h3 {
      margin: 0;
      font-size: 1rem;
    }
    .viewer-meta p {
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
    }
    .preview-card,
    .active-card {
      padding: 12px 14px;
      border-radius: 16px;
      border: 1px solid rgba(215, 204, 183, 0.95);
      background: rgba(255, 255, 255, 0.9);
      display: grid;
      gap: 8px;
    }
    .preview-card[hidden] { display: none; }
    .viewer-scroll {
      background: linear-gradient(180deg, rgba(246, 241, 232, 0.95), rgba(240, 233, 220, 0.95));
      padding: 18px 14px 24px;
      position: relative;
      display: grid;
      justify-items: center;
      align-content: flex-start;
      gap: 16px;
    }
    .viewer-page {
      position: relative;
      width: fit-content;
      max-width: 100%;
      border: 1px solid rgba(206, 220, 226, 0.98);
      border-radius: 18px;
      background: #fff;
      box-shadow: 0 18px 34px rgba(58, 44, 24, 0.14);
      overflow: hidden;
    }
    .viewer-page.is-target {
      border-color: rgba(28, 106, 99, 0.62);
      box-shadow: 0 0 0 2px rgba(28, 106, 99, 0.12), 0 18px 34px rgba(58, 44, 24, 0.14);
    }
    .viewer-page-label {
      position: absolute;
      top: 12px;
      left: 12px;
      z-index: 2;
      border-radius: 999px;
      padding: 5px 10px;
      background: rgba(24, 46, 42, 0.84);
      color: #fff;
      font-size: 0.72rem;
      font-weight: 800;
      letter-spacing: 0.02em;
    }
    .viewer-stage {
      position: relative;
      overflow: hidden;
      background: #fff;
    }
    .viewer-image {
      display: block;
      width: 100%;
      height: auto;
      background: #fff;
      user-select: none;
      -webkit-user-drag: none;
    }
    .viewer-caption {
      padding: 10px 14px;
      border-top: 1px solid rgba(215, 204, 183, 0.82);
      background: rgba(248, 244, 236, 0.96);
      color: var(--muted);
      font-size: 0.82rem;
      line-height: 1.45;
    }
    .overlay-box,
    .overlay-union {
      position: absolute;
      pointer-events: none;
      border-radius: 10px;
    }
    .overlay-union {
      border: 3px solid rgba(28, 106, 99, 0.8);
      background: rgba(28, 106, 99, 0.12);
      box-shadow: 0 0 0 9999px rgba(14, 21, 20, 0.08);
    }
    .overlay-box {
      border: 2px solid rgba(255, 255, 255, 0.84);
      background: rgba(28, 106, 99, 0.18);
    }
    .viewer-placeholder {
      min-height: 420px;
      width: 100%;
      display: grid;
      place-items: center;
      text-align: center;
      color: var(--muted);
      padding: 40px 20px;
    }
    .debug-block {
      border-top: 1px solid rgba(215, 204, 183, 0.8);
      padding: 14px 16px 18px;
      background: rgba(248, 244, 236, 0.94);
    }
    .debug-block[hidden] { display: none; }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid rgba(215, 204, 183, 0.9);
      border-radius: 14px;
      padding: 12px 14px;
      font-family: "IBM Plex Mono", monospace;
      font-size: 0.84rem;
      line-height: 1.5;
    }
    .artifacts {
      display: grid;
      gap: 8px;
      padding: 14px 16px 18px;
    }
    .artifact-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      border: 1px solid rgba(215, 204, 183, 0.86);
      border-radius: 14px;
      padding: 10px 12px;
      background: rgba(255,255,255,0.88);
    }
    .artifact-row strong { display: block; }
    .artifact-row small { color: var(--muted); word-break: break-all; }
    .status-banner {
      border: 1px solid rgba(155, 60, 47, 0.24);
      background: rgba(251, 233, 229, 0.92);
      color: var(--danger);
      border-radius: 16px;
      padding: 12px 14px;
      line-height: 1.45;
    }
    .muted { color: var(--muted); }
    .summary-shell {
      display: grid;
      grid-template-columns: minmax(520px, 1.08fr) minmax(380px, 0.92fr);
      gap: 16px;
      align-items: stretch;
    }
    .viewer-pane {
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr) auto;
      min-height: 78vh;
      background: #f6fbf9;
    }
    .summary-pane {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      min-height: 78vh;
    }
    .pdf-viewer {
      position: relative;
      border-top: 1px solid rgba(215, 204, 183, 0.82);
      background: #f8fdff;
      min-height: 620px;
      overflow: auto;
      overscroll-behavior: contain;
      display: grid;
      justify-items: center;
      align-content: flex-start;
      gap: 14px;
      padding: 12px;
    }
    .pdf-placeholder {
      min-height: 420px;
      width: 100%;
      display: grid;
      place-items: center;
      text-align: center;
      color: var(--muted);
      padding: 40px 20px;
    }
    .pdf-page {
      position: relative;
      width: fit-content;
      max-width: 100%;
      border: 1px solid #d9e8ef;
      border-radius: 12px;
      background: #fff;
      box-shadow: 0 10px 24px rgba(13, 76, 103, 0.1);
      overflow: hidden;
    }
    .pdf-page.is-target {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(28, 106, 99, 0.18), 0 18px 34px rgba(13, 76, 103, 0.12);
    }
    .pdf-page-label {
      position: absolute;
      top: 10px;
      left: 10px;
      z-index: 2;
      border-radius: 999px;
      padding: 4px 9px;
      background: rgba(24, 46, 42, 0.84);
      color: #fff;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.02em;
    }
    .pdf-stage {
      position: relative;
      overflow: hidden;
      background: #fff;
    }
    .pdf-page-image {
      display: block;
      width: 100%;
      height: auto;
      background: #fff;
      user-select: none;
      -webkit-user-drag: none;
    }
    .pdf-text-layer {
      position: absolute;
      inset: 0;
      pointer-events: none;
    }
    .citation-union {
      position: absolute;
      pointer-events: none;
      border: 3px solid rgba(28, 106, 99, 0.82);
      background: rgba(28, 106, 99, 0.12);
      border-radius: 12px;
      box-shadow: 0 0 0 9999px rgba(14, 21, 20, 0.08);
    }
    .citation-box {
      position: absolute;
      pointer-events: auto;
      appearance: none;
      padding: 0;
      margin: 0;
      border: 2px solid rgba(255, 163, 84, 0.96);
      background: rgba(255, 199, 140, 0.14);
      border-radius: 10px;
      box-shadow: 0 10px 22px rgba(95, 58, 22, 0.18);
      cursor: pointer;
      transition: transform 0.14s ease, box-shadow 0.18s ease, border-color 0.18s ease, background 0.18s ease;
    }
    .citation-box:hover,
    .citation-box:focus-visible,
    .citation-box[data-active="true"] {
      border-color: rgba(255, 138, 69, 1);
      background: rgba(255, 205, 150, 0.2);
      box-shadow: 0 14px 28px rgba(95, 58, 22, 0.24), 0 0 0 3px rgba(255, 138, 69, 0.18);
      outline: none;
      transform: translateY(-1px);
    }
    .pdf-page-meta {
      padding: 10px 14px;
      border-top: 1px solid rgba(215, 204, 183, 0.82);
      background: rgba(248, 244, 236, 0.96);
      color: var(--muted);
      font-size: 0.82rem;
      line-height: 1.45;
    }
    @media (max-width: 1100px) {
      .summary-shell { grid-template-columns: 1fr; }
      .panel { min-height: auto; }
      .viewer-pane { min-height: 60vh; }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="hero-top">
        <div class="title-wrap">
          <h1 id="page-title">Loading summary results…</h1>
          <p id="page-subtitle" class="muted">Preparing citation-backed summary output.</p>
        </div>
        <div class="chip-row">
          <button id="debug-toggle" class="toggle-chip" type="button" data-active="false">Debug</button>
          <a id="status-link" class="link-chip" href="#">JSON Status</a>
          <a id="trace-link" class="link-chip" href="#">Trace</a>
          <a id="summary-json-link" class="link-chip" href="#">Summary JSON</a>
        </div>
      </div>
      <div id="status-banner" class="status-banner" hidden></div>
    </section>

    <section class="summary-shell">
      <div class="panel viewer-pane">
        <div class="panel-head">
          <h2>Source Viewer</h2>
          <span id="viewer-chip" class="badge">No citation selected</span>
        </div>
        <div class="viewer-meta">
          <div class="active-card">
            <h3 id="active-title">Select a sentence to inspect the source.</h3>
            <p id="active-meta">The viewer will jump to the rendered PDF page and highlight the matched OCR region.</p>
            <div class="chip-row">
              <a id="source-pdf-link" class="link-chip" href="#" hidden>Open Source PDF</a>
            </div>
          </div>
        </div>
        <div id="viewer-scroll" class="viewer-scroll pdf-viewer">
          <div id="viewer-placeholder" class="pdf-placeholder">Select a summary sentence to open the cited evidence.</div>
        </div>
        <div id="debug-block" class="debug-block" hidden>
          <h3>Active Citation Debug</h3>
          <pre id="debug-json">{}</pre>
        </div>
      </div>

      <div class="panel summary-pane">
        <div class="panel-head">
          <h2>Grounded Summary</h2>
          <span id="summary-count" class="badge">0 cited sentences</span>
        </div>
        <div id="summary-scroll" class="summary-scroll"></div>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <h2>Artifacts</h2>
        <span class="badge">Run-side outputs</span>
      </div>
      <div id="artifact-scroll" class="artifacts"></div>
    </section>
  </div>

  <script>
    const jobId = "__JOB_ID__";
    const state = {
      model: null,
      debugMode: new URLSearchParams(window.location.search).get("debug") === "1",
      citationById: new Map(),
      citationByNumber: new Map(),
      pageByKey: new Map(),
      activeCitationId: null,
      activeItemId: null,
    };

    const el = {
      pageTitle: document.getElementById("page-title"),
      pageSubtitle: document.getElementById("page-subtitle"),
      statusBanner: document.getElementById("status-banner"),
      debugToggle: document.getElementById("debug-toggle"),
      statusLink: document.getElementById("status-link"),
      traceLink: document.getElementById("trace-link"),
      summaryJsonLink: document.getElementById("summary-json-link"),
      summaryCount: document.getElementById("summary-count"),
      summaryScroll: document.getElementById("summary-scroll"),
      viewerChip: document.getElementById("viewer-chip"),
      activeTitle: document.getElementById("active-title"),
      activeMeta: document.getElementById("active-meta"),
      sourcePdfLink: document.getElementById("source-pdf-link"),
      viewerScroll: document.getElementById("viewer-scroll"),
      viewerPlaceholder: document.getElementById("viewer-placeholder"),
      debugBlock: document.getElementById("debug-block"),
      debugJson: document.getElementById("debug-json"),
      artifactScroll: document.getElementById("artifact-scroll"),
    };

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function citationForId(id) {
      return id ? state.citationById.get(String(id)) || null : null;
    }

    function citationNumbersForItem(item) {
      return Array.isArray(item?.citation_numbers) ? item.citation_numbers : [];
    }

    function firstCitationForItem(item) {
      if (!Array.isArray(item?.citation_ids) || !item.citation_ids.length) return null;
      return citationForId(item.citation_ids[0]);
    }

    function pageForCitation(citation) {
      if (citation?.page_key && state.pageByKey.has(String(citation.page_key))) {
        return state.pageByKey.get(String(citation.page_key)) || null;
      }
      return Array.isArray(state.model?.source_pages) && state.model.source_pages.length
        ? state.model.source_pages[0]
        : null;
    }

    function getViewerPageElement(pageKey) {
      return Array.from(el.viewerScroll.querySelectorAll(".pdf-page")).find(
        (node) => node.dataset.pageKey === String(pageKey || "")
      ) || null;
    }

    function focusSummaryItem(itemId, smooth = true) {
      if (!itemId) return false;
      const target = el.summaryScroll.querySelector(`.sentence-card[data-item-id="${String(itemId)}"]`);
      if (!(target instanceof HTMLElement)) return false;
      target.scrollIntoView({ block: "center", behavior: smooth ? "smooth" : "auto" });
      return true;
    }

    function setDebugMode(nextValue) {
      state.debugMode = !!nextValue;
      el.debugToggle.dataset.active = String(state.debugMode);
      const url = new URL(window.location.href);
      if (state.debugMode) url.searchParams.set("debug", "1");
      else url.searchParams.delete("debug");
      window.history.replaceState({}, "", url);
      renderSections();
      renderArtifacts();
      renderActiveCitation();
    }

    function showStatusError(message) {
      el.statusBanner.hidden = false;
      el.statusBanner.textContent = String(message || "Unknown error");
    }

    function hideStatusError() {
      el.statusBanner.hidden = true;
      el.statusBanner.textContent = "";
    }

    function renderArtifacts() {
      const artifacts = state.model?.artifacts || {};
      el.artifactScroll.innerHTML = "";
      Object.entries(artifacts).forEach(([label, row]) => {
        if (!state.debugMode && (label === "trace_events" || label === "trace_summary")) return;
        const wrap = document.createElement("div");
        wrap.className = "artifact-row";
        const left = document.createElement("div");
        left.innerHTML = `<strong>${escapeHtml(label)}</strong><small>${escapeHtml(row.path || "")}</small>`;
        wrap.appendChild(left);
        if (row.url) {
          const link = document.createElement("a");
          link.className = "link-chip";
          link.href = row.url;
          link.textContent = "Open";
          wrap.appendChild(link);
        }
        el.artifactScroll.appendChild(wrap);
      });
      if (!el.artifactScroll.childElementCount) {
        el.artifactScroll.innerHTML = '<div class="muted">No artifact links available.</div>';
      }
    }

    function computeDisplayScale(citation, page) {
      const width = Number(page?.width || citation?.page_width || 1200);
      const availableWidth = Math.max(320, Math.min((el.viewerScroll.clientWidth || 980) - 28, 980));
      return Math.min(1, availableWidth / Math.max(width, 1));
    }

    function scrollViewerToElement(node, centerFraction = 0.2, smooth = true) {
      if (!(node instanceof HTMLElement)) return;
      const container = el.viewerScroll;
      const containerRect = container.getBoundingClientRect();
      const nodeRect = node.getBoundingClientRect();
      const targetTop = container.scrollTop + (nodeRect.top - containerRect.top) - (container.clientHeight * centerFraction);
      container.scrollTo({ top: Math.max(0, Math.floor(targetTop)), behavior: smooth ? "smooth" : "auto" });
    }

    function scrollViewerHitIntoView(hitNode, smooth = true) {
      scrollViewerToElement(hitNode, 0.35, smooth);
    }

    function focusViewerPage(pageKey, smooth = true) {
      const target = getViewerPageElement(pageKey);
      if (!(target instanceof HTMLElement)) return false;
      el.viewerScroll.querySelectorAll(".pdf-page.is-target").forEach((node) => node.classList.remove("is-target"));
      target.classList.add("is-target");
      scrollViewerToElement(target, 0.08, smooth);
      return true;
    }

    function centerOnCitation(citation, _scale, stage) {
      if (!(stage instanceof HTMLElement)) {
        el.viewerScroll.scrollTo({ top: 0, left: 0, behavior: "smooth" });
        return;
      }
      const firstHit = stage.querySelector('.citation-box[data-active="true"]') || stage.querySelector('.citation-union');
      if (firstHit instanceof HTMLElement) {
        scrollViewerHitIntoView(firstHit, true);
        return;
      }
      const pageNode = stage.closest('.pdf-page');
      if (pageNode instanceof HTMLElement) {
        scrollViewerToElement(pageNode, 0.12, true);
        return;
      }
      el.viewerScroll.scrollTo({ top: 0, left: 0, behavior: "smooth" });
    }

    function renderViewer(citation) {
      const sourcePages = Array.isArray(state.model?.source_pages) ? state.model.source_pages : [];
      const activePage = pageForCitation(citation);
      const sourcePdfUrl = citation?.source_pdf_url || activePage?.source_pdf_url || state.model?.urls?.source_pdf || null;

      el.viewerScroll.innerHTML = "";
      el.sourcePdfLink.hidden = !sourcePdfUrl;
      if (sourcePdfUrl) {
        const pageNumber = citation?.page || activePage?.page || 1;
        el.sourcePdfLink.href = `${sourcePdfUrl}#page=${pageNumber}`;
      }

      if (!citation) {
        el.viewerChip.textContent = "No citation selected";
      } else {
        el.viewerChip.textContent = `Citation ${citation.number}`;
      }

      if (!sourcePages.length) {
        const placeholder = document.createElement("div");
        placeholder.className = "pdf-placeholder";
        placeholder.textContent = "No rendered source page is available for this summary item.";
        el.viewerScroll.appendChild(placeholder);
        return;
      }

      const fragment = document.createDocumentFragment();
      const pagesToRender = activePage ? [activePage] : sourcePages.slice(0, 1);
      pagesToRender.forEach((page) => {
        const imageUrl = page.image_url || (citation?.page_key === page.page_key ? citation.page_image_url : null);
        if (!imageUrl) return;

        const scale = computeDisplayScale(citation, page);
        const width = Number(page.width || citation?.page_width || 1200);
        const height = Number(page.height || citation?.page_height || 1600);
        const displayWidth = Math.round(width * scale);
        const displayHeight = Math.round(height * scale);
        const isTargetPage = !!citation && String(page.page_key || "") === String(citation.page_key || "");

        const pageCard = document.createElement("section");
        pageCard.className = "pdf-page";
        pageCard.dataset.page = String(page.page || 0);
        pageCard.dataset.pageKey = String(page.page_key || "");
        if (isTargetPage) pageCard.classList.add("is-target");

        const label = document.createElement("div");
        label.className = "pdf-page-label";
        label.textContent = `${page.pdf_id} · Page ${page.page}`;
        pageCard.appendChild(label);

        const stage = document.createElement("div");
        stage.className = "pdf-stage";
        stage.style.width = `${displayWidth}px`;
        stage.style.height = `${displayHeight}px`;
        pageCard.appendChild(stage);

        const image = document.createElement("img");
        image.className = "pdf-page-image";
        image.src = imageUrl;
        image.alt = `${page.pdf_id} page ${page.page}`;
        image.width = displayWidth;
        image.height = displayHeight;
        stage.appendChild(image);

        const textLayer = document.createElement("div");
        textLayer.className = "pdf-text-layer";
        stage.appendChild(textLayer);

        if (isTargetPage && citation?.bbox?.width && citation?.bbox?.height && !citation.degraded) {
          const union = document.createElement("div");
          union.className = "citation-union";
          union.style.left = `${(citation.bbox.left || 0) * scale}px`;
          union.style.top = `${(citation.bbox.top || 0) * scale}px`;
          union.style.width = `${(citation.bbox.width || 0) * scale}px`;
          union.style.height = `${(citation.bbox.height || 0) * scale}px`;
          textLayer.appendChild(union);
        }

        const boxes = isTargetPage && Array.isArray(citation?.boxes) ? citation.boxes : [];
        boxes.forEach((box, index) => {
          const bbox = box?.bbox || {};
          if (!bbox.width || !bbox.height) return;
          const hit = document.createElement("button");
          hit.type = "button";
          hit.className = "citation-box";
          hit.dataset.active = "true";
          hit.dataset.boxIndex = String(index);
          hit.style.left = `${Math.max(0, (bbox.left || 0) * scale)}px`;
          hit.style.top = `${Math.max(0, (bbox.top || 0) * scale)}px`;
          hit.style.width = `${Math.max(10, (bbox.width || 0) * scale)}px`;
          hit.style.height = `${Math.max(10, (bbox.height || 0) * scale)}px`;
          hit.title = `[${citation.number}] ${citation.label}`;
          hit.setAttribute("aria-label", `Citation ${citation.number} region ${index + 1} on page ${page.page}`);
          hit.addEventListener("click", () => {
            activateCitation(citation.id, state.activeItemId);
            focusSummaryItem(state.activeItemId);
          });
          textLayer.appendChild(hit);
        });

        const caption = document.createElement("div");
        caption.className = "pdf-page-meta";
        if (!citation) {
          caption.textContent = `Available source page ${page.page} for ${page.pdf_id}.`;
        } else if (isTargetPage && citation.degraded) {
          caption.textContent = `Degraded citation mapping on ${page.pdf_id} page ${page.page}. The page is correct, but the highlight falls back to page-level evidence.`;
        } else if (isTargetPage) {
          caption.textContent = `Matched OCR paragraphs on ${page.pdf_id} page ${page.page}. Click a highlighted region to return to the linked summary sentence.`;
        } else {
          caption.textContent = `Showing source page ${page.page} for ${page.pdf_id}.`;
        }
        pageCard.appendChild(caption);

        image.addEventListener("load", () => {
          if (isTargetPage) centerOnCitation(citation, scale, stage);
        }, { once: true });

        fragment.appendChild(pageCard);
      });

      if (!fragment.childElementCount) {
        const placeholder = document.createElement("div");
        placeholder.className = "pdf-placeholder";
        placeholder.textContent = "No rendered source page is available for this summary item.";
        el.viewerScroll.appendChild(placeholder);
        return;
      }
      el.viewerScroll.appendChild(fragment);
      if (activePage?.page_key) {
        requestAnimationFrame(() => focusViewerPage(activePage.page_key, false));
      }
    }

    function renderActiveCitation() {
      const citation = citationForId(state.activeCitationId);
      if (!citation) {
        el.activeTitle.textContent = "Select a sentence to inspect the source.";
        el.activeMeta.textContent = "The viewer will jump to the rendered PDF page and highlight the matched OCR region.";
        el.debugBlock.hidden = true;
        renderViewer(null);
        return;
      }
      el.activeTitle.textContent = `${citation.pdf_id} · Page ${citation.page}`;
      el.activeMeta.textContent = citation.degraded
        ? `${citation.snippet || citation.label} • degraded mapping fallback on page ${citation.page}`
        : `${citation.snippet || citation.label}`;
      el.debugBlock.hidden = !state.debugMode;
      el.debugJson.textContent = JSON.stringify(citation, null, 2);
      renderViewer(citation);
    }

    function activateCitation(citationId, itemId = null) {
      const citation = citationForId(citationId);
      if (!citation) return;
      state.activeCitationId = citation.id;
      state.activeItemId = itemId || state.activeItemId;
      const url = new URL(window.location.href);
      url.searchParams.set("citation", String(citation.number));
      if (state.debugMode) url.searchParams.set("debug", "1");
      else url.searchParams.delete("debug");
      window.history.replaceState({}, "", url);
      renderSections();
      renderActiveCitation();
    }

    function renderSections() {
      const sections = Array.isArray(state.model?.sections) ? state.model.sections : [];
      el.summaryScroll.innerHTML = "";
      let visibleSentenceCount = 0;
      sections.forEach((section) => {
        if (section.debug_only && !state.debugMode) return;
        const wrap = document.createElement("section");
        wrap.className = "summary-section";
        wrap.dataset.debug = String(!!section.debug_only);
        const heading = document.createElement("h3");
        heading.textContent = section.title;
        wrap.appendChild(heading);

        const list = document.createElement("div");
        list.className = "sentence-list";
        (section.items || []).forEach((item) => {
          visibleSentenceCount += 1;
          const sentence = document.createElement("button");
          sentence.type = "button";
          sentence.className = "sentence-card";
          sentence.dataset.itemId = item.item_id;
          sentence.dataset.active = String(item.item_id === state.activeItemId);
          sentence.dataset.disabled = String(!item.citation_ids || !item.citation_ids.length);
          if (!item.citation_ids || !item.citation_ids.length) {
            sentence.disabled = true;
          }
          sentence.addEventListener("click", () => {
            const citation = firstCitationForItem(item);
            if (!citation) return;
            state.activeItemId = item.item_id;
            activateCitation(citation.id, item.item_id);
          });

          const text = document.createElement("p");
          text.className = "sentence-text";
          const copy = document.createElement("span");
          copy.className = "sentence-copy";
          copy.textContent = item.text;
          text.appendChild(copy);
          (citationNumbersForItem(item)).forEach((number) => {
            const citationRow = state.citationByNumber.get(number) || null;
            const marker = document.createElement("span");
            marker.className = "marker";
            marker.dataset.active = String(number === (citationForId(state.activeCitationId)?.number || null));
            marker.textContent = String(number);
            marker.title = citationRow?.label || `Citation ${number}`;
            text.appendChild(marker);
          });
          sentence.appendChild(text);

          const meta = document.createElement("div");
          meta.className = "sentence-meta";
          if (item.degraded) {
            const degraded = document.createElement("span");
            degraded.className = "badge warn";
            degraded.textContent = "Degraded";
            meta.appendChild(degraded);
          }
          if (item.supported === false) {
            const unsupported = document.createElement("span");
            unsupported.className = "badge err";
            unsupported.textContent = "Unsupported";
            meta.appendChild(unsupported);
          }
          if (meta.childElementCount) {
            sentence.appendChild(meta);
          }
          list.appendChild(sentence);
        });
        wrap.appendChild(list);
        el.summaryScroll.appendChild(wrap);
      });
      if (!el.summaryScroll.childElementCount) {
        el.summaryScroll.innerHTML = '<div class="muted">No summary sections are available yet.</div>';
      }
      el.summaryCount.textContent = `${visibleSentenceCount} cited sentence${visibleSentenceCount === 1 ? "" : "s"}`;
    }

    async function loadModel() {
      const response = await fetch(`/ui/runs/${encodeURIComponent(jobId)}/summary-data`);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `Failed to load summary data (${response.status})`);
      }
      return payload;
    }

    async function bootstrap() {
      try {
        const model = await loadModel();
        state.model = model;
        state.citationById = new Map();
        state.citationByNumber = new Map();
        state.pageByKey = new Map();
        (model.citation_index || []).forEach((citation) => {
          state.citationById.set(String(citation.id), citation);
          state.citationByNumber.set(Number(citation.number), citation);
        });
        (model.source_pages || []).forEach((page) => {
          state.pageByKey.set(String(page.page_key), page);
        });

        el.pageTitle.textContent = model.title || "Summary Results";
        el.pageSubtitle.textContent = `${model.scope?.title || "Summary"} • ${model.job?.status || "unknown status"}`;
        el.statusLink.href = model.urls?.status || "#";
        el.traceLink.href = model.urls?.trace || "#";
        el.summaryJsonLink.href = model.urls?.summary_json || "#";
        if (model.job?.summary_error) {
          showStatusError(model.job.summary_error);
        } else {
          hideStatusError();
        }

        renderArtifacts();
        renderSections();
        const queryCitation = Number(new URLSearchParams(window.location.search).get("citation") || 0);
        if (queryCitation && state.citationByNumber.has(queryCitation)) {
          const citation = state.citationByNumber.get(queryCitation);
          const item = (model.sections || []).flatMap((section) => section.items || []).find((row) => (row.citation_ids || []).includes(citation.id));
          state.activeItemId = item?.item_id || null;
          activateCitation(citation.id, state.activeItemId);
        } else if ((model.citation_index || []).length) {
          const first = model.citation_index[0];
          const item = (model.sections || []).flatMap((section) => section.items || []).find((row) => (row.citation_ids || []).includes(first.id));
          state.activeItemId = item?.item_id || null;
          activateCitation(first.id, state.activeItemId);
        } else {
          renderActiveCitation();
        }
        setDebugMode(state.debugMode);
      } catch (error) {
        showStatusError(error instanceof Error ? error.message : String(error));
        el.summaryScroll.innerHTML = '<div class="muted">Summary data could not be loaded.</div>';
        renderViewer(null);
      }
    }

    el.debugToggle.addEventListener("click", () => setDebugMode(!state.debugMode));
    bootstrap();
  </script>
</body>
</html>
"""
    return html_page.replace("__JOB_ID__", html.escape(job_id))


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _render_home()


@app.get("/health")
async def health() -> dict[str, object]:
    gateway = OpenRAGGateway(_settings())
    return await gateway.health()


@app.get("/diagnostics")
async def diagnostics() -> dict[str, object]:
    settings = _settings()
    gateway = OpenRAGGateway(settings)
    inspector = OpenSearchInspector(settings)
    langflow = LangflowGateway(settings)
    return {
        "openrag": await gateway.health(),
        "langflow": (await langflow.diagnostics()).model_dump(mode="json"),
        "opensearch": (await inspector.diagnostics()).model_dump(mode="json"),
    }


@app.post("/upgrade-openrag-flows")
async def upgrade_openrag_flows(request: FlowUpgradeRequest) -> dict[str, object]:
    gateway = LangflowGateway(_settings())
    result = await gateway.upgrade_flows(
        output_dir=Path(request.output_dir) if request.output_dir else None,
        patch_live=request.patch_live,
    )
    return result.model_dump(mode="json")


@app.post("/materialize")
async def materialize(request: MaterializeRequest):
    settings = _settings()
    if request.input_path:
        run_id, payload, source_kind = load_run_json(
            Path(request.input_path),
            run_id=request.run_id,
        )
    else:
        if not request.run_id or not request.mongo_uri or not request.mongo_database:
            raise ValueError("run_id, mongo_uri, and mongo_database are required for Mongo materialization")
        run_id, payload, source_kind = load_run_from_mongo(
            run_id=request.run_id,
            mongo_uri=request.mongo_uri,
            mongo_database=request.mongo_database,
            mongo_collection=request.mongo_collection,
        )
    output_dir = Path(request.output_dir) if request.output_dir else settings.materialized_root / run_id
    manifest = materialize_summary_payload(
        run_id=run_id,
        summary_payload=payload,
        source_kind=source_kind,
        output_dir=output_dir,
        include_non_survivors=request.include_non_survivors,
    )
    return manifest.model_dump()


@app.post("/ingest")
async def ingest(request: IngestRequest):
    gateway = OpenRAGGateway(_settings())
    manifest_path = Path(request.manifest_path)
    manifest = load_manifest(manifest_path)
    if request.apply_recommended_settings:
        await gateway.apply_recommended_settings()
    results = await gateway.ingest_manifest(manifest, manifest_dir=manifest_path.parent)
    return {
        "manifest": manifest.model_dump(),
        "results": [row.model_dump() for row in results],
    }


@app.post("/filters")
async def create_filter(request: ScopeRequest):
    gateway = OpenRAGGateway(_settings())
    manifest_path = Path(request.manifest_path)
    manifest = load_manifest(manifest_path)
    scope = load_scopes(Path(request.scope_path))[0]
    result = await gateway.upsert_scope_filter(
        manifest=manifest,
        scope=scope,
        data_sources=resolve_scope_retrieval_sources(manifest, scope),
    )
    return result.model_dump()


@app.post("/summaries")
async def run_summary(request: ScopeRequest):
    settings = _settings()
    gateway = OpenRAGGateway(settings)
    manifest_path = Path(request.manifest_path)
    manifest = load_manifest(manifest_path)
    scope = load_scopes(Path(request.scope_path))[0]
    if request.apply_recommended_settings:
        await gateway.apply_recommended_settings()
    result = await summarize_scope(
        gateway,
        manifest=manifest,
        scope=scope,
        settings=settings,
    )
    return result.model_dump()


@app.get("/ui/runs")
async def list_ui_runs() -> dict[str, Any]:
    return {"runs": list(_jobs.values())}


@app.post("/ui/runs")
async def create_ui_run(
    pdf_path: str | None = Form(default=None),
    credentials_path: str | None = Form(default=None),
    question: str | None = Form(default=None),
    max_pages: int | None = Form(default=None),
    upload: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    settings = _settings()
    resolved_pdf_path: Path | None = None
    if upload and upload.filename:
        job_id = uuid4().hex
        upload_dir = settings.extraction_root / "uploads" / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        resolved_pdf_path = upload_dir / upload.filename
        with resolved_pdf_path.open("wb") as handle:
            handle.write(await upload.read())
    elif pdf_path:
        resolved_pdf_path = Path(pdf_path)
        job_id = uuid4().hex
    else:
        raise HTTPException(status_code=400, detail="pdf_path or upload is required")

    if not resolved_pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {resolved_pdf_path}")

    resolved_credentials = Path(credentials_path) if credentials_path else settings.google_application_credentials
    if resolved_credentials and not resolved_credentials.exists():
        raise HTTPException(status_code=404, detail=f"Credentials not found: {resolved_credentials}")
    _set_job(
        job_id,
        status="queued",
        pdf_path=resolved_pdf_path.as_posix(),
        credentials_path=resolved_credentials.as_posix() if resolved_credentials else None,
        question=question,
        max_pages=max_pages,
    )
    asyncio.create_task(
        _run_job(
            job_id=job_id,
            pdf_path=resolved_pdf_path,
            credentials_path=resolved_credentials,
            question=question,
            max_pages=max_pages,
        )
    )
    return _job_payload(job_id)


@app.get("/ui/runs/{job_id}")
async def get_ui_run(job_id: str) -> dict[str, Any]:
    return _job_payload(job_id)


@app.get("/ui/runs/{job_id}/summary-data")
async def get_ui_summary_data(job_id: str) -> dict[str, Any]:
    payload = _job_payload(job_id)
    return _build_summary_view_model(job_id, payload)


@app.get("/ui/runs/{job_id}/summary", response_class=HTMLResponse)
async def get_ui_summary(job_id: str) -> str:
    _job_payload(job_id)
    return _render_summary_page(job_id)


@app.get("/ui/runs/{job_id}/trace", response_class=HTMLResponse)
async def get_ui_trace(job_id: str) -> str:
    payload = _job_payload(job_id)
    return _render_trace_page(job_id, payload)
