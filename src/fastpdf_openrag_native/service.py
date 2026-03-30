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
        summary_error=result.summary_error,
        debug_artifacts=_job_debug_artifacts(result.model_dump(mode="json")),
    )


def _render_home() -> str:
    jobs = sorted(_jobs.values(), key=lambda row: row.get("job_id", ""), reverse=True)
    job_rows = "\n".join(
        (
            "<tr>"
            f"<td><a href=\"/ui/runs/{html.escape(row['job_id'])}/trace\">{html.escape(row['job_id'])}</a></td>"
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
        if (payload.status === 'completed' || payload.status === 'failed') {{
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
    error_html = ""
    if payload.get("error"):
        error_html = (
            "<section>"
            "<h2>Failure</h2>"
            f"<pre>{html.escape(str(payload.get('error')))}</pre>"
            f"<pre>{html.escape(str(payload.get('error_traceback', '')))}</pre>"
            "</section>"
        )
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


@app.get("/ui/runs/{job_id}/trace", response_class=HTMLResponse)
async def get_ui_trace(job_id: str) -> str:
    payload = _job_payload(job_id)
    return _render_trace_page(job_id, payload)
