# PDF Pipeline Debug Guide

This file is the detailed runbook for the PDF workflow in `fastpdf-openrag-native`.

It covers:

- how a PDF is turned into HTML
- what OpenRAG endpoint is called at each stage
- where every artifact is stored
- how to inspect traces, chunks, and OpenSearch health
- how to run the debug UI

## End-to-end pipeline

For `uv run fastpdf-openrag-native process-pdf --pdf ... --credentials ...`, the pipeline is:

1. `pymupdf` opens the source PDF.
2. `pymupdf` renders each page to a PNG image.
3. `google.cloud.vision.ImageAnnotatorClient.document_text_detection` OCRs each rendered page image.
4. `fastpdf-openrag-native` writes:
   - page image
   - raw OCR JSON
   - OCR plain text
   - per-page HTML document
   - `manifest.json`
5. `fastpdf-openrag-native` calls OpenRAG `POST /api/v1/settings` if `--apply-recommended-settings` is active.
6. `fastpdf-openrag-native` calls OpenRAG `DELETE /api/v1/documents` per file before ingest.
7. `fastpdf-openrag-native` calls OpenRAG `POST /api/v1/documents/ingest` per file.
8. `fastpdf-openrag-native` polls OpenRAG `GET /api/v1/tasks/{task_id}` until completion or timeout.
9. `fastpdf-openrag-native` creates a knowledge filter over the selected filenames.
10. `fastpdf-openrag-native` summarizes each page with OpenRAG `POST /api/v1/chat`, relying on the upgraded Langflow agent flow to hybrid-search and backend-rerank the scoped indexed chunks.
11. `fastpdf-openrag-native` reduces all page summaries with OpenRAG `POST /api/v1/chat` through the same upgraded flow.
12. `fastpdf-openrag-native` verifies each final sentence with the hotfixed OpenRAG `POST /api/v1/search` endpoint, which widens and reranks the backend candidate set before responding.
13. `fastpdf-openrag-native` queries OpenSearch directly to dump stored chunks by filename when the index is healthy.

Before steps 10 and 11 matter, the Langflow flows on this machine should already be patched once with:

```bash
uv run fastpdf-openrag-native upgrade-openrag-flows
```

## Services and outputs

### `pymupdf`

Produces:

- `artifacts/<run-id>__p0001.png`

Debug record:

- trace event with `service: pymupdf`
- action `open_pdf`
- action `render_page_image`
- note that PyMuPDF is used only for rendering, not as a text source

### `google_vision`

Produces:

- `artifacts/<run-id>__p0001.google_vision.json`
- `artifacts/<run-id>__p0001.txt`

Debug record:

- trace event with `service: google_vision`
- action `document_text_detection`
- paragraph count
- OCR text length
- credential path used

### `fastpdf-openrag-native`

Produces:

- `pages/<run-id>__p0001.html`
- `manifest.json`

The generated HTML contains:

- source PDF name
- page image reference
- OCR paragraphs with bounding-box metadata attributes
- full OCR text

### `openrag`

Produces:

- task ids for ingest
- knowledge filter ids
- summary responses
- search verification responses

Debug record:

- trace events with `service: openrag`
- request filename, status, task id, delete error
- summary failure or success
- knowledge filter id and name

### `opensearch`

Produces:

- diagnostics snapshot
- chunk dump files when searchable

Debug record:

- cluster health
- shard allocation explanation
- mapping for the `documents` index
- raw chunk dump or `503` error

## Important directories

Given a run id like `merged_notes-20260329T181641Z`:

- `data/extracted/merged_notes-20260329T181641Z/manifest.json`
- `data/extracted/merged_notes-20260329T181641Z/pages/`
- `data/extracted/merged_notes-20260329T181641Z/artifacts/`
- `outputs/traces/merged_notes-20260329T181641Z/events.jsonl`
- `outputs/traces/merged_notes-20260329T181641Z/summary.json`
- `outputs/traces/merged_notes-20260329T181641Z/chunks/`
- `outputs/merged_notes-20260329T181641Z/all-pages.summary.json`

## Commands you will use most

### 1. Stack diagnostics

```bash
uv run fastpdf-openrag-native diagnose-stack
```

Use this first.

If `opensearch.cluster_health.status` is `red`, then retrieval is already broken before your summary request begins.

If the `langflow` section does not show `agent_flow_rerank_marker_present: true`, patch the mounted flows:

```bash
uv run fastpdf-openrag-native upgrade-openrag-flows
```

If you want the full local restart path, including the public-search hotfix mount and recommended settings reapply:

```bash
bash ./scripts/restart_openrag_stack.sh
```

### 2. Extract without OpenRAG ingest

```bash
uv run fastpdf-openrag-native extract-pdf \
  --pdf /home/divyesh-nandlal-vishwakarma/Downloads/chinchin/merged_notes.pdf \
  --credentials /home/divyesh-nandlal-vishwakarma/Downloads/ocr-neuralit-4e01e06ccf84\(2\).json \
  --max-pages 2
```

### 3. Full pipeline

```bash
uv run fastpdf-openrag-native process-pdf \
  --pdf /home/divyesh-nandlal-vishwakarma/Downloads/chinchin/merged_notes.pdf \
  --credentials /home/divyesh-nandlal-vishwakarma/Downloads/ocr-neuralit-4e01e06ccf84\(2\).json
```

If you pass `--max-pages N`, the run is intentionally partial. The CLI prints `warning=partial_run ...`, and `manifest.json` records `requested_max_pages` and `is_partial_run`.

### 4. Short diagnostic run

When the stack is unhealthy, use a bounded run:

```bash
FASTPDF_OPENRAG_INGEST_WAIT_TIMEOUT=30 \
uv run fastpdf-openrag-native process-pdf \
  --pdf /home/divyesh-nandlal-vishwakarma/Downloads/chinchin/merged_notes.pdf \
  --credentials /home/divyesh-nandlal-vishwakarma/Downloads/ocr-neuralit-4e01e06ccf84\(2\).json \
  --max-pages 1
```

### 5. Inspect a stored trace

```bash
tail -n 50 outputs/traces/merged_notes-20260329T181641Z/events.jsonl
cat outputs/traces/merged_notes-20260329T181641Z/summary.json
```

### 6. Inspect extracted OCR files

```bash
find data/extracted/merged_notes-20260329T181641Z -maxdepth 3 -type f | sort
sed -n '1,160p' data/extracted/merged_notes-20260329T181641Z/manifest.json
sed -n '1,160p' data/extracted/merged_notes-20260329T181641Z/pages/merged_notes-20260329T181641Z__p0001.html
```

### 7. Inspect chunks through the repo wrapper

```bash
uv run fastpdf-openrag-native inspect-chunks \
  --filename merged_notes-20260329T181641Z__p0001.html
```

This succeeds only if the `documents` index is searchable.

### 8. Run the debug UI

```bash
uv run fastpdf-openrag-native serve-ui --port 8077
```

Open:

- `http://127.0.0.1:8077/`

The UI shows:

- queued/running/completed/failed runs
- current JSON status
- a trace page for each run
- direct links to manifest, extracted pages, OCR artifacts, trace JSON, final summary JSON, and chunk dumps

## Current `merged_notes` manifest state

The repo-local file:

- `data/extracted/merged_notes/manifest.json`

currently materializes the full source PDF:

- `total_pages = 40`
- `materialized_pages = 40`
- `requested_max_pages = null`
- `is_partial_run = false`

If you see a run that only covers one or two pages, that is a separate run id created with `--max-pages`, not this full manifest.

## How OpenRAG retrieval works here

The installed OpenRAG backend on this machine is doing hybrid retrieval in its `search_service.py`:

- semantic KNN search over embedding fields
- keyword `multi_match` over `text` and `filename`
- semantic weighting boost `0.7`
- keyword weighting boost `0.3`
- OpenSearch aggregations over filenames, types, owners, connector types, embedding models

For `POST /api/v1/chat`, this repo upgrades the Langflow agent flow so the OpenSearch retrieval tool reranks hybrid candidates before the LLM sees them.

For `POST /api/v1/search`, this repo now mounts a public-search hotfix that widens the candidate pool and deterministically reranks the returned hybrid hits before responding.

## How summary generation is sent to OpenRAG

Single-page summary:

- one or more retrieval chunk filenames for the current page in `filters.data_sources`
- `document_types` includes `text/markdown` for retrieval docs
- the application sends the full scoped retrieval set into chat
- the upgraded Langflow OpenSearch tool hybrid-searches and backend-reranks those candidates
- prompt asks for JSON with `summary` and `key_facts`

Reduce summary:

- retrieval chunk filenames drawn from all selected pages in `filters.data_sources`
- prompt contains the page summaries produced earlier
- the same upgraded Langflow retrieval tool does the final candidate ordering

Verification:

- one search query per final sentence, answered by the hotfixed public search endpoint with reranked evidence and rank metadata
- exact sentence text is submitted to `/api/v1/search`

All of those prompts, responses, and returned sources are recorded in `events.jsonl` or the final summary JSON when the stack is healthy enough to return them.

The summarizer now also records:

- page-level OpenRAG search preflight hits
- whether a chat retry was needed because the agent initially skipped retrieval
- reduce-step OpenRAG search preflight hits
- the exact `data_sources` and limits used for each step

## Current known issue on this machine

As of March 29, 2026, the local OpenSearch `documents` index is red and the primary shard is unassigned.

The allocation explanation reports:

- `reason: ALLOCATION_FAILED`
- merge failure
- `ArrayIndexOutOfBoundsException`
- `max_retry` refusing reallocation

This causes:

- delete preflight `500/503`
- ingest tasks that do not finish
- summary search `503 search_phase_execution_exception`
- chunk dump `503`

## Recovery commands

### Safe retry

```bash
uv run fastpdf-openrag-native diagnose-stack
```

Then, if you want to retry shard allocation without deleting data:

```bash
python - <<'PY'
import httpx
r = httpx.post(
    "https://127.0.0.1:9200/_cluster/reroute?retry_failed=true&pretty",
    auth=("admin", "your-opensearch-password"),
    verify=False,
    timeout=30,
)
print(r.text)
PY
```

### Destructive reset

Only do this if you accept deleting all OpenRAG documents currently stored in the `documents` index.

```bash
python - <<'PY'
import httpx
r = httpx.delete(
    "https://127.0.0.1:9200/documents",
    auth=("admin", "your-opensearch-password"),
    verify=False,
    timeout=30,
)
print(r.status_code)
print(r.text)
PY
docker restart os openrag-backend langflow openrag-frontend
```

After that:

```bash
uv run fastpdf-openrag-native diagnose-stack
```

Do not continue until the index is queryable.
