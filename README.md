# fastpdf-openrag-native

`fastpdf-openrag-native` is a separate repository for moving fastpdf summary work onto OpenRAG's real ingestion, chunking, storage, search, and filter flow instead of the monitor's current local chunk-selection and `/chat` prompt-stuffing path.

It now also includes a direct PDF workflow:

- render PDF pages with PyMuPDF
- OCR each page with Google Vision `document_text_detection`
- emit per-page HTML, Google Vision text, raw OCR JSON, and page images
- ingest the generated HTML pages into OpenRAG
- create an OpenRAG knowledge filter over those pages
- summarize every page through OpenRAG retrieval
- write detailed traces for every stage
- expose a small debug UI

## Why this repo exists

The current fastpdf monitor summary path has two structural problems:

- evidence is prepared locally with heuristic chunking and selection, so clinically important spans can be dropped before the model sees them
- the response is normalized locally afterward, which can further degrade otherwise acceptable model output

This repository changes the boundary:

- fastpdf run pages are materialized into page artifacts plus structure-aware retrieval chunk documents
- those retrieval documents are ingested into OpenRAG through the official SDK
- reusable knowledge filters can be created for a run or scope
- scoped summaries are generated from OpenRAG hybrid retrieval plus backend reranking inside the Langflow agent flow
- final sentences are verified with OpenRAG search before being accepted

## Design choices

This repo is intentionally opinionated:

- one page artifact plus multiple structure-aware retrieval docs per page: retrieval can stay page-bounded without forcing whole-page chunks
- moderate OpenRAG chunk size recommendation: the repo pre-chunks pages before ingest, but the ingestion flow still uses overlap so longer section docs are preserved cleanly
- backend reranking in OpenRAG's Langflow agent flow: hybrid candidates are reranked inside the retrieval tool before the LLM sees them
- public-search reranking: the repo hotfix widens and reranks `/api/v1/search` results so verification can stay on the backend path
- map/reduce/verify summary flow: coverage is handled at the page level first, then reduced, then each sentence is checked with search
- filter-first chat: reusable knowledge filters give you an OpenRAG-native way to scope later chat/Q&A to a run or clinical section

## Repo layout

- `src/fastpdf_openrag_native/fastpdf_loader.py`: load fastpdf run JSON or Mongo documents and materialize page markdown
- `src/fastpdf_openrag_native/ocr_extract.py`: render PDFs and OCR them into HTML/text/JSON artifacts
- `src/fastpdf_openrag_native/openrag.py`: OpenRAG SDK wrapper for settings, ingestion, search, chat, and knowledge filters
- `src/fastpdf_openrag_native/opensearch.py`: direct OpenSearch diagnostics and chunk inspection
- `src/fastpdf_openrag_native/pdf_workflow.py`: end-to-end PDF OCR -> OpenRAG -> summary pipeline
- `src/fastpdf_openrag_native/summarizer.py`: page map, reduce, and verification pipeline
- `src/fastpdf_openrag_native/service.py`: FastAPI API plus debug UI
- `src/fastpdf_openrag_native/cli.py`: CLI entrypoint
- `OPENRAG_STARTUP.md`: local startup and troubleshooting runbook for OpenRAG, Docling, and this repo
- `PDF_PIPELINE_DEBUG.md`: detailed PDF workflow, traces, and debugging commands

## Prerequisites

1. Run a dedicated OpenRAG deployment.
2. Set `OPENRAG_URL` and `OPENRAG_API_KEY`.
3. Set `GOOGLE_APPLICATION_CREDENTIALS` for Google Vision OCR.
3. Prefer a dedicated OpenRAG index or deployment for this repo, because chunking and ingestion settings should be tuned for structure-aware medical/legal retrieval docs.

OpenRAG references:

- docs: <https://docs.openr.ag/>
- Python SDK: <https://github.com/langflow-ai/openrag/tree/main/sdks/python>
- environment/config reference: <https://docs.openr.ag/reference/configuration/>
- Langflow customization: <https://docs.openr.ag/agents/>

## Quick start

Install:

```bash
uv sync
cp .env.example .env
```

Restart the local OpenRAG stack through the repo helper when you want the full hotfix/override path applied:

```bash
bash ./scripts/restart_openrag_stack.sh
```

Run stack diagnostics first:

```bash
uv run fastpdf-openrag-native diagnose-stack
```

Upgrade the mounted OpenRAG flows before your first serious run, or after OpenRAG/Langflow updates overwrite them:

```bash
uv run fastpdf-openrag-native upgrade-openrag-flows
```

If the `documents` index is `red`, retrieval and summary will fail even if OpenRAG itself is up. The CLI and UI now record that explicitly in the trace output.

Repair the local OpenSearch index when that happens:

```bash
uv run fastpdf-openrag-native repair-opensearch
```

This rebuilds only the `documents` index from the currently running OpenRAG embedding configuration and normalizes single-node replicas for `documents` and `knowledge_filters`.

If `/api/v1/search` still returns empty results after the index is green, this machine also needs the OpenRAG no-auth retrieval hotfix. The root bug was that API-key requests were being converted to an anonymous OpenSearch JWT whenever OAuth was disabled. The repo now carries the patched backend files at `openrag-hotfixes/session_manager.py`, `openrag-hotfixes/api_v1_chat.py`, and `openrag-hotfixes/api_v1_search.py`. The new public-search hotfix is mounted through the repo-local Compose override at `docker/openrag-override.yml`, which the restart helper script uses automatically.

`upgrade-openrag-flows` patches the mounted Langflow flow JSON and, by default, PATCHes the running locked flows through the Langflow API. The agent flow upgrade does three things:

- replaces the stock prompt so the agent treats the knowledge filter as hard scope
- increases the OpenSearch tool candidate count to the configured backend rerank candidate limit
- injects a backend reranker into the OpenSearch tool code path, using `sentence_transformers` cross-encoder by default or Cohere rerank if configured

The chat hotfix also now tolerates sparse Langflow history payloads where a tool chunk contains `item: null` or `results: null`. Without that guard, non-stream `/api/v1/chat` could crash during source recovery after a successful ingest.

Recreate only the backend after changing the hotfix:

```bash
docker compose \
  -f "$HOME/.openrag/tui/docker-compose.yml" \
  --env-file "$HOME/.openrag/tui/.env" \
  up -d --no-build openrag-backend
```

Verify public API retrieval:

```bash
curl -s http://127.0.0.1:3000/api/v1/search \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: ${OPENRAG_API_KEY}" \
  -d '{"query":"*","limit":10}'
```

Verify non-stream chat returns grounded `sources` with real retrieval scores:

```bash
curl -s http://127.0.0.1:3000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: ${OPENRAG_API_KEY}" \
  -d '{"message":"What is the patient name and reason for visit in the indexed document?","stream":false,"limit":5}'
```

Expected behavior:

- the response text is grounded to the indexed page
- `sources[0].filename` matches the retrieved page file
- `sources[0].score` is a non-zero search score, not `0`

For a non-LLM smoke test that proves the public search hotfix is active and exposes rerank metadata:

```bash
bash ./scripts/smoke_test_openrag_stack.sh patient
```

If a pipeline finishes OCR and ingest but summary generation fails, the UI job status is now `completed_with_errors` instead of `completed`, and the JSON payload includes `summary_error`.

## PDF workflow quick start

Extract and OCR a PDF into per-page HTML:

```bash
uv run fastpdf-openrag-native extract-pdf \
  --pdf /home/divyesh-nandlal-vishwakarma/Downloads/chinchin/merged_notes.pdf \
  --credentials /home/divyesh-nandlal-vishwakarma/Downloads/ocr-neuralit-4e01e06ccf84\(2\).json \
  --max-pages 2
```

Run the full OCR -> OpenRAG -> summary pipeline:

```bash
uv run fastpdf-openrag-native process-pdf \
  --pdf /home/divyesh-nandlal-vishwakarma/Downloads/chinchin/merged_notes.pdf \
  --credentials /home/divyesh-nandlal-vishwakarma/Downloads/ocr-neuralit-4e01e06ccf84\(2\).json
```

If you pass `--max-pages N`, the run is intentionally partial. The CLI now prints a `warning=partial_run ...` line, and the generated `manifest.json` records `requested_max_pages` plus `is_partial_run`.

Start the debug UI:

```bash
uv run fastpdf-openrag-native serve-ui --port 8077
```

Open `http://127.0.0.1:8077/`.

The UI accepts either a PDF path or an uploaded PDF and writes the same trace files the CLI does.
Each run trace page now exposes direct links to:

- `manifest.json`
- extracted `pages/`
- extracted `artifacts/`
- trace `events.jsonl`
- trace `summary.json`
- final `all-pages.summary.json`
- OpenSearch chunk dumps

Materialize a fastpdf run JSON into page artifacts plus retrieval docs:

```bash
uv run fastpdf-openrag-native materialize-run \
  --input examples/sample_summary_payload.json \
  --run-id sample-run
```

Ingest the materialized retrieval docs into OpenRAG:

```bash
uv run fastpdf-openrag-native ingest-manifest \
  --manifest data/materialized/sample-run/manifest.json \
  --apply-recommended-settings
```

Create a reusable knowledge filter for a scope:

```bash
uv run fastpdf-openrag-native create-filter \
  --manifest data/materialized/sample-run/manifest.json \
  --scope examples/sample_scope.json
```

Run a scoped summary with verification:

```bash
uv run fastpdf-openrag-native summarize-scope \
  --manifest data/materialized/sample-run/manifest.json \
  --scope examples/sample_scope.json
```

Run the API service:

```bash
uv run fastpdf-openrag-native serve-ui
```

## What `ingest-manifest` actually does

`uv run fastpdf-openrag-native ingest-manifest --manifest ...` does the following:

1. Loads `manifest.json`.
2. Optionally updates deployment-wide OpenRAG settings through `POST /api/v1/settings`.
3. Deletes any stale page or retrieval-document filenames for that manifest.
4. Posts each retrieval document to `POST /api/v1/documents/ingest`.
5. Polls `GET /api/v1/tasks/{task_id}` until completion or timeout.
6. Prints one line per ingested retrieval document with the final task state.

This is the official OpenRAG ingest path. It is not using the old fastpdf monitor chunker.

## Output locations

For PDF runs:

- extracted pages: `data/extracted/<run-id>/pages/*.html`
- OCR text: `data/extracted/<run-id>/artifacts/*.txt`
- raw Google Vision JSON: `data/extracted/<run-id>/artifacts/*.google_vision.json`
- rendered page images: `data/extracted/<run-id>/artifacts/*.png`
- extraction manifest: `data/extracted/<run-id>/manifest.json`
- trace events: `outputs/traces/<run-id>/events.jsonl`
- trace summary: `outputs/traces/<run-id>/summary.json`
- summary output: `outputs/<run-id>/all-pages.summary.json` when retrieval is healthy
- chunk dumps: `outputs/traces/<run-id>/chunks/*.chunks.json` when OpenSearch chunk inspection succeeds

## Summary flow

For a PDF-wide summary, the repo does this:

1. Build page-scoped HTML documents from OCR output.
2. Build structure-aware retrieval markdown chunks from OCR paragraphs or page text.
3. Ingest those retrieval docs into OpenRAG.
4. Create one OpenRAG knowledge filter over the selected retrieval filenames.
5. Ask OpenRAG search for candidate chunks, then call `/api/v1/chat` on the scoped retrieval files so the Langflow agent tool can hybrid-search and backend-rerank them.
6. Repeat the same flow-backed retrieval step for the reduce summary.
7. Verify each sentence in the final draft with the hotfixed `/api/v1/search` endpoint, which widens and reranks the backend candidate set before returning evidence.

The page and reduce prompts now explicitly tell OpenRAG that the source files are already indexed and that it must use retrieval. In backend-rerank mode, the application passes the full scoped retrieval set into chat and lets the Langflow OpenSearch tool do the final candidate ordering.

The installed OpenRAG backend on this machine still uses hybrid search in `search_service.py`: semantic KNN over embedding fields plus keyword `multi_match` scoring in OpenSearch. The chat/summarization path is upgraded through Langflow flow patching, and the public `/api/v1/search` endpoint is upgraded through the repo hotfix so it reranks the returned hybrid candidate pool before responding.

## Verified local state on this machine

On March 30, 2026, the stack was re-verified locally after two fixes:

- `uv run fastpdf-openrag-native repair-opensearch`
- OpenRAG backend no-auth retrieval hotfix via `openrag-hotfixes/session_manager.py`

Verified results:

- `uv run pytest` passed
- `uv run fastpdf-openrag-native diagnose-stack` reported healthy OpenRAG and a green OpenSearch cluster
- `curl http://127.0.0.1:3000/api/v1/search ... '{"query":"*"}'` returned the indexed document
- `curl http://127.0.0.1:3000/api/v1/search ... '{"query":"motor vehicle crash"}'` returned the indexed document with a semantic score
- `curl http://127.0.0.1:3000/api/v1/chat ...` returned a grounded answer with an inline source citation

The stable local hotfix path on this machine is:

- `openrag-hotfixes/session_manager.py`: fixes API-key search running as `anonymous` in no-auth mode
- `openrag-hotfixes/api_v1_chat.py`: enriches non-stream `/api/v1/chat` responses with `sources` from the just-created Langflow session history and backfills fresh OpenSearch scores onto those sources

On March 30, 2026, the local stack was repaired by rebuilding `documents` from OpenRAG's active `text-embedding-3-large` configuration and forcing `number_of_replicas=0` on the single-node cluster. After that, OpenSearch returned to `green` and OpenRAG ingest completed again.

## Expected workflow with fastpdf

Near term:

1. Export or fetch a fastpdf run document or `summary_payload`.
2. Materialize page markdown docs from `summary_payload.pdfs[].pages[]`.
3. Ingest those docs into OpenRAG.
4. Create one knowledge filter per run or scope.
5. Use `summarize-scope` for page-known summary jobs.
6. Use filter-scoped OpenRAG chat for interactive Q&A.

Later integration:

- replace monitor-side local chunk scoring with this repo's ingestion pipeline
- replace prompt-stuffed summary jobs with page-scoped map/reduce/verify
- keep OpenRAG as the retrieval layer, not just the generator

## Important limitations

- OpenRAG settings are deployment-wide. `--apply-recommended-settings` should be used only against a dedicated OpenRAG deployment for this workload.
- Stock OpenRAG does not provide this public-search rerank behavior by default. You only get it when you restart with the repo override or `bash ./scripts/restart_openrag_stack.sh`.
- Verification currently uses evidence support checks, not symbolic citation reconstruction.
- The Langflow flow upgrade is durable only as long as the mounted flow JSON stays in place. Re-run `uv run fastpdf-openrag-native upgrade-openrag-flows` after OpenRAG upgrades that replace flow definitions.
- This repo does not modify fastpdf in place. It is a separate bridge repo intended to prove the OpenRAG-native path first.
