# OpenRAG Startup For fastpdf-openrag-native

This runbook is for this machine and this repository:

- repo: `/home/divyesh-nandlal-vishwakarma/Desktop/Divyesh/fastpdf-openrag-native`
- existing OpenRAG workspace: `~/.openrag`
- OpenRAG compose file: `~/.openrag/tui/docker-compose.yml`
- Docling venv: `~/.openrag/docling-venv`
- expected OpenRAG URL: `http://127.0.0.1:3000`
- expected Docling URL: `http://127.0.0.1:5001`

OpenRAG references used for this runbook:

- Quickstart: <https://docs.openr.ag/quickstart>
- Terminal management: <https://docs.openr.ag/tui/>
- Service management: <https://docs.openr.ag/manage-services/>
- Configuration variables: <https://docs.openr.ag/reference/configuration/>
- OpenSearch users/roles and DLS: <https://docs.opensearch.org/docs/latest/security/access-control/users-roles/>

## What this repo needs

`fastpdf-openrag-native` needs all of the following:

1. OpenRAG frontend/backend/OpenSearch containers running.
2. `docling serve` running on the host.
3. `OPENRAG_URL` set.
4. `OPENRAG_API_KEY` available.
5. Google Vision credentials available for the PDF OCR workflow.

The repo now tries to auto-discover the API key from the local `openrag-backend` container if you do not export it manually, but exporting it yourself is still the safest path.

The repo also upgrades the mounted Langflow flows so the OpenRAG agent can use backend reranking for chat and summarization. Start the stack first, then run the flow-upgrade command from this repo.

## Fastest Way To Start Everything

If you want fastpdf plus OpenRAG plus Docling together, use the existing fastpdf launcher:

```bash
cd /home/divyesh-nandlal-vishwakarma/Desktop/Divyesh/fastpdf
OPENRAG_COMPOSE_FILE="$HOME/.openrag/tui/docker-compose.yml" bash ./restart_all.sh
```

That script already knows how to:

- start the fastpdf infrastructure
- start host-side Docling from `~/.openrag/docling-venv/bin/docling-serve`
- start the OpenRAG containers with Docker Compose
- wait for health checks on Docling and OpenRAG

For this repo specifically, the preferred restart path is now the repo-local restart script because it also mounts the public-search rerank hotfix and reapplies the Langflow/OpenRAG settings:

```bash
cd /home/divyesh-nandlal-vishwakarma/Desktop/Divyesh/fastpdf-openrag-native
bash ./scripts/restart_openrag_stack.sh
```

## Start Only OpenRAG And Docling

If you want to start only the OpenRAG side for this repo:

### 1. Start Docling

Check health first:

```bash
curl -fsS http://127.0.0.1:5001/health
```

If it is not healthy, start it manually:

```bash
"$HOME/.openrag/docling-venv/bin/docling-serve" run --host 127.0.0.1 --port 5001
```

If you want it in the background:

```bash
nohup "$HOME/.openrag/docling-venv/bin/docling-serve" run --host 127.0.0.1 --port 5001 \
  >> "$HOME/.openrag/docling-serve.log" 2>&1 &
echo $! > "$HOME/.openrag/tui/.docling.pid"
```

Check it again:

```bash
curl -fsS http://127.0.0.1:5001/health
```

### 2. Start OpenRAG containers

```bash
docker compose -f "$HOME/.openrag/tui/docker-compose.yml" up -d
```

Check health:

```bash
curl -fsS http://127.0.0.1:3000/api/settings
docker ps --format 'table {{.Names}}\t{{.Status}}'
```

### 3. Inspect logs when needed

```bash
docker logs --tail 200 openrag-backend
docker logs --tail 200 langflow
tail -n 200 "$HOME/.openrag/docling-serve.log"
```

## Export The Correct Environment For This Repo

From `fastpdf-openrag-native`:

```bash
cd /home/divyesh-nandlal-vishwakarma/Desktop/Divyesh/fastpdf-openrag-native
export OPENRAG_URL="http://127.0.0.1:3000"
```

If you want to export the key explicitly, use a real OpenRAG public API key. The SDK endpoints only accept `orag_...` keys.

Generate one from the running backend:

```bash
export OPENRAG_API_KEY="$(
  docker exec -i openrag-backend env PYTHONPATH=/app/src /app/.venv/bin/python - <<'"'"'PY'"'"' \
  | awk -F= '/^OPENRAG_API_KEY=/{print $2}'
import asyncio
from config.settings import clients
from services.api_key_service import APIKeyService

async def main():
    await clients.initialize()
    result = await APIKeyService().create_key(
        user_id="fastpdf-openrag-native",
        user_email="fastpdf-openrag-native@local",
        name="fastpdf-openrag-native local cli",
    )
    print("OPENRAG_API_KEY=" + (result.get("api_key") or ""))

asyncio.run(main())
PY
)"
```

If you do not export `OPENRAG_API_KEY`, this repo now attempts to discover it automatically from `openrag-backend`.

You can also write a local `.env` file:

```bash
cat > .env <<'EOF'
OPENRAG_URL=http://127.0.0.1:3000
OPENRAG_API_KEY=paste_the_key_here
FASTPDF_OPENRAG_LANGFLOW_URL=http://127.0.0.1:7860
FASTPDF_OPENRAG_MATERIALIZED_ROOT=data/materialized
FASTPDF_OPENRAG_OUTPUT_ROOT=outputs
FASTPDF_OPENRAG_EXTRACTION_ROOT=data/extracted
FASTPDF_OPENRAG_TRACE_ROOT=outputs/traces
FASTPDF_OPENRAG_DEFAULT_CHUNK_SIZE=1200
FASTPDF_OPENRAG_DEFAULT_CHUNK_OVERLAP=150
FASTPDF_OPENRAG_RETRIEVAL_RERANK_ENABLED=false
FASTPDF_OPENRAG_BACKEND_RERANK_ENABLED=true
FASTPDF_OPENRAG_BACKEND_RERANK_PROVIDER=cross_encoder
FASTPDF_OPENRAG_BACKEND_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
FASTPDF_OPENRAG_BACKEND_RERANK_TOP_N=8
FASTPDF_OPENRAG_BACKEND_RERANK_CANDIDATE_LIMIT=16
FASTPDF_OPENRAG_VERIFICATION_LIMIT=3
FASTPDF_OPENRAG_VERIFICATION_SCORE_THRESHOLD=0.15
FASTPDF_OPENRAG_FILTER_PREFIX=fastpdf-openrag-native
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/google-vision-service-account.json
FASTPDF_OPENRAG_OPENSEARCH_URL=https://127.0.0.1:9200
FASTPDF_OPENRAG_OPENSEARCH_USERNAME=admin
FASTPDF_OPENRAG_OPENSEARCH_PASSWORD=paste_the_password_here
FASTPDF_OPENRAG_OPENSEARCH_INDEX_NAME=documents
FASTPDF_OPENRAG_OPENSEARCH_VERIFY_SSL=false
FASTPDF_OPENRAG_UI_HOST=127.0.0.1
FASTPDF_OPENRAG_UI_PORT=8077
FASTPDF_OPENRAG_INGEST_WAIT_TIMEOUT=180
EOF
```

## Upgrade The Langflow Flows

After the containers are healthy, patch the mounted OpenRAG flow JSON and the running locked flows:

```bash
uv run fastpdf-openrag-native upgrade-openrag-flows
```

What this changes:

- the agent prompt is rewritten so the knowledge filter is treated as hard scope
- the OpenSearch retrieval tool candidate count is raised to the backend rerank candidate limit
- backend reranking is injected into the Langflow OpenSearch tool code path
- the ingestion flow `Split Text` node is tuned to `chunk_size=1200`, `chunk_overlap=150`, `separator="\\n\\n"`

Verify it:

```bash
uv run fastpdf-openrag-native diagnose-stack
```

The `langflow` section should report:

- `agent_flow_rerank_marker_present: true`
- `agent_flow_prompt_upgraded: true`
- `ingestion_chunk_size: 1200`
- `ingestion_chunk_overlap: 150`
- `backend_reranking_enabled: true`
- `backend_reranker_provider: "cross_encoder"`

## Public API Retrieval Hotfix On This Machine

This machine had a real OpenRAG backend bug in the no-auth path:

- `/api/v1/search` authenticated the API key correctly
- but `SessionManager.get_effective_jwt_token()` still fell back to the anonymous JWT whenever OAuth was disabled
- so OpenSearch retrieval ran as `anonymous` instead of the API-key user
- result: indexed docs existed, raw OpenSearch queries worked, but `/api/v1/search` returned `{"results":[]}`

The durable fix is now mounted into the backend from this repo:

- hotfix file: `/home/divyesh-nandlal-vishwakarma/Desktop/Divyesh/fastpdf-openrag-native/openrag-hotfixes/session_manager.py`
- hotfix file: `/home/divyesh-nandlal-vishwakarma/Desktop/Divyesh/fastpdf-openrag-native/openrag-hotfixes/api_v1_chat.py`
- hotfix file: `/home/divyesh-nandlal-vishwakarma/Desktop/Divyesh/fastpdf-openrag-native/openrag-hotfixes/api_v1_search.py`
- compose override: `/home/divyesh-nandlal-vishwakarma/Desktop/Divyesh/fastpdf-openrag-native/docker/openrag-override.yml`

`api_v1_chat.py` also guards against Langflow history chunks that contain `item: null` or `results: null`. That specific shape previously caused `/api/v1/chat` to crash during source recovery even though retrieval itself had succeeded.

`api_v1_search.py` now widens the raw hybrid candidate pool and applies deterministic reranking inside the public search endpoint, then returns:

- `score`: reranked score
- `base_score`: original OpenSearch hybrid score
- `rerank_score`: deterministic rerank score
- `retrieval_rank`: final reranked order

The old backend compose file already mounts `session_manager.py` and `api_v1_chat.py`. The new search hotfix is applied through the repo-local Compose override, so use the repo restart script or the manual override command below when you restart the stack.

These backend hotfixes fix API-key retrieval and chat-source recovery. The actual reranking upgrade lives in the Langflow flow patch, not in `search_service.py`.

If you change either hotfix file, recreate only the backend so Python does not keep serving stale bytecode from `openrag-backend`:

```bash
export FASTPDF_OPENRAG_NATIVE_ROOT="/home/divyesh-nandlal-vishwakarma/Desktop/Divyesh/fastpdf-openrag-native"
docker compose \
  -f "$HOME/.openrag/tui/docker-compose.yml" \
  -f "$FASTPDF_OPENRAG_NATIVE_ROOT/docker/openrag-override.yml" \
  --env-file "$HOME/.openrag/tui/.env" \
  up -d --no-build openrag-backend
```

Verify the mount is active:

```bash
docker exec openrag-backend sh -lc 'grep -n "API-key users still need" /app/src/session_manager.py'
```

Verify public API retrieval with your exported `OPENRAG_API_KEY`:

```bash
curl -s http://127.0.0.1:3000/api/v1/search \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: ${OPENRAG_API_KEY}" \
  -d '{"query":"*","limit":10}'
```

You should now get actual documents back, not `{"results":[]}`.

For a rerank-aware search smoke test with pretty JSON output:

```bash
cd /home/divyesh-nandlal-vishwakarma/Desktop/Divyesh/fastpdf-openrag-native
bash ./scripts/smoke_test_openrag_stack.sh patient
```

You can also verify grounded chat:

```bash
curl -s http://127.0.0.1:3000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: ${OPENRAG_API_KEY}" \
  -d '{"message":"What is the patient name and reason for visit in the indexed document?","stream":false,"limit":5}'
```

Expected behavior after the second hotfix:

- the JSON `response` still contains the grounded answer text
- the top-level `sources` array is no longer empty
- `sources[0].filename` should match the retrieved page file
- `sources[0].score` should be a real non-zero retrieval score

If a full PDF pipeline run ingests correctly but summary generation still fails, the fastpdf UI now marks the job as `completed_with_errors` and includes `summary_error` in the run JSON instead of showing a misleading plain `completed` status.

## Run This Repo

### 1. Install dependencies

```bash
cd /home/divyesh-nandlal-vishwakarma/Desktop/Divyesh/fastpdf-openrag-native
uv sync --extra dev
```

### 2. Upgrade the Langflow flows

Patch the mounted OpenRAG flows first if you have not done it already:

```bash
uv run fastpdf-openrag-native upgrade-openrag-flows
```

If you want the full backend restart plus search/chat hotfix mounts plus flow/settings reapply in one command:

```bash
bash ./scripts/restart_openrag_stack.sh
```

### 3. Materialize sample pages

```bash
uv run fastpdf-openrag-native materialize-run \
  --input examples/sample_summary_payload.json \
  --run-id sample-run
```

### 4. Ingest sample pages into OpenRAG

```bash
uv run fastpdf-openrag-native ingest-manifest \
  --manifest data/materialized/sample-run/manifest.json \
  --apply-recommended-settings
```

### 5. Create a reusable scope filter

```bash
uv run fastpdf-openrag-native create-filter \
  --manifest data/materialized/sample-run/manifest.json \
  --scope examples/sample_scope.json
```

### 6. Run a scoped summary

```bash
uv run fastpdf-openrag-native summarize-scope \
  --manifest data/materialized/sample-run/manifest.json \
  --scope examples/sample_scope.json
```

### 7. Run the API service

```bash
uv run fastpdf-openrag-native serve-ui
```

Open `http://127.0.0.1:8077/` and use the trace page for a run to open:

- `manifest.json`
- extracted `pages/`
- extracted `artifacts/`
- `events.jsonl`
- trace `summary.json`
- final `all-pages.summary.json`
- chunk dump files

### 8. Diagnose the stack before a PDF run

```bash
uv run fastpdf-openrag-native diagnose-stack
```

If the `documents` index is `red`, OCR extraction can still work, but OpenRAG ingest/search/summary will not be reliable.

### 8a. Repair a red `documents` index

On this machine, the observed hard failure was:

- OpenSearch `3.2.0`
- `documents` primary shard `UNASSIGNED`
- allocation reason `ALLOCATION_FAILED`
- OpenSearch log stack with `opensearch-jvector` merge failure:
  `ArrayIndexOutOfBoundsException: Index 4 out of bounds for length 1`

That is an OpenSearch-side shard failure, not a fastpdf repo bug. When it happens, use the repo repair command:

```bash
uv run fastpdf-openrag-native repair-opensearch
```

What it does:

1. Captures before/after diagnostics into `outputs/opensearch-repair/<timestamp>/`.
2. Asks the running `openrag-backend` container for the current target index body based on the active OpenRAG embedding configuration.
3. Deletes and recreates only the `documents` index.
4. Forces single-node-safe replica settings on `documents` and `knowledge_filters`.

Expected result after a good repair:

```bash
uv run fastpdf-openrag-native diagnose-stack
curl -sk -u admin:'Mrugakshi1225*' https://127.0.0.1:9200/_cluster/health?pretty
```

You want:

- cluster `status: green`
- `documents` with `rep: 0`
- no `search_phase_execution_exception` from `/documents/_search`

### 9. Run the PDF OCR + OpenRAG pipeline

```bash
uv run fastpdf-openrag-native process-pdf \
  --pdf /home/divyesh-nandlal-vishwakarma/Downloads/chinchin/merged_notes.pdf \
  --credentials /home/divyesh-nandlal-vishwakarma/Downloads/ocr-neuralit-4e01e06ccf84\(2\).json
```

Important behavior:

- PyMuPDF is used only to render page images.
- Google Vision is the only OCR/text source emitted and ingested by this repo.
- If you omit `--max-pages`, the whole PDF is processed.
- The current repo-local `data/extracted/merged_notes/manifest.json` is already a full 40-page manifest.

For a shorter diagnostic run:

```bash
FASTPDF_OPENRAG_INGEST_WAIT_TIMEOUT=30 \
uv run fastpdf-openrag-native process-pdf \
  --pdf /home/divyesh-nandlal-vishwakarma/Downloads/chinchin/merged_notes.pdf \
  --credentials /home/divyesh-nandlal-vishwakarma/Downloads/ocr-neuralit-4e01e06ccf84\(2\).json \
  --max-pages 1
```

## Known Pitfalls

### `OPENRAG_API_KEY is required`

Do this:

```bash
export OPENRAG_URL="http://127.0.0.1:3000"
export OPENRAG_API_KEY="your_key_here"
```

Do not do this:

```bash
export $OPENRAG_URL="http://127.0.0.1:3000"
```

The `$` is only for reading an existing variable, not defining one.

### The generated `sk-...` Langflow key still fails

That key is for Langflow. `fastpdf-openrag-native` uses the authenticated OpenRAG public API endpoints, which validate only `orag_...` keys.

Use the key-generation command above instead of `get_langflow_api_key()`.

### `ServerError: An internal error has occurred while deleting documents`

This is currently an OpenRAG-side delete problem and shows up as a 500/503 from the delete endpoint while OpenSearch is warming or when the filename is not indexed yet.

This repo now ignores that delete failure and continues with ingest. If ingest still fails:

```bash
docker logs --tail 200 openrag-backend
docker logs --tail 200 os
sleep 10
uv run fastpdf-openrag-native ingest-manifest \
  --manifest data/materialized/sample-run/manifest.json \
  --apply-recommended-settings
```

### Docling is not healthy

Check:

```bash
curl -fsS http://127.0.0.1:5001/health
tail -n 200 "$HOME/.openrag/docling-serve.log"
```

Restart it:

```bash
pkill -f 'docling-serve run' || true
nohup "$HOME/.openrag/docling-venv/bin/docling-serve" run --host 127.0.0.1 --port 5001 \
  >> "$HOME/.openrag/docling-serve.log" 2>&1 &
```

### OpenRAG is up but this repo still cannot talk to it

Check:

```bash
curl -fsS http://127.0.0.1:3000/api/settings
uv run fastpdf-openrag-native health
uv run fastpdf-openrag-native diagnose-stack
```

If `health` fails, confirm:

- the containers are actually up
- the repo is using `OPENRAG_URL=http://127.0.0.1:3000`
- the repo has a valid `OPENRAG_API_KEY`

### OpenSearch `documents` index is red

This repo now exposes that directly through:

```bash
uv run fastpdf-openrag-native diagnose-stack
```

The current machine has already shown this failure mode:

- `cluster_health.status = red`
- `allocation_explain.index = documents`
- `allocation_explain.primary = true`
- `search_phase_execution_exception` on search

Try the safe retry first:

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

If you accept a destructive reset of all knowledge-base documents in the `documents` index:

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

Then rerun:

```bash
uv run fastpdf-openrag-native diagnose-stack
```
