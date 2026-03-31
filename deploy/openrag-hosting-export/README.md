# OpenRAG Hosting Export

This directory is a standalone deployment/export package for the OpenRAG stack currently used by `fastpdf-openrag-native`.

It exists so you can:
- deploy the same OpenRAG stack without depending on `~/.openrag/tui`
- keep the native repo hotfixes and rerank wiring
- export your current local OpenRAG config without editing it

This package does not modify your live config. It only gives you:
- a portable `docker-compose.yml`
- a sanitized `.env.example`
- helper scripts for export, restart, and health checks

## What this package deploys

The compose file here includes the same major pieces used by the native repo:
- `openrag-backend`
- `openrag-frontend`
- `langflow`
- `docling`
- `opensearch`
- `opensearch-dashboards`
- the native repo backend hotfix mounts:
  - `openrag-hotfixes/api_v1_chat.py`
  - `openrag-hotfixes/api_v1_search.py`
  - `openrag-hotfixes/session_manager.py`
- backend/langflow rerank environment wiring

This deploy package now includes a `docling` container in Compose, so the default server deployment is self-contained.

## Is the fastpdf bridge code needed here?

No, not for hosting this stack.

The bridge code in `fastpdf/openrag_native_bridge` is only needed when the legacy `fastpdf` batch monitor / batch system wants to delegate jobs into `fastpdf-openrag-native`.

If you are deploying and using `fastpdf-openrag-native` directly, the native repo already does its own:
- OCR/materialization
- OpenRAG ingestion
- retrieval/rerank
- verification
- citation rendering
- UI/API/CLI orchestration

So for standalone hosting of the native system, do not bring the bridge.

Use the bridge only if you want old `fastpdf` services to call into the native stack.

## Files in this directory

- `docker-compose.yml`
  Current portable compose for the OpenRAG services used by the native repo.
- `.env.example`
  Sanitized environment template with placeholders.
- `scripts/restart_stack.sh`
  Restarts Docling plus the OpenRAG services, then reapplies native flow upgrades/settings.
- `scripts/healthcheck.sh`
  Quick health check for the deployed services.
- `scripts/export_local_openrag_config.sh`
  Copies your current local `~/.openrag` state into an ignored `exports/` directory without editing the live setup.

## Quick start

### 1. Create a local env file

```bash
cd /home/divyesh-nandlal-vishwakarma/Desktop/Divyesh/fastpdf-openrag-native/deploy/openrag-hosting-export
cp .env.example .env
```

Then edit `.env` and set at minimum:
- `FASTPDF_OPENRAG_NATIVE_ROOT` must be the native repo root, for example `/srv/fastpdf-openrag-native`, not `/srv/fastpdf/openrag_native_bridge`
- `FASTPDF_OPENRAG_NATIVE_ROOT`
- `OPENSEARCH_PASSWORD`
- `LANGFLOW_SECRET_KEY`
- `SESSION_SECRET`
- `OPENAI_API_KEY` if you are using OpenAI

Optional Langflow auth:
- Leave `LANGFLOW_SUPERUSER_PASSWORD` unset if you want Langflow/OpenRAG auto-login mode.
- Set `LANGFLOW_SUPERUSER_PASSWORD` only if you want password-based Langflow auth; if you do, also set `LANGFLOW_AUTO_LOGIN=false`.

Optional flow overrides:
- Built-in `LANGFLOW_CHAT_FLOW_ID`, `LANGFLOW_INGEST_FLOW_ID`, and `LANGFLOW_URL_INGEST_FLOW_ID` are auto-detected from the bundled flow JSON or handled by OpenRAG defaults. Set them only if you are overriding the built-in flows.

### 2. Export your current local config if you want exact parity

This copies your current `~/.openrag/tui/.env`, compose, flows, config, documents, and data into `exports/`.
It does not edit the live setup.

```bash
cd /home/divyesh-nandlal-vishwakarma/Desktop/Divyesh/fastpdf-openrag-native/deploy/openrag-hosting-export
bash ./scripts/export_local_openrag_config.sh
```

If you also want to export the current key material, do this carefully:

```bash
cd /home/divyesh-nandlal-vishwakarma/Desktop/Divyesh/fastpdf-openrag-native/deploy/openrag-hosting-export
INCLUDE_KEYS=true bash ./scripts/export_local_openrag_config.sh
```

### 3. Start or restart the stack

```bash
cd /home/divyesh-nandlal-vishwakarma/Desktop/Divyesh/fastpdf-openrag-native/deploy/openrag-hosting-export
bash ./scripts/restart_stack.sh
```

That script will:
- optionally restart an external host Docling process if you explicitly enable it
- start the compose services, including `docling`
- run `fastpdf-openrag-native upgrade-openrag-flows`
- apply the recommended knowledge settings
- run `diagnose-stack`

### 4. Check health

```bash
cd /home/divyesh-nandlal-vishwakarma/Desktop/Divyesh/fastpdf-openrag-native/deploy/openrag-hosting-export
bash ./scripts/healthcheck.sh
```

## Service URLs

With the default ports:
- OpenRAG frontend: `http://127.0.0.1:3000/`
- Langflow: `http://127.0.0.1:7860/`
- OpenSearch Dashboards: `http://127.0.0.1:5601/`
- OpenSearch HTTPS API: `https://127.0.0.1:9200/`

## Notes about flows and IDs

This deployment package does not embed your live Langflow flow IDs or private env values.
Those come from your actual deployment state.

If you want exact parity with the stack you are already running, use `scripts/export_local_openrag_config.sh` first and copy the needed values from that export into `.env`.

## Notes about Docling

The deploy package now includes a Compose-managed `docling` service on port `5001`, and the default internal URL is `http://docling:5001`.

Leave `DOCLING_SERVE_API_KEY` unset unless you intentionally want Docling API-key auth. Current Docling images fail startup if that variable is present but empty.

For Langflow, leave `LANGFLOW_SUPERUSER_PASSWORD` unset if you want default auto-login mode. In that mode, this deploy package now omits the password variable entirely instead of passing an empty string, which matches official OpenRAG/Langflow behavior more closely.

The built-in OpenRAG flow IDs can be left unset. `scripts/restart_stack.sh` now auto-detects them from `state/flows/*.json`, and direct `docker compose up` no longer forces them to empty strings.

If you want to use an external Docling instance instead, set `DOCLING_SERVE_URL` in `.env` to that endpoint. You only need `DOCLING_MANAGED=true` if you explicitly want `scripts/restart_stack.sh` to launch a separate host Docling process instead of relying on the bundled container.
