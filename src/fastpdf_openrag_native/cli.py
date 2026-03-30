from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
import uvicorn

from .fastpdf_loader import load_run_from_mongo, load_run_json, materialize_summary_payload
from .langflow import LangflowGateway
from .ocr_extract import extract_pdf_to_html
from .openrag import OpenRAGGateway
from .opensearch import OpenSearchInspector
from .pdf_workflow import run_pdf_pipeline
from .settings import get_settings
from .summarizer import load_manifest, load_scopes, resolve_scope_retrieval_sources, summarize_scope
from .trace import TraceRecorder

app = typer.Typer(no_args_is_help=True)


@app.command("materialize-run")
def materialize_run(
    input: Path | None = typer.Option(default=None, exists=True, file_okay=True, dir_okay=False),
    run_id: str | None = typer.Option(default=None),
    output_dir: Path | None = typer.Option(default=None),
    include_non_survivors: bool = typer.Option(default=False),
    mongo_uri: str | None = typer.Option(default=None),
    mongo_database: str | None = typer.Option(default=None),
    mongo_collection: str = typer.Option(default="runs"),
) -> None:
    settings = get_settings()
    if input:
        resolved_run_id, payload, source_kind = load_run_json(input, run_id=run_id)
    else:
        if not run_id or not mongo_uri or not mongo_database:
            raise typer.BadParameter("run-id, mongo-uri, and mongo-database are required when --input is omitted")
        resolved_run_id, payload, source_kind = load_run_from_mongo(
            run_id=run_id,
            mongo_uri=mongo_uri,
            mongo_database=mongo_database,
            mongo_collection=mongo_collection,
        )

    destination = output_dir or settings.materialized_root / resolved_run_id
    manifest = materialize_summary_payload(
        run_id=resolved_run_id,
        summary_payload=payload,
        source_kind=source_kind,
        output_dir=destination,
        include_non_survivors=include_non_survivors,
    )
    typer.echo((destination / "manifest.json").as_posix())
    typer.echo(f"materialized_pages={manifest.materialized_pages}")


@app.command("ingest-manifest")
def ingest_manifest(
    manifest: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    apply_recommended_settings: bool = typer.Option(default=False),
) -> None:
    gateway = OpenRAGGateway(get_settings())

    async def _run() -> None:
        loaded_manifest = load_manifest(manifest)
        if apply_recommended_settings:
            await gateway.apply_recommended_settings()
        results = await gateway.ingest_manifest(loaded_manifest, manifest_dir=manifest.parent)
        for row in results:
            typer.echo(f"{row.filename}: {row.status}")

    asyncio.run(_run())


@app.command("create-filter")
def create_filter(
    manifest: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    scope: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
) -> None:
    gateway = OpenRAGGateway(get_settings())

    async def _run() -> None:
        loaded_manifest = load_manifest(manifest)
        loaded_scope = load_scopes(scope)[0]
        result = await gateway.upsert_scope_filter(
            manifest=loaded_manifest,
            scope=loaded_scope,
            data_sources=resolve_scope_retrieval_sources(loaded_manifest, loaded_scope),
        )
        typer.echo(result.model_dump_json(indent=2))

    asyncio.run(_run())


@app.command("summarize-scope")
def summarize_scope_command(
    manifest: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    scope: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    output: Path | None = typer.Option(default=None),
    apply_recommended_settings: bool = typer.Option(default=False),
) -> None:
    settings = get_settings()
    gateway = OpenRAGGateway(settings)

    async def _run() -> None:
        loaded_manifest = load_manifest(manifest)
        loaded_scope = load_scopes(scope)[0]
        if apply_recommended_settings:
            await gateway.apply_recommended_settings()
        result = await summarize_scope(
            gateway,
            manifest=loaded_manifest,
            scope=loaded_scope,
            settings=settings,
        )
        destination = output or (
            settings.output_root / loaded_manifest.run_id / f"{loaded_scope.scope_id}.summary.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        typer.echo(destination.as_posix())

    asyncio.run(_run())


@app.command("health")
def health() -> None:
    gateway = OpenRAGGateway(get_settings())
    langflow = LangflowGateway(get_settings())

    async def _run() -> None:
        payload = {
            "openrag": await gateway.health(),
            "langflow": (await langflow.diagnostics()).model_dump(mode="json"),
        }
        typer.echo(json.dumps(payload, indent=2))

    asyncio.run(_run())


@app.command("extract-pdf")
def extract_pdf(
    pdf: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    credentials: Path | None = typer.Option(default=None, exists=True, file_okay=True, dir_okay=False),
    output_dir: Path | None = typer.Option(default=None),
    max_pages: int | None = typer.Option(default=None, min=1),
) -> None:
    settings = get_settings()
    destination = output_dir or (settings.extraction_root / pdf.stem)
    trace = TraceRecorder(settings.trace_root / f"{pdf.stem}-extract")
    manifest = extract_pdf_to_html(
        pdf,
        output_dir=destination,
        trace=trace,
        settings=settings,
        credentials_path=credentials,
        max_pages=max_pages,
    )
    typer.echo((destination / "manifest.json").as_posix())
    typer.echo(f"total_pages={manifest.total_pages}")
    typer.echo(f"materialized_pages={manifest.materialized_pages}")
    if manifest.is_partial_run:
        typer.echo(
            "warning=partial_run "
            f"requested_max_pages={manifest.requested_max_pages} "
            f"materialized_pages={manifest.materialized_pages} "
            f"total_pages={manifest.total_pages}"
        )


@app.command("process-pdf")
def process_pdf(
    pdf: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    credentials: Path | None = typer.Option(default=None, exists=True, file_okay=True, dir_okay=False),
    question: str | None = typer.Option(default=None),
    max_pages: int | None = typer.Option(default=None, min=1),
    apply_recommended_settings: bool = typer.Option(default=True),
) -> None:
    settings = get_settings()

    async def _run() -> None:
        result = await run_pdf_pipeline(
            pdf_path=pdf,
            credentials_path=credentials,
            settings=settings,
            question=question,
            max_pages=max_pages,
            apply_recommended_settings=apply_recommended_settings,
        )
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))
        if result.is_partial_run:
            typer.echo(
                "warning=partial_run "
                f"requested_max_pages={result.requested_max_pages} "
                f"materialized_pages={result.materialized_pages} "
                f"total_pages={result.total_pages}"
            )

    asyncio.run(_run())


@app.command("inspect-chunks")
def inspect_chunks(
    filename: str = typer.Option(..., help="Exact OpenRAG/OpenSearch filename, e.g. merged_notes-...__p0001.html"),
) -> None:
    inspector = OpenSearchInspector(get_settings())

    async def _run() -> None:
        chunks = await inspector.list_chunks_for_filename(filename)
        typer.echo(json.dumps([row.model_dump(mode="json") for row in chunks], indent=2))

    asyncio.run(_run())


@app.command("diagnose-stack")
def diagnose_stack() -> None:
    settings = get_settings()
    gateway = OpenRAGGateway(settings)
    inspector = OpenSearchInspector(settings)
    langflow = LangflowGateway(settings)

    async def _run() -> None:
        payload = {
            "openrag": await gateway.health(),
            "langflow": (await langflow.diagnostics()).model_dump(mode="json"),
            "opensearch": (await inspector.diagnostics()).model_dump(mode="json"),
        }
        typer.echo(json.dumps(payload, indent=2))

    asyncio.run(_run())


@app.command("upgrade-openrag-flows")
def upgrade_openrag_flows(
    output_dir: Path | None = typer.Option(default=None),
    patch_live: bool = typer.Option(
        default=True,
        help="Patch the running Langflow flow definitions after writing the mounted flow files.",
    ),
) -> None:
    gateway = LangflowGateway(get_settings())

    async def _run() -> None:
        result = await gateway.upgrade_flows(output_dir=output_dir, patch_live=patch_live)
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))

    asyncio.run(_run())


@app.command("repair-opensearch")
def repair_opensearch(
    output_dir: Path | None = typer.Option(default=None),
) -> None:
    inspector = OpenSearchInspector(get_settings())

    async def _run() -> None:
        result = await inspector.repair_documents_index(output_dir=output_dir)
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))

    asyncio.run(_run())


@app.command("serve-ui")
def serve_ui(
    host: str | None = typer.Option(default=None),
    port: int | None = typer.Option(default=None),
) -> None:
    settings = get_settings()
    uvicorn.run(
        "fastpdf_openrag_native.service:app",
        host=host or settings.ui_host,
        port=port or settings.ui_port,
        reload=False,
    )


if __name__ == "__main__":
    app()
