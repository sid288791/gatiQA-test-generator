import datetime
import json
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.config import Settings
from app.db import TestCaseStore
from app.embeddings import Embedder
from app.observability import build_tracer
from app.models import (
    AnalysisSummaryRequest,
    AnalysisSummaryResponse,
    MetricsDashboard,
    SaveTestCaseRequest,
    TestGenerationRequest,
    TestGenerationResponse,
)
from app.service import TestGenerationService

logger = logging.getLogger(__name__)


def _jsonable(row: dict) -> dict:
    """Make a DB row dict JSON-serializable (UUID/datetime/jsonb-str -> plain)."""
    out = {}
    for k, v in row.items():
        if isinstance(v, uuid.UUID):
            out[k] = str(v)
        elif isinstance(v, (datetime.datetime, datetime.date)):
            out[k] = v.isoformat()
        elif k == "test_spec" and isinstance(v, str):
            try:
                out[k] = json.loads(v)
            except json.JSONDecodeError:
                out[k] = v
        else:
            out[k] = v
    return out


def create_app(
    service: TestGenerationService | None = None,
    store: TestCaseStore | None = None,
) -> FastAPI:
    """App factory — inject a pre-built service/store for testing or use defaults."""
    settings = service.settings if service is not None else Settings()
    db = store or TestCaseStore(settings)
    tracer = service._tracer if service is not None else build_tracer(settings)
    embedder = Embedder(settings, tracer=tracer)
    svc = service or TestGenerationService(
        settings, store=db, embedder=embedder, tracer=tracer
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            await db.connect()
        except Exception as exc:
            logger.warning("Postgres unavailable at startup: %s", exc)
        yield
        await db.close()
        tracer.shutdown()

    app = FastAPI(
        title="gatiQA Test Generator",
        version=__version__,
        description="Internal service: drafts structured gatiQA tests from natural language.",
        lifespan=lifespan,
    )
    app.state.service = svc
    app.state.store = db
    app.state.embedder = embedder
    app.state.tracer = tracer

    @app.post(
        "/internal/v1/test-generations",
        response_model=TestGenerationResponse,
        response_model_exclude_none=True,
        summary="Generate a structured test draft (never executes or approves it)",
    )
    async def create_test_generation(
        request: TestGenerationRequest, http_request: Request
    ) -> TestGenerationResponse:
        trace = None
        try:
            with app.state.tracer.request_trace(
                "test_generation", headers=dict(http_request.headers),
                metadata={"operation": "test-generation", **svc.metadata.model_dump(),
                          "profileId": request.profile.profileId,
                          "profileVersion": request.profile.version},
            ) as trace:
                return await app.state.service.generate_async(
                    request, trace_headers=dict(http_request.headers))
        finally:
            if trace is not None:
                await persist_metrics(trace.metrics())

    @app.post(
        "/internal/v1/test-cases",
        status_code=201,
        summary="Persist a test case (generated draft JSON) into Postgres",
    )
    async def save_test_case(request: SaveTestCaseRequest):
        profile_id = None
        if request.evaluation_profile_id:
            try:
                profile_id = uuid.UUID(request.evaluation_profile_id)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="evaluation_profile_id must be a valid UUID.",
                )
        profile_ids: list[uuid.UUID] = []
        for raw_id in request.evaluation_profile_ids:
            try:
                pid = uuid.UUID(raw_id)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"evaluation_profile_ids entry '{raw_id}' is not a valid UUID.",
                )
            if pid not in profile_ids:
                profile_ids.append(pid)
        # Semantic-cache fields: use explicit source_input, else fall back to
        # description/name (the NL text the spec was generated from).
        source_input = request.source_input or request.description or request.name
        embedding = await app.state.embedder.embed(source_input)
        try:
            # Dedupe: if a matching test_case already exists, return it
            # instead of inserting a duplicate row.
            existing = await app.state.store.find_similar(
                source_input=source_input, embedding=embedding
            )
            if existing is not None:
                logger.info(
                    "Duplicate test_case detected: %s (%s match, similarity=%s)",
                    existing["id"], existing["match_type"], existing["similarity"],
                )
                # Still attach any selected profiles to the existing row —
                # a duplicate save may carry new profile selections.
                all_ids = list(profile_ids)
                if profile_id is not None and profile_id not in all_ids:
                    all_ids.append(profile_id)
                if all_ids:
                    await app.state.store.link_profiles(existing["id"], all_ids)
                return JSONResponse(
                    status_code=200,
                    content={
                        "duplicate": True,
                        "matchedTestCaseId": str(existing["id"]),
                        "similarity": existing["similarity"],
                        "matchType": existing["match_type"],
                        "linkedProfileIds": [str(p) for p in all_ids],
                        "existing": _jsonable(existing),
                    },
                )
            return await app.state.store.save_test_case(
                name=request.name,
                description=request.description,
                test_spec=request.test_spec,
                evaluation_profile_id=profile_id,
                evaluation_profile_ids=profile_ids,
                status=request.status,
                created_by=request.created_by,
                source_input=source_input,
                embedding=embedding,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:
            logger.exception("Failed to persist test case")
            raise HTTPException(status_code=500, detail=f"Persist failed: {exc}")

    async def persist_metrics(metrics) -> bool:
        try:
            await app.state.store.save_request_metrics(metrics)
            return True
        except Exception as exc:
            logger.warning("Request metrics persistence unavailable: %s", type(exc).__name__)
            return False

    async def dashboard(request: AnalysisSummaryRequest) -> MetricsDashboard:
        try:
            return await app.state.store.analysis_dashboard(request)
        except Exception as exc:
            logger.warning("Metrics dashboard unavailable: %s", type(exc).__name__)
            start, end = request.time_window()
            return MetricsDashboard(
                available=False, unavailableReason="METRICS_STORAGE_UNAVAILABLE",
                fromTime=start, toTime=end, operation=request.operation,
                limit=request.limit, offset=request.offset,
            )

    @app.post(
        "/internal/v1/analysis-summaries",
        response_model=AnalysisSummaryResponse,
        response_model_exclude_unset=True,
        responses={502: {"model": AnalysisSummaryResponse}},
        summary="Analysis summary, request metrics, and project-wide trace dashboard",
        description="Omit analysis for dashboard-only access without an LLM call. "
                    "History defaults to the last 7 days (maximum 90-day window); "
                    "totals cover all matching requests, not just the paginated traces. "
                    "Tokens and estimated USD costs include LLM and embedding calls. "
                    "Null totals indicate missing usage or pricing; known totals are partial. "
                    "Latency covers processing before metrics storage and dashboard queries. "
                    "This internal endpoint exposes project-wide data and must remain behind "
                    "the application's authenticated backend.",
    )
    async def summarize_analysis(request: AnalysisSummaryRequest, http_request: Request):
        if request.analysis is None:
            return AnalysisSummaryResponse.model_validate({
                "dashboard": (await dashboard(request)).model_dump(),
            })
        status_code = 200
        try:
            with app.state.tracer.request_trace(
                "analysis_summary", headers=dict(http_request.headers),
                metadata={"operation": "analysis-summary", **svc.metadata.model_dump()},
            ) as trace:
                result = await create_summary(request, http_request)
        except HTTPException as exc:
            status_code = exc.status_code
            result = {"detail": exc.detail, "errorCode": "SUMMARIZATION_FAILED",
                      "summaryId": trace.attributes.get("summaryId")}
        metrics = trace.metrics()
        result["metrics"] = metrics.model_dump()
        result["metricsPersisted"] = await persist_metrics(metrics)
        result["dashboard"] = (await dashboard(request)).model_dump()
        response = AnalysisSummaryResponse.model_validate(result)
        if status_code != 200:
            return JSONResponse(status_code=status_code,
                                content=response.model_dump(mode="json", exclude_unset=True))
        return response

    async def create_summary(request: AnalysisSummaryRequest, http_request: Request):
        tracer = app.state.tracer
        with tracer.request_trace(
            "analysis_summary",
            headers=dict(http_request.headers),
            metadata={
                "operation": "analysis-summary",
                "provider": svc.provider,
                "model": svc.model_name,
                "promptVersion": svc.metadata.promptVersion,
            },
        ) as trace:
            # Mint the summaryId up front so every response — cache hit,
            # LLM, or fallback — carries the same stable reference.
            summary_id = str(uuid.uuid4())
            trace.attributes["summaryId"] = summary_id
            trace.set_metadata({"summaryId": summary_id})
            # Canonical form of the analysis JSON — dedupe key for the cache.
            with tracer.span("canonicalization") as obs:
                source_input = json.dumps(request.analysis, sort_keys=True)
                obs.update(metadata={"inputChars": len(source_input)})
            embedding = await app.state.embedder.embed(source_input)
            with tracer.span("report_cache_lookup") as obs:
                try:
                    existing = await app.state.store.find_similar_report(
                        source_input=source_input, embedding=embedding)
                except Exception as exc:
                    obs.set_error(exc)
                    logger.warning("Report cache lookup failed: %s", exc)
                    existing = None
                obs.update(metadata={
                    "cache_hit": existing is not None,
                    "embeddingAvailable": embedding is not None,
                    **({"match_type": existing["match_type"],
                        "similarity": existing["similarity"],
                        "matchedReportId": str(existing["id"])}
                       if existing is not None else {}),
                })
            if existing is not None:
                # Cached summary — recorded as a cache hit, never as an
                # LLM invocation.
                trace.set_metadata({
                    "result.source": "cache",
                    "result.cached": True,
                    "result.matchedReportId": str(existing["id"]),
                    "result.similarity": existing["similarity"],
                    "result.matchType": existing["match_type"],
                })
                result = {
                    "summary": existing["summary"],
                    "cached": True,
                    "matchedReportId": str(existing["id"]),
                    "similarity": existing["similarity"],
                    "matchType": existing["match_type"],
                }
                result["summaryId"] = summary_id
                if trace.trace_id:
                    result["traceId"] = trace.trace_id
                if trace.observation_id:
                    result["observationId"] = trace.observation_id
                return result
            try:
                summary = await app.state.service.summarize_analysis(
                    request.analysis, trace_headers=dict(http_request.headers))
            except Exception as exc:
                logger.exception("Analysis summarization failed")
                trace.set_error(f"{type(exc).__name__}: {exc}")
                raise HTTPException(status_code=502, detail="Analysis summarization failed.")
            with tracer.span("report_cache_persist") as obs:
                try:
                    await app.state.store.save_analysis_report(
                        source_input=source_input, summary=summary, embedding=embedding)
                    obs.update(metadata={"persisted": True})
                except Exception as exc:
                    obs.set_error(exc)
                    logger.warning("Failed to cache analysis report: %s", exc)
                    obs.update(metadata={"persisted": False,
                                         "error.type": type(exc).__name__})
            result = {"summary": summary, "cached": False,
                      "summaryId": summary_id}
            if trace.trace_id:
                result["traceId"] = trace.trace_id
            if trace.observation_id:
                result["observationId"] = trace.observation_id
            if trace.last_generation_id:
                result["generationId"] = trace.last_generation_id
            return result

    @app.get("/internal/v1/health", summary="Liveness probe")
    async def health():
        return {
            "status": "ok",
            "provider": svc.provider,
            "model": svc.model_name,
            "promptVersion": svc.metadata.promptVersion,
            "db": "connected" if db._pool is not None else "unavailable",
            "tracing": "enabled" if tracer.enabled else "disabled",
        }

    return app
