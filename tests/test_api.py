import asyncio
import os
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.models import AnalysisSummaryRequest, MetricsDashboard, MetricsTotals
from app.observability import Tracer

from app.server import create_app
from app.service import TestGenerationService
from tests.fake_provider import scripted_call

BODY = {
    "input": (
        "Search for cordless drill in store 123. SKU-100 should appear in the "
        "top 3 and cost USD 99.99."
    ),
    "profile": {
        "profileId": "search-basic",
        "version": 1,
        "enabledEvaluators": ["sku_match", "top_k", "pricing"],
    },
    "approvedDefaults": {"priceTolerance": "0.01"},
}

GOOD_OUTPUT = {
    "request": {"query": "cordless drill", "storeId": "123"},
    "expectations": {"products": [
        {"sku": "SKU-100", "maximumRank": 3,
         "price": {"value": "99.99", "currency": "USD"}}
    ]},
    "clarifications": [],
}


def client(test_settings, response) -> TestClient:
    svc = TestGenerationService(test_settings, _call_fn=scripted_call(response))
    return TestClient(create_app(svc))


def test_complete_draft(test_settings):
    resp = client(test_settings, GOOD_OUTPUT).post(
        "/internal/v1/test-generations", json=BODY)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "READY_FOR_REVIEW"
    assert body["draft"]["request"] == {"query": "cordless drill", "storeId": "123"}
    p = body["draft"]["expectations"]["products"][0]
    assert p["price"] == {"value": "99.99", "currency": "USD", "tolerance": "0.01"}


def test_ambiguous_currency(test_settings):
    out = {
        "request": {"query": "hammer", "storeId": "55"},
        "expectations": {"products": [
            {"sku": "SKU-200", "maximumRank": 5, "price": {"value": "19.99"}}
        ]},
        "clarifications": [],
    }
    resp = client(test_settings, out).post(
        "/internal/v1/test-generations",
        json={
            "input": "Search for hammer in store 55. SKU-200 top 5 cost 19.99.",
            "profile": {"profileId": "p", "version": 1,
                         "enabledEvaluators": ["sku_match", "top_k", "pricing"]},
        },
    )
    assert resp.json()["status"] == "NEEDS_CLARIFICATION"


def test_invalid_body_422(test_settings):
    # "input" is required (min_length=1) — a body without it is rejected.
    resp = client(test_settings, GOOD_OUTPUT).post(
        "/internal/v1/test-generations", json={"profile": {"profileId": "p", "version": 1}})
    assert resp.status_code == 422


def test_provider_failure(test_settings):
    resp = client(test_settings, RuntimeError("down")).post(
        "/internal/v1/test-generations", json=BODY)
    assert resp.json()["status"] == "FAILED"
    assert resp.json()["validationIssues"][0]["code"] == "PROVIDER_ERROR"


def test_health(test_settings):
    resp = client(test_settings, GOOD_OUTPUT).get("/internal/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


class MetricsStore:
    _pool = None

    def __init__(self, hit=None):
        self.metrics = []
        self.hit = hit
        self.queries = []

    async def find_similar_report(self, **kwargs):
        return self.hit

    async def save_analysis_report(self, **kwargs):
        return {"id": "report-1"}

    async def save_request_metrics(self, metrics):
        self.metrics.append(metrics)

    async def analysis_dashboard(self, request):
        self.queries.append(request)
        start, end = request.time_window()
        return MetricsDashboard(
            available=True, fromTime=start, toTime=end, limit=request.limit,
            offset=request.offset, totals=MetricsTotals(totalRequests=len(self.metrics)),
            traces=self.metrics,
        )


def summary_client(test_settings, monkeypatch, *, content=None, hit=None, failure=False):
    from tests.test_observability import GOOD_SUMMARY, _FakeHttpClient, _FakeOllamaResponse

    class HttpClient(_FakeHttpClient):
        async def post(self, url, json=None):
            if url.endswith("/api/embed"):
                return _FakeOllamaResponse({"embeddings": [[0.1] * 768], "prompt_eval_count": 20})
            if failure:
                raise RuntimeError("private provider error")
            return _FakeOllamaResponse({
                "message": {"content": GOOD_SUMMARY if content is None else content},
                "prompt_eval_count": 100, "eval_count": 50,
                "eval_duration": 500_000_000,
            })

    monkeypatch.setattr("app.service.httpx.AsyncClient", HttpClient)
    settings = test_settings.model_copy(update={
        "langfuse_input_cost_per_mtok": 2,
        "langfuse_output_cost_per_mtok": 4,
        "langfuse_embedding_cost_per_mtok": 1,
    })
    store = MetricsStore(hit)
    app = create_app(TestGenerationService(settings), store=store)
    return TestClient(app), store


def test_combined_summary_metrics_and_dashboard(test_settings, monkeypatch):
    from tests.test_observability import ANALYSIS
    api, store = summary_client(test_settings, monkeypatch)
    response = api.post("/internal/v1/analysis-summaries", json={"analysis": ANALYSIS})
    assert response.status_code == 200
    body = response.json()
    metrics = body["metrics"]
    assert metrics["totalTokens"] == 170
    assert metrics["inputTokens"] == 120
    assert metrics["outputTokens"] == 50
    assert metrics["totalCostUsd"] == pytest.approx(0.00042)
    assert metrics["costStatus"] == "estimated"
    assert metrics["llmCalls"] == 1
    assert metrics["embeddingCalls"] == 1
    assert metrics["source"] == "llm"
    assert metrics["summaryId"] == body["summaryId"]
    assert metrics["latencyMs"] >= 0
    assert len(metrics["observations"]) == 6
    generation = next(obs for obs in metrics["observations"] if obs["kind"] == "generation")
    assert generation["outputTokensPerSecond"] == 100
    assert generation["providerTimingsMs"]["eval_duration"] == 500
    assert "input" not in generation
    assert "output" not in generation
    assert body["metricsPersisted"] is True
    assert body["dashboard"]["totals"]["totalRequests"] == 1
    assert body["dashboard"]["traces"][0]["requestId"] == metrics["requestId"]
    assert len(store.metrics) == 1


def test_combined_api_openapi_contract(test_settings):
    schema = client(test_settings, GOOD_OUTPUT).get("/openapi.json").json()
    endpoint = schema["paths"]["/internal/v1/analysis-summaries"]["post"]
    for code in ("200", "502"):
        assert endpoint["responses"][code]["content"]["application/json"]["schema"]["$ref"].endswith(
            "/AnalysisSummaryResponse")
    metrics = schema["components"]["schemas"]["RequestMetrics"]["properties"]
    assert {"totalTokens", "totalCostUsd", "latencyMs", "observations"} <= metrics.keys()


def test_dashboard_only_does_not_generate_or_persist(test_settings, monkeypatch):
    api, store = summary_client(test_settings, monkeypatch, failure=True)
    response = api.post("/internal/v1/analysis-summaries", json={
        "operation": "test-generation", "limit": 5, "offset": 10,
    })
    assert response.status_code == 200
    assert set(response.json()) == {"dashboard"}
    assert store.metrics == []
    assert store.queries[0].operation == "test-generation"
    assert store.queries[0].limit == 5
    assert store.queries[0].offset == 10


def test_cache_hit_counts_embedding_but_not_llm(test_settings, monkeypatch):
    api, store = summary_client(test_settings, monkeypatch, hit={
        "id": "cached-report", "summary": "Cached", "match_type": "normalized", "similarity": 1,
    })
    body = api.post("/internal/v1/analysis-summaries", json={"analysis": {}}).json()
    assert body["cached"] is True
    assert "generationId" not in body
    assert body["metrics"]["source"] == "cache"
    assert body["metrics"]["llmCalls"] == 0
    assert body["metrics"]["totalTokens"] == 20
    assert body["metrics"]["totalCostUsd"] == pytest.approx(0.00002)
    assert len(store.metrics) == 1


def test_summary_fallback_retains_usage(test_settings, monkeypatch):
    from tests.test_observability import ANALYSIS
    api, _ = summary_client(test_settings, monkeypatch, content="invalid summary")
    body = api.post("/internal/v1/analysis-summaries", json={"analysis": ANALYSIS}).json()
    assert body["metrics"]["source"] == "fallback"
    assert body["metrics"]["fallbackReason"] == "no_anchor_line"
    assert body["metrics"]["totalTokens"] == 170


def test_failed_summary_retains_metrics_and_dashboard(test_settings, monkeypatch):
    api, store = summary_client(test_settings, monkeypatch, failure=True)
    response = api.post("/internal/v1/analysis-summaries", json={"analysis": {}})
    assert response.status_code == 502
    body = response.json()
    assert body["errorCode"] == "SUMMARIZATION_FAILED"
    assert body["metrics"]["status"] == "error"
    assert body["metrics"]["errorType"] == "RuntimeError"
    assert body["metrics"]["llmCalls"] == 1
    assert body["metrics"]["totalTokens"] is None
    assert body["metrics"]["knownTotalTokens"] == 20
    assert body["metrics"]["totalCostUsd"] is None
    assert "private provider error" not in response.text
    assert body["metricsPersisted"] is True
    assert len(store.metrics) == 1


def test_summary_survives_metrics_storage_failure(test_settings, monkeypatch):
    api, store = summary_client(test_settings, monkeypatch)

    async def unavailable(*args):
        raise RuntimeError("private database connection details")

    monkeypatch.setattr(store, "save_request_metrics", unavailable)
    monkeypatch.setattr(store, "analysis_dashboard", unavailable)
    response = api.post("/internal/v1/analysis-summaries", json={"analysis": {}})
    assert response.status_code == 200
    body = response.json()
    assert body["metricsPersisted"] is False
    assert body["dashboard"]["available"] is False
    assert body["dashboard"]["totals"] is None
    assert body["metrics"]["llmCalls"] == 1
    assert "private database" not in response.text


@pytest.mark.parametrize("body", [
    {"limit": 0}, {"limit": 101}, {"offset": -1}, {"operation": "invalid"},
    {"fromTime": "2026-01-01T00:00:00Z", "toTime": "2026-01-01T00:00:00Z"},
    {"fromTime": "2026-01-01T00:00:00Z", "toTime": "2026-05-01T00:00:00Z"},
    {"fromTime": "2026-01-01T00:00:00"},
])
def test_dashboard_invalid_filters(test_settings, body):
    response = client(test_settings, GOOD_OUTPUT).post("/internal/v1/analysis-summaries", json=body)
    assert response.status_code == 422


def test_test_generations_are_recorded_in_same_history(test_settings, monkeypatch):
    import json
    api, store = summary_client(test_settings, monkeypatch, content=json.dumps(GOOD_OUTPUT))
    response = api.post("/internal/v1/test-generations", json=BODY)
    assert response.status_code == 200
    assert store.metrics[0].operation == "test-generation"
    assert store.metrics[0].totalTokens == 150
    assert store.metrics[0].status == "READY_FOR_REVIEW"
    response = api.post("/internal/v1/analysis-summaries", json={})
    assert response.json()["dashboard"]["traces"][0]["operation"] == "test-generation"


def test_test_generation_failure_tracks_all_attempts(test_settings, monkeypatch):
    api, store = summary_client(test_settings, monkeypatch, failure=True)
    response = api.post("/internal/v1/test-generations", json=BODY)
    assert response.json()["status"] == "FAILED"
    metrics = store.metrics[0]
    assert metrics.llmCalls == test_settings.max_result_retries + 1
    assert metrics.retries == test_settings.max_result_retries
    assert metrics.status == "error"
    assert metrics.errorType == "PROVIDER_ERROR"
    assert metrics.totalTokens is None
    assert all(obs.errorType == "RuntimeError" for obs in metrics.observations
               if obs.kind == "generation")


def test_postgres_metrics_aggregation_and_pagination(test_settings):
    import asyncpg
    from app.db import TestCaseStore, _METRICS_SCHEMA

    dsn = os.environ.get("GATIQA_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("Set GATIQA_TEST_DATABASE_URL to run temporary-table Postgres verification.")

    async def check():
        store = TestCaseStore(test_settings)
        store._pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1)
        try:
            await store._pool.execute(_METRICS_SCHEMA.replace(
                "CREATE TABLE IF NOT EXISTS request_metrics", "CREATE TEMP TABLE request_metrics"))
            tracer = Tracer(test_settings)
            end = datetime.now(timezone.utc)
            for index, operation in enumerate(("test-generation", "analysis-summary", "analysis-summary")):
                with tracer.request_trace(operation, metadata={"operation": operation}) as trace:
                    if index < 2:
                        with tracer.generation("llm", provider="ollama", model="test") as gen:
                            gen.update(usage_details={"input": 10, "output": 5, "total": 15})
                            if index == 0:
                                gen.update(cost_details={"total": 0.002})
                        if index == 1:
                            trace.set_error("PROVIDER_ERROR")
                    else:
                        trace.set_metadata({"result.cached": True, "result.source": "cache"})
                metrics = trace.metrics()
                metrics.startedAt = end - timedelta(minutes=3 - index)
                metrics.completedAt = metrics.startedAt + timedelta(milliseconds=(index + 1) * 100)
                metrics.latencyMs = (index + 1) * 100
                await store.save_request_metrics(metrics)
                await store.save_request_metrics(metrics)
            request = AnalysisSummaryRequest(fromTime=end - timedelta(hours=1), toTime=end, limit=1)
            dashboard = await store.analysis_dashboard(request)
            totals = dashboard.totals
            assert totals.totalRequests == 3
            assert totals.failedRequests == 1
            assert totals.cacheHits == 1
            assert totals.totalTokens == 30
            assert totals.totalCostUsd is None
            assert totals.knownCostUsd == pytest.approx(0.002)
            assert totals.requestsWithUnknownCost == 1
            assert totals.errorRate == pytest.approx(1 / 3)
            assert totals.averageLatencyMs == 200
            assert totals.p50LatencyMs == 200
            assert totals.p95LatencyMs == 290
            assert dashboard.hasMore is True
            assert len(dashboard.traces) == 1
            assert dashboard.traces[0].cached is True
            assert dashboard.byOperation["analysis-summary"].totalRequests == 2
            filtered = await store.analysis_dashboard(request.model_copy(update={
                "operation": "analysis-summary", "offset": 1,
            }))
            assert filtered.totals.totalRequests == 2
            assert filtered.hasMore is False
            assert filtered.traces[0].status == "error"
            empty_page = await store.analysis_dashboard(request.model_copy(update={"offset": 10}))
            assert empty_page.traces == []
            assert empty_page.totals.totalRequests == 3
            empty_window = await store.analysis_dashboard(request.model_copy(update={
                "fromTime": end, "toTime": end + timedelta(hours=1),
            }))
            assert empty_window.totals.totalRequests == 0
            assert empty_window.totals.totalTokens == 0
            assert empty_window.totals.averageLatencyMs is None
            assert empty_window.byOperation == {}
        finally:
            await store.close()

    asyncio.run(check())
