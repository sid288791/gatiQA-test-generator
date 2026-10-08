"""Langfuse tracing tests — fake client only, no live Langfuse or model calls."""

import asyncio
import json
from contextlib import contextmanager

import pytest

from app.config import Settings
from app.models import GenerationStatus, TestGenerationRequest
from app.observability import (
    Tracer,
    build_tracer,
    ollama_metadata,
    ollama_usage_details,
    parse_traceparent,
)
from app.service import TestGenerationService
from tests.fake_provider import scripted_call

REQUEST = {
    "input": (
        "Search for cordless drill in store 123. SKU-100 should appear in the "
        "top 3 and cost USD 99.99."
    ),
    "profile": {"profileId": "search-basic", "version": 1,
                "enabledEvaluators": ["sku_match", "top_k", "pricing"]},
}

GOOD_OUTPUT = {
    "request": {"query": "cordless drill", "storeId": "123"},
    "expectations": {"products": [
        {"sku": "SKU-100", "maximumRank": 3,
         "price": {"value": "99.99", "currency": "USD"}}
    ]},
    "clarifications": [],
}


# ---- fake Langfuse client --------------------------------------------------

class FakeSpan:
    def __init__(self, recorder, kwargs):
        self._recorder = recorder
        self.kwargs = kwargs
        self.updates = []
        self.trace_id = "f" * 32
        self.id = f"obs-{len(recorder)}"

    def update(self, **kwargs):
        self.updates.append(kwargs)


class FakeLangfuse:
    """Records start_as_current_observation calls; spans record updates."""

    def __init__(self, fail_on_start=False):
        self.observations = []   # kwargs per started observation
        self.spans = []          # FakeSpan per started observation
        self.fail_on_start = fail_on_start
        self.flushed = False
        self.shutdown_called = False

    def start_as_current_observation(self, **kwargs):
        if self.fail_on_start:
            raise RuntimeError("langfuse backend unreachable")
        span = FakeSpan(self.spans, kwargs)
        self.observations.append(kwargs)
        self.spans.append(span)

        @contextmanager
        def cm():
            yield span

        return cm()

    def flush(self):
        self.flushed = True

    def shutdown(self):
        self.shutdown_called = True


def traced_settings(test_settings, **overrides):
    """Fresh Settings copy — never mutate the session-scoped fixture."""
    return test_settings.model_copy(update={"langfuse_enabled": True, **overrides})


def traced_service(test_settings, response, fake=None, capture_io=True):
    settings = traced_settings(test_settings, langfuse_capture_io=capture_io)
    tracer = build_tracer(settings, client=fake or FakeLangfuse())
    return TestGenerationService(
        settings, _call_fn=scripted_call(response), tracer=tracer)


def run(service, request=REQUEST, **kwargs):
    return asyncio.run(service.generate_async(
        TestGenerationRequest.model_validate(request), **kwargs))


# ---- disabled-by-default ----------------------------------------------------

def test_tracing_disabled_by_default(test_settings):
    svc = TestGenerationService(test_settings, _call_fn=scripted_call(GOOD_OUTPUT))
    assert svc._tracer.enabled is False
    resp = run(svc)
    assert resp.status is GenerationStatus.READY_FOR_REVIEW
    assert resp.traceId is None
    assert resp.generationId is None


def test_enabled_without_keys_disables(test_settings):
    settings = traced_settings(
        test_settings, langfuse_public_key=None, langfuse_secret_key=None)
    tracer = build_tracer(settings)
    assert tracer.enabled is False


# ---- enabled path -----------------------------------------------------------

def test_root_trace_and_child_spans(test_settings):
    fake = FakeLangfuse()
    resp = run(traced_service(test_settings, GOOD_OUTPUT, fake))

    assert resp.status is GenerationStatus.READY_FOR_REVIEW
    assert resp.traceId == "f" * 32

    names = [o["name"] for o in fake.observations]
    assert names[0] == "test_generation"
    for expected in ("input_sanitization", "semantic_cache_lookup",
                     "draft_parse_validate", "semantic_cache_persist"):
        assert expected in names

    root = fake.observations[0]
    assert root["as_type"] == "span"
    md = root["metadata"]
    assert md["operation"] == "test-generation"
    assert md["profileId"] == "search-basic"
    assert md["profileVersion"] == 1
    assert md["provider"] == "ollama"
    assert md["promptVersion"] == test_settings.prompt_version

    # Terminal result metadata recorded on the root observation.
    root_span = fake.spans[0]
    result_md = next(u["metadata"] for u in root_span.updates
                     if "result.status" in u.get("metadata", {}))
    assert result_md["result.status"] == "READY_FOR_REVIEW"
    assert result_md["result.cached"] is False
    assert result_md["result.rejected"] is False
    assert result_md["result.requiredClarification"] is False
    assert result_md["result.validationIssues"] == 0
    assert result_md["result.appliedDefaults"] == 0  # REQUEST has no approvedDefaults


def test_traceparent_propagated(test_settings):
    fake = FakeLangfuse()
    headers = {"traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"}
    run(traced_service(test_settings, GOOD_OUTPUT, fake), trace_headers=headers)
    assert fake.observations[0]["trace_context"] == {
        "trace_id": "0af7651916cd43dd8448eb211c80319c",
        "parent_span_id": "b7ad6b7169203331",
    }


def test_no_traceparent_gives_no_context(test_settings):
    fake = FakeLangfuse()
    run(traced_service(test_settings, GOOD_OUTPUT, fake))
    assert fake.observations[0]["trace_context"] is None


def test_instrumentation_failure_does_not_break_generation(test_settings):
    fake = FakeLangfuse(fail_on_start=True)
    resp = run(traced_service(test_settings, GOOD_OUTPUT, fake))
    assert resp.status is GenerationStatus.READY_FOR_REVIEW
    assert resp.draft is not None


def test_capture_io_disabled_still_records_metadata(test_settings):
    fake = FakeLangfuse()
    resp = run(traced_service(test_settings, GOOD_OUTPUT, fake, capture_io=False))
    assert resp.status is GenerationStatus.READY_FOR_REVIEW
    for obs in fake.observations:
        assert obs.get("input") is None
    # metadata still flows
    assert fake.observations[0]["metadata"]["operation"] == "test-generation"


# ---- generation observation on the real _chat_ollama path -------------------

class _FakeOllamaResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeHttpClient:
    """Stands in for httpx.AsyncClient inside _chat_ollama."""

    payload = {}

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, json=None):
        return _FakeOllamaResponse(self.payload)


def test_chat_ollama_generation_observation(test_settings, monkeypatch):
    _FakeHttpClient.payload = {
        "message": {"role": "assistant", "content": json.dumps(GOOD_OUTPUT)},
        "done_reason": "stop",
        "prompt_eval_count": 120,
        "eval_count": 45,
        "total_duration": 900_000_000,
        "load_duration": 100_000_000,
        "prompt_eval_duration": 200_000_000,
        "eval_duration": 600_000_000,
    }
    monkeypatch.setattr("app.service.httpx.AsyncClient", _FakeHttpClient)

    fake = FakeLangfuse()
    settings = traced_settings(test_settings)
    tracer = build_tracer(settings, client=fake)
    svc = TestGenerationService(settings, tracer=tracer)  # no _call_fn -> real path

    resp = run(svc)
    assert resp.status is GenerationStatus.READY_FOR_REVIEW
    assert resp.traceId == "f" * 32
    assert resp.generationId is not None

    gens = [o for o in fake.observations if o["as_type"] == "generation"]
    assert len(gens) == 1
    gen = gens[0]
    assert gen["name"] == "llm_generate"
    assert gen["model"] == settings.llm_model
    assert gen["metadata"]["provider"] == "ollama"
    assert gen["metadata"]["attempt"] == 1
    assert gen["metadata"]["generationId"] == resp.generationId
    assert gen["metadata"]["promptVersion"] == settings.prompt_version
    assert gen["model_parameters"]["format"] == "json"
    assert gen["model_parameters"]["think"] is False
    # prompt content captured when capture_io is on
    assert gen["input"][0]["role"] == "system"

    # usage + timing mapped from the Ollama response on span.update
    gen_span = fake.spans[fake.observations.index(gen)]
    update = gen_span.updates[-1]
    assert update["usage_details"] == {"input": 120, "output": 45, "total": 165}
    assert update["metadata"]["ollama.done_reason"] == "stop"
    assert update["metadata"]["ollama.total_duration.ns"] == 900_000_000
    assert update["completion_start_time"] is not None
    assert "cost_details" not in update  # no fabricated cost for local Ollama


def test_generation_error_marked_and_reraised(test_settings, monkeypatch):
    class _FailingHttpClient(_FakeHttpClient):
        async def post(self, url, json=None):
            raise RuntimeError("connection refused")

    monkeypatch.setattr("app.service.httpx.AsyncClient", _FailingHttpClient)
    fake = FakeLangfuse()
    settings = traced_settings(test_settings)
    svc = TestGenerationService(settings, tracer=build_tracer(settings, client=fake))

    resp = run(svc)
    assert resp.status is GenerationStatus.FAILED
    assert resp.validationIssues[0].code == "PROVIDER_ERROR"

    gens = [o for o in fake.observations if o["as_type"] == "generation"]
    assert len(gens) == settings.max_result_retries + 1
    gen_span = fake.spans[fake.observations.index(gens[0])]
    assert any(u.get("level") == "ERROR" for u in gen_span.updates)


def test_capture_io_off_hides_prompt_and_output(test_settings, monkeypatch):
    _FakeHttpClient.payload = {
        "message": {"role": "assistant", "content": json.dumps(GOOD_OUTPUT)},
        "prompt_eval_count": 10,
        "eval_count": 5,
    }
    monkeypatch.setattr("app.service.httpx.AsyncClient", _FakeHttpClient)
    fake = FakeLangfuse()
    settings = traced_settings(test_settings, langfuse_capture_io=False)
    svc = TestGenerationService(settings, tracer=build_tracer(settings, client=fake))

    resp = run(svc)
    assert resp.status is GenerationStatus.READY_FOR_REVIEW
    gens = [o for o in fake.observations if o["as_type"] == "generation"]
    assert gens[0]["input"] is None
    gen_span = fake.spans[fake.observations.index(gens[0])]
    for update in gen_span.updates:
        assert "output" not in update
    # usage counts still recorded
    assert gen_span.updates[-1]["usage_details"] == {"input": 10, "output": 5, "total": 15}


# ---- cache hit: no LLM generation observation --------------------------------

class _FakeEmbedder:
    async def embed(self, text):
        return [0.1] * 8


class _FakeStore:
    """Minimal TestCaseStore stand-in — one canned hit, records saves."""

    available = True
    HIT = {
        "id": "11111111-2222-3333-4444-555555555555",
        "test_spec": GOOD_OUTPUT,
        "match_type": "semantic",
        "similarity": 0.97,
    }

    def __init__(self, hit=HIT, saved_id="99999999-8888-7777-6666-555555555555"):
        self._hit = hit
        self._saved_id = saved_id
        self.saved = []

    async def find_similar(self, *, source_input, embedding):
        return self._hit

    async def save_test_case(self, **kwargs):
        self.saved.append(kwargs)
        return {"id": self._saved_id}


def test_cache_hit_creates_no_generation(test_settings):
    fake = FakeLangfuse()
    settings = traced_settings(test_settings)
    svc = TestGenerationService(
        settings,
        _call_fn=scripted_call(GOOD_OUTPUT),
        store=_FakeStore(),
        embedder=_FakeEmbedder(),
        tracer=build_tracer(settings, client=fake),
    )
    resp = run(svc)

    assert resp.cached is True
    assert resp.status is GenerationStatus.READY_FOR_REVIEW
    assert resp.matchedTestCaseId == "11111111-2222-3333-4444-555555555555"
    assert resp.matchType == "semantic"
    assert resp.testCaseId == "11111111-2222-3333-4444-555555555555"
    assert resp.traceId == "f" * 32
    assert resp.observationId == fake.spans[0].id
    # A cache hit is not an LLM call — no generation observation, no generationId.
    assert not any(o["as_type"] == "generation" for o in fake.observations)
    assert resp.generationId is None

    lookup = next(o for o in fake.observations if o["name"] == "semantic_cache_lookup")
    lookup_span = fake.spans[fake.observations.index(lookup)]
    md = lookup_span.updates[-1]["metadata"]
    assert md["cache_hit"] is True
    assert md["match_type"] == "semantic"
    assert md["similarity"] == 0.97
    assert md["matchedTestCaseId"] == "11111111-2222-3333-4444-555555555555"

    root_md = next(u["metadata"] for u in fake.spans[0].updates
                   if "result.status" in u.get("metadata", {}))
    assert root_md["result.cached"] is True
    assert root_md["result.matchType"] == "semantic"


# ---- terminal failure / clarification statuses --------------------------------

def test_malformed_output_records_error_and_failed(test_settings):
    fake = FakeLangfuse()
    resp = run(traced_service(test_settings, "not json at all", fake))

    assert resp.status is GenerationStatus.FAILED
    assert resp.validationIssues[0].code == "MODEL_OUTPUT_INVALID"

    # Each failed parse attempt is marked ERROR on draft_parse_validate.
    parse_spans = [
        fake.spans[i] for i, o in enumerate(fake.observations)
        if o["name"] == "draft_parse_validate"
    ]
    assert len(parse_spans) == test_settings.max_result_retries + 1
    assert all(any(u.get("level") == "ERROR" for u in s.updates) for s in parse_spans)

    # Root observation carries terminal FAILED status + error.
    root_updates = fake.spans[0].updates
    result_md = next(u["metadata"] for u in root_updates
                     if "result.status" in u.get("metadata", {}))
    assert result_md["result.status"] == "FAILED"
    assert result_md["result.rejected"] is True
    assert any(u.get("level") == "ERROR" and u.get("status_message") == "MODEL_OUTPUT_INVALID"
               for u in root_updates)


def test_needs_clarification_recorded(test_settings):
    fake = FakeLangfuse()
    ambiguous = {
        "input": "Search for hammer in store 55. SKU-200 should appear in the top 5 and cost 19.99.",
        "profile": {"profileId": "search-basic", "version": 1,
                    "enabledEvaluators": ["sku_match", "top_k", "pricing"]},
    }
    no_currency = {
        "request": {"query": "hammer", "storeId": "55"},
        "expectations": {"products": [
            {"sku": "SKU-200", "maximumRank": 5,
             "price": {"value": "19.99"}}
        ]},
        "clarifications": [],
    }
    resp = run(traced_service(test_settings, no_currency, fake), request=ambiguous)

    assert resp.status is GenerationStatus.NEEDS_CLARIFICATION
    assert resp.clarifications

    root_md = next(u["metadata"] for u in fake.spans[0].updates
                   if "result.status" in u.get("metadata", {}))
    assert root_md["result.status"] == "NEEDS_CLARIFICATION"
    assert root_md["result.requiredClarification"] is True
    assert root_md["result.clarifications"] == len(resp.clarifications)
    assert root_md["result.rejected"] is False


def test_missing_credentials_fail_closed(test_settings):
    """Enabled without keys -> disabled tracer, generation still works."""
    settings = traced_settings(
        test_settings, langfuse_public_key=None, langfuse_secret_key=None)
    tracer = build_tracer(settings)
    assert tracer.enabled is False
    svc = TestGenerationService(
        settings, _call_fn=scripted_call(GOOD_OUTPUT), tracer=tracer)
    resp = run(svc)
    assert resp.status is GenerationStatus.READY_FOR_REVIEW
    assert resp.traceId is None
    assert resp.generationId is None


def test_disabled_tracer_makes_no_external_calls(test_settings):
    """Disabled tracer never touches a client — zero observations recorded."""
    fake = FakeLangfuse()
    tracer = Tracer(test_settings, client=None)  # disabled path
    svc = TestGenerationService(
        test_settings, _call_fn=scripted_call(GOOD_OUTPUT), tracer=tracer)
    resp = run(svc)
    assert resp.status is GenerationStatus.READY_FOR_REVIEW
    assert fake.observations == []  # client never invoked
    assert resp.traceId is None


# ---- analysis-summary flow ----------------------------------------------------

ANALYSIS = {
    "totalProducts": 50,
    "analyzedProducts": 10,
    "findings": [
        {"type": "MISSING_PRICE", "severity": "ERROR", "count": 1,
         "reason": "price field is null",
         "details": [{"sku": "X1", "reason": "price null"}]},
        {"type": "LOW_STOCK", "severity": "WARNING", "count": 2,
         "skus": ["A2", "B3"]},
    ],
}

# Accepted by _extract_summary: anchor line with analyzed/total, all real
# SKUs mentioned, no reasoning markers.
GOOD_SUMMARY = (
    "Analyzed 10 of 50 products and found 2 issues.\n\n"
    "ERROR — MISSING_PRICE: 1 product (X1) — price field is null.\n\n"
    "WARNING — LOW_STOCK: 2 products (A2, B3)."
)
# _extract_summary strips blank lines — the accepted candidate is
# single-newline-joined.
EXTRACTED_SUMMARY = GOOD_SUMMARY.replace("\n\n", "\n")


def _summary_service(test_settings, monkeypatch, payload, fake=None,
                     capture_io=True):
    _FakeHttpClient.payload = payload
    monkeypatch.setattr("app.service.httpx.AsyncClient", _FakeHttpClient)
    settings = traced_settings(test_settings, langfuse_capture_io=capture_io)
    tracer = build_tracer(settings, client=fake or FakeLangfuse())
    return TestGenerationService(settings, tracer=tracer)


def _run_summary(service):
    return asyncio.run(service.summarize_analysis(ANALYSIS))


def test_summary_llm_success(test_settings, monkeypatch):
    fake = FakeLangfuse()
    svc = _summary_service(test_settings, monkeypatch, {
        "message": {"role": "assistant", "content": GOOD_SUMMARY},
        "prompt_eval_count": 300, "eval_count": 60,
    }, fake)

    summary = _run_summary(svc)
    assert summary == EXTRACTED_SUMMARY

    names = [o["name"] for o in fake.observations]
    assert names[0] == "analysis_summary"
    assert "llm_summarize" in names
    assert "summary_extraction" in names

    gen = next(o for o in fake.observations if o["as_type"] == "generation")
    assert gen["name"] == "llm_summarize"
    assert gen["metadata"]["operation"] == "analysis-summary"
    assert gen["metadata"]["provider"] == "ollama"
    assert gen["metadata"]["promptVersion"] == test_settings.prompt_version
    assert gen["metadata"]["generationId"]
    gen_span = fake.spans[fake.observations.index(gen)]
    assert gen_span.updates[-1]["usage_details"] == {
        "input": 300, "output": 60, "total": 360}

    ext = next(o for o in fake.observations if o["name"] == "summary_extraction")
    ext_span = fake.spans[fake.observations.index(ext)]
    ext_md = ext_span.updates[-1]["metadata"]
    assert ext_md["source"] == "llm"
    assert ext_md["usedFallback"] is False
    assert ext_md["fallbackReason"] is None

    # Root carries summaryId + aggregate analysis stats + result source.
    root_md = {}
    for u in fake.spans[0].updates:
        root_md.update(u.get("metadata") or {})
    assert root_md["summaryId"]
    assert root_md["analysis.analyzedProducts"] == 10
    assert root_md["analysis.totalProducts"] == 50
    assert root_md["analysis.findingCount"] == 2
    assert root_md["analysis.severityDistribution"] == {"ERROR": 1, "WARNING": 1}
    assert root_md["analysis.typeDistribution"] == {
        "MISSING_PRICE": 1, "LOW_STOCK": 1}
    assert root_md["result.source"] == "llm"


def test_summary_malformed_falls_back(test_settings, monkeypatch):
    fake = FakeLangfuse()
    svc = _summary_service(test_settings, monkeypatch, {
        "message": {"role": "assistant",
                    "content": "let me think about this problem..."},
    }, fake)

    summary = _run_summary(svc)
    assert "Analyzed 10 of 50 products" in summary  # deterministic fallback text

    # The LLM was invoked (generation exists) but the fallback is a span,
    # not an LLM observation.
    assert any(o["as_type"] == "generation" and o["name"] == "llm_summarize"
               for o in fake.observations)
    ext = next(o for o in fake.observations if o["name"] == "summary_extraction")
    assert ext["as_type"] == "span"
    ext_span = fake.spans[fake.observations.index(ext)]
    ext_md = ext_span.updates[-1]["metadata"]
    assert ext_md["source"] == "fallback"
    assert ext_md["usedFallback"] is True
    assert ext_md["fallbackReason"] == "no_anchor_line"

    root_md = {}
    for u in fake.spans[0].updates:
        root_md.update(u.get("metadata") or {})
    assert root_md["result.source"] == "fallback"
    assert root_md["result.fallbackReason"] == "no_anchor_line"


def test_cost_details_recorded_when_priced(test_settings, monkeypatch):
    """Configured per-MTok pricing -> cost_details on the generation."""
    fake = FakeLangfuse()
    settings = traced_settings(
        test_settings,
        langfuse_input_cost_per_mtok=3.0,   # $3 / 1M input tokens
        langfuse_output_cost_per_mtok=15.0,  # $15 / 1M output tokens
    )
    _FakeHttpClient.payload = {
        "message": {"role": "assistant", "content": GOOD_SUMMARY},
        "prompt_eval_count": 1000, "eval_count": 100,
    }
    monkeypatch.setattr("app.service.httpx.AsyncClient", _FakeHttpClient)
    svc = TestGenerationService(settings, tracer=build_tracer(settings, client=fake))

    _run_summary(svc)

    gen = next(o for o in fake.observations if o["as_type"] == "generation")
    gen_span = fake.spans[fake.observations.index(gen)]
    last = gen_span.updates[-1]
    assert last["usage_details"] == {"input": 1000, "output": 100, "total": 1100}
    assert last["cost_details"] == pytest.approx(
        {"input": 0.003, "output": 0.0015, "total": 0.0045})


def test_cost_details_absent_by_default(test_settings, monkeypatch):
    """No pricing configured -> no cost_details (local Ollama is free)."""
    fake = FakeLangfuse()
    svc = _summary_service(test_settings, monkeypatch, {
        "message": {"role": "assistant", "content": GOOD_SUMMARY},
        "prompt_eval_count": 1000, "eval_count": 100,
    }, fake)

    _run_summary(svc)

    gen = next(o for o in fake.observations if o["as_type"] == "generation")
    gen_span = fake.spans[fake.observations.index(gen)]
    for u in gen_span.updates:
        assert "cost_details" not in u


def test_summary_provider_error(test_settings, monkeypatch):
    class _FailingHttpClient(_FakeHttpClient):
        async def post(self, url, json=None):
            raise RuntimeError("connection refused")

    monkeypatch.setattr("app.service.httpx.AsyncClient", _FailingHttpClient)
    fake = FakeLangfuse()
    settings = traced_settings(test_settings)
    svc = TestGenerationService(settings, tracer=build_tracer(settings, client=fake))

    with pytest.raises(RuntimeError):
        _run_summary(svc)

    gen = next(o for o in fake.observations if o["as_type"] == "generation")
    gen_span = fake.spans[fake.observations.index(gen)]
    assert any(u.get("level") == "ERROR" for u in gen_span.updates)
    # Root observation marked ERROR by the request_trace context manager.
    assert any(u.get("level") == "ERROR" for u in fake.spans[0].updates)


def test_summary_disabled_no_calls(test_settings, monkeypatch):
    _FakeHttpClient.payload = {
        "message": {"role": "assistant", "content": GOOD_SUMMARY}}
    monkeypatch.setattr("app.service.httpx.AsyncClient", _FakeHttpClient)
    fake = FakeLangfuse()
    svc = TestGenerationService(test_settings, tracer=Tracer(test_settings, client=None))

    summary = _run_summary(svc)
    assert summary == EXTRACTED_SUMMARY
    assert fake.observations == []


class _FakeReportStore:
    """TestCaseStore stand-in for the analysis-report cache."""

    available = True
    _pool = None

    def __init__(self, hit=None):
        self._hit = hit
        self.saved = []

    async def find_similar_report(self, *, source_input, embedding):
        return self._hit

    async def save_analysis_report(self, **kwargs):
        self.saved.append(kwargs)


def test_summary_endpoint_cache_hit(test_settings):
    from fastapi.testclient import TestClient
    from app.server import create_app

    fake = FakeLangfuse()
    settings = traced_settings(test_settings)
    tracer = build_tracer(settings, client=fake)
    svc = TestGenerationService(
        settings, _call_fn=scripted_call(GOOD_OUTPUT), tracer=tracer)
    store = _FakeReportStore(hit={
        "id": "rep-1", "summary": "cached summary text",
        "similarity": 0.99, "match_type": "exact",
    })
    app = create_app(svc, store=store)
    app.state.tracer = tracer

    resp = TestClient(app).post(
        "/internal/v1/analysis-summaries", json={"analysis": ANALYSIS})
    assert resp.status_code == 200
    body = resp.json()
    assert body["cached"] is True
    assert body["summary"] == "cached summary text"
    assert body["traceId"] == "f" * 32
    assert body["summaryId"]  # stable reference even on cache hits
    assert body["observationId"] == fake.spans[0].id
    assert "generationId" not in body  # cache hit is not an LLM invocation

    names = [o["name"] for o in fake.observations]
    assert names[0] == "analysis_summary"
    assert "canonicalization" in names
    assert "report_cache_lookup" in names
    assert not any(o["as_type"] == "generation" for o in fake.observations)

    lookup = next(o for o in fake.observations if o["name"] == "report_cache_lookup")
    lookup_md = fake.spans[fake.observations.index(lookup)].updates[-1]["metadata"]
    assert lookup_md["cache_hit"] is True
    assert lookup_md["match_type"] == "exact"
    assert lookup_md["similarity"] == 0.99
    assert lookup_md["matchedReportId"] == "rep-1"

    root_md = {}
    for u in fake.spans[0].updates:
        root_md.update(u.get("metadata") or {})
    assert root_md["result.source"] == "cache"
    assert root_md["result.cached"] is True


def test_summary_endpoint_full_flow(test_settings, monkeypatch):
    from fastapi.testclient import TestClient
    from app.server import create_app

    _FakeHttpClient.payload = {
        "message": {"role": "assistant", "content": GOOD_SUMMARY},
        "prompt_eval_count": 10, "eval_count": 5,
    }
    monkeypatch.setattr("app.service.httpx.AsyncClient", _FakeHttpClient)

    fake = FakeLangfuse()
    settings = traced_settings(test_settings)
    tracer = build_tracer(settings, client=fake)
    svc = TestGenerationService(settings, tracer=tracer)
    store = _FakeReportStore(hit=None)
    app = create_app(svc, store=store)
    app.state.tracer = tracer

    resp = TestClient(app).post(
        "/internal/v1/analysis-summaries", json={"analysis": ANALYSIS})
    assert resp.status_code == 200
    body = resp.json()
    assert body["cached"] is False
    assert body["summary"] == EXTRACTED_SUMMARY
    assert body["traceId"] == "f" * 32
    assert body["observationId"] == fake.spans[0].id
    assert body["generationId"]
    assert body["summaryId"]
    assert len(store.saved) == 1

    names = [o["name"] for o in fake.observations]
    for expected in ("analysis_summary", "canonicalization",
                     "report_cache_lookup", "llm_summarize",
                     "summary_extraction", "report_cache_persist"):
        assert expected in names


# ---- trace correlation --------------------------------------------------------

def test_all_observations_share_one_trace(test_settings):
    """Every observation in a request belongs to the same trace id."""
    fake = FakeLangfuse()
    resp = run(traced_service(test_settings, GOOD_OUTPUT, fake))
    assert resp.status is GenerationStatus.READY_FOR_REVIEW
    trace_ids = {s.trace_id for s in fake.spans}
    assert trace_ids == {"f" * 32}
    assert resp.traceId == "f" * 32
    assert resp.observationId == fake.spans[0].id


def test_missing_traceparent_starts_fresh_trace(test_settings):
    """No incoming context -> a new trace is created safely."""
    fake = FakeLangfuse()
    resp = run(traced_service(test_settings, GOOD_OUTPUT, fake))
    assert fake.observations[0]["trace_context"] is None
    assert resp.traceId == "f" * 32  # fresh trace id still assigned
    assert resp.observationId is not None


def test_persisted_test_case_id_on_response(test_settings):
    """A persisted draft exposes testCaseId on the response + persist span."""
    fake = FakeLangfuse()
    settings = traced_settings(test_settings)
    store = _FakeStore(hit=None)
    svc = TestGenerationService(
        settings,
        _call_fn=scripted_call(GOOD_OUTPUT),
        store=store,
        embedder=_FakeEmbedder(),
        tracer=build_tracer(settings, client=fake),
    )
    resp = run(svc)
    assert resp.status is GenerationStatus.READY_FOR_REVIEW
    assert resp.testCaseId == "99999999-8888-7777-6666-555555555555"
    assert len(store.saved) == 1

    persist = next(o for o in fake.observations
                   if o["name"] == "semantic_cache_persist")
    persist_md = fake.spans[fake.observations.index(persist)].updates[-1]["metadata"]
    assert persist_md["persisted"] is True
    assert persist_md["testCaseId"] == "99999999-8888-7777-6666-555555555555"


def test_langfuse_import_failure_disables(test_settings, monkeypatch):
    """langfuse package missing -> fail closed, generation unaffected."""
    import sys
    monkeypatch.setitem(sys.modules, "langfuse", None)
    settings = traced_settings(
        test_settings,
        langfuse_public_key="pk-test", langfuse_secret_key="sk-test")
    tracer = build_tracer(settings)
    assert tracer.enabled is False
    svc = TestGenerationService(
        settings, _call_fn=scripted_call(GOOD_OUTPUT), tracer=tracer)
    resp = run(svc)
    assert resp.status is GenerationStatus.READY_FOR_REVIEW
    assert resp.traceId is None


def test_langfuse_init_failure_disables(test_settings, monkeypatch):
    """Langfuse() constructor raising -> fail closed, generation unaffected."""
    import sys
    import types
    broken = types.ModuleType("langfuse")

    class _BrokenLangfuse:
        def __init__(self, **kwargs):
            raise RuntimeError("cannot reach langfuse")

    broken.Langfuse = _BrokenLangfuse
    monkeypatch.setitem(sys.modules, "langfuse", broken)
    settings = traced_settings(
        test_settings,
        langfuse_public_key="pk-test", langfuse_secret_key="sk-test")
    tracer = build_tracer(settings)
    assert tracer.enabled is False
    svc = TestGenerationService(
        settings, _call_fn=scripted_call(GOOD_OUTPUT), tracer=tracer)
    resp = run(svc)
    assert resp.status is GenerationStatus.READY_FOR_REVIEW


# ---- helpers -----------------------------------------------------------------

def test_local_metrics_without_langfuse(test_settings):
    tracer = Tracer(test_settings)
    with tracer.request_trace("analysis_summary", metadata={
        "operation": "analysis-summary", "provider": "ollama", "model": "test",
    }) as trace:
        with tracer.request_trace("nested") as nested:
            assert nested is trace
        with tracer.generation("llm_summarize", model="test", provider="ollama") as gen:
            gen.update(usage_details={"input": 10, "output": 5, "total": 15},
                       cost_details={"total": 0.002})
        trace.set_metadata({"result.source": "llm"})
    metrics = trace.metrics()
    assert metrics.totalTokens == 15
    assert metrics.totalCostUsd == 0.002
    assert metrics.llmCalls == 1
    assert metrics.latencyMs >= 0
    assert metrics.source == "llm"
    assert metrics.observations[0].latencyMs >= 0


def test_parse_traceparent():
    assert parse_traceparent(None) is None
    assert parse_traceparent({}) is None
    assert parse_traceparent({"traceparent": "garbage"}) is None
    assert parse_traceparent({"traceparent": "00-" + "0" * 32 + "-b7ad6b7169203331-01"}) is None
    assert parse_traceparent({
        "Traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    }) == {"trace_id": "0af7651916cd43dd8448eb211c80319c",
           "parent_span_id": "b7ad6b7169203331"}


def test_ollama_usage_details():
    assert ollama_usage_details({}) is None
    assert ollama_usage_details({"prompt_eval_count": 7}) == {"input": 7}
    assert ollama_usage_details({"prompt_eval_count": 7, "eval_count": 3}) == {
        "input": 7, "output": 3, "total": 10}
    assert ollama_metadata({"done_reason": "stop", "total_duration": 5}) == {
        "ollama.done_reason": "stop", "ollama.total_duration.ns": 5}


@pytest.mark.parametrize("fake", [None, FakeLangfuse(), FakeLangfuse(fail_on_start=True)])
def test_metrics_capture_independent_of_sdk(test_settings, fake):
    tracer = Tracer(test_settings, client=fake)
    with tracer.request_trace("analysis_summary", metadata={"operation": "analysis-summary"}) as trace:
        with tracer.generation("llm", model="test", provider="ollama") as gen:
            gen.update(input="private prompt", output="private response",
                       usage_details={"input": 10})
    metrics = trace.metrics()
    assert metrics.inputTokens == 10
    assert metrics.totalTokens is None
    assert metrics.knownTotalTokens == 10
    assert metrics.totalCostUsd is None
    assert metrics.costStatus == "unknown"
    assert metrics.llmCalls == 1
    assert "private prompt" not in metrics.model_dump_json()
    assert "private response" not in metrics.model_dump_json()


def test_embedding_exports_usage_as_embedding(test_settings, monkeypatch):
    from app.embeddings import Embedder

    _FakeHttpClient.payload = {"embeddings": [[0.1] * 768], "prompt_eval_count": 12}
    monkeypatch.setattr("app.embeddings.httpx.AsyncClient", _FakeHttpClient)
    fake = FakeLangfuse()
    tracer = Tracer(test_settings, client=fake)
    with tracer.request_trace("analysis_summary") as trace:
        result = asyncio.run(Embedder(test_settings, tracer=tracer).embed("private text"))
    assert len(result) == 768
    assert trace.metrics().embeddingCalls == 1
    assert trace.metrics().totalTokens == 12
    assert fake.observations[1]["as_type"] == "embedding"
    assert fake.observations[1]["model"] == test_settings.embedding_model
    assert fake.spans[1].updates[-1]["usage_details"]["input"] == 12


def test_zero_usage_is_not_unknown(test_settings):
    tracer = Tracer(test_settings)
    with tracer.request_trace("cached") as trace:
        trace.set_metadata({"result.cached": True, "result.source": "cache"})
    metrics = trace.metrics()
    assert metrics.totalTokens == 0
    assert metrics.totalCostUsd == 0
    assert metrics.costStatus == "no_model_calls"
    assert metrics.llmCalls == 0


def test_parallel_requests_have_isolated_metrics(test_settings):
    tracer = Tracer(test_settings)

    async def request(count):
        with tracer.request_trace("analysis_summary") as trace:
            with tracer.generation("llm", model=str(count), provider="ollama") as gen:
                await asyncio.sleep(0)
                gen.update(usage_details={"input": count, "output": 1, "total": count + 1})
        return trace.metrics()

    async def run_requests():
        return await asyncio.gather(request(10), request(20))

    first, second = asyncio.run(run_requests())
    assert first.requestId != second.requestId
    assert first.totalTokens == 11
    assert second.totalTokens == 21
    assert len(first.observations) == len(second.observations) == 1
    with tracer.request_trace("next") as trace:
        assert trace.observations == []


@pytest.mark.parametrize("payload", [
    {"prompt_eval_count": True}, {"prompt_eval_count": -1},
    {"eval_count": False}, {"eval_count": -10},
])
def test_invalid_usage_is_unknown(payload):
    assert ollama_usage_details(payload) is None
