"""Langfuse tracing facade — no-op unless enabled and configured.

Built on the Langfuse Python SDK (v4, OpenTelemetry-based). Design rules:

- ``build_tracer(settings)`` returns a ``Tracer``. When Langfuse is disabled
  or misconfigured the tracer is a cheap no-op, so service code paths are
  identical to untraced behavior.
- One root trace per request (``request_trace``); child observations for
  pipeline stages (``span``); an explicit *generation* observation for each
  LLM call (``generation``). The Ollama path uses raw httpx, which no
  auto-instrumentation covers — hence the explicit generation observation
  in ``TestGenerationService._chat_ollama``.
- ``LANGFUSE_CAPTURE_IO=false`` withholds prompt/output content; metadata,
  token counts, latency, status and error types are still recorded.
- Incoming W3C ``traceparent`` headers are honored via Langfuse
  ``trace_context`` so callers can correlate their request with this trace.
- Instrumentation failures never break generation: every SDK interaction is
  wrapped and degrades to a no-op on error.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from time import perf_counter
from typing import Any, Iterator, Mapping

from app.config import Settings
from app.models import ObservationMetrics, RequestMetrics

logger = logging.getLogger(__name__)

# Active per-request trace handle. ContextVar keeps concurrent async
# requests isolated — each task sees only its own handle.
_current_trace: ContextVar["TraceHandle | None"] = ContextVar(
    "gatiqa_langfuse_trace", default=None
)


# ---------------------------------------------------------------------------
# W3C traceparent propagation
# ---------------------------------------------------------------------------


def parse_traceparent(headers: Mapping[str, str] | None) -> dict | None:
    """Extract a Langfuse ``trace_context`` from a W3C traceparent header.

    Returns ``{"trace_id": ..., "parent_span_id": ...}`` or None when the
    header is absent/malformed. Never raises.
    """
    if not headers:
        return None
    raw = None
    try:
        for key, value in headers.items():
            if key.lower() == "traceparent":
                raw = value
                break
    except Exception:
        return None
    if not raw:
        return None
    parts = raw.strip().split("-")
    # Format: version-trace_id-span_id-flags (e.g. "00-<32hex>-<16hex>-01")
    if len(parts) != 4:
        return None
    _version, trace_id, span_id, _flags = parts
    if len(trace_id) != 32 or len(span_id) != 16:
        return None
    if trace_id == "0" * 32 or span_id == "0" * 16:
        return None
    try:
        int(trace_id, 16)
        int(span_id, 16)
    except ValueError:
        return None
    return {"trace_id": trace_id, "parent_span_id": span_id}


# ---------------------------------------------------------------------------
# Ollama response -> Langfuse usage/timing mapping
# ---------------------------------------------------------------------------


def ollama_usage_details(data: dict) -> dict[str, int] | None:
    """Map Ollama /api/chat eval counts to Langfuse usage_details.

    prompt_eval_count -> input tokens, eval_count -> output tokens.
    Cost is only attached when per-token pricing is configured — see
    ``cost_details_for``.
    """
    prompt_tokens = data.get("prompt_eval_count")
    completion_tokens = data.get("eval_count")
    usage: dict[str, int] = {}
    if type(prompt_tokens) is int and prompt_tokens >= 0:
        usage["input"] = prompt_tokens
    if type(completion_tokens) is int and completion_tokens >= 0:
        usage["output"] = completion_tokens
    if "input" in usage and "output" in usage:
        usage["total"] = usage["input"] + usage["output"]
    return usage or None


def cost_details_for(
    usage: dict[str, int] | None,
    *,
    input_per_mtok: float | None,
    output_per_mtok: float | None,
) -> dict[str, float] | None:
    """Estimated USD cost from token usage and configured per-MTok pricing.

    Returns None unless both rates are configured and usage is present — a
    local Ollama call records no cost by default.
    """
    if not usage or input_per_mtok is None or output_per_mtok is None:
        return None
    cost: dict[str, float] = {}
    if isinstance(usage.get("input"), int):
        cost["input"] = usage["input"] * input_per_mtok / 1_000_000
    if isinstance(usage.get("output"), int):
        cost["output"] = usage["output"] * output_per_mtok / 1_000_000
    if "input" in cost and "output" in cost:
        cost["total"] = cost["input"] + cost["output"]
    return cost or None


def ollama_metadata(data: dict) -> dict[str, Any]:
    """Map Ollama duration fields (nanoseconds) and done_reason to metadata."""
    md: dict[str, Any] = {}
    for key in ("total_duration", "load_duration",
                "prompt_eval_duration", "eval_duration"):
        value = data.get(key)
        if isinstance(value, int):
            md[f"ollama.{key}.ns"] = value
    if data.get("done_reason"):
        md["ollama.done_reason"] = data["done_reason"]
    return md


def ollama_completion_start(started: datetime, data: dict) -> datetime | None:
    """Approximate completion_start_time: request start + load + prompt-eval.

    Ollama durations are nanoseconds; load+prompt_eval is the pre-generation
    phase, so completion begins roughly when eval starts.
    """
    pre_ns = (data.get("load_duration") or 0) + (data.get("prompt_eval_duration") or 0)
    if not isinstance(pre_ns, int) or pre_ns <= 0:
        return None
    return started + timedelta(microseconds=pre_ns / 1000)


# ---------------------------------------------------------------------------
# Observation / trace handles
# ---------------------------------------------------------------------------


def _safe_call(fn, *args, **kwargs) -> None:
    """Invoke an SDK call, swallowing instrumentation errors."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - tracing must never break requests
        logger.debug("Langfuse call failed (%s): %s", getattr(fn, "__name__", fn), exc)


class Observation:
    """Handle for a child observation (span or generation).

    ``_span`` is None when tracing is disabled or the SDK failed to start
    the observation — every method is then a no-op.
    """

    __slots__ = ("_span", "_capture_io", "generation_id", "observation_id", "trace_id", "_metrics")

    def __init__(self, *, capture_io: bool, generation_id: str | None = None):
        self._span = None
        self._capture_io = capture_io
        self.generation_id = generation_id
        self.observation_id: str | None = None
        self.trace_id: str | None = None
        self._metrics: ObservationMetrics | None = None

    def update(self, *, input=None, output=None, metadata=None, level=None,
               status_message=None, usage_details=None, cost_details=None,
               completion_start_time=None, model=None, model_parameters=None) -> None:
        metrics = self._metrics
        if metrics is not None:
            metrics.metadata.update(metadata or {})
            if level == "ERROR":
                metrics.status = "error"
            metrics.errorType = metrics.metadata.get("error.type")
            for key, field in (("input", "inputTokens"), ("output", "outputTokens"),
                               ("total", "totalTokens")):
                if usage_details and key in usage_details:
                    setattr(metrics, field, usage_details[key])
            for key, field in (("input", "inputCostUsd"), ("output", "outputCostUsd"),
                               ("total", "totalCostUsd")):
                if cost_details and key in cost_details:
                    setattr(metrics, field, cost_details[key])
            for key, value in (metadata or {}).items():
                if key.startswith("ollama.") and key.endswith(".ns"):
                    metrics.providerTimingsMs[key[7:-3]] = value / 1_000_000
            duration = metrics.providerTimingsMs.get("eval_duration")
            if duration and metrics.outputTokens is not None:
                metrics.outputTokensPerSecond = metrics.outputTokens * 1000 / duration
        if self._span is None:
            return
        kwargs: dict[str, Any] = {}
        if self._capture_io:
            if input is not None:
                kwargs["input"] = input
            if output is not None:
                kwargs["output"] = output
        for key, value in (
            ("metadata", metadata), ("level", level),
            ("status_message", status_message), ("usage_details", usage_details),
            ("cost_details", cost_details),
            ("completion_start_time", completion_start_time), ("model", model),
            ("model_parameters", model_parameters),
        ):
            if value is not None:
                kwargs[key] = value
        if kwargs:
            _safe_call(self._span.update, **kwargs)

    def set_error(self, exc: BaseException) -> None:
        self.update(
            level="ERROR",
            status_message=f"{type(exc).__name__}: {exc}",
            metadata={"error.type": type(exc).__name__},
        )


class TraceHandle:
    """Per-request trace state: trace id + generated generation ids.

    ``attributes`` is a generic bag for values produced inside the traced
    operation that the caller wants to expose (e.g. a minted summaryId).
    """

    __slots__ = ("trace_id", "observation_id", "generation_ids",
                 "attributes", "_span", "_capture_io", "_metadata", "observations",
                 "request_id", "started_at", "completed_at", "_started", "_elapsed",
                 "error_type")

    def __init__(self, *, capture_io: bool = True):
        self.trace_id: str | None = None
        self.observation_id: str | None = None  # root observation id
        self.generation_ids: list[str] = []
        self.attributes: dict[str, Any] = {}
        self._span = None
        self._capture_io = capture_io
        self._metadata: dict[str, Any] = {}
        self.observations: list[ObservationMetrics] = []
        self.request_id = str(uuid.uuid4())
        self.started_at = datetime.now(timezone.utc)
        self.completed_at: datetime | None = None
        self._started = perf_counter()
        self._elapsed: float | None = None
        self.error_type: str | None = None

    def finish(self) -> None:
        if self.completed_at is None:
            self.completed_at = datetime.now(timezone.utc)
            self._elapsed = (perf_counter() - self._started) * 1000

    def metrics(self) -> RequestMetrics:
        calls = [obs for obs in self.observations if obs.kind != "span"]

        def total(field):
            values = [getattr(obs, field) for obs in calls]
            return sum(values) if all(value is not None for value in values) else None

        md = self._metadata
        cost = total("totalCostUsd")
        llm_calls = sum(obs.kind == "generation" for obs in calls)
        return RequestMetrics(
            requestId=self.request_id, operation=md.get("operation", "unknown"),
            provider=md.get("provider"), model=md.get("model"),
            promptVersion=md.get("promptVersion"), traceId=self.trace_id,
            observationId=self.observation_id, summaryId=self.attributes.get("summaryId"),
            testCaseId=md.get("result.testCaseId"), startedAt=self.started_at,
            completedAt=self.completed_at or datetime.now(timezone.utc),
            latencyMs=self._elapsed if self._elapsed is not None else (perf_counter() - self._started) * 1000,
            status="error" if self.error_type else md.get("result.status", "success"),
            errorType=self.error_type, source=md.get("result.source"),
            cached=md.get("result.cached", False), fallbackReason=md.get("result.fallbackReason"),
            llmCalls=llm_calls, embeddingCalls=sum(obs.kind == "embedding" for obs in calls),
            retries=max(llm_calls - 1, 0), inputTokens=total("inputTokens"),
            outputTokens=total("outputTokens"), totalTokens=total("totalTokens"),
            knownTotalTokens=sum(obs.totalTokens if obs.totalTokens is not None else
                                 (obs.inputTokens or 0) + (obs.outputTokens or 0) for obs in calls),
            totalCostUsd=cost,
            knownCostUsd=sum(obs.totalCostUsd if obs.totalCostUsd is not None else
                             (obs.inputCostUsd or 0) + (obs.outputCostUsd or 0) for obs in calls),
            costStatus="no_model_calls" if not calls else "unknown" if cost is None else "estimated",
            observations=self.observations, metadata=dict(md),
        )

    @property
    def last_generation_id(self) -> str | None:
        return self.generation_ids[-1] if self.generation_ids else None

    def attach_response(self, response) -> None:
        """Expose observability references on a TestGenerationResponse."""
        if self.trace_id and getattr(response, "traceId", None) is None:
            response.traceId = self.trace_id
        if (self.observation_id
                and getattr(response, "observationId", None) is None):
            response.observationId = self.observation_id
        if self.last_generation_id and getattr(response, "generationId", None) is None:
            response.generationId = self.last_generation_id

    def record_result(self, response, *, error: str | None = None) -> None:
        """Record terminal result metadata on the root observation.

        Captures final status, result counts, rejection/clarification flags
        and cache-hit provenance. ``error`` marks the root observation ERROR
        with the given status message (e.g. terminal PROVIDER_ERROR).
        """
        status = getattr(getattr(response, "status", None), "value", None)
        metadata = {
            "result.status": status,
            "result.cached": bool(getattr(response, "cached", False)),
            "result.rejected": status == "FAILED",
            "result.requiredClarification": status == "NEEDS_CLARIFICATION",
            "result.clarifications": len(getattr(response, "clarifications", None) or []),
            "result.missingExpectations": len(getattr(response, "missingExpectations", None) or []),
            "result.validationIssues": len(getattr(response, "validationIssues", None) or []),
            "result.appliedDefaults": len(getattr(response, "appliedDefaults", None) or []),
        }
        matched = getattr(response, "matchedTestCaseId", None)
        if matched:
            metadata["result.matchedTestCaseId"] = matched
        match_type = getattr(response, "matchType", None)
        if match_type:
            metadata["result.matchType"] = match_type
        similarity = getattr(response, "similarity", None)
        if similarity is not None:
            metadata["result.similarity"] = similarity
        metadata["result.source"] = (
            "rejected" if error == "INPUT_REJECTED" else "cache" if metadata["result.cached"] else "llm"
        )
        metadata["result.testCaseId"] = getattr(response, "testCaseId", None)
        self._metadata.update(metadata)
        self.error_type = error
        if self._span is not None:
            kwargs: dict[str, Any] = {"metadata": metadata}
            if error:
                kwargs["level"] = "ERROR"
                kwargs["status_message"] = error
            _safe_call(self._span.update, **kwargs)

    def set_output(self, output) -> None:
        if self._span is not None and self._capture_io:
            _safe_call(self._span.update, output=output)

    def set_metadata(self, metadata: dict) -> None:
        self._metadata.update(metadata)
        if self._span is not None:
            _safe_call(self._span.update, metadata=metadata)

    def set_error(self, status_message: str) -> None:
        self.error_type = status_message.split(":", 1)[0]
        if self._span is not None:
            _safe_call(self._span.update, level="ERROR",
                       status_message=status_message)


# ---------------------------------------------------------------------------
# Tracer facade
# ---------------------------------------------------------------------------


class Tracer:
    """Thin facade over the Langfuse client; no-op when ``client`` is None."""

    def __init__(self, settings: Settings, client=None):
        self._settings = settings
        self._client = client
        self._capture_io = settings.langfuse_capture_io

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @property
    def capture_io(self) -> bool:
        return self._capture_io

    # ------------------------------------------------------------------
    # Root trace (one per request; reentrant — a nested call reuses the
    # active handle so endpoint-level and service-level tracing compose)
    # ------------------------------------------------------------------

    @contextmanager
    def request_trace(self, name: str, *, headers=None, metadata=None, input=None):
        active = _current_trace.get()
        if active is not None:
            yield active
            return
        with self._export_request_trace(name, headers=headers, metadata=metadata, input=input) as handle:
            handle.set_metadata(metadata or {})
            token = _current_trace.set(handle)
            try:
                yield handle
            except BaseException as exc:
                if handle.error_type is None:
                    handle.set_error(type(exc).__name__)
                raise
            finally:
                handle.finish()
                _current_trace.reset(token)

    @contextmanager
    def _export_request_trace(
        self,
        name: str,
        *,
        headers: Mapping[str, str] | None = None,
        metadata: dict | None = None,
        input=None,
    ) -> Iterator[TraceHandle]:
        active = _current_trace.get()
        if active is not None:
            yield active
            return

        handle = TraceHandle(capture_io=self._capture_io)
        if self._client is None:
            yield handle
            return

        try:
            cm = self._client.start_as_current_observation(
                name=name,
                as_type="span",
                trace_context=parse_traceparent(headers),
                input=input if self._capture_io else None,
                metadata=metadata,
            )
            span = cm.__enter__()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse root observation failed to start; "
                           "tracing degraded for this request: %s", exc)
            yield handle
            return

        handle._span = span
        handle.trace_id = getattr(span, "trace_id", None)
        handle.observation_id = getattr(span, "id", None)
        token = _current_trace.set(handle)
        exc_info = (None, None, None)
        try:
            yield handle
        except BaseException as exc:  # noqa: BLE001
            exc_info = sys.exc_info()
            _safe_call(span.update, level="ERROR",
                       status_message=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            _current_trace.reset(token)
            _safe_call(cm.__exit__, *exc_info)

    # ------------------------------------------------------------------
    # Child span observation
    # ------------------------------------------------------------------

    @contextmanager
    def _measure(self, obs, name, kind, *, metadata=None, **kwargs):
        metrics = ObservationMetrics(
            id=str(uuid.uuid4()), name=name, kind=kind,
            startedAt=datetime.now(timezone.utc), metadata=dict(metadata or {}),
            observationId=obs.observation_id, generationId=obs.generation_id, **kwargs,
        )
        obs._metrics = metrics
        handle = _current_trace.get()
        if handle is not None:
            handle.observations.append(metrics)
        started = perf_counter()
        try:
            yield obs
        except BaseException as exc:
            obs.set_error(exc)
            raise
        finally:
            metrics.latencyMs = (perf_counter() - started) * 1000

    @contextmanager
    def span(self, name: str, *, input=None, metadata=None, kind="span") -> Iterator[Observation]:
        with self._export_span(name, input=input, metadata=metadata, kind=kind) as obs:
            with self._measure(obs, name, kind, metadata=metadata,
                               model=(metadata or {}).get("model"),
                               provider=(metadata or {}).get("provider")):
                yield obs

    @contextmanager
    def _export_span(self, name: str, *, input=None, metadata=None, kind="span") -> Iterator[Observation]:
        obs = Observation(capture_io=self._capture_io)
        if self._client is None:
            yield obs
            return
        try:
            cm = self._client.start_as_current_observation(
                name=name,
                as_type=kind,
                input=input if self._capture_io else None,
                metadata=metadata,
                **({"model": (metadata or {}).get("model")} if kind == "embedding" else {}),
            )
            inner = cm.__enter__()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Langfuse span '%s' failed to start: %s", name, exc)
            yield obs
            return
        obs._span = inner
        obs.trace_id = getattr(inner, "trace_id", None)
        obs.observation_id = getattr(inner, "id", None)
        exc_info = (None, None, None)
        try:
            yield obs
        except BaseException as exc:  # noqa: BLE001
            exc_info = sys.exc_info()
            obs.set_error(exc)
            raise
        finally:
            _safe_call(cm.__exit__, *exc_info)

    # ------------------------------------------------------------------
    # Generation observation (explicit LLM-call tracing)
    # ------------------------------------------------------------------

    @contextmanager
    def generation(self, name: str, *, model: str, provider: str,
                   generation_id=None, attempt=None, input=None,
                   model_parameters=None, metadata=None) -> Iterator[Observation]:
        with self._export_generation(
            name, model=model, provider=provider, generation_id=generation_id,
            attempt=attempt, input=input, model_parameters=model_parameters, metadata=metadata,
        ) as obs:
            with self._measure(obs, name, "generation", model=model, provider=provider,
                               attempt=attempt, metadata=metadata):
                yield obs

    @contextmanager
    def _export_generation(
        self,
        name: str,
        *,
        model: str,
        provider: str,
        generation_id: str | None = None,
        attempt: int | None = None,
        input=None,
        model_parameters: dict | None = None,
        metadata: dict | None = None,
    ) -> Iterator[Observation]:
        """Explicit generation observation for one LLM call attempt.

        A stable ``generation_id`` is created when the caller does not
        supply one, recorded as observation metadata, and registered on the
        active trace handle so the response can expose it.
        """
        generation_id = generation_id or str(uuid.uuid4())
        obs = Observation(capture_io=self._capture_io, generation_id=generation_id)
        handle = _current_trace.get()
        if handle is not None:
            handle.generation_ids.append(generation_id)

        if self._client is None:
            yield obs
            return

        md = dict(metadata or {})
        md.setdefault("generationId", generation_id)
        md.setdefault("provider", provider)
        if attempt is not None:
            md.setdefault("attempt", attempt)
        try:
            cm = self._client.start_as_current_observation(
                name=name,
                as_type="generation",
                model=model,
                input=input if self._capture_io else None,
                model_parameters=model_parameters,
                metadata=md,
            )
            inner = cm.__enter__()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Langfuse generation '%s' failed to start: %s", name, exc)
            yield obs
            return
        obs._span = inner
        obs.trace_id = getattr(inner, "trace_id", None)
        obs.observation_id = getattr(inner, "id", None)
        exc_info = (None, None, None)
        try:
            yield obs
        except BaseException as exc:  # noqa: BLE001
            exc_info = sys.exc_info()
            obs.set_error(exc)
            raise
        finally:
            _safe_call(cm.__exit__, *exc_info)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def flush(self) -> None:
        if self._client is not None:
            _safe_call(self._client.flush)

    def shutdown(self) -> None:
        if self._client is not None:
            _safe_call(self._client.shutdown)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _instrument_pydantic_ai() -> None:
    """Route real pydantic-ai Agent calls through Langfuse's OTel provider.

    Langfuse registers a global TracerProvider when enabled, so
    ``Agent.instrument_all()`` forwards Agent spans to it. The direct Ollama
    httpx path is NOT covered by this — it gets explicit generation
    observations in ``_chat_ollama``. If an Agent path is added, rely on
    these auto-instrumented spans and do NOT wrap the same model call in a
    manual ``generation()`` observation — that would double-count it.
    """
    try:
        from pydantic_ai import Agent
        Agent.instrument_all()
        logger.debug("pydantic-ai Agent instrumentation enabled for Langfuse.")
    except Exception as exc:  # noqa: BLE001
        logger.debug("pydantic-ai instrumentation unavailable: %s", exc)


def build_tracer(settings: Settings, *, client=None) -> Tracer:
    """Create a Tracer from settings; no-op unless enabled AND configured.

    ``client`` is injectable for tests — passing a fake client exercises the
    enabled path without a live Langfuse instance.
    """
    if not settings.langfuse_enabled:
        return Tracer(settings, client=None)

    if client is None:
        if not (settings.langfuse_public_key and settings.langfuse_secret_key):
            logger.warning(
                "LANGFUSE_ENABLED=true but LANGFUSE_PUBLIC_KEY/"
                "LANGFUSE_SECRET_KEY are not set — tracing disabled."
            )
            return Tracer(settings, client=None)
        try:
            # OTel resource service.name: standard OTEL_SERVICE_NAME env var
            # is honored natively by the SDK; fall back to the configured
            # alias (GATIQA_SERVICE_NAME) when unset.
            if settings.otel_service_name:
                os.environ.setdefault("OTEL_SERVICE_NAME", settings.otel_service_name)
            from langfuse import Langfuse  # lazy import — optional dep

            from app import __version__

            client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                base_url=settings.langfuse_base_url,
                environment=settings.langfuse_environment,
                release=__version__,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse init failed — tracing disabled: %s", exc)
            return Tracer(settings, client=None)
        _instrument_pydantic_ai()
        logger.info(
            "Langfuse tracing enabled (env=%s, capture_io=%s).",
            settings.langfuse_environment, settings.langfuse_capture_io,
        )
    return Tracer(settings, client=client)
