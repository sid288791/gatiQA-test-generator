from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Literal

from pydantic import AliasChoices, AwareDatetime, BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Request models (Spring-facing contract)
# ---------------------------------------------------------------------------


class EvaluationProfile(BaseModel):
    profileId: str = Field(min_length=1, max_length=128)
    version: int = Field(ge=1)
    enabledEvaluators: list[str] = Field(default_factory=list)


class ExpectationSchemaSpec(BaseModel):
    evaluator: str = Field(min_length=1, max_length=64)
    requiredFields: list[str] = Field(default_factory=list)


class TestGenerationRequest(BaseModel):
    input: str = Field(min_length=1, max_length=100_000)
    profile: EvaluationProfile = Field(
        default_factory=lambda: EvaluationProfile(
            profileId="default", version=1, enabledEvaluators=[]
        )
    )
    expectationSchemas: list[ExpectationSchemaSpec] | None = None
    approvedDefaults: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Draft models (structured LLM output, validated by pydantic-ai)
# ---------------------------------------------------------------------------


class PriceSpec(BaseModel):
    value: str = Field(min_length=1, max_length=32)
    currency: str | None = None
    tolerance: str | None = None


class ProductExpectation(BaseModel):
    sku: str = Field(min_length=1, max_length=64)
    maximumRank: int | None = Field(default=None, ge=1)
    price: PriceSpec | None = None
    availability: str | None = None
    badges: list[str] = Field(default_factory=list)


class DraftRequest(BaseModel):
    query: str | None = None
    storeId: str | None = None
    endpointUrl: str | None = None


class ResultCountSpec(BaseModel):
    greaterThan: int | None = Field(default=None, ge=0)
    lessThan: int | None = Field(default=None, ge=0)
    equalTo: int | None = Field(default=None, ge=0)


class DraftExpectations(BaseModel):
    products: list[ProductExpectation] = Field(default_factory=list)
    resultCount: ResultCountSpec | None = None

    @model_validator(mode="before")
    @classmethod
    def _drop_sku_less_products(cls, data):
        """LLMs sometimes emit placeholder products like {"sku": null, ...}.
        A product without a sku is meaningless — drop it instead of failing
        the whole draft validation."""
        if isinstance(data, dict) and isinstance(data.get("products"), list):
            data["products"] = [
                p for p in data["products"]
                if isinstance(p, dict) and p.get("sku")
            ]
        return data


class Draft(BaseModel):
    request: DraftRequest = Field(default_factory=DraftRequest)
    expectations: DraftExpectations = Field(default_factory=DraftExpectations)
    clarifications: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class GenerationStatus(str, Enum):
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    FAILED = "FAILED"


class AppliedDefault(BaseModel):
    field: str
    value: str


class MissingExpectation(BaseModel):
    evaluator: str
    field: str
    productSku: str | None = None
    message: str


class ValidationIssue(BaseModel):
    code: str
    message: str
    path: str | None = None


class GenerationMetadata(BaseModel):
    provider: str
    model: str
    promptVersion: str


class TestGenerationResponse(BaseModel):
    status: GenerationStatus
    draft: Draft | None = None
    appliedDefaults: list[AppliedDefault] = Field(default_factory=list)
    clarifications: list[str] = Field(default_factory=list)
    missingExpectations: list[MissingExpectation] = Field(default_factory=list)
    validationIssues: list[ValidationIssue] = Field(default_factory=list)
    metadata: GenerationMetadata
    # Semantic-cache fields — populated when an existing test_case row was
    # returned instead of calling the LLM.
    cached: bool = False
    matchedTestCaseId: str | None = None
    similarity: float | None = None
    matchType: str | None = None
    # Persistence — id of the test_case row this response was persisted as
    # or linked to via the semantic cache, when available.
    testCaseId: str | None = None
    # Observability — populated only when Langfuse tracing is enabled.
    # generationId identifies the LLM generation attempt; traceId correlates
    # with the Langfuse/W3C trace for this request; observationId is the
    # root observation inside that trace.
    generationId: str | None = None
    traceId: str | None = None
    observationId: str | None = None


# ---------------------------------------------------------------------------
# Persistence models (test_case table)
# ---------------------------------------------------------------------------


class SaveTestCaseRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    test_spec: dict = Field(
        validation_alias=AliasChoices("test_spec", "testSpec"))
    evaluation_profile_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("evaluation_profile_id", "evaluationProfileId"))
    # Many-to-many: a test case can link to multiple evaluation profiles.
    evaluation_profile_ids: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("evaluation_profile_ids", "evaluationProfileIds"))
    status: str = Field(default="DRAFT", max_length=50)
    created_by: str | None = Field(
        default=None, max_length=200,
        validation_alias=AliasChoices("created_by", "createdBy"))
    # Original NL input — used for normalized_input + embedding_vec so the
    # semantic cache can match future generations against this row.
    source_input: str | None = Field(
        default=None, max_length=2000,
        validation_alias=AliasChoices("source_input", "sourceInput"))
    # UI may send the same profile selector shape as /test-generations:
    # {"profile": {"profileId": "...", "version": 1, ...}}
    profile: dict | None = None

    @model_validator(mode="before")
    @classmethod
    def _extract_profile_id(cls, data):
        """Pull profile.profileId into evaluation_profile_id when the UI
        sends the generation-style profile object."""
        if isinstance(data, dict):
            profile = data.get("profile")
            if isinstance(profile, dict) and profile.get("profileId"):
                data.setdefault("evaluation_profile_id", profile["profileId"])
        return data


# ---------------------------------------------------------------------------
# Analysis summarization
# ---------------------------------------------------------------------------


class AnalysisSummaryRequest(BaseModel):
    """Analysis results JSON — kept free-form since finding shapes vary."""
    analysis: dict | None = None
    fromTime: AwareDatetime | None = Field(default=None, description="Inclusive history start; defaults to 7 days before toTime.")
    toTime: AwareDatetime | None = Field(default=None, description="Exclusive history end; defaults to query time. Reuse for stable pagination.")
    operation: Literal["test-generation", "analysis-summary"] | None = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0, le=100_000)

    def time_window(self) -> tuple[datetime, datetime]:
        end = self.toTime or datetime.now(timezone.utc)
        return self.fromTime or end - timedelta(days=7), end

    @model_validator(mode="after")
    def validate_window(self):
        start, end = self.time_window()
        if start >= end or end - start > timedelta(days=90):
            raise ValueError("Time window must be positive and no longer than 90 days.")
        return self


class ObservationMetrics(BaseModel):
    id: str
    name: str
    kind: Literal["span", "generation", "embedding"]
    startedAt: datetime
    latencyMs: float = 0
    status: Literal["success", "error"] = "success"
    errorType: str | None = None
    observationId: str | None = None
    generationId: str | None = None
    model: str | None = None
    provider: str | None = None
    attempt: int | None = None
    inputTokens: int | None = None
    outputTokens: int | None = None
    totalTokens: int | None = None
    inputCostUsd: float | None = None
    outputCostUsd: float | None = None
    totalCostUsd: float | None = None
    providerTimingsMs: dict[str, float] = Field(default_factory=dict)
    outputTokensPerSecond: float | None = None
    metadata: dict = Field(default_factory=dict)


class RequestMetrics(BaseModel):
    requestId: str
    operation: str
    provider: str | None = None
    model: str | None = None
    promptVersion: str | None = None
    traceId: str | None = None
    observationId: str | None = None
    summaryId: str | None = None
    testCaseId: str | None = None
    startedAt: datetime
    completedAt: datetime
    latencyMs: float
    status: str = "success"
    errorType: str | None = None
    source: str | None = None
    cached: bool = False
    fallbackReason: str | None = None
    llmCalls: int = 0
    embeddingCalls: int = 0
    retries: int = 0
    inputTokens: int | None = None
    outputTokens: int | None = None
    totalTokens: int | None = None
    knownTotalTokens: int = 0
    totalCostUsd: float | None = None
    knownCostUsd: float = 0
    costStatus: Literal["estimated", "unknown", "no_model_calls"] = "unknown"
    currency: Literal["USD"] = "USD"
    observations: list[ObservationMetrics] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class MetricsTotals(BaseModel):
    totalRequests: int = 0
    failedRequests: int = 0
    cacheHits: int = 0
    fallbackRequests: int = 0
    llmCalls: int = 0
    embeddingCalls: int = 0
    retries: int = 0
    inputTokens: int | None = 0
    outputTokens: int | None = 0
    totalTokens: int | None = 0
    knownTotalTokens: int = 0
    requestsWithUnknownTokens: int = 0
    totalCostUsd: float | None = 0
    knownCostUsd: float = 0
    requestsWithUnknownCost: int = 0
    currency: Literal["USD"] = "USD"
    averageLatencyMs: float | None = None
    p50LatencyMs: float | None = None
    p95LatencyMs: float | None = None
    errorRate: float = 0
    cacheHitRate: float = 0


class MetricsDashboard(BaseModel):
    available: bool
    unavailableReason: str | None = None
    fromTime: datetime
    toTime: datetime
    operation: str | None = None
    limit: int
    offset: int
    hasMore: bool = False
    totals: MetricsTotals | None = None
    byOperation: dict[str, MetricsTotals] = Field(default_factory=dict)
    traces: list[RequestMetrics] = Field(default_factory=list)


class AnalysisSummaryResponse(BaseModel):
    detail: str | None = None
    errorCode: str | None = None
    summary: str | None = None
    cached: bool | None = None
    summaryId: str | None = None
    traceId: str | None = None
    observationId: str | None = None
    generationId: str | None = None
    matchedReportId: str | None = None
    similarity: float | None = None
    matchType: str | None = None
    metrics: RequestMetrics | None = None
    metricsPersisted: bool | None = None
    dashboard: MetricsDashboard
