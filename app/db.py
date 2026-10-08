"""Postgres persistence layer — asyncpg pool, test_case inserts, semantic lookup."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import asyncpg

from app.config import Settings
from app.embeddings import normalize_text
from app.models import AnalysisSummaryRequest, MetricsDashboard, MetricsTotals, RequestMetrics

logger = logging.getLogger(__name__)

# Columns added for the semantic cache. Applied idempotently at connect time
# so this works against the existing schema without a migration framework.
# embedding_vec requires the pgvector extension (provided by the
# pgvector/pgvector:pg16 docker image).
_CACHE_COLUMNS = """
CREATE EXTENSION IF NOT EXISTS vector;
ALTER TABLE test_case
    ADD COLUMN IF NOT EXISTS source_input text,
    ADD COLUMN IF NOT EXISTS normalized_input text,
    ADD COLUMN IF NOT EXISTS embedding_vec vector(768)
;
CREATE TABLE IF NOT EXISTS test_case_evaluation_profile (
    test_case_id  UUID NOT NULL REFERENCES test_case(id) ON DELETE CASCADE,
    profile_id    UUID NOT NULL REFERENCES evaluation_profile(id) ON DELETE CASCADE,
    created_at    TIMESTAMP NOT NULL DEFAULT now(),
    PRIMARY KEY (test_case_id, profile_id)
);
CREATE INDEX IF NOT EXISTS idx_tcep_profile
    ON test_case_evaluation_profile(profile_id);
CREATE TABLE IF NOT EXISTS analysis_report (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_input     TEXT NOT NULL,
    normalized_input TEXT,
    embedding_vec    vector(768),
    summary          TEXT NOT NULL,
    created_at       TIMESTAMP NOT NULL DEFAULT now()
);
"""

_METRICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS request_metrics (
    id UUID PRIMARY KEY,
    operation TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    trace_data JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_request_metrics_started_at
    ON request_metrics (started_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_request_metrics_operation_started_at
    ON request_metrics (operation, started_at DESC, id DESC);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_test_case_normalized_input
    ON test_case (normalized_input);
CREATE INDEX IF NOT EXISTS idx_test_case_embedding_vec
    ON test_case USING hnsw (embedding_vec vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_analysis_report_normalized_input
    ON analysis_report (normalized_input);
CREATE INDEX IF NOT EXISTS idx_analysis_report_embedding_vec
    ON analysis_report USING hnsw (embedding_vec vector_cosine_ops)
"""


class TestCaseStore:
    """Persists generated test drafts and finds semantically similar ones."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                dsn=self._settings.database_url,
                min_size=self._settings.db_pool_min_size,
                max_size=self._settings.db_pool_max_size,
            )
            await self._pool.execute(_CACHE_COLUMNS)
            await self._pool.execute(_CREATE_INDEX)
            await self._pool.execute(_METRICS_SCHEMA)
            logger.info("Postgres pool created (%s)", self._settings.database_url)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def available(self) -> bool:
        return self._pool is not None

    # ------------------------------------------------------------------
    # Semantic cache lookup
    # ------------------------------------------------------------------

    async def find_similar(
        self,
        *,
        source_input: str,
        embedding: list[float] | None,
    ) -> dict[str, Any] | None:
        """Return the best-matching test_case row, or None.

        Two passes:
        1. Exact match on normalized text (free, catches trivial rewording).
        2. Cosine similarity over stored embeddings (catches paraphrases).
        """
        if self._pool is None:
            return None

        normalized = normalize_text(source_input)

        # Pass 1 — exact normalized match.
        row = await self._pool.fetchrow(
            """
            SELECT id, name, test_spec, source_input
            FROM test_case
            WHERE normalized_input = $1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            normalized,
        )
        if row is not None:
            result = dict(row)
            result["similarity"] = 1.0
            result["match_type"] = "normalized"
            return result

        if embedding is None:
            return None

        # Pass 2 — pgvector cosine distance (<=>). similarity = 1 - distance.
        # HNSW index accelerates this; the threshold filter keeps only
        # sufficiently similar rows.
        vec_literal = "[" + ",".join(str(x) for x in embedding) + "]"
        row = await self._pool.fetchrow(
            """
            SELECT id, name, test_spec, source_input,
                   1 - (embedding_vec <=> $1::vector) AS similarity
            FROM test_case
            WHERE embedding_vec IS NOT NULL
              AND 1 - (embedding_vec <=> $1::vector) >= $2
            ORDER BY embedding_vec <=> $1::vector
            LIMIT 1
            """,
            vec_literal,
            self._settings.similarity_threshold,
        )
        if row is not None:
            result = dict(row)
            result["similarity"] = round(float(result["similarity"]), 4)
            result["match_type"] = "embedding"
            return result
        return None

    # ------------------------------------------------------------------
    # Inserts
    # ------------------------------------------------------------------

    async def save_test_case(
        self,
        *,
        name: str,
        test_spec: dict[str, Any],
        description: str | None = None,
        evaluation_profile_id: uuid.UUID | None = None,
        evaluation_profile_ids: list[uuid.UUID] | None = None,
        status: str = "DRAFT",
        created_by: str | None = None,
        source_input: str | None = None,
        embedding: list[float] | None = None,
    ) -> dict[str, Any]:
        """Insert a test_case row (+ profile links) and return it.

        evaluation_profile_id (singular) populates the legacy FK column and is
        merged into evaluation_profile_ids for the join table — the join table
        is the source of truth for the many-to-many relation.
        """
        if self._pool is None:
            raise RuntimeError("Database pool is not initialized.")

        profile_ids: list[uuid.UUID] = list(evaluation_profile_ids or [])
        if evaluation_profile_id is not None and evaluation_profile_id not in profile_ids:
            profile_ids.insert(0, evaluation_profile_id)
        # Keep the legacy single-FK column pointing at the first profile.
        primary_profile_id = profile_ids[0] if profile_ids else None

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO test_case
                        (name, description, test_spec, evaluation_profile_id, status,
                         created_by, source_input, normalized_input, embedding_vec)
                    VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7, $8, $9::vector)
                    RETURNING id, name, description, test_spec, evaluation_profile_id,
                              status, created_by, created_at, updated_at
                    """,
                    name,
                    description,
                    json.dumps(test_spec),
                    primary_profile_id,
                    status,
                    created_by,
                    source_input,
                    normalize_text(source_input) if source_input else None,
                    ("[" + ",".join(str(x) for x in embedding) + "]") if embedding else None,
                )
                if profile_ids:
                    await conn.executemany(
                        """
                        INSERT INTO test_case_evaluation_profile (test_case_id, profile_id)
                        VALUES ($1, $2)
                        ON CONFLICT (test_case_id, profile_id) DO NOTHING
                        """,
                        [(row["id"], pid) for pid in profile_ids],
                    )
        result = dict(row)
        result["evaluation_profile_ids"] = [str(pid) for pid in profile_ids]
        return result

    # ------------------------------------------------------------------
    # Analysis report cache
    # ------------------------------------------------------------------

    async def find_similar_report(
        self,
        *,
        source_input: str,
        embedding: list[float] | None,
    ) -> dict[str, Any] | None:
        """Return a cached analysis report matching this input, or None.

        Only exact canonical-JSON matches are returned.  Embedding-based
        similarity is deliberately *not* used here: structured JSON analyses
        that differ only in SKUs, counts or totals embed very closely yet
        require completely different summaries.  The canonical JSON
        (``json.dumps(sort_keys=True)``) already normalises key order, so an
        exact match on the normalised text is the correct deduplication key.
        """
        if self._pool is None:
            return None

        normalized = normalize_text(source_input)
        row = await self._pool.fetchrow(
            """
            SELECT id, summary, source_input
            FROM analysis_report
            WHERE normalized_input = $1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            normalized,
        )
        if row is not None:
            result = dict(row)
            result["similarity"] = 1.0
            result["match_type"] = "normalized"
            return result

        return None

    async def save_analysis_report(
        self,
        *,
        source_input: str,
        summary: str,
        embedding: list[float] | None,
    ) -> dict[str, Any]:
        """Persist an analysis report row and return it."""
        if self._pool is None:
            raise RuntimeError("Database pool is not initialized.")
        row = await self._pool.fetchrow(
            """
            INSERT INTO analysis_report
                (source_input, normalized_input, embedding_vec, summary)
            VALUES ($1, $2, $3::vector, $4)
            RETURNING id, summary, created_at
            """,
            source_input,
            normalize_text(source_input),
            ("[" + ",".join(str(x) for x in embedding) + "]") if embedding else None,
            summary,
        )
        return dict(row)

    async def save_request_metrics(self, metrics: RequestMetrics) -> None:
        if self._pool is None:
            raise RuntimeError("Database pool is not initialized.")
        await self._pool.execute(
            """
            INSERT INTO request_metrics (id, operation, started_at, trace_data)
            VALUES ($1, $2, $3, $4::jsonb)
            ON CONFLICT (id) DO NOTHING
            """,
            uuid.UUID(metrics.requestId), metrics.operation, metrics.startedAt,
            metrics.model_dump_json(),
        )

    async def analysis_dashboard(self, request: AnalysisSummaryRequest) -> MetricsDashboard:
        if self._pool is None:
            raise RuntimeError("Database pool is not initialized.")
        start, end = request.time_window()
        args = (start, end, request.operation)
        async with self._pool.acquire() as conn:
            async with conn.transaction(isolation="repeatable_read", readonly=True):
                aggregates = await conn.fetch(
                    """
                    SELECT operation,
                        count(*) AS "totalRequests",
                        count(*) FILTER (WHERE trace_data->>'errorType' IS NOT NULL
                            OR trace_data->>'status' IN ('error', 'FAILED')) AS "failedRequests",
                        count(*) FILTER (WHERE (trace_data->>'cached')::boolean) AS "cacheHits",
                        count(*) FILTER (WHERE trace_data->>'source' = 'fallback') AS "fallbackRequests",
                        coalesce(sum((trace_data->>'llmCalls')::bigint), 0) AS "llmCalls",
                        coalesce(sum((trace_data->>'embeddingCalls')::bigint), 0) AS "embeddingCalls",
                        coalesce(sum((trace_data->>'retries')::bigint), 0) AS retries,
                        CASE WHEN count(*) FILTER (WHERE trace_data->>'inputTokens' IS NULL) > 0
                            THEN NULL ELSE coalesce(sum((trace_data->>'inputTokens')::bigint), 0)
                            END AS "inputTokens",
                        CASE WHEN count(*) FILTER (WHERE trace_data->>'outputTokens' IS NULL) > 0
                            THEN NULL ELSE coalesce(sum((trace_data->>'outputTokens')::bigint), 0)
                            END AS "outputTokens",
                        CASE WHEN count(*) FILTER (WHERE trace_data->>'totalTokens' IS NULL) > 0
                            THEN NULL ELSE coalesce(sum((trace_data->>'totalTokens')::bigint), 0)
                            END AS "totalTokens",
                        coalesce(sum((trace_data->>'knownTotalTokens')::bigint), 0) AS "knownTotalTokens",
                        count(*) FILTER (WHERE trace_data->>'totalTokens' IS NULL) AS "requestsWithUnknownTokens",
                        CASE WHEN count(*) FILTER (WHERE trace_data->>'totalCostUsd' IS NULL) > 0
                            THEN NULL ELSE coalesce(sum((trace_data->>'totalCostUsd')::double precision), 0)
                            END AS "totalCostUsd",
                        coalesce(sum((trace_data->>'knownCostUsd')::double precision), 0) AS "knownCostUsd",
                        count(*) FILTER (WHERE trace_data->>'totalCostUsd' IS NULL) AS "requestsWithUnknownCost",
                        avg((trace_data->>'latencyMs')::double precision) AS "averageLatencyMs",
                        percentile_cont(0.5) WITHIN GROUP (ORDER BY (trace_data->>'latencyMs')::double precision)
                            AS "p50LatencyMs",
                        percentile_cont(0.95) WITHIN GROUP (ORDER BY (trace_data->>'latencyMs')::double precision)
                            AS "p95LatencyMs"
                    FROM request_metrics
                    WHERE started_at >= $1 AND started_at < $2
                        AND ($3::text IS NULL OR operation = $3)
                    GROUP BY GROUPING SETS ((), (operation))
                    """, *args,
                )
                rows = await conn.fetch(
                    """
                    SELECT trace_data FROM request_metrics
                    WHERE started_at >= $1 AND started_at < $2
                        AND ($3::text IS NULL OR operation = $3)
                    ORDER BY started_at DESC, id DESC LIMIT $4 OFFSET $5
                    """, *args, request.limit, request.offset,
                )
        totals = MetricsTotals()
        by_operation = {}
        for row in aggregates:
            item = MetricsTotals.model_validate(dict(row))
            if item.totalRequests:
                item.errorRate = item.failedRequests / item.totalRequests
                item.cacheHitRate = item.cacheHits / item.totalRequests
            if row["operation"] is None:
                totals = item
            else:
                by_operation[row["operation"]] = item
        return MetricsDashboard(
            available=True, fromTime=start, toTime=end, operation=request.operation,
            limit=request.limit, offset=request.offset,
            hasMore=request.offset + len(rows) < totals.totalRequests,
            totals=totals, byOperation=by_operation,
            traces=[RequestMetrics.model_validate_json(row["trace_data"])
                    if isinstance(row["trace_data"], str)
                    else RequestMetrics.model_validate(row["trace_data"]) for row in rows],
        )

    async def link_profiles(
        self,
        test_case_id: uuid.UUID,
        profile_ids: list[uuid.UUID],
    ) -> None:
        """Attach profiles to an existing test case (idempotent)."""
        if self._pool is None or not profile_ids:
            return
        await self._pool.executemany(
            """
            INSERT INTO test_case_evaluation_profile (test_case_id, profile_id)
            VALUES ($1, $2)
            ON CONFLICT (test_case_id, profile_id) DO NOTHING
            """,
            [(test_case_id, pid) for pid in profile_ids],
        )
