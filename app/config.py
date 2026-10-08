from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration — override any field via GATIQA_GEN_* env vars or .env file."""

    model_config = SettingsConfigDict(
        env_prefix="GATIQA_GEN_", env_file=".env", extra="ignore",
        populate_by_name=True,
    )

    # --- LLM provider (pydantic-ai "provider:model" format) ----------------
    # Default: Ollama running locally with qwen3:8b.
    # To use OpenAI instead: GATIQA_GEN_LLM_PROVIDER=openai GATIQA_GEN_LLM_MODEL=gpt-4o-mini
    llm_provider: str = "openai"
    llm_model: str = "qwen3:8b"
    # Ollama exposes an OpenAI-compatible API on this base URL.
    ollama_base_url: str = "http://localhost:11434/v1"

    # --- Model budgets -----------------------------------------------------
    model_timeout_seconds: float = 120.0
    max_output_tokens: int = 2048

    # --- Input safety ------------------------------------------------------
    max_input_chars: int = 4000

    # --- Bounded retries for structured-output validation failures ---------
    max_result_retries: int = 2

    prompt_version: str = "2026-09.1"

    # --- Postgres ----------------------------------------------------------
    database_url: str = "postgresql://gatiqa:gatiqa@localhost:5432/gatiqa"
    db_pool_min_size: int = 1
    db_pool_max_size: int = 5

    # --- Semantic cache (dedupe before LLM) --------------------------------
    # Embedding model served by Ollama (nomic-embed-text, 768 dims).
    embedding_model: str = "nomic-embed-text"
    # Cosine similarity >= this means "same test case" — return cached spec.
    similarity_threshold: float = 0.92
    # Max rows scanned per lookup (Phase 1: brute-force cosine is fine).
    cache_scan_limit: int = 5000

    # --- Server ------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8090

    # --- Langfuse observability --------------------------------------------
    # Disabled by default. Standard LANGFUSE_* env names are supported
    # directly; GATIQA_GEN_LANGFUSE_* variants also work for consistency
    # with the rest of this service's config. Keys are never logged.
    langfuse_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("LANGFUSE_ENABLED", "GATIQA_GEN_LANGFUSE_ENABLED"),
    )
    langfuse_base_url: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias=AliasChoices(
            "LANGFUSE_BASE_URL", "LANGFUSE_HOST", "GATIQA_GEN_LANGFUSE_BASE_URL"),
    )
    langfuse_public_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGFUSE_PUBLIC_KEY", "GATIQA_GEN_LANGFUSE_PUBLIC_KEY"),
    )
    langfuse_secret_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGFUSE_SECRET_KEY", "GATIQA_GEN_LANGFUSE_SECRET_KEY"),
    )
    langfuse_environment: str = Field(
        default="development",
        validation_alias=AliasChoices("LANGFUSE_ENVIRONMENT", "GATIQA_GEN_LANGFUSE_ENVIRONMENT"),
    )
    # When false, prompt/output content is NOT sent to Langfuse — only
    # metadata, latency, token counts, status and error types are recorded.
    langfuse_capture_io: bool = Field(
        default=True,
        validation_alias=AliasChoices("LANGFUSE_CAPTURE_IO", "GATIQA_GEN_LANGFUSE_CAPTURE_IO"),
    )
    # Optional USD-per-million-token pricing. Local Ollama has no real cost —
    # leave unset to record no cost_details. Set both to report estimated
    # cost on generation observations (e.g. when proxying a paid provider).
    langfuse_input_cost_per_mtok: float | None = Field(
        default=None, ge=0, allow_inf_nan=False,
        validation_alias=AliasChoices(
            "LANGFUSE_INPUT_COST_PER_MTOK", "GATIQA_GEN_LANGFUSE_INPUT_COST_PER_MTOK"),
    )
    langfuse_output_cost_per_mtok: float | None = Field(
        default=None, ge=0, allow_inf_nan=False,
        validation_alias=AliasChoices(
            "LANGFUSE_OUTPUT_COST_PER_MTOK", "GATIQA_GEN_LANGFUSE_OUTPUT_COST_PER_MTOK"),
    )
    langfuse_embedding_cost_per_mtok: float | None = Field(
        default=None, ge=0, allow_inf_nan=False,
        validation_alias=AliasChoices(
            "LANGFUSE_EMBEDDING_COST_PER_MTOK", "GATIQA_GEN_LANGFUSE_EMBEDDING_COST_PER_MTOK"),
    )
    # OTel service.name — standard OTEL_SERVICE_NAME wins, GATIQA_SERVICE_NAME
    # is the deployment-friendly alias.
    otel_service_name: str = Field(
        default="gatiqa-test-generator",
        validation_alias=AliasChoices(
            "OTEL_SERVICE_NAME", "GATIQA_SERVICE_NAME", "GATIQA_GEN_OTEL_SERVICE_NAME"),
    )
