import pytest

from app.config import Settings


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    # langfuse_enabled=False keeps tests hermetic even when a developer's
    # .env enables tracing — no live Langfuse calls in unit tests.
    return Settings(
        max_input_chars=500,
        max_result_retries=2,
        model_timeout_seconds=5.0,
        max_output_tokens=1024,
        langfuse_enabled=False,
    )
