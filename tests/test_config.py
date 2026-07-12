import pytest
from pydantic import ValidationError
from src.common.config import Settings


def required_settings() -> dict[str, str]:
    return {
        "confluent_bootstrap_servers": "broker:9092",
        "confluent_api_key": "key",
        "confluent_api_secret": "secret",
        "upstash_redis_rest_url": "https://redis.example.com",
        "upstash_redis_rest_token": "token",
        "openrouter_api_key": "openrouter",
        "tavily_api_key": "tavily",
        "market_data_api_key": "market",
        "langfuse_public_key": "public",
        "langfuse_secret_key": "secret",
        "gcp_project_id": "project",
        "gcp_region": "us-central1",
        "gcs_bucket_name": "bucket",
        "artifact_registry_repo": "repo",
        "orchestrator_api_token": "token",
    }


def test_langfuse_base_url_is_accepted_as_legacy_alias() -> None:
    settings = Settings(**required_settings(), LANGFUSE_BASE_URL="https://cloud.langfuse.com")
    assert str(settings.langfuse_host) == "https://cloud.langfuse.com/"


def test_market_data_provider_is_massive_only() -> None:
    with pytest.raises(ValidationError, match="market_data_provider"):
        Settings(**required_settings(), LANGFUSE_BASE_URL="https://cloud.langfuse.com", market_data_provider="polygon")


def test_polygon_base_url_is_accepted_as_legacy_market_data_alias() -> None:
    settings = Settings(
        **required_settings(),
        LANGFUSE_BASE_URL="https://cloud.langfuse.com",
        POLYGON_BASE_URL="https://api.polygon.io",
    )
    assert settings.market_data_base_url == "https://api.polygon.io"


def test_principal_policy_mode_defaults_to_shadow() -> None:
    settings = Settings(**required_settings(), LANGFUSE_BASE_URL="https://cloud.langfuse.com")

    assert settings.principal_policy_mode == "shadow"


def test_evidence_max_sources_is_bounded() -> None:
    settings = Settings(
        **required_settings(),
        LANGFUSE_BASE_URL="https://cloud.langfuse.com",
        evidence_max_sources=7,
    )
    assert settings.evidence_max_sources == 7

    with pytest.raises(ValidationError, match="evidence_max_sources"):
        Settings(
            **required_settings(),
            LANGFUSE_BASE_URL="https://cloud.langfuse.com",
            evidence_max_sources=26,
        )
