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
