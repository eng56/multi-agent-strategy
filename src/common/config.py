from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Every stateful dependency is a managed service."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    confluent_bootstrap_servers: str
    confluent_api_key: str
    confluent_api_secret: str
    confluent_security_protocol: Literal["SASL_SSL"] = "SASL_SSL"
    confluent_sasl_mechanism: Literal["PLAIN"] = "PLAIN"
    upstash_redis_rest_url: AnyHttpUrl
    upstash_redis_rest_token: str
    market_data_provider: Literal["polygon", "massive", "alpha_vantage"] = "polygon"
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: AnyHttpUrl
    gcp_project_id: str
    gcp_region: str
    gcs_bucket_name: str
    artifact_registry_repo: str
    orchestrator_api_token: str
    kms_key_name: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    runtime_topic: str = "agent-runtime"
    polygon_base_url: str = "https://api.polygon.io"
    tavily_base_url: str = "https://api.tavily.com"
    environment: str = "dev"
    default_max_parallel_agents: int = Field(default=10, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
