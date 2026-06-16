from dataclasses import dataclass

from src.common.budget import PersistentBudget
from src.common.config import Settings, get_settings
from src.integrations.artifacts import GCSArtifactStore
from src.integrations.credentials import EphemeralCredentialStore
from src.integrations.events import ConfluentEventBus
from src.integrations.llm import LangfuseRecorder, OpenRouterLLM
from src.integrations.tools import ResearchTools
from src.integrations.upstash import UpstashBlackboard


@dataclass
class Runtime:
    settings: Settings
    blackboard: UpstashBlackboard
    events: ConfluentEventBus
    artifacts: GCSArtifactStore
    credentials: EphemeralCredentialStore
    llm: OpenRouterLLM
    tools: ResearchTools


def build_runtime() -> Runtime:
    settings = get_settings()
    blackboard = UpstashBlackboard(str(settings.upstash_redis_rest_url), settings.upstash_redis_rest_token)
    budget = PersistentBudget(blackboard)
    credential_store = EphemeralCredentialStore(blackboard, settings.kms_key_name)
    recorder = LangfuseRecorder(str(settings.langfuse_host), settings.langfuse_public_key, settings.langfuse_secret_key)
    return Runtime(
        settings=settings,
        blackboard=blackboard,
        events=ConfluentEventBus(settings.confluent_bootstrap_servers, settings.confluent_api_key, settings.confluent_api_secret),
        artifacts=GCSArtifactStore(settings.gcs_bucket_name),
        credentials=credential_store,
        llm=OpenRouterLLM(settings.openrouter_base_url, budget, credential_store, recorder),
        tools=ResearchTools(budget, credential_store, recorder, settings.tavily_base_url, settings.polygon_base_url),
    )
