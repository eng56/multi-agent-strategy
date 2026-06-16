import base64
from uuid import UUID

from google.cloud import kms

from src.common.models import UserCredentials
from src.integrations.upstash import UpstashBlackboard


class EphemeralCredentialStore:
    """KMS-encrypted BYOK credentials in Upstash, hard-expiring after six hours."""

    TTL_SECONDS = 6 * 60 * 60

    def __init__(self, blackboard: UpstashBlackboard, kms_key_name: str) -> None:
        self.blackboard = blackboard
        self.kms_key_name = kms_key_name
        self.kms = kms.KeyManagementServiceClient()

    def _encrypt(self, value: str) -> str:
        result = self.kms.encrypt(request={"name": self.kms_key_name, "plaintext": value.encode()})
        return base64.b64encode(result.ciphertext).decode()

    def _decrypt(self, value: str) -> str:
        ciphertext = base64.b64decode(value)
        result = self.kms.decrypt(request={"name": self.kms_key_name, "ciphertext": ciphertext})
        return result.plaintext.decode()

    async def put(self, run_id: UUID, credentials: UserCredentials) -> None:
        encrypted = {name: self._encrypt(value) for name, value in credentials.model_dump().items()}
        await self.blackboard.command("SET", f"run:{run_id}:credentials", __import__("json").dumps(encrypted), "EX", self.TTL_SECONDS)

    async def get_key(self, run_id: UUID, provider: str) -> str:
        encrypted = await self.blackboard.get_json(f"run:{run_id}:credentials")
        field = f"{provider}_api_key"
        if not encrypted or field not in encrypted:
            raise RuntimeError("run credentials expired")
        return self._decrypt(str(encrypted[field]))

    async def delete(self, run_id: UUID) -> None:
        await self.blackboard.command("DEL", f"run:{run_id}:credentials")
