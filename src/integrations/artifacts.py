import hashlib
import json
from typing import Any
from uuid import UUID, uuid4

from google.cloud import storage

from src.common.models import ArtifactPointer


class GCSArtifactStore:
    """Stores large raw tool outputs in managed Google Cloud Storage."""

    def __init__(self, bucket_name: str) -> None:
        self.bucket = storage.Client().bucket(bucket_name)

    def put_json(self, run_id: UUID, category: str, value: Any) -> ArtifactPointer:
        content = json.dumps(value, separators=(",", ":")).encode()
        name = f"runs/{run_id}/{category}/{uuid4()}.json"
        self.bucket.blob(name).upload_from_string(content, content_type="application/json")
        return ArtifactPointer(
            uri=f"gs://{self.bucket.name}/{name}",
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
