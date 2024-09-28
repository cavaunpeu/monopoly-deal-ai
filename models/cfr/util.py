from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional

from google.cloud import storage

from app.settings import CHECKPOINTS_DIRNAME, GCP_POLICY_BUCKET_NAME


@dataclass(frozen=True)
class CheckpointPath:
    """Represents a checkpoint path with both full path and optional blob name."""

    full_path: str
    blob_name: Optional[str] = None

    @property
    def game_idx(self) -> int:
        """Extract game index from checkpoint path."""
        return self.extract_game_idx(self.full_path)

    @staticmethod
    def extract_game_idx(path: str) -> int:
        """Extract game index from checkpoint path."""
        filename = path.split("/")[-1] if path.startswith("gs://") else Path(path).name
        match = re.search(r"game_idx_(\d+)", filename)
        if not match:
            raise ValueError(f"Game idx not found in checkpoint path: {path}")
        return int(match.group(1))

    @classmethod
    def create(cls, experiment_name: str, game_idx: int, remote: bool) -> "CheckpointPath":
        """Create a CheckpointPath for a given experiment and game index."""
        filename = f"game_idx_{game_idx}.json"
        base_path = f"{CHECKPOINTS_DIRNAME}/{experiment_name}/{filename}"

        if remote:
            full_path = f"gs://{GCP_POLICY_BUCKET_NAME}/{base_path}"
            return cls(full_path=full_path, blob_name=base_path)
        else:
            return cls(full_path=base_path)


class CheckpointFinder:
    """Finds the latest checkpoint by game index."""

    def __init__(self, experiment_name: str, remote: bool):
        self.experiment_name = experiment_name
        self.remote = remote

    def find_latest(self) -> Optional[CheckpointPath]:
        """Find the latest checkpoint by game index."""
        game_idx_to_path = self._build_dict()
        if not game_idx_to_path:
            return None

        latest_game_idx = max(game_idx_to_path.keys())
        return CheckpointPath.create(self.experiment_name, latest_game_idx, self.remote)

    def _build_dict(self) -> dict[int, str]:
        """Build dict mapping game_idx -> full_path."""
        if self.remote:
            return self._build_remote_dict()
        return self._build_local_dict()

    def _build_remote_dict(self) -> dict[int, str]:
        """Build dict from GCS blobs."""
        try:
            client = storage.Client()
            bucket = client.bucket(GCP_POLICY_BUCKET_NAME)
            prefix = f"{CHECKPOINTS_DIRNAME}/{self.experiment_name}/game_idx_"
            blobs = list(bucket.list_blobs(prefix=prefix))

            return {
                CheckpointPath.extract_game_idx(
                    f"gs://{GCP_POLICY_BUCKET_NAME}/{blob.name}"
                ): f"gs://{GCP_POLICY_BUCKET_NAME}/{blob.name}"
                for blob in blobs
                if blob.name.endswith(".json")
            }
        except Exception as e:
            print(f"Error searching for checkpoints in GCS: {e}")
            return {}

    def _build_local_dict(self) -> dict[int, str]:
        """Build dict from local files."""
        checkpoint_dir = Path(CHECKPOINTS_DIRNAME) / self.experiment_name
        if not checkpoint_dir.exists():
            return {}

        checkpoint_files = list(checkpoint_dir.glob("game_idx_*.json"))
        if not checkpoint_files:
            return {}

        return {CheckpointPath.extract_game_idx(str(file_path)): str(file_path) for file_path in checkpoint_files}
