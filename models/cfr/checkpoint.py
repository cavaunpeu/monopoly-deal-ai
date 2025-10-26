from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Optional

from google.cloud import storage

from app.settings import CHECKPOINTS_DIRNAME, GCP_POLICY_BUCKET_NAME


GCP_PREFIX = "gs://"


@dataclass
class CheckpointManager:
    """Manages checkpoint saving and loading for CFR experiments.

    Handles both local and remote (GCS) checkpoint storage, providing a unified
    interface for saving and retrieving CFR model states.
    """

    experiment_name: str
    remote: bool
    gcp_policy_bucket_name: str = GCP_POLICY_BUCKET_NAME
    checkpoints_dirname: str = CHECKPOINTS_DIRNAME

    @property
    def experiment_dir(self) -> str:
        """Get the experiment directory path (local or remote)."""
        if self.remote:
            return f"{GCP_PREFIX}{self.gcp_policy_bucket_name}/{self.checkpoints_dirname}/{self.experiment_name}"
        return f"{self.checkpoints_dirname}/{self.experiment_name}"

    @property
    def experiment_dir_empty(self) -> bool:
        """Check if the experiment directory is empty."""
        if self.remote:
            client = storage.Client()
            bucket = client.bucket(self.gcp_policy_bucket_name)
            prefix = f"{self.checkpoints_dirname}/{self.experiment_name}/"
            blobs = list(bucket.list_blobs(prefix=prefix))
            return len(blobs) == 0
        return len(list(Path(self.experiment_dir).glob("*"))) == 0

    def find_latest_checkpoint_by_game_idx(self) -> Optional["CheckpointPath"]:
        """Find the latest checkpoint by game index.

        Returns:
            The latest checkpoint path, or None if no checkpoints exist.
        """
        game_idx_to_path = self._build_dict()
        if not game_idx_to_path:
            return None

        latest_game_idx = max(game_idx_to_path.keys())
        return self.get_checkpoint_path(latest_game_idx)

    def get_checkpoint_path(self, game_idx: int) -> "CheckpointPath":
        """Get a CheckpointPath for a given game index.

        Args:
            game_idx: The game index for the checkpoint.

        Returns:
            CheckpointPath object for the specified game index.
        """
        filename = f"game_idx_{game_idx}.json"
        return CheckpointPath(experiment_dir=self.experiment_dir, filename=filename)

    def save_cfr_state(self, game_idx: int, cfr) -> None:
        """Save CFR state to a JSON file.

        Args:
            game_idx: The game index for this checkpoint.
            cfr: The CFR object to save.
        """
        checkpoint_path = self.get_checkpoint_path(game_idx)

        if self.remote:
            client = storage.Client()
            bucket = client.bucket(self.gcp_policy_bucket_name)
            blob_name = f"{self.checkpoints_dirname}/{self.experiment_name}/{checkpoint_path.filename}"
            blob = bucket.blob(blob_name)
            blob.upload_from_string(json.dumps(cfr.to_json(), indent=4, default=str))
            print(f"✅ CFR state saved to GCS: {checkpoint_path.full_path}", flush=True)
        else:
            os.makedirs(f"{self.checkpoints_dirname}/{self.experiment_name}", exist_ok=True)
            with open(checkpoint_path.full_path, "w") as f:
                json.dump(cfr.to_json(), f, indent=4)
            print(f"✅ CFR state saved to {checkpoint_path.full_path}", flush=True)

    def _build_dict(self) -> dict[int, str]:
        """Build dict mapping game_idx -> full_path.

        Returns:
            Dictionary mapping game indices to checkpoint file paths.
        """
        if self.remote:
            return self._build_remote_dict()
        return self._build_local_dict()

    def _build_remote_dict(self) -> dict[int, str]:
        """Build dict from GCS blobs.

        Returns:
            Dictionary mapping game indices to GCS blob paths.
        """
        try:
            client = storage.Client()
            bucket = client.bucket(self.gcp_policy_bucket_name)
            prefix = f"{self.checkpoints_dirname}/{self.experiment_name}/game_idx_"
            blobs = list(bucket.list_blobs(prefix=prefix))

            return {
                CheckpointPath.extract_game_idx(
                    f"{GCP_PREFIX}{self.gcp_policy_bucket_name}/{blob.name}"
                ): f"{GCP_PREFIX}{self.gcp_policy_bucket_name}/{blob.name}"
                for blob in blobs
                if blob.name.endswith(".json")
            }
        except Exception as e:
            print(f"Error searching for checkpoints in GCS: {e}")
            return {}

    def _build_local_dict(self) -> dict[int, str]:
        """Build dict from local files.

        Returns:
            Dictionary mapping game indices to local file paths.
        """
        checkpoint_dir = Path(self.checkpoints_dirname) / self.experiment_name
        if not checkpoint_dir.exists():
            return {}

        checkpoint_files = list(checkpoint_dir.glob("game_idx_*.json"))
        if not checkpoint_files:
            return {}

        return {CheckpointPath.extract_game_idx(str(fp)): str(fp) for fp in checkpoint_files}


@dataclass(frozen=True)
class CheckpointPath:
    """Represents a checkpoint path with both full path and optional blob name."""

    experiment_dir: str
    filename: str

    @property
    def full_path(self) -> str:
        """Get the full path to the checkpoint file."""
        return f"{self.experiment_dir}/{self.filename}"

    @property
    def game_idx(self) -> int:
        """Extract game index from checkpoint path.

        Returns:
            The game index extracted from the filename.
        """
        return self.extract_game_idx(self.full_path)

    @staticmethod
    def extract_game_idx(path: str) -> int:
        """Extract game index from checkpoint path.

        Args:
            path: The checkpoint file path.

        Returns:
            The game index extracted from the filename.

        Raises:
            ValueError: If the game index cannot be found in the path.
        """
        filename = path.split("/")[-1] if path.startswith(GCP_PREFIX) else Path(path).name
        match = re.search(r"game_idx_(\d+)", filename)
        if not match:
            raise ValueError(f"Game idx not found in checkpoint path: {path}")
        return int(match.group(1))

    def __str__(self) -> str:
        """String representation of the checkpoint path."""
        return self.full_path

    def __repr__(self) -> str:
        """String representation of the checkpoint path."""
        return self.full_path
