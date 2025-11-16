from unittest.mock import Mock, patch

import pytest

from models.checkpoint import CheckpointManager, CheckpointPath


class TestCheckpointPath:
    """Test the CheckpointPath dataclass."""

    def test_remote_path(self):
        """Test creating a remote checkpoint path."""
        path = CheckpointPath(
            experiment_dir="gs://monopoly-deal-agent/checkpoints/test-experiment", filename="game_idx_123.json"
        )

        assert path.full_path == "gs://monopoly-deal-agent/checkpoints/test-experiment/game_idx_123.json"
        assert path.game_idx == 123

    def test_local_path(self):
        """Test creating a local checkpoint path."""
        path = CheckpointPath(experiment_dir="checkpoints/test-experiment", filename="game_idx_456.json")

        assert path.full_path == "checkpoints/test-experiment/game_idx_456.json"
        assert path.game_idx == 456

    def test_game_idx_property_invalid_path(self):
        """Test that invalid paths raise ValueError."""
        path = CheckpointPath(experiment_dir="invalid/path", filename="file.json")

        with pytest.raises(ValueError, match="Game idx not found in checkpoint path"):
            _ = path.game_idx

    def test_frozen_dataclass(self):
        """Test that CheckpointPath is immutable."""
        path = CheckpointPath(experiment_dir="checkpoints/test-experiment", filename="game_idx_123.json")

        # Verify immutability by attempting to modify attributes
        for attr_name in ("experiment_dir", "filename"):
            with pytest.raises(AttributeError):
                setattr(path, attr_name, f"new_{attr_name}")

    def test_get_checkpoint_path(self):
        """Test CheckpointManager.get_checkpoint_path method."""
        manager = CheckpointManager("test-experiment", remote=False)
        path = manager.get_checkpoint_path(123)

        assert path.experiment_dir == "checkpoints/test-experiment"
        assert path.filename == "game_idx_123.json"
        assert path.game_idx == 123


class TestCheckpointManager:
    """Test the CheckpointManager class."""

    def test_init(self):
        """Test CheckpointManager initialization."""
        manager = CheckpointManager("test-experiment", remote=True)
        assert manager.experiment_name == "test-experiment"
        assert manager.remote is True

    @patch("models.checkpoint.storage.Client")
    def test_find_latest_remote_success(self, mock_client):
        """Test finding latest checkpoint in remote storage."""
        # Mock GCS client and blobs
        mock_bucket = Mock()
        mock_client.return_value.bucket.return_value = mock_bucket

        # Create mock blobs
        mock_blob1 = Mock()
        mock_blob1.name = "checkpoints/test-experiment/game_idx_100.json"
        mock_blob2 = Mock()
        mock_blob2.name = "checkpoints/test-experiment/game_idx_200.json"
        mock_blob3 = Mock()
        mock_blob3.name = "checkpoints/test-experiment/game_idx_150.json"

        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2, mock_blob3]

        manager = CheckpointManager("test-experiment", remote=True)
        result = manager.find_latest_checkpoint_by_game_idx()

        assert result is not None
        assert result.game_idx == 200
        assert result.full_path == "gs://monopoly-deal-agent/checkpoints/test-experiment/game_idx_200.json"

    @patch("models.checkpoint.storage.Client")
    def test_find_latest_remote_no_checkpoints(self, mock_client):
        """Test finding latest checkpoint when none exist."""
        mock_bucket = Mock()
        mock_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.list_blobs.return_value = []

        manager = CheckpointManager("test-experiment", remote=True)
        result = manager.find_latest_checkpoint_by_game_idx()

        assert result is None

    @patch("models.checkpoint.storage.Client")
    def test_find_latest_remote_gcs_error(self, mock_client):
        """Test handling GCS errors gracefully."""
        mock_client.side_effect = Exception("GCS connection failed")

        manager = CheckpointManager("test-experiment", remote=True)
        result = manager.find_latest_checkpoint_by_game_idx()

        assert result is None

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.glob")
    def test_find_latest_local_success(self, mock_glob, mock_exists):
        """Test finding latest checkpoint in local storage."""
        mock_exists.return_value = True

        # Create mock file paths
        mock_file1 = Mock()
        mock_file1.stem = "game_idx_100"
        mock_file1.__str__ = Mock(return_value="checkpoints/test-experiment/game_idx_100.json")

        mock_file2 = Mock()
        mock_file2.stem = "game_idx_200"
        mock_file2.__str__ = Mock(return_value="checkpoints/test-experiment/game_idx_200.json")

        mock_file3 = Mock()
        mock_file3.stem = "game_idx_150"
        mock_file3.__str__ = Mock(return_value="checkpoints/test-experiment/game_idx_150.json")

        mock_glob.return_value = [mock_file1, mock_file2, mock_file3]

        manager = CheckpointManager("test-experiment", remote=False)
        result = manager.find_latest_checkpoint_by_game_idx()

        assert result is not None
        assert result.game_idx == 200
        assert result.full_path == "checkpoints/test-experiment/game_idx_200.json"

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.glob")
    def test_find_latest_local_no_results(self, mock_glob, mock_exists):
        """Test finding latest checkpoint when directory doesn't exist or no files exist."""
        # Test case 1: Directory doesn't exist
        mock_exists.return_value = False
        manager = CheckpointManager("test-experiment", remote=False)
        result = manager.find_latest_checkpoint_by_game_idx()
        assert result is None

        # Test case 2: Directory exists but no files
        mock_exists.return_value = True
        mock_glob.return_value = []
        result = manager.find_latest_checkpoint_by_game_idx()
        assert result is None

    @patch("models.checkpoint.storage.Client")
    def test_experiment_dir_empty_remote(self, mock_client):
        """Test experiment_dir_empty property for remote storage."""
        mock_bucket = Mock()
        mock_client.return_value.bucket.return_value = mock_bucket

        # Test empty directory
        mock_bucket.list_blobs.return_value = []
        manager = CheckpointManager("test-experiment", remote=True)
        assert manager.experiment_dir_empty is True

        # Test non-empty directory
        mock_blob = Mock()
        mock_blob.name = "checkpoints/test-experiment/game_idx_1.json"
        mock_bucket.list_blobs.return_value = [mock_blob]
        assert manager.experiment_dir_empty is False

    @patch("pathlib.Path.glob")
    def test_experiment_dir_empty_local(self, mock_glob):
        """Test experiment_dir_empty property for local storage."""
        # Test empty directory
        mock_glob.return_value = []
        manager = CheckpointManager("test-experiment", remote=False)
        assert manager.experiment_dir_empty is True

        # Test non-empty directory
        mock_file = Mock()
        mock_glob.return_value = [mock_file]
        assert manager.experiment_dir_empty is False

    def test_extract_game_idx_static_method(self):
        """Test the static extract_game_idx method."""
        # Test GCS path
        gcs_path = "gs://bucket/checkpoints/experiment/game_idx_123.json"
        assert CheckpointPath.extract_game_idx(gcs_path) == 123

        # Test local path
        local_path = "checkpoints/experiment/game_idx_456.json"
        assert CheckpointPath.extract_game_idx(local_path) == 456

        # Test invalid path
        with pytest.raises(ValueError, match="Game idx not found in checkpoint path"):
            CheckpointPath.extract_game_idx("invalid/path.json")
