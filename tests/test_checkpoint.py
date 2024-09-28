from unittest.mock import Mock, patch

import pytest

from models.cfr.util import CheckpointFinder, CheckpointPath


class TestCheckpointPath:
    """Test the CheckpointPath dataclass."""

    def test_create_remote(self):
        """Test creating a remote checkpoint path."""
        path = CheckpointPath.create("test-experiment", 123, remote=True)

        assert path.full_path == "gs://monopoly-deal-agent/checkpoints/test-experiment/game_idx_123.json"
        assert path.blob_name == "checkpoints/test-experiment/game_idx_123.json"

    def test_create_local(self):
        """Test creating a local checkpoint path."""
        path = CheckpointPath.create("test-experiment", 456, remote=False)

        assert path.full_path == "checkpoints/test-experiment/game_idx_456.json"
        assert path.blob_name is None

    def test_game_idx_property_remote(self):
        """Test extracting game index from remote path."""
        path = CheckpointPath.create("test-experiment", 789, remote=True)
        assert path.game_idx == 789

    def test_game_idx_property_local(self):
        """Test extracting game index from local path."""
        path = CheckpointPath.create("test-experiment", 101, remote=False)
        assert path.game_idx == 101

    def test_game_idx_property_invalid_path(self):
        """Test that invalid paths raise ValueError."""
        path = CheckpointPath(full_path="invalid/path/file.json", blob_name=None)

        with pytest.raises(ValueError, match="Game idx not found in checkpoint path"):
            _ = path.game_idx

    def test_frozen_dataclass(self):
        """Test that CheckpointPath is immutable."""
        path = CheckpointPath.create("test-experiment", 123, remote=True)

        # Verify immutability by attempting to modify attributes
        for attr_name in ("full_path", "blob_name"):
            with pytest.raises(AttributeError):
                setattr(path, attr_name, f"new_{attr_name}")


class TestCheckpointFinder:
    """Test the CheckpointFinder class."""

    def test_init(self):
        """Test CheckpointFinder initialization."""
        finder = CheckpointFinder("test-experiment", remote=True)
        assert finder.experiment_name == "test-experiment"
        assert finder.remote is True

    @patch("models.cfr.util.storage.Client")
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

        finder = CheckpointFinder("test-experiment", remote=True)
        result = finder.find_latest()

        assert result is not None
        assert result.game_idx == 200
        assert result.full_path == "gs://monopoly-deal-agent/checkpoints/test-experiment/game_idx_200.json"

    @patch("models.cfr.util.storage.Client")
    def test_find_latest_remote_no_checkpoints(self, mock_client):
        """Test finding latest checkpoint when none exist."""
        mock_bucket = Mock()
        mock_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.list_blobs.return_value = []

        finder = CheckpointFinder("test-experiment", remote=True)
        result = finder.find_latest()

        assert result is None

    @patch("models.cfr.util.storage.Client")
    def test_find_latest_remote_gcs_error(self, mock_client):
        """Test handling GCS errors gracefully."""
        mock_client.side_effect = Exception("GCS connection failed")

        finder = CheckpointFinder("test-experiment", remote=True)
        result = finder.find_latest()

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

        finder = CheckpointFinder("test-experiment", remote=False)
        result = finder.find_latest()

        assert result is not None
        assert result.game_idx == 200
        assert result.full_path == "checkpoints/test-experiment/game_idx_200.json"

    @patch("pathlib.Path.exists")
    def test_find_latest_local_no_directory(self, mock_exists):
        """Test finding latest checkpoint when directory doesn't exist."""
        mock_exists.return_value = False

        finder = CheckpointFinder("test-experiment", remote=False)
        result = finder.find_latest()

        assert result is None

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.glob")
    def test_find_latest_local_no_files(self, mock_glob, mock_exists):
        """Test finding latest checkpoint when no files exist."""
        mock_exists.return_value = True
        mock_glob.return_value = []

        finder = CheckpointFinder("test-experiment", remote=False)
        result = finder.find_latest()

        assert result is None

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
