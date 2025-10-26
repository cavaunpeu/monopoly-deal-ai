from abc import ABC, abstractmethod
import hashlib
import json
import random
import subprocess
from typing import TypeVar

import numpy as np


T = TypeVar("T")


def set_random_seed(seed: int) -> None:
    """Set the random seed for both Python and NumPy random number generators.

    Args:
        seed: The seed value to use for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)


def md5_json(data: dict) -> str:
    """Return a deterministic MD5 hash for a JSON-compatible dict.

    Args:
        data: Dictionary to hash (must be JSON-serializable).

    Returns:
        MD5 hash as a hexadecimal string.
    """
    json_str = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.md5(json_str.encode("utf-8")).hexdigest()


class Serializable(ABC):
    """Abstract base class for objects that can be serialized to/from JSON."""

    @abstractmethod
    def to_json(self) -> dict:
        """Convert the object to a JSON-serializable dictionary.

        Returns:
            Dictionary representation of the object.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_json(cls, data: dict) -> "Serializable":
        """Create an instance from a JSON dictionary.

        Args:
            data: Dictionary containing object data.

        Returns:
            New instance of the class.
        """
        raise NotImplementedError

    @property
    def key(self) -> str:
        """Generate a unique key for this object based on its JSON representation.

        Returns:
            MD5 hash of the object's JSON representation.
        """
        return md5_json(self.to_json())


def get_git_commit() -> str:
    """Get the current git commit hash.

    Returns:
        The current commit hash as a string.
    """
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def has_uncommitted_changes() -> bool:
    """Check if there are uncommitted changes in the git repository.

    Returns:
        True if there are uncommitted changes, False otherwise.
    """
    status = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
    return bool(status)
