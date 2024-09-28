from abc import ABC, abstractmethod
import hashlib
import json
import random
import subprocess
from typing import TypeVar

import numpy as np


T = TypeVar("T")


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def md5_json(data: dict) -> str:
    """Return a deterministic MD5 hash for a JSON-compatible dict."""
    json_str = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.md5(json_str.encode("utf-8")).hexdigest()


class Serializable(ABC):
    @abstractmethod
    def to_json(self):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_json(cls, data: dict):
        raise NotImplementedError

    @property
    def key(self):
        return md5_json(self.to_json())


def get_git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def has_uncommitted_changes():
    status = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
    return bool(status)
