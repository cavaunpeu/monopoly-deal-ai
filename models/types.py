from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from game.action import ABSTRACT_ACTION_TO_IDX, AbstractAction, BaseActionResolver, WrappedAction
from game.config import GameConfig
from game.game import PlayerSpec
from game.state import BaseStateAbstraction, GameState
from game.util import Serializable


if TYPE_CHECKING:
    from models.selector import BaseActionSelector


@dataclass(frozen=True)
class Policy(Serializable, Mapping[WrappedAction, float]):
    """A policy representing a probability distribution over actions.

    This class represents a policy as a mapping from actions to probabilities,
    with methods for sampling, serialization, and policy operations.
    """

    actions: list[WrappedAction]
    probs: list[float]

    def __post_init__(self) -> None:
        """Validate policy data after initialization."""
        if len(self.actions) == 0:
            raise ValueError("Actions must be provided for a policy.")
        if len(self.actions) != len(self.probs):
            raise ValueError("Actions and probabilities must have the same length.")

    @staticmethod
    def _sample(actions: list[WrappedAction], probs: list[float]) -> WrappedAction:
        """Sample a WrappedAction from the policy.

        Args:
            actions: List of available actions.
            probs: List of probabilities corresponding to actions.

        Returns:
            A randomly sampled action according to the probability distribution.
        """
        return np.random.choice(np.array(actions, dtype=object), p=probs)

    def sample(self) -> WrappedAction:
        """Sample a WrappedAction from the policy.

        Returns:
            A randomly sampled action according to the policy's probability distribution.
        """
        return self._sample(self.actions, self.probs)

    def argmax(self) -> WrappedAction:
        """Return the action with the highest probability.

        For uniform policies, prefers card actions over non-card actions.

        Returns:
            The action with the highest probability.
        """
        # If policy is uniform, prefer card actions over non-card actions
        if len(set(self.probs)) == 1:
            card_actions = [action for action in self.actions if action.action.plays_card]
            if card_actions:
                # Choose a card randomly
                return self._sample(card_actions, [1 / len(card_actions)] * len(card_actions))

        max_idx = np.argmax(self.probs)
        return self.actions[max_idx]

    def to_json(self) -> dict:
        """Convert policy to JSON-serializable dictionary.

        Returns:
            Dictionary containing encoded actions and probabilities.
        """
        return {
            "actions": [action.encode() for action in self.actions],
            "probs": self.probs,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Policy":
        """Create policy from JSON dictionary.

        Args:
            data: Dictionary containing encoded actions and probabilities.

        Returns:
            New Policy instance.
        """
        return cls(
            actions=[WrappedAction.decode(action) for action in data["actions"]],
            probs=data["probs"],
        )

    @classmethod
    def build_uniform_policy(cls, state: GameState) -> "Policy":
        """Build a uniform policy for the given game state.

        Args:
            state: The game state to build a policy for.

        Returns:
            Uniform policy over all available actions.
        """
        wrapped = state.get_player_actions()
        return Policy(actions=wrapped, probs=[1 / len(wrapped)] * len(wrapped))

    def to_human_readable(self) -> dict[str, float]:
        """Convert policy to human-readable format.

        Returns:
            Dictionary mapping action names to rounded probabilities.
            If multiple wrapped actions share the same abstract action, their
            probabilities are summed. Probabilities are guaranteed to sum to 1.0.
        """
        # Aggregate probabilities by abstract action name (handles duplicates from dedupe=False)
        probs = {}
        for action, prob in zip(self.actions, self.probs):
            probs[action.abstract_action.name] = probs.get(action.abstract_action.name, 0.0) + prob

        # Normalize, round, and ensure sum is 1.0
        total = sum(probs.values()) or len(probs)
        probs = {name: float(np.round(p / total, 3)) for name, p in probs.items()}

        # Adjust largest value to account for rounding errors
        if (diff := 1.0 - sum(probs.values())) and abs(diff) > 1e-6:
            largest_key = max(probs.keys(), key=lambda k: probs[k])
            probs[largest_key] = float(np.round(probs[largest_key] + diff, 3))

        return {name: prob for name, prob in probs.items() if prob > 0}

    def encode_probs(self) -> list[float]:
        """Encode probabilities as a fixed-length vector for all abstract actions.

        Returns:
            List of probabilities indexed by abstract action enum values.
        """
        d = {a.abstract_action: p for a, p in zip(self.actions, self.probs)}
        return [float(d.get(a, 0)) for a in AbstractAction]

    @classmethod
    def from_encoded_probs(cls, encoded_probs: list[float], state: GameState) -> "Policy":
        """Create policy from encoded probabilities and game state.

        Args:
            encoded_probs: List of probabilities indexed by abstract action enum.
            state: Game state to get available actions from.

        Returns:
            New Policy instance.
        """
        actions, probs = state.get_player_actions(), []
        for a in actions:
            idx = ABSTRACT_ACTION_TO_IDX[a.abstract_action]
            probs.append(encoded_probs[idx])
        return cls(actions=actions, probs=probs)

    def __getitem__(self, key: WrappedAction) -> float:
        """Get probability for a specific action.

        Args:
            key: The action to get probability for.

        Returns:
            Probability of the action.

        Raises:
            KeyError: If the action is not in this policy.
        """
        if key not in self.actions:
            raise KeyError(key)
        return self.probs[self.actions.index(key)]

    def __iter__(self):
        """Iterate over actions in the policy."""
        return iter(self.actions)

    def __len__(self) -> int:
        """Get number of actions in the policy."""
        return len(self.actions)


class SelectorModel(Protocol):
    """Protocol for all models that create action selectors.

    All models must implement this interface to ensure uniform access
    to required fields and methods, eliminating the need for isinstance()
    checks throughout the codebase.
    """

    # Required fields (must be stored in checkpoint)
    target_player_index: int
    game_config: GameConfig
    abstraction_cls: type[BaseStateAbstraction]
    resolver_cls: type[BaseActionResolver]

    # Required methods
    def create_selector(self) -> "BaseActionSelector":
        """Create an action selector for this model.

        Returns:
            BaseActionSelector instance configured for this model.
        """
        ...

    def get_player_spec(self) -> PlayerSpec:
        """Get the PlayerSpec for this model's target player.

        Returns:
            PlayerSpec with the model's abstraction and resolver classes.
        """
        ...


@dataclass(frozen=True)
class ModelAction:
    """One model action."""

    state_key: str
    state_vector_encoding: np.ndarray
    turn_idx: int
    streak_idx: int
    streaking_player_idx: int
    valid_actions: list[AbstractAction]
    action: AbstractAction
    reward: float


@dataclass(frozen=True)
class Trajectory:
    """One game trajectory. Model actions correspond to target player actions.

    Similarly, reward is defined for the target player."""

    model_actions: list[ModelAction]
    reward: float
