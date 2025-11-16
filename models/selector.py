from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import random
from typing import Any, Optional

import numpy as np

from game.action import AbstractAction, WrappedAction
from game.state import GameState
from models.types import Policy


@dataclass(frozen=True)
class ActionSelectorInfo:
    """Information about the action selector."""

    model_type: Optional[str]
    state_key: Optional[str]
    state_update_count: Optional[int]
    human_readable_policy: Optional[dict[str, float]]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class BaseActionSelector(ABC):
    """Abstract base class for action selectors in CFR evaluation."""

    @abstractmethod
    def select(
        self,
        actions: list[WrappedAction],
        state: GameState,
        deterministic: bool = False,
    ) -> WrappedAction:
        """Select an action from the available actions.

        Args:
            actions: List of available actions to choose from.
            state: Current game state.

        Returns:
            Selected action.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the selector's internal state."""
        pass

    def info(self, actions: list[WrappedAction], state: GameState) -> ActionSelectorInfo:
        """Return selector information about the model's decision-making process.

        Returns:
            dict: A dictionary containing model-specific selector information.
                 Default implementation returns an empty dict.
        """
        return ActionSelectorInfo(
            model_type=None,
            state_key=state.key,
            state_update_count=None,
            human_readable_policy=None,
        )


class RandomSelector(BaseActionSelector):
    """Action selector that randomly chooses from available actions."""

    def select(
        self,
        actions: list[WrappedAction],
        state: GameState,
        deterministic: bool = False,
    ) -> WrappedAction:
        """Randomly select a valid action.

        Args:
            actions: List of available actions to choose from.
            state: Current game state.
            deterministic: Whether to select a deterministic action.

        Returns:
            Randomly selected action.
        """
        # NB: This selector doesn't have a deterministic option, so we just return a random action.
        return random.choice(actions)

    @classmethod
    def from_model(cls, **kwargs: Any) -> "RandomSelector":
        return cls()

    def __repr__(self) -> str:
        """String representation of the selector."""
        return "RandomSelector()"


class RiskAwareSelector(BaseActionSelector):
    """Action selector that uses risk-aware heuristics based on aggressiveness.

    This selector balances between property-focused and cash-focused strategies
    based on a configurable aggressiveness parameter.
    """

    MIN_AGGRESSIVENESS = 0
    MAX_AGGRESSIVENESS = 1
    MEAN_AGGRESSIVENESS = (MAX_AGGRESSIVENESS + MIN_AGGRESSIVENESS) / 2

    def __init__(
        self,
        aggressiveness: float = MEAN_AGGRESSIVENESS,
        temperature: float = 2,
    ) -> None:
        """Initialize the risk-aware selector.

        Args:
            aggressiveness: Controls preference for property vs. cash cards.
                           Must be in the range [0, 1].
            temperature: Adjusts the sensitivity of the aggressiveness effect.
                        Higher values make aggressiveness impact stronger.

        Raises:
            ValueError: If aggressiveness is not in the valid range.
        """
        if not (0 <= aggressiveness <= 1):
            raise ValueError("Aggressiveness must be between 0 and 1.")
        self.aggressiveness = aggressiveness
        self.temperature = temperature

    def select(self, actions: list[WrappedAction], state: GameState, deterministic: bool = False) -> WrappedAction:
        """Select an action based on risk-aware heuristics.

        Args:
            actions: List of available actions to choose from.
            state: Current game state.
            deterministic: Whether to select a deterministic action.

        Returns:
            Selected action based on aggressiveness and action type.
        """
        action_scores = []

        for wrapped in actions:
            action = wrapped.abstract_action
            match action:
                case (
                    AbstractAction.COMPLETE_PROPERTY_SET
                    | AbstractAction.ADD_TO_PROPERTY_SET
                    | AbstractAction.START_NEW_PROPERTY_SET
                ):
                    # Exponential weighting favoring cards that would complete a property based on aggressiveness
                    score = np.exp(self.aggressiveness * self.temperature)

                case AbstractAction.CASH:
                    # Exponential weighting favoring cash cards based on 1 - aggressiveness
                    score = np.exp((1 - self.aggressiveness) * self.temperature)

                case (
                    AbstractAction.ATTEMPT_COLLECT_RENT
                    | AbstractAction.JUST_SAY_NO
                    | AbstractAction.GIVE_OPPONENT_PROPERTY
                    | AbstractAction.GIVE_OPPONENT_CASH
                ):
                    # Constant score for rent, and response abstract actions
                    score = self.MEAN_AGGRESSIVENESS * self.temperature

                case _ if not wrapped.action.plays_card or action == AbstractAction.OTHER:
                    # Set a score of 1 if this is the only action available
                    score = len(actions) == 1

                case _:
                    raise ValueError(f"Unknown abstract action type: {action}")

            action_scores.append(score)

        # Add a small constant to avoid division by zero
        action_scores = [score + 1e-6 for score in action_scores]

        # Convert scores to probabilities and sample an action based on the distribution
        probs = np.array(action_scores) / sum(action_scores)
        if deterministic:
            return Policy(actions=actions, probs=list(probs)).argmax()
        else:
            return Policy(actions=actions, probs=list(probs)).sample()

    @classmethod
    def from_model(cls, **kwargs: Any) -> "RiskAwareSelector":
        return cls()

    def __repr__(self) -> str:
        """String representation of the selector."""
        return f"RiskPreferenceSampler(aggressiveness={self.aggressiveness}, temperature={self.temperature})"
