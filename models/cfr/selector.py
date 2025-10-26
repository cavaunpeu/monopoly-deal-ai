from abc import ABC, abstractmethod
from dataclasses import dataclass
import random

import numpy as np

from game.action import AbstractAction, WrappedAction
from game.game import Game
from models.cfr.cfr import Policy, RuntimePolicyManager


class BaseActionSelector(ABC):
    """Abstract base class for action selectors in CFR evaluation."""

    @abstractmethod
    def select(
        self,
        actions: list[WrappedAction],
        game: Game,
    ) -> WrappedAction:
        """Select an action from the available actions.

        Args:
            actions: List of available actions to choose from.
            game: Current game state.

        Returns:
            Selected action.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the selector's internal state."""
        pass

    def info(self, actions: list[WrappedAction], game: Game) -> dict:
        """Return selector information about the model's decision-making process.

        Returns:
            dict: A dictionary containing model-specific selector information.
                 Default implementation returns an empty dict.
        """
        return {}


class RandomSelector(BaseActionSelector):
    """Action selector that randomly chooses from available actions."""

    def select(
        self,
        actions: list[WrappedAction],
        game: Game,
    ) -> WrappedAction:
        """Randomly select a valid action.

        Args:
            actions: List of available actions to choose from.
            game: Current game state (unused).

        Returns:
            Randomly selected action.
        """
        return random.choice(actions)

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

    def select(self, actions: list[WrappedAction], game: Game) -> WrappedAction:
        """Select an action based on risk-aware heuristics.

        Args:
            actions: List of available actions to choose from.
            game: Current game state (unused).

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
        return Policy(actions=actions, probs=list(probs)).sample()

    def __repr__(self) -> str:
        """String representation of the selector."""
        return f"RiskPreferenceSampler(aggressiveness={self.aggressiveness}, temperature={self.temperature})"


@dataclass
class CFRActionSelector(BaseActionSelector):
    """Action selector that uses CFR-trained policies for action selection."""

    policy_manager: RuntimePolicyManager

    def select(
        self,
        actions: list[WrappedAction],
        game: Game,
    ) -> WrappedAction:
        """Select action using the CFR policy's argmax strategy.

        Args:
            actions: List of available actions (unused, policy determines action).
            game: Current game state.

        Returns:
            Action with highest probability according to CFR policy.
        """
        return self.policy_manager.get(game.state).argmax()

    def info(self, actions: list[WrappedAction], game: Game) -> dict:
        """Return CFR-specific selector information.

        Args:
            actions: List of available actions (unused).
            game: Current game state.

        Returns:
            Dictionary containing policy information, state key, and update count.
        """
        # Get the current policy
        policy = self.policy_manager.get(game.state)

        # Get the game state key
        state_key = game.state.key

        # Get update counts from the policy manager
        update_count = self.policy_manager.update_count.get(state_key, 0)

        # Convert policy to human-readable format
        policy_dict = policy.to_human_readable()

        return {"policy": policy_dict, "state_key": state_key, "update_count": update_count, "model_type": "CFR"}
