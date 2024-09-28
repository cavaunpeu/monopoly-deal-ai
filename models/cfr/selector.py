from abc import ABC, abstractmethod
from dataclasses import dataclass
import random

import numpy as np

from game.action import AbstractAction, WrappedAction
from game.game import Game
from models.cfr.cfr import Policy, RuntimePolicyManager


class BaseActionSelector(ABC):
    @abstractmethod
    def select(
        self,
        actions: list[WrappedAction],
        game: Game,
    ) -> WrappedAction:
        raise NotImplementedError

    def reset(self):
        pass

    def info(self, actions: list[WrappedAction], game: Game) -> dict:
        """Return selector information about the model's decision-making process.

        Returns:
            dict: A dictionary containing model-specific selector information.
                 Default implementation returns an empty dict.
        """
        return {}


class RandomSelector(BaseActionSelector):
    def select(
        self,
        actions: list[WrappedAction],
        game: Game,
    ):
        """Randomly select a valid action."""
        return random.choice(actions)

    def __repr__(self) -> str:
        return "RandomSelector()"


class RiskAwareSelector(BaseActionSelector):
    MIN_AGGRESSIVENESS = 0
    MAX_AGGRESSIVENESS = 1
    MEAN_AGGRESSIVENESS = (MAX_AGGRESSIVENESS + MIN_AGGRESSIVENESS) / 2

    def __init__(
        self,
        aggressiveness=MEAN_AGGRESSIVENESS,
        temperature=2,
    ):
        """
        aggressiveness (float): Controls preference for property vs. cash cards.
                                Must be in the range [0, 1].
        temperature (float): Adjusts the sensitivity of the aggressiveness effect.
                             Higher values make aggressiveness impact stronger.
        """
        if not (0 <= aggressiveness <= 1):
            raise ValueError("Aggressiveness must be between 0 and 1.")
        self.aggressiveness = aggressiveness
        self.temperature = temperature

    def select(self, actions: list[WrappedAction], game: Game):
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

    def __repr__(self):
        return f"RiskPreferenceSampler(aggressiveness={self.aggressiveness}, temperature={self.temperature})"


@dataclass
class CFRActionSelector(BaseActionSelector):
    policy_manager: RuntimePolicyManager

    def select(
        self,
        actions: list[WrappedAction],
        game: Game,
    ):
        return self.policy_manager.get(game.state).argmax()

    def info(self, actions: list[WrappedAction], game: Game) -> dict:
        """Return CFR-specific selector information."""
        # Get the current policy
        policy = self.policy_manager.get(game.state)

        # Get the game state key
        state_key = game.state.key

        # Get update counts from the policy manager
        update_count = self.policy_manager.update_count.get(state_key, 0)

        # Convert policy to human-readable format
        policy_dict = policy.to_human_readable()

        return {"policy": policy_dict, "state_key": state_key, "update_count": update_count, "model_type": "CFR"}
