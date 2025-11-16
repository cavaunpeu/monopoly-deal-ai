from dataclasses import dataclass
from typing import Any

from game.action import WrappedAction
from game.state import GameState
from models.cfr.cfr import CFR, RuntimePolicyManager
from models.selector import ActionSelectorInfo, BaseActionSelector


@dataclass
class CFRActionSelector(BaseActionSelector):
    """Action selector that uses CFR-trained policies for action selection."""

    policy_manager: RuntimePolicyManager

    def select(
        self,
        actions: list[WrappedAction],
        state: GameState,
        deterministic: bool = False,
    ) -> WrappedAction:
        """Select action using the CFR policy's argmax strategy.

        Args:
            actions: List of available actions (unused, policy determines action).
            state: Current game state.
            deterministic: Whether to select a deterministic action.

        Returns:
            Action with highest probability according to CFR policy.
        """
        if deterministic:
            return self.policy_manager.get(state).argmax()
        else:
            return self.policy_manager.get(state).sample()

    def info(self, actions: list[WrappedAction], state: GameState) -> ActionSelectorInfo:
        """Return CFR-specific selector information.

        Args:
            actions: List of available actions (unused).
            state: Current game state.

        Returns:
            Dictionary containing policy information, state key, and update count.
        """
        # Get the current policy
        policy = self.policy_manager.get(state)

        # Get the game state key
        state_key = state.key

        # Get update counts from the policy manager
        update_count = self.policy_manager.update_count.get(state_key, 0)

        return ActionSelectorInfo(
            model_type="CFR",
            state_key=state_key,
            state_update_count=update_count,
            human_readable_policy=policy.to_human_readable(),
        )

    @classmethod
    def from_model(cls, model: CFR, opponent_player_index: int, **kwargs: Any) -> "CFRActionSelector":
        return cls(policy_manager=model.policy_manager.get_runtime_policy_manager(opponent_player_index))
