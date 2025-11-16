from __future__ import annotations

from typing import TYPE_CHECKING, Any

from game.action import WrappedAction
from game.state import GameState
from models.selector import ActionSelectorInfo, BaseActionSelector
from models.types import Policy


if TYPE_CHECKING:
    from models.gae.model import PolicyAndValueNetwork


class GAEActionSelector(BaseActionSelector):
    """Action selector that uses a GAE model to select actions."""

    def __init__(self, model: PolicyAndValueNetwork):
        self.model = model

    def select(self, actions: list[WrappedAction], state: GameState, argmax: bool = False) -> WrappedAction:
        # Get abstract actions
        abstract_actions = [a.abstract_action for a in actions]
        # Compute action probabilities
        probs = self.model.actions_to_probs(abstract_actions, state.symmetric_key, state.vector_encoding())
        # Use argmax if requested, otherwise sample from the policy
        if argmax:
            return Policy(actions=actions, probs=probs.tolist()).aggregated_argmax()
        else:
            return Policy(actions=actions, probs=probs.tolist()).sample()

    def info(self, actions: list[WrappedAction], state: GameState) -> ActionSelectorInfo:
        """Return selector information."""
        # Get abstract actions
        abstract_actions = [a.abstract_action for a in actions]
        # Get symmetric state key (no player index)
        symmetric_key = state.symmetric_key
        # Get the action probabilities
        probs = self.model.actions_to_probs(abstract_actions, symmetric_key, state.vector_encoding())
        # Define policy
        policy = Policy(actions=actions, probs=probs.tolist())
        # Get update count for the symmetric state key
        update_count = self.model.update_count.get(symmetric_key, 0)

        # Return selector information (use full state.key for display/provenance)
        return ActionSelectorInfo(
            model_type="GAE",
            state_key=state.key,
            state_update_count=update_count,
            human_readable_policy=policy.to_human_readable(),
        )

    @classmethod
    def from_model(cls, model: PolicyAndValueNetwork, **kwargs: Any) -> "GAEActionSelector":
        return cls(model=model)
