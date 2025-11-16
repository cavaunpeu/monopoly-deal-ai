from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from itertools import chain
import random
from typing import TYPE_CHECKING

from game.cards import Card, CashCard, PropertyTypeCard, RentCard, SpecialCard
from game.pile import Pile


if TYPE_CHECKING:
    from game.state import PlayerState


class BaseAction(ABC):
    """Abstract base class for all game actions."""

    def encode(self) -> int:
        """Encode this action to an integer representation.

        Returns:
            Integer encoding of the action.
        """
        return encode_action(self)

    @classmethod
    def decode(cls, encoded: int) -> "BaseAction":
        """Decode an integer to an action instance.

        Args:
            encoded: Integer encoding of the action.

        Returns:
            Decoded action instance.
        """
        return decode_action(encoded)

    @property
    @abstractmethod
    def plays_card(self) -> bool:
        """Check if this action plays a card.

        Returns:
            True if the action plays a card, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def is_legal(self) -> bool:
        """Check if this action is legal in the current game state.

        Returns:
            True if the action is legal, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def is_response(self) -> bool:
        """Check if this action is a response to another action.

        Returns:
            True if this is a response action, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def is_draw(self) -> bool:
        """Check if this action draws cards.

        Returns:
            True if the action draws cards, False otherwise.
        """
        pass


@dataclass(frozen=True)
class PassAction(BaseAction):
    """Action representing a player passing their turn."""

    @property
    def is_legal(self) -> bool:
        """Pass action is always legal."""
        return True

    @property
    def is_response(self) -> bool:
        """Pass action is not a response."""
        return False

    @property
    def plays_card(self) -> bool:
        """Pass action does not play a card."""
        return False

    @property
    def is_draw(self) -> bool:
        """Pass action does not draw cards."""
        return False


@dataclass(frozen=True)
class YieldAction(BaseAction):
    """Action representing a player yielding — the only legal response action
    when the player has no valid responses."""

    @property
    def is_legal(self) -> bool:
        """Yield action is always legal."""
        return True

    @property
    def is_response(self) -> bool:
        """Yield action is a response action."""
        return True

    @property
    def plays_card(self) -> bool:
        """Yield action does not play a card."""
        return False

    @property
    def is_draw(self) -> bool:
        """Yield action does not draw cards."""
        return False


@dataclass(frozen=True)
class IllegalAction(BaseAction):
    """Action representing an illegal move (used for data model consistency)."""

    @property
    def is_legal(self) -> bool:
        """Illegal action is never legal."""
        return False

    @property
    def is_response(self) -> bool:
        """Illegal action is not a response."""
        return False

    @property
    def plays_card(self) -> bool:
        """Illegal action does not play a card."""
        return False

    @property
    def is_draw(self) -> bool:
        """Illegal action does not draw cards."""
        return False


@dataclass(frozen=True)
class GameAction(BaseAction):
    """Base class for actions that move cards between piles."""

    src: Pile
    dst: Pile
    card: Card
    response_def_cls: type["BaseResponseDefinition"]

    @property
    def response_def(self) -> BaseResponseDefinition:
        """Get the response definition for this action.

        Returns:
            Response definition instance for the card.
        """
        return self.response_def_cls(self.card)

    @property
    def is_legal(self) -> bool:
        return True

    @property
    def is_response(self) -> bool:
        return False

    @property
    def plays_card(self) -> bool:
        return True

    @property
    def is_draw(self) -> bool:
        return False


@dataclass(frozen=True)
class ResponseGameAction(BaseAction):
    src: Pile
    dst: Pile
    card: Card

    def is_playable_by(self, player_state: "PlayerState"):
        return player_state.contains(self.card, self.src)

    @property
    def is_legal(self) -> bool:
        return True

    @property
    def is_response(self) -> bool:
        return True

    @property
    def plays_card(self) -> bool:
        return True

    @property
    def is_draw(self) -> bool:
        return False


@dataclass(frozen=True)
class DrawAction(BaseAction):
    card: Card

    @property
    def is_legal(self) -> bool:
        return True

    @property
    def is_response(self) -> bool:
        return False

    @property
    def plays_card(self) -> bool:
        return False

    @property
    def is_draw(self) -> bool:
        return True


CardAction = GameAction | ResponseGameAction


################################################################################
# Define responses
################################################################################


@dataclass(frozen=True)
class BaseResponseDefinition(ABC):
    card: Card

    @abstractmethod
    def get_valid_responses(
        self,
        streaking_player_init_state: "PlayerState",
    ) -> set[ResponseGameAction]:
        raise NotImplementedError

    @abstractmethod
    def response_complete(
        self,
        streaking_player_init_state: "PlayerState",
        streaking_player_curr_state: "PlayerState",
        response_actions_taken: list[ResponseGameAction],
    ) -> bool:
        raise NotImplementedError

    def response_required(
        self,
        streaking_player_init_state: "PlayerState",
    ) -> bool:
        return bool(self.get_valid_responses(streaking_player_init_state))


class NoResponseDefinition(BaseResponseDefinition):
    def get_valid_responses(
        self,
        streaking_player_init_state: "PlayerState",
    ) -> set[ResponseGameAction]:
        return set()

    def response_complete(
        self,
        streaking_player_init_state: "PlayerState",
        streaking_player_curr_state: "PlayerState",
        response_actions_taken: list[ResponseGameAction],
    ) -> bool:
        return True


class RentCardResponseDefinition(BaseResponseDefinition):
    card: RentCard

    def get_rent_amount(self, streaking_player_init_state: "PlayerState"):
        return streaking_player_init_state.get_rent_amount(self.card)

    def get_valid_responses(self, streaking_player_init_state: "PlayerState") -> set[ResponseGameAction]:
        if self.get_rent_amount(streaking_player_init_state) > 0:
            responses = {
                ResponseGameAction(
                    src=Pile.HAND,
                    dst=Pile.DISCARD,
                    card=SpecialCard.JUST_SAY_NO,
                )
            }
            # Add cash responses
            for card in CashCard:
                responses.add(
                    ResponseGameAction(
                        src=Pile.CASH,
                        dst=Pile.OPPONENT_CASH,
                        card=card,
                    )
                )
            # Add property responses
            for card in PropertyTypeCard:
                responses.add(
                    ResponseGameAction(
                        src=Pile.PROPERTY,
                        dst=Pile.OPPONENT_CASH,
                        card=card,
                    )
                )
                responses.add(
                    ResponseGameAction(
                        src=Pile.CASH,
                        dst=Pile.OPPONENT_CASH,
                        card=card,
                    )
                )
            return responses
        return set()

    def response_complete(
        self,
        streaking_player_init_state: "PlayerState",
        streaking_player_curr_state: "PlayerState",
        response_actions_taken: list[ResponseGameAction],
    ) -> bool:
        # Just say no
        if response_actions_taken[-1] == ResponseGameAction(
            src=Pile.HAND,
            dst=Pile.DISCARD,
            card=SpecialCard.JUST_SAY_NO,
        ):
            return True
        # Yield (can only be played if opponent has no valid responses)
        if response_actions_taken[-1] == YieldAction():
            return True
        # Collecting rent
        rent_amount = self.get_rent_amount(streaking_player_init_state)
        return streaking_player_curr_state.cash.cash_total - streaking_player_init_state.cash.cash_total >= rent_amount


################################################################################
# Map card types to actions
################################################################################

CARD_TYPE_TO_GAME_ACTION = {}
CARD_TYPE_TO_RESPONSE_ACTIONS = defaultdict(set)
CARD_TYPE_TO_DRAW_ACTIONS = {}

# Property types
for card in PropertyTypeCard:
    # Not responding
    CARD_TYPE_TO_GAME_ACTION[card] = GameAction(
        src=Pile.HAND,
        dst=Pile.PROPERTY,
        card=card,
        response_def_cls=NoResponseDefinition,
    )
    # Responding
    CARD_TYPE_TO_RESPONSE_ACTIONS[card].add(
        ResponseGameAction(
            src=Pile.PROPERTY,
            dst=Pile.OPPONENT_CASH,
            card=card,
        )
    )
    CARD_TYPE_TO_RESPONSE_ACTIONS[card].add(
        ResponseGameAction(
            src=Pile.CASH,
            dst=Pile.OPPONENT_CASH,
            card=card,
        )
    )
    # Draw
    CARD_TYPE_TO_DRAW_ACTIONS[card] = DrawAction(
        card=card,
    )


# Cash cards
for card in CashCard:
    # Not responding
    CARD_TYPE_TO_GAME_ACTION[card] = GameAction(
        src=Pile.HAND,
        dst=Pile.CASH,
        card=card,
        response_def_cls=NoResponseDefinition,
    )
    # Responding
    CARD_TYPE_TO_RESPONSE_ACTIONS[card].add(
        ResponseGameAction(
            src=Pile.CASH,
            dst=Pile.OPPONENT_CASH,
            card=card,
        )
    )
    # Draw
    CARD_TYPE_TO_DRAW_ACTIONS[card] = DrawAction(
        card=card,
    )

# Rent cards
for card in RentCard:
    # Not responding
    CARD_TYPE_TO_GAME_ACTION[card] = GameAction(
        src=Pile.HAND,
        dst=Pile.DISCARD,
        card=card,
        response_def_cls=RentCardResponseDefinition,
    )
    # Responding
    CARD_TYPE_TO_RESPONSE_ACTIONS[card].add(IllegalAction())
    # Draw
    CARD_TYPE_TO_DRAW_ACTIONS[card] = DrawAction(
        card=card,
    )

# Special cards
for card in SpecialCard:
    # Not responding
    CARD_TYPE_TO_GAME_ACTION[card] = IllegalAction()
    # Responding
    CARD_TYPE_TO_RESPONSE_ACTIONS[card].add(
        ResponseGameAction(
            src=Pile.HAND,
            dst=Pile.DISCARD,
            card=card,
        )
    )
    # Draw
    CARD_TYPE_TO_DRAW_ACTIONS[card] = DrawAction(
        card=card,
    )

################################################################################
# Map actions to indices, indices to actions
################################################################################

actions = [
    PassAction(),
    YieldAction(),
    *CARD_TYPE_TO_GAME_ACTION.values(),
    *chain.from_iterable(CARD_TYPE_TO_RESPONSE_ACTIONS.values()),
    *CARD_TYPE_TO_DRAW_ACTIONS.values(),
]

ACTION_TO_IDX = {}
IDX_TO_ACTION = {}

for i, action in enumerate(actions):
    ACTION_TO_IDX[action] = i
    IDX_TO_ACTION[i] = action

################################################################################
# Define abstract actions
################################################################################


@dataclass(frozen=True)
class WrappedAction:
    action: BaseAction
    abstract_action: "AbstractAction"

    def encode(self):
        return {
            "action": self.action.encode(),
            "abstract_action": self.abstract_action.encode(),
        }

    @classmethod
    def decode(cls, encoded: dict):
        return cls(
            action=decode_action(encoded["action"]),
            abstract_action=decode_abstract_action(encoded["abstract_action"]),
        )


class AbstractAction(Enum):
    """Abstract action enum representing high-level game actions."""

    # Intent-based actions
    START_NEW_PROPERTY_SET = "START_NEW_PROPERTY_SET"
    ADD_TO_PROPERTY_SET = "ADD_TO_PROPERTY_SET"
    COMPLETE_PROPERTY_SET = "COMPLETE_PROPERTY_SET"
    CASH = "CASH"
    ATTEMPT_COLLECT_RENT = "ATTEMPT_COLLECT_RENT"
    JUST_SAY_NO = "JUST_SAY_NO"
    GIVE_OPPONENT_CASH = "GIVE_OPPONENT_CASH"
    GIVE_OPPONENT_PROPERTY = "GIVE_OPPONENT_PROPERTY"
    PASS = "PASS"
    YIELD = "YIELD"
    OTHER = "OTHER"

    # PLAY_ prefixed card actions - PropertyTypeCard
    PLAY_PROPERTY_BROWN = "PLAY_PROPERTY_BROWN"
    PLAY_PROPERTY_GREEN = "PLAY_PROPERTY_GREEN"
    PLAY_PROPERTY_PINK = "PLAY_PROPERTY_PINK"

    # PLAY_ prefixed card actions - RentCard
    PLAY_RENT_BROWN = "PLAY_RENT_BROWN"
    PLAY_RENT_GREEN = "PLAY_RENT_GREEN"
    PLAY_RENT_PINK = "PLAY_RENT_PINK"

    # PLAY_ prefixed card actions - CashCard
    PLAY_CASH_ONE = "PLAY_CASH_ONE"
    PLAY_CASH_TWO = "PLAY_CASH_TWO"
    PLAY_CASH_THREE = "PLAY_CASH_THREE"
    PLAY_CASH_FOUR = "PLAY_CASH_FOUR"

    # PLAY_ prefixed card actions - SpecialCard
    PLAY_JUST_SAY_NO = "PLAY_JUST_SAY_NO"

    def encode(self) -> int:
        return encode_abstract_action(self)

    @classmethod
    def decode(cls, idx: int | str) -> "AbstractAction":
        return decode_abstract_action(int(idx))


def card_to_play_action(card: Card) -> AbstractAction:
    """Map a Card to its corresponding PLAY_ AbstractAction enum value.

    Args:
        card: The card to map.

    Returns:
        The corresponding PLAY_ AbstractAction enum value.

    Raises:
        ValueError: If the card doesn't have a corresponding PLAY_ action.
    """
    match card:
        case PropertyTypeCard():
            return AbstractAction[f"PLAY_PROPERTY_{card.name}"]
        case RentCard():
            return AbstractAction[f"PLAY_RENT_{card.name}"]
        case CashCard():
            return AbstractAction[f"PLAY_CASH_{card.name}"]
        case SpecialCard():
            return AbstractAction[f"PLAY_{card.name}"]


class BaseActionResolver(ABC):
    @classmethod
    @abstractmethod
    def resolve(
        cls,
        wrapped_actions: list[WrappedAction],
        player_state: "PlayerState",
        random_seed: int,
    ) -> list[WrappedAction]:
        raise NotImplementedError


class GreedyActionResolver(BaseActionResolver):
    @classmethod
    def resolve(
        cls,
        wrapped_actions: list[WrappedAction],
        player_state: "PlayerState",
        random_seed: int,
    ) -> list[WrappedAction]:
        # Map abstract actions to actions
        abstract_action_to_actions = defaultdict(list)
        for wrapped in wrapped_actions:
            abstract_action_to_actions[wrapped.abstract_action].append(wrapped.action)

        # Map abstract actions to single actions
        resolved = []

        # Resolve with greedy logic
        for aa, actions in abstract_action_to_actions.items():
            match aa:
                case _ if len(actions) == 1:
                    (action,) = actions
                    res = WrappedAction(action=action, abstract_action=aa)
                case AbstractAction.ATTEMPT_COLLECT_RENT:
                    # Return argmax card w.r.t. rent amount
                    res = WrappedAction(
                        action=max(actions, key=lambda x: player_state.get_rent_amount(x.card)),
                        abstract_action=aa,
                    )
                case AbstractAction.CASH:
                    # Return argmax card w.r.t. value (could be property or cash card)
                    res = WrappedAction(
                        action=max(actions, key=lambda x: x.card.cash_value),
                        abstract_action=aa,
                    )
                case _:
                    # Otherwise, return a random action
                    rng = random.Random(random_seed)
                    res = WrappedAction(
                        action=rng.choice(
                            actions,
                        ),
                        abstract_action=aa,
                    )
            resolved.append(res)

        return resolved


################################################################################
# Map abstraction actions to indices, indices to actions
################################################################################

ABSTRACT_ACTION_TO_IDX = {}
IDX_TO_ABSTRACT_ACTION = {}

for i, action in enumerate(AbstractAction):
    ABSTRACT_ACTION_TO_IDX[action] = i
    IDX_TO_ABSTRACT_ACTION[i] = action

################################################################################
# Define functions to encode and decode actions
################################################################################


def encode_action(action: BaseAction) -> int:
    """Encode a BaseAction to its integer representation.

    Args:
        action: The action to encode.

    Returns:
        Integer index representing the action.
    """
    return ACTION_TO_IDX[action]


def decode_action(idx: int) -> BaseAction:
    """Decode an integer to its corresponding BaseAction.

    Args:
        idx: Integer index of the action.

    Returns:
        The decoded action instance.
    """
    return IDX_TO_ACTION[idx]


################################################################################
# Define functions to encode and decode abstract actions
################################################################################


def encode_abstract_action(action: AbstractAction) -> int:
    """Encode an AbstractAction to its integer representation.

    Args:
        action: The abstract action to encode.

    Returns:
        Integer index representing the abstract action.
    """
    return ABSTRACT_ACTION_TO_IDX[action]


def decode_abstract_action(idx: int) -> AbstractAction:
    """Decode an integer to its corresponding AbstractAction.

    Args:
        idx: Integer index of the abstract action.

    Returns:
        The decoded abstract action enum value.
    """
    return IDX_TO_ABSTRACT_ACTION[idx]


@dataclass(frozen=True)
class SelectedAction:
    """Represents an action that was selected and executed in the game with context"""

    turn_idx: int
    streak_idx: int
    player_idx: int
    action: BaseAction
