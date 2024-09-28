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
from game.util import Serializable


if TYPE_CHECKING:
    from game.state import PlayerState


class BaseAction(ABC):
    def encode(self):
        return encode_action(self)

    @classmethod
    def decode(cls, encoded: int):
        return decode_action(encoded)

    @property
    @abstractmethod
    def plays_card(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_legal(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_response(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_draw(self) -> bool:
        pass


@dataclass(frozen=True)
class PassAction(BaseAction):
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
        return False


@dataclass(frozen=True)
class YieldAction(BaseAction):
    @property
    def is_legal(self) -> bool:
        return True

    @property
    def is_response(self) -> bool:
        return True

    @property
    def plays_card(self) -> bool:
        return False

    @property
    def is_draw(self) -> bool:
        return False


@dataclass(frozen=True)
class IllegalAction(BaseAction):
    @property
    def is_legal(self) -> bool:
        return False

    @property
    def is_response(self) -> bool:
        return False

    @property
    def plays_card(self) -> bool:
        return False

    @property
    def is_draw(self) -> bool:
        return False


@dataclass(frozen=True)
class GameAction(BaseAction):
    src: Pile
    dst: Pile
    card: Card
    response_def_cls: type["BaseResponseDefinition"]

    @property
    def response_def(self) -> BaseResponseDefinition:
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

    def encode(self):
        return encode_abstract_action(self)

    @classmethod
    def decode(cls, idx: int | str):
        return decode_abstract_action(int(idx))


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


def encode_action(action: BaseAction):
    return ACTION_TO_IDX[action]


def decode_action(idx: int):
    return IDX_TO_ACTION[idx]


################################################################################
# Define functions to encode and decode abstract actions
################################################################################


def encode_abstract_action(action: AbstractAction):
    return ABSTRACT_ACTION_TO_IDX[action]


def decode_abstract_action(idx: int):
    return IDX_TO_ABSTRACT_ACTION[idx]


################################################################################
# Define a PlayerAction class
################################################################################


@dataclass(frozen=True)
class PlayerAction(Serializable):
    player_idx: int
    action: BaseAction

    def to_json(self):
        return {
            "player_idx": self.player_idx,
            "action": self.action.encode(),
        }

    @classmethod
    def from_json(cls, data: dict):
        return cls(
            player_idx=data["player_idx"],
            action=decode_action(data["action"]),
        )

    def clone(self):
        return PlayerAction.from_json(self.to_json())


@dataclass(frozen=True)
class SelectedAction:
    """Represents an action that was selected and executed in the game with context"""

    turn_idx: int
    streak_idx: int
    player_idx: int
    action: BaseAction
