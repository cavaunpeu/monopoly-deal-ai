from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import cast

from game.action import (
    CARD_TYPE_TO_GAME_ACTION,
    AbstractAction,
    BaseAction,
    BaseActionResolver,
    CardAction,
    GameAction,
    GreedyActionResolver,
    PassAction,
    ResponseGameAction,
    WrappedAction,
    YieldAction,
    decode_abstract_action,
    decode_action,
)
from game.cards import Card, CashCard, PropertyTypeCard, RentCard, SpecialCard
from game.config import GameConfig
from game.pile import CashPile, Hand, Pile, PropertyPile
from game.util import Serializable


@dataclass(frozen=True)
class PlayerState(Serializable):
    """Represents the state of a player in the game."""

    hand: Hand
    properties: PropertyPile
    cash: CashPile

    def contains(self, card: Card, pile: Pile):
        match pile:
            case Pile.HAND:
                return self.hand.contains(card)
            case Pile.PROPERTY:
                return self.properties.contains(cast(PropertyTypeCard, card))
            case Pile.CASH:
                return self.cash.contains(cast(CashCard, card))

    def get_rent_amount(self, rent_card: RentCard):
        """Returns the amount of rent to collect."""
        property_type = rent_card.value
        num_properties = self.properties[property_type]
        max_num_chargable = len(property_type.rent_progression)
        num_chargable = min(num_properties, max_num_chargable)
        # Return rent amount
        return property_type.rent_progression[num_chargable - 1] if num_chargable > 0 else 0

    def to_json(self):
        return {
            "hand": self.hand.encode(),
            "properties": self.properties.encode(),
            "cash": self.cash.encode(),
        }

    @classmethod
    def from_json(cls, data: dict):
        return cls(
            hand=Hand.decode(data["hand"]),
            properties=PropertyPile.decode(data["properties"]),
            cash=CashPile.decode(data["cash"]),
        )

    def clone(self):
        return PlayerState(
            hand=self.hand.clone(),
            properties=self.properties.clone(),
            cash=self.cash.clone(),
        )


@dataclass(frozen=True)
class OpponentState(Serializable):
    """Represents the state of the opponent in the game."""

    properties: PropertyPile
    cash: CashPile

    def to_json(self):
        return {
            "properties": self.properties.encode(),
            "cash": self.cash.encode(),
        }

    @classmethod
    def from_json(cls, data: dict):
        return cls(
            properties=PropertyPile.decode(data["properties"]),
            cash=CashPile.decode(data["cash"]),
        )


class BaseResponseContext(Serializable, ABC):
    @property
    def is_empty(self):
        return not bool(self.to_json())

    @abstractmethod
    def clone(self):
        raise NotImplementedError


class EmptyResponseContext(BaseResponseContext):
    def to_json(self):
        return {}

    @classmethod
    def from_json(cls, data: dict):
        return cls()

    def clone(self):
        return self


@dataclass
class ResponseContext(BaseResponseContext):
    streaking_player_init_state: PlayerState
    init_action_taken: GameAction
    response_actions_taken: list[ResponseGameAction] = field(default_factory=list)

    def add_response(self, action: ResponseGameAction):
        self.response_actions_taken.append(action)

    def to_json(self):
        return {
            "streaking_player_init_state": self.streaking_player_init_state.to_json(),
            "init_action_taken": self.init_action_taken.encode(),
            "response_actions_taken": [action.encode() for action in self.response_actions_taken],
        }

    @classmethod
    def from_json(cls, data: dict):
        return cls(
            streaking_player_init_state=PlayerState.from_json(data["streaking_player_init_state"]),
            init_action_taken=decode_action(data["init_action_taken"]),
            response_actions_taken=[decode_action(idx) for idx in data["response_actions_taken"]],
        )

    def clone(self):
        return ResponseContext(
            streaking_player_init_state=self.streaking_player_init_state.clone(),
            init_action_taken=self.init_action_taken,
            response_actions_taken=list(self.response_actions_taken),
        )


def build_response_context_from_json(data: dict) -> BaseResponseContext:
    if not data:
        return EmptyResponseContext()
    return ResponseContext.from_json(data)


@dataclass
class TurnState(Serializable):
    turn_idx: int
    streak_idx: int
    streaking_player_idx: int

    def __post_init__(self):
        self._set_empty_response_ctx()

    def _set_empty_response_ctx(self):
        self.response_ctx = EmptyResponseContext()

    def set_response_ctx(self, response_ctx: BaseResponseContext):
        if not self.response_ctx.is_empty:
            raise ValueError("Response environment already set.")
        self.response_ctx = response_ctx

    @property
    def responding(self) -> bool:
        return not self.response_ctx.is_empty

    @property
    def acting_player_index(self) -> int:
        if self.responding:
            return 1 - self.streaking_player_idx
        return self.streaking_player_idx

    @property
    def other_player_index(self) -> int:
        return 1 - self.acting_player_index

    def complete_response(self):
        self._set_empty_response_ctx()

    def complete_streak(self):
        self.streak_idx = 0
        self.streaking_player_idx = 1 - self.streaking_player_idx

    def increment_streak_idx(self):
        self.streak_idx += 1

    def increment_turn_idx(self):
        self.turn_idx += 1

    def to_json(self):
        return {
            "turn_idx": self.turn_idx,
            "streak_idx": self.streak_idx,
            "streaking_player_idx": self.streaking_player_idx,
            "response_ctx": self.response_ctx.to_json(),
        }

    @classmethod
    def from_json(cls, data: dict):
        response_ctx = build_response_context_from_json(data["response_ctx"])
        turn_state = cls(
            turn_idx=data["turn_idx"],
            streak_idx=data["streak_idx"],
            streaking_player_idx=data["streaking_player_idx"],
        )
        turn_state.set_response_ctx(response_ctx)
        return turn_state

    def clone(self):
        turn_state = TurnState(
            turn_idx=self.turn_idx,
            streak_idx=self.streak_idx,
            streaking_player_idx=self.streaking_player_idx,
        )
        turn_state.set_response_ctx(self.response_ctx.clone())
        return turn_state


@dataclass(frozen=True)
class GameState(Serializable):
    """
    Represents the information set available to a player in the game.

    Note: Despite the name "GameState", this is technically an information set (infoset)
    rather than a complete game state. In game theory terms, an information set represents
    all the information available to a player at a decision point, which includes:
    - Complete information about the current player (hand, properties, cash)
    - Partial information about the opponent (properties, cash, but NOT hand)
    - Public game state (turn information, deck size, etc.)

    The opponent's hand is intentionally hidden, making this an imperfect information game.
    This design allows algorithms like CFR to work with the same information sets that
    human players would have, ensuring fair evaluation and training.
    """

    turn: TurnState
    player: PlayerState
    opponent: OpponentState
    config: GameConfig
    random_seed: int
    abstraction_cls: type["BaseStateAbstraction"]
    resolver_cls: type["BaseActionResolver"]

    @property
    def abstraction(self) -> "BaseStateAbstraction":
        return self.abstraction_cls.from_game_state(self)

    def get_player_actions(self, dedupe: bool = True) -> list[WrappedAction]:
        wrapped_actions = []
        # If player is responding, iterate through valid, playable responses
        if self.turn.responding:
            ctx = cast(ResponseContext, self.turn.response_ctx)
            for action in ctx.init_action_taken.response_def.get_valid_responses(ctx.streaking_player_init_state):
                if action.is_playable_by(self.player):
                    wrapped = self.abstraction_cls.action_to_wrapped_action(action, self)
                    wrapped_actions.append(wrapped)
            # If no valid responses, add YieldAction
            if not wrapped_actions:
                wrapped_actions.append(
                    WrappedAction(
                        action=YieldAction(),
                        abstract_action=AbstractAction.YIELD,
                    )
                )
        # Otherwise, map all cards to abstract actions
        else:
            for action in self.abstraction_cls.hand_to_actions(self.player.hand):
                if action.is_legal:
                    wrapped = self.abstraction_cls.action_to_wrapped_action(action, self)
                    wrapped_actions.append(wrapped)
            # Add pass action
            wrapped_actions.append(WrappedAction(action=PassAction(), abstract_action=AbstractAction.PASS))

        if dedupe:
            return self.resolver_cls.resolve(wrapped_actions, self.player, self.random_seed)
        return wrapped_actions

    def to_json(self):
        return {
            "turn": self.turn.to_json(),
            "player": self.player.to_json(),
            "opponent": self.opponent.to_json(),
            "config": self.config.to_json(),
            "abstraction_cls": self.abstraction_cls.__name__,
            "resolver_cls": self.resolver_cls.__name__,
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_json(cls, data: dict):
        abstraction_cls = ABSTRACTION_NAME_TO_CLS[data["abstraction_cls"]]
        resolver_cls = RESOLVER_NAME_TO_CLS[data["resolver_cls"]]
        return cls(
            turn=TurnState.from_json(data["turn"]),
            player=PlayerState.from_json(data["player"]),
            opponent=OpponentState.from_json(data["opponent"]),
            config=GameConfig.from_json(data["config"]),
            abstraction_cls=abstraction_cls,
            resolver_cls=resolver_cls,
            random_seed=data["random_seed"],
        )

    @property
    def key(self) -> str:
        return f"{self.turn.acting_player_index}@{self.abstraction_cls.__name__}@{self.abstraction.key}"

    @staticmethod
    def player_idx_from_key(key: str) -> int:
        return int(key.split("@")[0])

    @staticmethod
    def parse_key(key: str) -> "ParsedGameStateKey":
        player_idx, abstraction_cls_name, abstraction_key = key.split("@")
        return ParsedGameStateKey(
            player_idx=int(player_idx),
            abstraction_cls_name=abstraction_cls_name,
            abstraction_key=abstraction_key,
        )


@dataclass(frozen=True)
class ParsedGameStateKey:
    player_idx: int
    abstraction_cls_name: str
    abstraction_key: str


@dataclass(frozen=True)
class BasePlayerStateAbstraction(Serializable):
    pass


@dataclass(frozen=True)
class IntentPlayerStateAbstraction(BasePlayerStateAbstraction):
    actions: tuple[AbstractAction, ...]

    def to_json(self):
        return {
            "actions": [action.encode() for action in self.actions],
        }

    @classmethod
    def from_json(cls, data: dict):
        return cls(
            actions=tuple([decode_abstract_action(idx) for idx in data["actions"]]),
        )


@dataclass(frozen=True)
class BaseOpponentStateAbstraction(Serializable):
    pass


@dataclass(frozen=True)
class IntentOpponentStateAbstraction(BaseOpponentStateAbstraction):
    def to_json(self):
        return {}

    @classmethod
    def from_json(cls, data: dict):
        return cls()


@dataclass(frozen=True)
class BaseStateAbstraction(Serializable):
    turn: TurnState
    player: BasePlayerStateAbstraction
    opponent: BaseOpponentStateAbstraction

    @classmethod
    @abstractmethod
    def from_game_state(cls, game_state: "GameState") -> "BaseStateAbstraction":
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def action_to_wrapped_action(cls, action: BaseAction, game_state: "GameState"):
        raise NotImplementedError

    @classmethod
    def hand_to_actions(cls, hand: Hand) -> list[BaseAction]:
        return [CARD_TYPE_TO_GAME_ACTION[card] for card in hand]

    def to_json(self):
        return {
            "streak_idx": self.turn.streak_idx,
            "player": self.player.to_json(),
            "opponent": self.opponent.to_json(),
        }


class IntentStateAbstraction(BaseStateAbstraction):
    @classmethod
    def action_to_wrapped_action(cls, action: BaseAction, game_state: "GameState") -> WrappedAction:
        # Match responses
        if action.is_response:
            match action:
                # Just say no
                case ResponseGameAction(
                    src=Pile.HAND,
                    dst=Pile.DISCARD,
                    card=SpecialCard.JUST_SAY_NO,
                ):
                    return WrappedAction(action=action, abstract_action=AbstractAction.JUST_SAY_NO)
                # Give opponent cash
                case ResponseGameAction(
                    src=Pile.CASH,
                    dst=Pile.OPPONENT_CASH,
                    card=card,
                ):
                    return WrappedAction(action=action, abstract_action=AbstractAction.GIVE_OPPONENT_CASH)
                # Give opponent property
                case ResponseGameAction(
                    src=Pile.PROPERTY,
                    dst=Pile.OPPONENT_CASH,
                    card=card,
                ):
                    return WrappedAction(
                        action=action,
                        abstract_action=AbstractAction.GIVE_OPPONENT_PROPERTY,
                    )
                # Yield
                case YieldAction():
                    return WrappedAction(action=action, abstract_action=AbstractAction.YIELD)
                case _:
                    raise ValueError(f"Invalid response action: {action}")

        # Match pass
        elif not action.plays_card:
            return WrappedAction(action=action, abstract_action=AbstractAction.PASS)

        # Match non responses
        card = cast(CardAction, action).card
        match card:
            # Store cash, via property
            case PropertyTypeCard() as prop if game_state.player.properties.get(prop, 0) >= prop.num_to_complete:
                return WrappedAction(action=action, abstract_action=AbstractAction.CASH)

            # Complete property set
            case PropertyTypeCard() as prop if game_state.player.properties.get(prop, 0) + 1 == prop.num_to_complete:
                return WrappedAction(action=action, abstract_action=AbstractAction.COMPLETE_PROPERTY_SET)

            # Add to property set
            case PropertyTypeCard() as prop if game_state.player.properties.get(prop, 0) > 0:
                return WrappedAction(action=action, abstract_action=AbstractAction.ADD_TO_PROPERTY_SET)

            # Start new property set
            case PropertyTypeCard():
                return WrappedAction(action=action, abstract_action=AbstractAction.START_NEW_PROPERTY_SET)

            # Store cash
            case CashCard():
                return WrappedAction(action=action, abstract_action=AbstractAction.CASH)

            # Collect rent
            case RentCard() as rent if game_state.player.properties.get(rent.value, 0) > 0:
                return WrappedAction(action=action, abstract_action=AbstractAction.ATTEMPT_COLLECT_RENT)

            # Fallback
            case _:
                return WrappedAction(action=action, abstract_action=AbstractAction.OTHER)

    @classmethod
    def from_game_state(cls, game_state: "GameState") -> "IntentStateAbstraction":
        # Define player abstraction
        player = IntentPlayerStateAbstraction(
            actions=tuple(
                sorted(
                    [action.abstract_action for action in game_state.get_player_actions()],
                    key=lambda a: a.encode(),
                )
            )
        )

        # Define opponent abstraction
        opponent = IntentOpponentStateAbstraction()

        # Return state abstraction
        return cls(
            turn=game_state.turn,
            player=player,
            opponent=opponent,
        )

    @classmethod
    def from_json(cls, data: dict):
        return cls(
            turn=TurnState.from_json(data["turn"]),
            player=IntentPlayerStateAbstraction.from_json(data["player"]),
            opponent=IntentOpponentStateAbstraction.from_json(data["opponent"]),
        )


ABSTRACTION_NAME_TO_CLS = {"IntentStateAbstraction": IntentStateAbstraction}
RESOLVER_NAME_TO_CLS = {"GreedyActionResolver": GreedyActionResolver}
