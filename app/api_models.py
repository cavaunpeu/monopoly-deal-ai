from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict

from game.action import BaseAction, GameAction, PassAction, ResponseGameAction, YieldAction, decode_action
from game.cards import Card, CashCard, PropertyTypeCard, RentCard, SpecialCard


class CreateGameRequest(BaseModel):
    model_name: str


class CreateGameResponse(BaseModel):
    game_id: str


class TakeStepRequest(BaseModel):
    action_id: int


class AvailableActionsResponse(BaseModel):
    actions: list["SerializedAction"]


class GameOverResponse(BaseModel):
    over: bool


class HumanState(BaseModel):
    hand: list["SerializedCard"]
    properties: list["PropertyCardModel"]
    cash: list["CashCardModel | PropertyCardModel"]


class AiState(BaseModel):
    properties: list["PropertyCardModel"]
    cash: list["CashCardModel | PropertyCardModel"]
    hand_count: int
    hand: list["SerializedCard"] | None = None


class SerializedAction(BaseModel):
    id: int
    is_response: bool
    card: "SerializedCard | None"
    src: str | None
    dst: str | None


class PublicPileSizesResponse(BaseModel):
    deck: int
    discard: int


class CardModel(BaseModel):
    kind: Any

    model_config = ConfigDict(extra="forbid")


class CashCardModel(CardModel):
    kind: Literal["CASH"] = "CASH"
    name: str  # e.g. "ONE", "TWO"
    value: int

    @classmethod
    def from_card(cls, card: CashCard) -> "CashCardModel":
        return cls(
            name=card.name,
            value=card.cash_value,
        )


class PropertyCardModel(CardModel):
    kind: Literal["PROPERTY"] = "PROPERTY"
    name: str  # e.g. "GREEN", "PINK"
    rent_progression: tuple[int, ...]
    value: int

    @classmethod
    def from_card(cls, card: PropertyTypeCard) -> "PropertyCardModel":
        return cls(
            name=card.name,
            rent_progression=card.rent_progression,
            value=card.cash_value,
        )


class RentCardModel(CardModel):
    kind: Literal["RENT"] = "RENT"
    name: str  # e.g. "GREEN", "PINK"
    property_name: str


class SpecialCardModel(CardModel):
    kind: Literal["SPECIAL"] = "SPECIAL"
    name: str  # e.g. "JUST_SAY_NO"


SerializedCard = Union[
    CashCardModel,
    PropertyCardModel,
    RentCardModel,
    SpecialCardModel,
]


class ResponseInfo(BaseModel):
    initiating_card: SerializedCard
    initiating_player: str  # "human" or "bot"
    responding_player: str  # "human" or "bot"
    response_cards_played: list[SerializedCard]


class SelectedActionEntry(BaseModel):
    turn_idx: int
    streak_idx: int
    player_idx: int
    action: SerializedAction


class TurnStateResponse(BaseModel):
    turn_idx: int
    streak_idx: int
    streaking_player_idx: int
    acting_player_idx: int
    is_human_turn: bool
    human_player_index: int
    cards_played_this_turn: int
    max_cards_per_turn: int
    remaining_cards: int
    selected_actions: list[SelectedActionEntry]
    game_over: bool
    winner: str | None
    is_responding: bool
    response_info: ResponseInfo | None


class PropertyTypeResponse(BaseModel):
    """Property type response model"""

    name: str
    num_to_complete: int
    rent_progression: list[int]
    cash_value: int


class GameConfigResponse(BaseModel):
    """Game configuration response"""

    required_property_sets: int
    max_consecutive_player_actions: int
    cash_card_values: list[int]
    rent_cards_per_property_type: int
    deck_size_multiplier: int
    total_deck_size: int
    initial_hand_size: int
    new_cards_per_turn: int
    card_to_special_card_ratio: float
    required_property_sets_map: dict[str, int]
    property_types: list[PropertyTypeResponse]


class GameStateResponse(BaseModel):
    """Unified game state response combining all game data"""

    turn: TurnStateResponse
    human: HumanState
    ai: AiState
    piles: PublicPileSizesResponse
    config: GameConfigResponse
    actions: list[SerializedAction]


def serialize_card(card: Card) -> SerializedCard:
    """Serialize a game card to a Pydantic model.

    Args:
        card: The card to serialize.

    Returns:
        SerializedCard representation of the card.
    """
    match card:
        case CashCard() as c:
            return CashCardModel(
                name=c.name,
                value=c.cash_value,
            )

        case PropertyTypeCard() as p:
            return PropertyCardModel(
                name=p.name,
                rent_progression=p.rent_progression,
                value=p.cash_value,
            )

        case RentCard() as r:
            return RentCardModel(
                name=r.name,
                property_name=r.value.name,
            )

        case SpecialCard() as s:
            return SpecialCardModel(
                name=s.name,
            )


def serialize_cash_pile_card(
    card: CashCard | PropertyTypeCard,
) -> CashCardModel | PropertyCardModel:
    match card:
        case CashCard() as c:
            return CashCardModel(
                name=c.name,
                value=c.cash_value,
            )

        case PropertyTypeCard() as p:
            return PropertyCardModel(
                name=p.name,
                rent_progression=p.rent_progression,
                value=p.cash_value,
            )


def serialize_action(action: BaseAction) -> "SerializedAction":
    """Serialize a game action to a Pydantic model.

    Args:
        action: The action to serialize.

    Returns:
        SerializedAction representation of the action.
    """
    match action:
        case PassAction():
            return SerializedAction(
                id=action.encode(),
                is_response=False,
                card=None,
                src=None,
                dst=None,
            )
        case YieldAction():
            return SerializedAction(
                id=action.encode(),
                is_response=True,
                card=None,
                src=None,
                dst=None,
            )
        case GameAction(src=src, dst=dst, card=card, response_def=_):
            return SerializedAction(
                id=action.encode(),
                is_response=False,
                card=serialize_card(card),
                src=src.value,
                dst=dst.value,
            )
        case ResponseGameAction(src=src, dst=dst, card=card):
            return SerializedAction(
                id=action.encode(),
                is_response=True,
                card=serialize_card(card),
                src=src.value,
                dst=dst.value,
            )
        case _:
            raise ValueError(f"Unsupported action type: {action}")


def deserialize_action(action_id: int) -> BaseAction:
    """Deserialize an action ID back to a BaseAction instance.

    Args:
        action_id: Integer ID of the action.

    Returns:
        BaseAction instance corresponding to the ID.
    """
    return decode_action(int(action_id))


def serialize_turn_state(turn_state_data: dict) -> TurnStateResponse:
    """Serialize turn state data into TurnStateResponse"""
    # Serialize the selected actions
    selected_actions = []
    for selected_action in turn_state_data["selected_actions"]:
        action_data = {
            "turn_idx": selected_action.turn_idx,
            "streak_idx": selected_action.streak_idx,
            "player_idx": selected_action.player_idx,
            "action": {
                "id": selected_action.action.encode(),
                "card": serialize_card(selected_action.action.card) if selected_action.action.plays_card else None,
                "src": selected_action.action.src if hasattr(selected_action.action, "src") else None,
                "dst": selected_action.action.dst if hasattr(selected_action.action, "dst") else None,
                "is_response": selected_action.action.is_response
                if hasattr(selected_action.action, "is_response")
                else False,
            },
        }
        selected_actions.append(SelectedActionEntry(**action_data))

    # Serialize response info cards if present
    response_info = turn_state_data.get("response_info")
    if response_info:
        if response_info["initiating_card"]:
            response_info["initiating_card"] = serialize_card(response_info["initiating_card"])
        response_info["response_cards_played"] = [
            serialize_card(card) for card in response_info["response_cards_played"]
        ]

    return TurnStateResponse(**{**turn_state_data, "selected_actions": selected_actions})


def serialize_game_config(game_config_obj, property_types) -> GameConfigResponse:
    """Serialize game config into GameConfigResponse"""
    return GameConfigResponse(
        required_property_sets=game_config_obj.required_property_sets,
        max_consecutive_player_actions=game_config_obj.max_consecutive_player_actions,
        cash_card_values=game_config_obj.cash_card_values,
        rent_cards_per_property_type=game_config_obj.rent_cards_per_property_type,
        deck_size_multiplier=game_config_obj.deck_size_multiplier,
        total_deck_size=game_config_obj.get_total_deck_size(),
        initial_hand_size=game_config_obj.initial_hand_size,
        new_cards_per_turn=game_config_obj.new_cards_per_turn,
        card_to_special_card_ratio=game_config_obj.card_to_special_card_ratio,
        required_property_sets_map={prop_type.name: prop_type.num_to_complete for prop_type in property_types},
        property_types=[
            PropertyTypeResponse(
                name=prop_type.name,
                num_to_complete=prop_type.num_to_complete,
                rent_progression=list(prop_type.rent_progression),
                cash_value=prop_type.cash_value,
            )
            for prop_type in property_types
        ],
    )
