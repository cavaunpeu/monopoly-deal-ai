from functools import wraps

from fastapi import APIRouter, Depends, HTTPException

from app.api_models import (
    AiState,
    CreateGameResponse,
    GameConfigResponse,
    GameOverResponse,
    GameStateResponse,
    HumanState,
    PropertyCardModel,
    PublicPileSizesResponse,
    SerializedAction,
    TakeStepRequest,
    TurnStateResponse,
    deserialize_action,
    serialize_action,
    serialize_card,
    serialize_cash_pile_card,
    serialize_game_config,
    serialize_turn_state,
)
from app.dependencies import get_db, get_game_service
from app.game.service import GameService


router = APIRouter()


def handle_game_not_found_errors(func):
    """Decorator to handle game not found errors and convert them to HTTP 404."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if isinstance(e, ValueError):
                if "could not be recreated" in str(e) or "No database connection" in str(e):
                    raise HTTPException(status_code=404, detail="Game not found")
            raise e

    return wrapper


@router.post("/", response_model=CreateGameResponse)
def create_game(db=Depends(get_db), service: GameService = Depends(get_game_service)):
    """Create a new game instance.

    Returns:
        CreateGameResponse containing the new game ID.
    """
    service.db = db
    game_id = service.create_game()
    return CreateGameResponse(game_id=game_id)


@router.post("/{game_id}/step")
@handle_game_not_found_errors
def take_game_step(
    game_id: str,
    req: TakeStepRequest,
    db=Depends(get_db),
    service: GameService = Depends(get_game_service),
):
    """Execute a game step with the specified action.

    Args:
        game_id: Unique identifier for the game.
        req: Request containing the action to execute.

    Returns:
        Status confirmation of the step execution.
    """
    service.db = db
    action = deserialize_action(req.action_id)
    service.take_game_step(game_id, action)
    return {"status": "ok"}


@router.get("/{game_id}/ai_action", response_model=SerializedAction)
@handle_game_not_found_errors
def get_ai_action(game_id: str, db=Depends(get_db), service: GameService = Depends(get_game_service)):
    """Get the AI's selected action."""
    service.db = db
    action = service.select_bot_action(game_id)
    return serialize_action(action)


@router.get("/{game_id}/selection_info")
@handle_game_not_found_errors
def get_selection_info(game_id: str, db=Depends(get_db), service: GameService = Depends(get_game_service)):
    """Get selection information for the current player's available actions.

    Args:
        game_id: Unique identifier for the game.

    Returns:
        Selection information including available actions and their details.
    """
    service.db = db
    game = service._get_game(game_id)
    wrapped_actions = service._get_player_wrapped_actions(game_id)
    selection_info = service.selector.info(wrapped_actions, game)
    return selection_info


@router.get("/{game_id}/bot_is_acting", response_model=bool)
@handle_game_not_found_errors
def bot_is_acting(game_id: str, db=Depends(get_db), service: GameService = Depends(get_game_service)):
    service.db = db
    return service.bot_is_acting_player(game_id)


@router.get("/{game_id}/over", response_model=GameOverResponse)
@handle_game_not_found_errors
def game_is_over(game_id: str, db=Depends(get_db), service: GameService = Depends(get_game_service)):
    service.db = db
    return GameOverResponse(over=service.game_is_over(game_id))


@router.get("/{game_id}/state", response_model=GameStateResponse)
@handle_game_not_found_errors
def get_game_state(
    game_id: str, show_ai_hand: bool = False, db=Depends(get_db), service: GameService = Depends(get_game_service)
):
    """Unified endpoint that returns all game state in a single call"""
    service.db = db

    # Fetch all game data
    turn_state_data = service.get_enhanced_turn_state(game_id)
    human_state = service.get_human_state(game_id)
    ai_state, hand_count, ai_hand = service.get_bot_state(game_id, show_hand=show_ai_hand)
    pile_sizes_obj = service.get_public_pile_sizes(game_id)
    game_config_obj = service.get_game_config()
    actions = service.get_player_actions(game_id)
    property_types = service.get_property_types()

    # Return the game state
    return GameStateResponse(
        turn=serialize_turn_state(turn_state_data),
        human=HumanState(
            hand=[serialize_card(c) for c in human_state.hand.cards],
            properties=[PropertyCardModel.from_card(c) for c in human_state.properties.cards],
            cash=[serialize_cash_pile_card(c) for c in human_state.cash.cards],
        ),
        ai=AiState(
            properties=[PropertyCardModel.from_card(c) for c in ai_state.properties.cards],
            cash=[serialize_cash_pile_card(c) for c in ai_state.cash.cards],
            hand_count=hand_count,
            hand=ai_hand if show_ai_hand else None,
        ),
        piles=PublicPileSizesResponse(deck=pile_sizes_obj.deck, discard=pile_sizes_obj.discard),
        config=serialize_game_config(game_config_obj, property_types),
        actions=[serialize_action(a) for a in actions],
    )


@router.get("/{game_id}/turn_state", response_model=TurnStateResponse)
@handle_game_not_found_errors
def get_turn_state(game_id: str, db=Depends(get_db), service: GameService = Depends(get_game_service)):
    """Get just the turn state"""
    service.db = db
    turn_state_data = service.get_enhanced_turn_state(game_id)
    return serialize_turn_state(turn_state_data)


@router.get("/models")
def get_models():
    """Get the available model manifest with descriptions and checkpoint paths."""
    return GameService.get_model_manifest()


@router.get("/config", response_model=GameConfigResponse)
def get_default_game_config(service: GameService = Depends(get_game_service)):
    """Get the default game configuration."""
    game_config_obj = service.get_game_config()
    property_types = service.get_property_types()
    return serialize_game_config(game_config_obj, property_types)
