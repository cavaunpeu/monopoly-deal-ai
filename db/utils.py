import datetime
from typing import cast

from sqlalchemy import update
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import BinaryExpression

from app.db_models import Action, Config
from game.action import ACTION_TO_IDX
from game.config import GameConfigType


def upsert_game_config_types(db: Session) -> None:
    """
    Upsert all game config types from GameConfigType enum into the configs table.

    Args:
        db: Database session
    """
    for config_type in GameConfigType:
        config = config_type.value

        # Build a dict of field values
        field_values = {
            "cash_card_values": config.cash_card_values,
            "rent_cards_per_property_type": config.rent_cards_per_property_type,
            "required_property_sets": config.required_property_sets,
            "deck_size_multiplier": config.deck_size_multiplier,
            "initial_hand_size": config.initial_hand_size,
            "new_cards_per_turn": config.new_cards_per_turn,
            "max_consecutive_player_actions": config.max_consecutive_player_actions,
        }

        # Check if config already exists
        existing_config = get_config_by_name(db, config_type.name)

        if existing_config:
            # Update existing config using SQLAlchemy update statement
            config_name = config_type.name
            where_clause = cast(BinaryExpression[bool], Config.name == config_name)
            # Add updated_at for updates
            update_values = {
                **field_values,
                "updated_at": datetime.datetime.now(datetime.timezone.utc),
            }
            stmt = update(Config).where(where_clause).values(**update_values)
            db.execute(stmt)
        else:
            # Create new config - updated_at will be None for new records
            new_config = Config(name=config_type.name, **field_values)
            db.add(new_config)

    db.commit()


def get_config_by_name(db: Session, name: str) -> Config | None:
    """
    Get a config by name from the database.

    Args:
        db: Database session
        name: Config name (e.g., 'TINY', 'SMALL')

    Returns:
        Config object or None if not found
    """
    where_clause = cast(BinaryExpression[bool], Config.name == name)
    return db.query(Config).filter(where_clause).first()


def get_all_configs(db: Session) -> list[Config]:
    """
    Get all configs from the database.

    Args:
        db: Database session

    Returns:
        List of all Config objects
    """
    return db.query(Config).all()


def insert_actions(db: Session) -> None:
    """
    Insert all actions from ACTION_TO_IDX into the actions table.
    This is an append-only operation - existing actions are not updated.
    Application-level protection against updates/deletes.

    Args:
        db: Database session
    """
    for action, action_id in ACTION_TO_IDX.items():
        # Check if action already exists
        existing_action = db.query(Action).filter(Action.id == action_id).first()

        if existing_action:
            # Skip if already exists (append-only)
            continue

        # Extract action properties
        plays_card = action.plays_card
        is_legal = action.is_legal
        is_response = action.is_response
        is_draw = action.is_draw

        # Extract optional fields based on action type
        src = None
        dst = None
        card = None
        response_def_cls = None

        if hasattr(action, "src"):
            src = action.src.value if action.src else None
        if hasattr(action, "dst"):
            dst = action.dst.value if action.dst else None
        if hasattr(action, "card"):
            card = action.card.name if action.card else None
        if hasattr(action, "response_def_cls"):
            response_def_cls = action.response_def_cls.__name__ if action.response_def_cls else None

        # Create new action
        new_action = Action(
            id=action_id,
            plays_card=plays_card,
            is_legal=is_legal,
            is_response=is_response,
            is_draw=is_draw,
            src=src,
            dst=dst,
            card=card,
            response_def_cls=response_def_cls,
        )
        db.add(new_action)

    db.commit()
