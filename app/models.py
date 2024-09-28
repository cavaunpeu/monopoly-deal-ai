from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlmodel import JSON, Column, Field, SQLModel


class ConfigBase(SQLModel):
    """Base model for Config with shared fields"""

    cash_card_values: List[int] = Field(default_factory=list, sa_column=Column(JSON))
    rent_cards_per_property_type: int
    required_property_sets: int
    deck_size_multiplier: int
    initial_hand_size: int
    new_cards_per_turn: int
    max_consecutive_player_actions: int


class Config(ConfigBase, table=True):
    """Config model for database table"""

    __tablename__ = "configs"  # type: ignore

    name: str = Field(primary_key=True, max_length=32)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


class GameBase(SQLModel):
    """Base model for Game with shared fields"""

    init_player_index: Optional[int] = None
    abstraction_cls: Optional[str] = None
    resolver_cls: Optional[str] = None
    random_seed: int


class Game(GameBase, table=True):
    """Game model for database table"""

    __tablename__ = "games"  # type: ignore

    id: str = Field(primary_key=True, max_length=32)
    config_name: str = Field(foreign_key="configs.name")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Action(SQLModel, table=True):
    """Action model for database table - append-only reference data"""

    __tablename__ = "actions"  # type: ignore

    id: int = Field(primary_key=True)
    plays_card: bool
    is_legal: bool
    is_response: bool
    is_draw: bool
    # Optional fields for actions that have them
    src: Optional[str] = Field(default=None, max_length=20)  # Pile enum value
    dst: Optional[str] = Field(default=None, max_length=20)  # Pile enum value
    card: Optional[str] = Field(default=None, max_length=50)  # Card enum value
    response_def_cls: Optional[str] = Field(default=None, max_length=100)  # Response definition class
    created_at: datetime = Field(default_factory=datetime.utcnow)
