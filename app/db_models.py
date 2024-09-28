from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, TypeDecorator, func
from sqlalchemy.dialects.sqlite import JSON as SQLite_JSON
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, relationship


Base = declarative_base()


class IntegerArray(TypeDecorator[list[int]]):
    """Cross-dialect integer array type that uses JSON for both PostgreSQL and SQLite"""

    impl = String
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        # Use JSON for both PostgreSQL and SQLite to match the database schema
        return dialect.type_descriptor(SQLite_JSON())

    def process_bind_param(self, value: Optional[list[int]], dialect: Dialect) -> Any:
        if value is None:
            return value
        # Always store as JSON
        return value

    def process_result_value(self, value: Any, dialect: Dialect) -> Optional[list[int]]:
        if value is None:
            return value
        # Always return the JSON value (which is already a list)
        return value


class Config(Base):
    __tablename__ = "configs"

    name: Mapped[str] = mapped_column(String(32), primary_key=True)
    cash_card_values: Mapped[list[int]] = mapped_column(IntegerArray())
    rent_cards_per_property_type: Mapped[int] = mapped_column(Integer)
    required_property_sets: Mapped[int] = mapped_column(Integer)
    deck_size_multiplier: Mapped[int] = mapped_column(Integer)
    initial_hand_size: Mapped[int] = mapped_column(Integer)
    new_cards_per_turn: Mapped[int] = mapped_column(Integer)
    max_consecutive_player_actions: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[Any] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[Optional[Any]] = mapped_column(DateTime, default=None, onupdate=func.now(), nullable=True)

    # Relationship
    games: Mapped[list["Game"]] = relationship("Game", back_populates="config")


class Game(Base):
    __tablename__ = "games"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    config_name: Mapped[str] = mapped_column(String(32), ForeignKey("configs.name", ondelete="CASCADE"))
    init_player_index: Mapped[int] = mapped_column(Integer)
    abstraction_cls: Mapped[str] = mapped_column(String(80))
    resolver_cls: Mapped[str] = mapped_column(String(80))
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[Any] = mapped_column(DateTime, default=func.now(), nullable=False)

    # Relationship
    config: Mapped[Config] = relationship("Config", back_populates="games")
    selected_actions: Mapped[list["SelectedAction"]] = relationship("SelectedAction", back_populates="game")


class SelectedAction(Base):
    __tablename__ = "selected_actions"

    turn_idx: Mapped[int] = mapped_column(Integer, primary_key=True)
    streak_idx: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_idx: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[str] = mapped_column(String(32), ForeignKey("games.id", ondelete="CASCADE"), primary_key=True)
    action_id: Mapped[int] = mapped_column(Integer, ForeignKey("actions.id", ondelete="CASCADE"))
    created_at: Mapped[Any] = mapped_column(DateTime, default=func.now(), nullable=False, primary_key=True)

    # Relationships
    game: Mapped["Game"] = relationship("Game", back_populates="selected_actions")


class Action(Base):
    """Action model for database table - append-only reference data"""

    __tablename__ = "actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    plays_card: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_legal: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_response: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_draw: Mapped[bool] = mapped_column(Boolean, nullable=False)
    # Optional fields
    src: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # Pile enum value
    dst: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # Pile enum value
    card: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # Card enum value
    response_def_cls: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)  # Response definition class
    created_at: Mapped[Any] = mapped_column(DateTime, default=func.now(), nullable=False)
