from functools import lru_cache
import os

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlmodel import SQLModel


@lru_cache(maxsize=1)
def get_database_url() -> str:
    """Get database URL from environment variable"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    # Convert postgres:// to postgresql:// for SQLAlchemy compatibility
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    return database_url


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create and return SQLAlchemy engine"""
    database_url = get_database_url()
    return create_engine(database_url)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker:
    """Create and return session factory"""
    engine = get_engine()
    return sessionmaker(bind=engine)


def get_db_session() -> Session:
    """Get a database session"""
    session_factory = get_session_factory()
    return session_factory()


def create_tables() -> None:
    """Create all tables defined in models"""
    engine = get_engine()
    SQLModel.metadata.create_all(bind=engine)
