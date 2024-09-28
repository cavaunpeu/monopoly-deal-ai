import os
import tempfile
from typing import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlmodel import SQLModel

from app.db_models import Base


@pytest.fixture
def test_db_session() -> Generator[Session, None, None]:
    """Create a temporary SQLite database for testing"""
    # Create a temporary SQLite database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_path = tmp_file.name

    try:
        # Create engine and session
        engine = create_engine(f"sqlite:///{db_path}")

        # Create tables from both SQLModel and SQLAlchemy models
        SQLModel.metadata.create_all(bind=engine)
        Base.metadata.create_all(bind=engine)

        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()

        yield session

        session.close()
    finally:
        # Clean up the temporary database file
        if os.path.exists(db_path):
            os.unlink(db_path)
