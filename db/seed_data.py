from pathlib import Path
import sys


# Add the project root to the Python path first
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.database import get_database_url  # noqa: E402
from db.utils import insert_actions, upsert_game_config_types  # noqa: E402


def get_database_url_for_seeding():
    """Get database URL for seeding from environment."""
    try:
        database_url = get_database_url()
        # Convert postgres:// to postgresql:// for SQLAlchemy compatibility
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        return database_url
    except ValueError:
        print("❌ DATABASE_URL not set")
        raise


def seed_database():
    """Seed the database with initial data."""
    print("🌱 Seeding database...")

    database_url = get_database_url_for_seeding()
    engine = create_engine(database_url)
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as db:
        try:
            upsert_game_config_types(db)
            insert_actions(db)
            print("✅ Database seeded successfully")
        except Exception as e:
            print(f"❌ Error: {e}")
            db.rollback()
            raise


if __name__ == "__main__":
    seed_database()
