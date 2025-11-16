from typing import Generator

from sqlalchemy.orm import Session

from app.database import get_db_session
from app.game.cache import GameCache
from app.game.service import GameService
from game.config import GameConfig
from models.selector import BaseActionSelector


# Shared cache instance across all game services
_shared_cache = GameCache()


def get_selector_and_config(model_name: str | None = None) -> tuple[BaseActionSelector, GameConfig, int]:
    """Get selector and config for a specific model name, or the first model if None.

    Args:
        model_name: Name of the model to use. If None, uses the first model.

    Returns:
        Tuple of (selector, config, target_player_index)
    """
    # Get model manifest to check available models
    manifest = GameService.get_model_manifest()
    available_models = manifest.get("models", {})

    if not available_models:
        raise RuntimeError("No models available in models.yaml")

    # Use specified model or default to first
    selected_model_name: str
    if model_name is None:
        selected_model_name = GameService.get_default_model_name()
    elif model_name not in available_models:
        # Model not in models.yaml
        raise ValueError(
            f"Model '{model_name}' not found in models.yaml. Available models: {list(available_models.keys())}"
        )
    else:
        selected_model_name = model_name

    # Load the model on demand (will be cached)
    try:
        model = GameService._load_model(selected_model_name)
    except ValueError as e:
        # Model failed to load
        raise ValueError(f"Model '{selected_model_name}' failed to load: {str(e)}")

    # Use uniform interface - all models implement SelectorModel protocol
    selector = model.create_selector()
    config = model.game_config
    target_player_index = model.target_player_index

    return (selector, config, target_player_index)


def get_game_service(model_name: str | None = None) -> GameService:
    """Get game service for a specific model, or default model if None.

    Checks for FastAPI dependency overrides first (for testing), then creates
    a service with the specified model_name.
    """
    # Check for dependency override (used in tests)
    try:
        from app.main import app

        if hasattr(app, "dependency_overrides") and get_game_service in app.dependency_overrides:
            # Return the overridden service (tests provide a mock)
            return app.dependency_overrides[get_game_service]()
    except (ImportError, AttributeError):
        # Not in FastAPI context, proceed normally
        pass

    # Create service with specified model
    selector, config, target_player_index = get_selector_and_config(model_name)
    # If model_name was None, get_selector_and_config will have selected the first model
    selected_model_name: str
    if model_name is None:
        selected_model_name = GameService.get_default_model_name()
    else:
        selected_model_name = model_name
    return GameService(
        cache=_shared_cache,
        selector=selector,
        config=config,
        target_player_index=target_player_index,
        model_name=selected_model_name,
        db=None,  # Will be injected per request
    )


def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session"""
    db = get_db_session()
    try:
        yield db
    finally:
        db.close()
