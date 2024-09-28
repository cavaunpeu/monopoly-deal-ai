from functools import lru_cache
from typing import Generator

from sqlalchemy.orm import Session

from app.database import get_db_session
from app.game.cache import GameCache
from app.game.service import GameService
from game.config import GameConfig
from models.cfr.selector import CFRActionSelector


@lru_cache(maxsize=1)
def get_selector_and_config() -> tuple[CFRActionSelector, GameConfig, int]:
    # Get the first available model from models.yaml
    models = GameService.get_loaded_models()

    if not models:
        raise RuntimeError("No models available in models.yaml")

    # Use the first model
    model_name = list(models.keys())[0]
    cfr = models[model_name]

    # Get selector
    return (
        CFRActionSelector(policy_manager=cfr.policy_manager.get_runtime_policy_manager(cfr.target_player_index)),
        cfr.game_config,
        cfr.target_player_index,
    )


@lru_cache(maxsize=1)
def get_game_service() -> GameService:
    selector, config, target_player_index = get_selector_and_config()
    return GameService(
        cache=GameCache(),
        selector=selector,
        config=config,
        target_player_index=target_player_index,
        db=None,  # Will be injected per request
    )


def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session"""
    db = get_db_session()
    try:
        yield db
    finally:
        db.close()
