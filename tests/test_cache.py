import pytest

from app.game.cache import GameCache
from game.action import GreedyActionResolver
from game.config import GameConfigType
from game.game import Game, PlayerSpec
from game.state import IntentStateAbstraction


def _default_player_specs() -> dict[int, PlayerSpec]:
    """Create default player specs for testing (same for both players)."""
    default_spec = PlayerSpec(
        abstraction_cls=IntentStateAbstraction,
        resolver_cls=GreedyActionResolver,
    )
    return {0: default_spec, 1: default_spec}


@pytest.fixture(scope="function")
def cache():
    return GameCache()


def test_cache(cache):
    # Create game
    game = Game(
        config=GameConfigType.TINY.value,
        init_player_index=0,
        player_specs=_default_player_specs(),
    )

    # Set game
    key = "my_game_id"
    cache.set(key, game)

    # Get game
    assert cache.get(key) == game

    # Delete game
    cache.delete(key)

    # Get game
    assert cache.get(key) is None
