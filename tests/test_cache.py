import pytest

from app.game.cache import GameCache
from game.config import GameConfigType
from game.game import Game


@pytest.fixture(scope="function")
def cache():
    return GameCache()


def test_cache(cache):
    # Create game
    game = Game(
        config=GameConfigType.TINY.value,
        init_player_index=0,
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
