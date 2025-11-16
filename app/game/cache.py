from typing import Optional

from game.game import Game


class GameCache:
    def __init__(self):
        self._cache: dict[str, Game] = {}

    def get(self, game_id: str) -> Optional[Game]:
        """Get a game from cache."""
        return self._cache.get(game_id)

    def set(self, game_id: str, game: Game) -> None:
        """Store a game in cache."""
        self._cache[game_id] = game

    def delete(self, game_id: str) -> None:
        """Delete a game from cache."""
        self._cache.pop(game_id, None)
