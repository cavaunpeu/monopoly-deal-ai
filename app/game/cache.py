from typing import Optional

from game.game import Game


class GameCache:
    def __init__(self):
        self._cache: dict[str, Game] = {}

    def get(self, game_id: str) -> Optional[Game]:
        return self._cache.get(game_id)

    def set(self, game_id: str, game: Game) -> None:
        self._cache[game_id] = game

    def delete(self, game_id: str) -> None:
        self._cache.pop(game_id, None)
