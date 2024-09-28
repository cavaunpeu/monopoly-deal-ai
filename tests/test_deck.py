import pytest

from game.config import GameConfigType
from game.deck import Deck


@pytest.mark.parametrize(
    "random_seed1,random_seed2,expected_equal",
    [(0, 0, True), (0, 1, False), (1, 0, False), (1, 1, True)],
)
def test_random_seed_determinism(random_seed1, random_seed2, expected_equal):
    # Deck 1
    deck1 = Deck.build(
        game_config=GameConfigType.TINY.value,
        random_seed=random_seed1,
    )
    deck1_cards = [deck1.pick() for _ in range(len(deck1))]

    # Deck 2
    deck2 = Deck.build(
        game_config=GameConfigType.TINY.value,
        random_seed=random_seed2,
    )
    deck2_cards = [deck2.pick() for _ in range(len(deck2))]

    if expected_equal:
        assert deck1_cards == deck2_cards
    else:
        assert deck1_cards != deck2_cards
