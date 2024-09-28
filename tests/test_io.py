from deepdiff import DeepDiff
import pytest

from game.cards import CashCard, PropertyTypeCard, RentCard
from game.pile import CashPile, Hand, PropertyPile
from game.state import OpponentState, PlayerState, TurnState


@pytest.fixture
def turn_state():
    return TurnState(
        turn_idx=3,
        streak_idx=2,
        streaking_player_idx=1,
    )


@pytest.fixture
def player_state(hand, properties, cash):
    return PlayerState(
        hand=hand,
        properties=properties,
        cash=cash,
    )


@pytest.fixture
def opponent_state(properties, cash):
    return OpponentState(
        properties=properties,
        cash=cash,
    )


@pytest.fixture
def hand():
    return Hand([PropertyTypeCard.BROWN, CashCard.ONE, RentCard.GREEN])


@pytest.fixture
def properties():
    return PropertyPile([PropertyTypeCard.BROWN, PropertyTypeCard.GREEN])


@pytest.fixture
def cash():
    return CashPile([CashCard.TWO, PropertyTypeCard.BROWN, PropertyTypeCard.BROWN])


@pytest.fixture
def serializable(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    "serializable",
    ["turn_state", "player_state", "opponent_state"],
    indirect=True,
)
def test_serializable_io(serializable):
    assert not DeepDiff(serializable, serializable.from_json(serializable.to_json()), ignore_order=True)
