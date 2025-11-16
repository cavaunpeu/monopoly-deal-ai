import random

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


def create_small_game():
    return Game(
        config=GameConfigType.SMALL.value,
        init_player_index=0,
        player_specs=_default_player_specs(),
        random_seed=0,
    )


def test_rehypothecate_game_from_player_actions():
    # Set random seed
    random.seed(0)

    # Create game
    game1 = create_small_game()

    # Store actions
    actions_taken = []

    while not game1.over and game1.turn_state.turn_idx < 25:
        # Get player actions
        actions = game1.state.get_player_actions()
        # Select random action
        action = random.choice(actions).action
        # Take game step
        game1.step(selected_action=action)
        # Store action
        actions_taken.append(action)

    # Create new game
    game2 = create_small_game()

    # Take actions
    for action in actions_taken:
        game2.step(selected_action=action)

    # Assert games are equal
    assert game1.state == game2.state


def test_clone():
    """Test that the clone method correctly re-applies selected actions."""
    random.seed(42)

    # Create original game
    original_game = create_small_game()

    # Play ~10 actions
    action_count = 0
    while not original_game.over and action_count < 10:
        # Get player actions
        actions = original_game.state.get_player_actions()
        # Select random action
        action = random.choice(actions).action
        # Take game step
        original_game.step(selected_action=action)
        action_count += 1

    # Clone the game
    cloned_game = original_game.clone()

    # Assert that both games have identical states
    assert original_game.state == cloned_game.state


def test_game_initialization():
    """Test that game initialization correctly distributes cards according to config."""
    # Test with medium config
    game = Game(
        config=GameConfigType.MEDIUM.value,
        init_player_index=0,
        player_specs=_default_player_specs(),
        random_seed=42,
    )

    # Check initial hand sizes
    player_0_hand_size = len(game.players[0].hand)
    player_1_hand_size = len(game.players[1].hand)

    # With medium config: initial_hand_size=5, new_cards_per_turn=2
    # Player 0 (init player) should have: initial_hand_size + new_cards_per_turn = 5 + 2 = 7
    # Player 1 should have: initial_hand_size = 5
    expected_player_0_hand = (
        GameConfigType.MEDIUM.value.initial_hand_size + GameConfigType.MEDIUM.value.new_cards_per_turn
    )
    expected_player_1_hand = GameConfigType.MEDIUM.value.initial_hand_size

    print(f"Player 0 hand size: {player_0_hand_size} (expected: {expected_player_0_hand})")
    print(f"Player 1 hand size: {player_1_hand_size} (expected: {expected_player_1_hand})")

    assert player_0_hand_size == expected_player_0_hand, (
        f"Player 0 should have {expected_player_0_hand} cards, got {player_0_hand_size}"
    )
    assert player_1_hand_size == expected_player_1_hand, (
        f"Player 1 should have {expected_player_1_hand} cards, got {player_1_hand_size}"
    )

    # Check that the init player is correct
    assert game.init_player_index == 0, f"Init player should be 0, got {game.init_player_index}"

    # Check that the current player is the init player
    assert game.player.index == 0, f"Current player should be 0, got {game.player.index}"
