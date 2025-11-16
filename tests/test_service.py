import random

import pytest

from app.game.cache import GameCache
from app.game.service import GameService
from db.utils import upsert_game_config_types
from game.config import GameConfig
from models.cfr.cfr import CFR
from models.cfr.selector import CFRActionSelector


@pytest.fixture(scope="function")
def cache():
    return GameCache()


@pytest.fixture(scope="function")
def cfr_action_selector_and_game_config() -> tuple[CFRActionSelector, GameConfig, CFR]:
    cfr = CFR.from_checkpoint(
        "tests/fixtures/tiny-checkpoint.json",
    )
    return (
        CFRActionSelector(
            policy_manager=cfr.policy_manager.get_runtime_policy_manager(cfr.target_player_index),
        ),
        cfr.game_config,
        cfr,
    )


@pytest.fixture(scope="function")
def service(cache, cfr_action_selector_and_game_config):
    selector, config, cfr = cfr_action_selector_and_game_config
    # Register the model in GameService for the test
    model_name = "test-model"
    GameService.register_model_for_testing(model_name, cfr)
    return GameService(cache=cache, selector=selector, config=config, target_player_index=0, model_name=model_name)


def test_create_game(service):
    _ = service.create_game()


def test_get_game_state(service):
    # Create game
    game_id = service.create_game()

    # Get game state
    _ = service.get_game_state(game_id)


def test_get_player_actions(service):
    # Create game
    game_id = service.create_game()

    # Get player actions
    _ = service.get_player_actions(game_id)


def test_take_game_step(service):
    # Create game
    game_id = service.create_game()

    # Get player actions
    actions = service.get_player_actions(game_id)

    # Take action
    service.take_game_step(game_id, actions[0])


def test_get_player_states(service):
    # Create game
    game_id = service.create_game()

    # Get human state
    _ = service.get_human_state(game_id)

    # Get bot state
    _ = service.get_bot_state(game_id)


def test_game_flow(service):
    actions_taken = 0
    game_id = service.create_game()
    while actions_taken < 25 and not service.game_is_over(game_id):
        # Select action
        if service.bot_is_acting_player(game_id):
            action = service.select_bot_action(game_id)
        else:
            action = random.choice(service.get_player_actions(game_id))
        # Take action
        service.take_game_step(game_id, action)
        actions_taken += 1


def test_cache_consistency_multiple_operations(service):
    """Test that cache maintains consistency with multiple operations"""
    game_id = service.create_game()

    # Multiple operations on same game should be consistent
    state1 = service.get_human_state(game_id)
    state2 = service.get_human_state(game_id)

    # States should be identical (no side effects from get_player_actions)
    assert len(state1.hand.cards) == len(state2.hand.cards), "Hand size should be consistent"
    assert len(state1.properties.cards) == len(state2.properties.cards), "Properties should be consistent"
    assert len(state1.cash.cards) == len(state2.cash.cards), "Cash should be consistent"

    # Test that we can get bot state consistently too
    bot_state1, hand_count1, _ = service.get_bot_state(game_id)
    bot_state2, hand_count2, _ = service.get_bot_state(game_id)

    assert hand_count1 == hand_count2, "Bot hand count should be consistent"
    assert len(bot_state1.properties.cards) == len(bot_state2.properties.cards), "Bot properties should be consistent"
    assert len(bot_state1.cash.cards) == len(bot_state2.cash.cards), "Bot cash should be consistent"


def test_rehypothecate_game_via_service(test_db_session):
    """Test rehypothecating a game via the service with database persistence"""
    # Set random seed for reproducibility
    random.seed(0)

    # 1. Set up a new DB and insert configs and actions
    upsert_game_config_types(test_db_session)
    from db.utils import insert_actions

    insert_actions(test_db_session)

    # 2. Create first service instance with database
    cfr = CFR.from_checkpoint("tests/fixtures/tiny-checkpoint.json")
    selector = CFRActionSelector(
        policy_manager=cfr.policy_manager.get_runtime_policy_manager(cfr.target_player_index),
    )
    config = cfr.game_config

    # Register the model for the test
    model_name_1 = "test-model-1"
    GameService.register_model_for_testing(model_name_1, cfr)

    service1 = GameService(
        model_name=model_name_1,
        cache=GameCache(),
        selector=selector,
        config=config,
        target_player_index=0,
        db=test_db_session,
    )

    # 3. Create game and play it (reimplementing test_rehypothecate_game_from_player_actions)
    game_id = service1.create_game()
    total_turns = 0

    # Play the game until it's over or we hit the turn limit
    while not service1.game_is_over(game_id) and total_turns < 25:
        if service1.bot_is_acting_player(game_id):
            # Bot's turn - use the selector
            action = service1.select_bot_action(game_id)
        else:
            # Human's turn - select random action
            actions = service1.get_player_actions(game_id)
            action = random.choice(actions)

        # Take the action
        service1.take_game_step(game_id, action)
        total_turns += 1

    # 4. Get the final game state from service1
    service1_state = service1.get_game_state(game_id)

    # 5. Create a new service instance (simulating service restart)
    model_name_2 = "test-model-2"
    GameService.register_model_for_testing(model_name_2, cfr)
    service2 = GameService(
        cache=GameCache(),  # Fresh cache
        selector=selector,
        config=config,
        target_player_index=0,
        model_name=model_name_2,
        db=test_db_session,  # Same database
    )

    # 6. Try to get the game state from service2 using the same game_id
    service2_state = service2.get_game_state(game_id)
    assert service2_state == service1_state, "Game states should match after service restart"
