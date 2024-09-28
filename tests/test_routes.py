from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from app.dependencies import get_db, get_game_service
from app.game.service import GameService
from app.main import app
from game.action import BaseAction
from game.pile import Hand, Pile
from game.state import OpponentState, PlayerState


@pytest.fixture(autouse=True)
def setup_env():
    """Set up environment variables for testing"""


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_service():
    """Create mock game service"""
    return Mock(spec=GameService)


class TestGameRoutes:
    """Route tests focusing on critical API logic and edge cases"""

    @pytest.fixture
    def mock_service(self):
        """Mock game service for testing"""
        service = Mock(spec=GameService)

        # Default return values
        service.create_game.return_value = "test-game-id"

        # Mock human state
        mock_hand = Mock(spec=Hand)
        mock_hand.cards = []
        mock_properties = Mock(spec=Pile)
        mock_properties.cards = []
        mock_cash = Mock(spec=Pile)
        mock_cash.cards = []

        mock_human_state = Mock(spec=PlayerState)
        mock_human_state.hand = mock_hand
        mock_human_state.properties = mock_properties
        mock_human_state.cash = mock_cash
        service.get_human_state.return_value = mock_human_state

        # Mock bot state
        mock_bot_properties = Mock(spec=Pile)
        mock_bot_properties.cards = []
        mock_bot_cash = Mock(spec=Pile)
        mock_bot_cash.cards = []

        mock_bot_state = Mock(spec=OpponentState)
        mock_bot_state.properties = mock_bot_properties
        mock_bot_state.cash = mock_bot_cash
        service.get_bot_state.return_value = (mock_bot_state, 5, None)

        # Mock actions
        service.get_player_actions.return_value = []
        service.bot_is_acting_player.return_value = False
        service.game_is_over.return_value = False

        # Mock public pile sizes
        mock_pile_sizes = Mock()
        mock_pile_sizes.deck = 50
        mock_pile_sizes.discard = 0
        service.get_public_pile_sizes.return_value = mock_pile_sizes

        # Mock enhanced turn state
        service.get_enhanced_turn_state.return_value = {
            "turn_idx": 0,
            "streak_idx": 0,
            "streaking_player_idx": 0,
            "acting_player_idx": 0,
            "is_human_turn": True,
            "human_player_index": 0,
            "cards_played_this_turn": 0,
            "max_cards_per_turn": 3,
            "remaining_cards": 3,
            "selected_actions": [],
            "game_over": False,
            "winner": None,
            "is_responding": False,
            "response_info": None,
        }

        # Mock AI action
        mock_ai_action = Mock(spec=BaseAction)
        mock_ai_action.encode.return_value = 67890
        service.select_bot_action.return_value = mock_ai_action

        # Mock game config
        service.get_game_config.return_value = Mock(
            required_property_sets=3,
            max_consecutive_player_actions=3,
            cash_card_values=[1, 2, 3, 4, 5],
            rent_cards_per_property_type=2,
            deck_size_multiplier=1,
            get_total_deck_size=lambda: 100,
            initial_hand_size=5,
            new_cards_per_turn=2,
            card_to_special_card_ratio=0.1,
        )
        service.get_property_types.return_value = []

        return service

    def test_complete_game_flow_via_api(self, client, mock_service):
        """Test complete game flow through API endpoints"""
        # Mock the database dependency
        mock_db = Mock()
        app.dependency_overrides[get_db] = lambda: mock_db
        # Mock the game service dependency
        app.dependency_overrides[get_game_service] = lambda: mock_service
        try:
            # Create game
            response = client.post("/game/")
            assert response.status_code == 200
            data = response.json()
            assert "game_id" in data
            assert data["game_id"] == "test-game-id"
            mock_service.create_game.assert_called_once()

            # Get initial game state (unified endpoint)
            response = client.get("/game/test-game-id/state")
            assert response.status_code == 200
            data = response.json()
            assert "human" in data
            assert "hand" in data["human"]
            assert "properties" in data["human"]
            assert "cash" in data["human"]
            assert "actions" in data
            assert isinstance(data["actions"], list)
            mock_service.get_human_state.assert_called_with("test-game-id")
            mock_service.get_player_actions.assert_called_with("test-game-id")

            # Check if bot is acting
            response = client.get("/game/test-game-id/bot_is_acting")
            assert response.status_code == 200
            assert response.json() is False
            mock_service.bot_is_acting_player.assert_called_with("test-game-id")

            # Check game over status
            response = client.get("/game/test-game-id/over")
            assert response.status_code == 200
            data = response.json()
            assert "over" in data
            assert data["over"] is False
            mock_service.game_is_over.assert_called_with("test-game-id")
        finally:
            app.dependency_overrides.clear()

    def test_service_errors_propagate_to_api(self, client):
        """Test that service errors properly become HTTP errors"""
        mock_service = Mock(spec=GameService)
        mock_service.get_human_state.side_effect = ValueError("Game test-id not found")

        app.dependency_overrides[get_game_service] = lambda: mock_service
        try:
            # The error should be raised and handled by FastAPI
            with pytest.raises(Exception):  # Any exception is fine, we're testing error propagation
                client.get("/game/test-id/state")
        finally:
            app.dependency_overrides.clear()


class TestDatabasePersistence:
    """Test database persistence and server restart resilience"""

    def test_game_state_persistence_across_requests(self, client, test_db_session):
        """Test that game state persists across multiple API requests"""
        # Set up real database and service
        app.dependency_overrides[get_db] = lambda: test_db_session

        try:
            # Create a game
            response = client.post("/game/")
            assert response.status_code == 200
            game_id = response.json()["game_id"]

            # Get initial game state
            response = client.get(f"/game/{game_id}/state")
            assert response.status_code == 200
            initial_hand = response.json()["human"]["hand"]
            initial_hand_count = len(initial_hand)
            actions = response.json()["actions"]

            # Find a card-playing action (not a pass action)
            card_action = None
            for action in actions:
                # Pass actions have no card and is_response=False
                if action["card"] is not None:
                    card_action = action
                    break

            if card_action:
                # Take the action
                response = client.post(f"/game/{game_id}/step", json={"action_id": card_action["id"]})
                assert response.status_code == 200

                # Get human state again - hand should have one fewer card
                response = client.get(f"/game/{game_id}/state")
                assert response.status_code == 200
                new_hand = response.json()["human"]["hand"]
                assert len(new_hand) == initial_hand_count - 1

            # Get turn state - should show selected actions
            response = client.get(f"/game/{game_id}/turn_state")
            assert response.status_code == 200
            turn_state = response.json()
            assert "selected_actions" in turn_state
            assert len(turn_state["selected_actions"]) == 1  # One action taken

        finally:
            app.dependency_overrides.clear()

    def test_game_recreation_after_server_restart(self, client, test_db_session):
        """Test that games can be recreated from database after service restart"""
        # Set up real database and service
        app.dependency_overrides[get_db] = lambda: test_db_session

        try:
            # Create a game and play some moves
            response = client.post("/game/")
            assert response.status_code == 200
            game_id = response.json()["game_id"]

            # Play a few moves
            for _ in range(3):
                response = client.get(f"/game/{game_id}/state")
                if response.status_code == 200:
                    actions = response.json()["actions"]
                    if actions:
                        # Take the first available action
                        action = actions[0]
                        response = client.post(
                            f"/game/{game_id}/step",
                            json={"action_id": action["id"]},
                        )
                        if response.status_code != 200:
                            break  # Game might be over or bot's turn

                # Get the current game state
                response = client.get(f"/game/{game_id}/state")
                assert response.status_code == 200
                original_state = response.json()["human"]

            response = client.get(f"/game/{game_id}/turn_state")
            assert response.status_code == 200
            original_turn_state = response.json()

            # Simulate server restart by clearing the service cache
            # (In real scenario, this would happen when the server restarts)
            service = app.dependency_overrides.get(get_game_service, get_game_service)()
            if hasattr(service, "cache"):
                service.cache = type(service.cache)()  # Create new empty cache

            # Now try to access the same game - it should be recreated from database
            response = client.get(f"/game/{game_id}/state")
            assert response.status_code == 200
            recreated_state = response.json()["human"]

            # States should be identical
            assert recreated_state["hand"] == original_state["hand"]
            assert recreated_state["properties"] == original_state["properties"]
            assert recreated_state["cash"] == original_state["cash"]

            # Turn state should also be recreated correctly
            response = client.get(f"/game/{game_id}/turn_state")
            assert response.status_code == 200
            recreated_turn_state = response.json()

            # Key turn state fields should match
            assert recreated_turn_state["turn_idx"] == original_turn_state["turn_idx"]
            assert recreated_turn_state["streak_idx"] == original_turn_state["streak_idx"]
            assert recreated_turn_state["streaking_player_idx"] == original_turn_state["streaking_player_idx"]
            assert recreated_turn_state["acting_player_idx"] == original_turn_state["acting_player_idx"]

        finally:
            app.dependency_overrides.clear()

    def test_played_cards_persistence(self, client, test_db_session):
        """Test that played cards are properly persisted and restored"""
        # Set up real database and service
        app.dependency_overrides[get_db] = lambda: test_db_session

        try:
            # Create a game
            response = client.post("/game/")
            assert response.status_code == 200
            game_id = response.json()["game_id"]

            # Play a card action
            response = client.get(f"/game/{game_id}/state")
            assert response.status_code == 200
            actions = response.json()["actions"]

            # Find a card-playing action (not a pass action)
            card_action = None
            for action in actions:
                # Pass actions have no card and is_response=False
                if action["card"] is not None:
                    card_action = action
                    break

            if card_action:
                # Take the action
                response = client.post(f"/game/{game_id}/step", json={"action_id": card_action["id"]})
                assert response.status_code == 200

                # Get turn state - should show played cards
                response = client.get(f"/game/{game_id}/turn_state")
                assert response.status_code == 200
                turn_state = response.json()
                original_selected_actions = turn_state["selected_actions"]
                assert len(original_selected_actions) == 1

                # Simulate server restart
                service = app.dependency_overrides.get(get_game_service, get_game_service)()
                if hasattr(service, "cache"):
                    service.cache = type(service.cache)()

                # Get turn state again - selected actions should still be there
                response = client.get(f"/game/{game_id}/turn_state")
                assert response.status_code == 200
                recreated_turn_state = response.json()
                recreated_selected_actions = recreated_turn_state["selected_actions"]

                # Selected actions should be identical
                assert len(recreated_selected_actions) == len(original_selected_actions)
                assert recreated_selected_actions == original_selected_actions

        finally:
            app.dependency_overrides.clear()

    def test_multiple_games_persistence(self, client, test_db_session):
        """Test that multiple games can coexist and be recreated independently"""
        # Set up real database and service
        app.dependency_overrides[get_db] = lambda: test_db_session

        try:
            # Create two games
            response1 = client.post("/game/")
            assert response1.status_code == 200
            game_id1 = response1.json()["game_id"]

            response2 = client.post("/game/")
            assert response2.status_code == 200
            game_id2 = response2.json()["game_id"]

            assert game_id1 != game_id2

            # Play different moves in each game
            # Game 1: play one action
            response = client.get(f"/game/{game_id1}/state")
            if response.status_code == 200:
                actions = response.json()["actions"]
                if actions:
                    response = client.post(
                        f"/game/{game_id1}/step",
                        json={"action_id": actions[0]["id"]},
                    )

            # Game 2: play two actions
            for _ in range(2):
                response = client.get(f"/game/{game_id2}/state")
                if response.status_code == 200:
                    actions = response.json()["actions"]
                    if actions:
                        response = client.post(
                            f"/game/{game_id2}/step",
                            json={"action_id": actions[0]["id"]},
                        )
                        if response.status_code != 200:
                            break

            # Get states of both games
            response1 = client.get(f"/game/{game_id1}/turn_state")
            response2 = client.get(f"/game/{game_id2}/turn_state")

            assert response1.status_code == 200
            assert response2.status_code == 200

            state1 = response1.json()
            state2 = response2.json()

            # Simulate server restart
            service = app.dependency_overrides.get(get_game_service, get_game_service)()
            if hasattr(service, "cache"):
                service.cache = type(service.cache)()

            # Recreate both games
            response1 = client.get(f"/game/{game_id1}/turn_state")
            response2 = client.get(f"/game/{game_id2}/turn_state")

            assert response1.status_code == 200
            assert response2.status_code == 200

            recreated_state1 = response1.json()
            recreated_state2 = response2.json()

            # Each game should maintain its own state
            assert recreated_state1["turn_idx"] == state1["turn_idx"]
            assert recreated_state1["streak_idx"] == state1["streak_idx"]
            assert recreated_state2["turn_idx"] == state2["turn_idx"]
            assert recreated_state2["streak_idx"] == state2["streak_idx"]

            # Games should be independent
            assert (
                recreated_state1["turn_idx"] != recreated_state2["turn_idx"]
                or recreated_state1["streak_idx"] != recreated_state2["streak_idx"]
            )

        finally:
            app.dependency_overrides.clear()

    def test_game_not_found_after_restart_without_db(self, client):
        """Test that games not in database raise appropriate errors"""
        # Use a mock service without database
        mock_service = Mock(spec=GameService)
        mock_service.get_human_state.side_effect = ValueError(
            "No database connection found. Could not recreate game test-id."
        )

        app.dependency_overrides[get_game_service] = lambda: mock_service

        try:
            # Try to access a non-existent game
            # The error should be handled by FastAPI and return a 404 response
            response = client.get("/game/test-id/state")
            assert response.status_code == 404, f"Expected 404 status code, got {response.status_code}"
            assert "Game not found" in response.json()["detail"], "Expected 'Game not found' in error detail"
        finally:
            app.dependency_overrides.clear()

    def test_unified_game_state_endpoint(self, client):
        """Test the new unified game state endpoint"""
        # Create a fresh mock service
        mock_service = Mock(spec=GameService)

        # Mock the database dependency
        mock_db = Mock()
        app.dependency_overrides[get_db] = lambda: mock_db
        # Mock the game service dependency
        app.dependency_overrides[get_game_service] = lambda: mock_service

        # Set up all the required mocks
        # Mock human state
        mock_hand = Mock(spec=Hand)
        mock_hand.cards = []
        mock_properties = Mock(spec=Pile)
        mock_properties.cards = []
        mock_cash = Mock(spec=Pile)
        mock_cash.cards = []

        mock_human_state = Mock(spec=PlayerState)
        mock_human_state.hand = mock_hand
        mock_human_state.properties = mock_properties
        mock_human_state.cash = mock_cash
        mock_service.get_human_state.return_value = mock_human_state

        # Mock bot state
        mock_bot_properties = Mock(spec=Pile)
        mock_bot_properties.cards = []
        mock_bot_cash = Mock(spec=Pile)
        mock_bot_cash.cards = []

        mock_bot_state = Mock(spec=OpponentState)
        mock_bot_state.properties = mock_bot_properties
        mock_bot_state.cash = mock_bot_cash
        mock_service.get_bot_state.return_value = (mock_bot_state, 5, None)

        # Mock other required methods
        mock_service.get_player_actions.return_value = []
        mock_service.get_public_pile_sizes.return_value = Mock(deck=50, discard=0)
        mock_service.get_enhanced_turn_state.return_value = {
            "turn_idx": 0,
            "streak_idx": 0,
            "streaking_player_idx": 0,
            "acting_player_idx": 0,
            "is_human_turn": True,
            "human_player_index": 0,
            "cards_played_this_turn": 0,
            "max_cards_per_turn": 3,
            "remaining_cards": 3,
            "selected_actions": [],
            "game_over": False,
            "winner": None,
            "is_responding": False,
            "response_info": None,
        }

        # Mock game config
        mock_service.get_game_config.return_value = Mock(
            required_property_sets=3,
            max_consecutive_player_actions=3,
            cash_card_values=[1, 2, 3, 4, 5],
            rent_cards_per_property_type=2,
            deck_size_multiplier=1,
            get_total_deck_size=lambda: 100,
            initial_hand_size=5,
            new_cards_per_turn=2,
            card_to_special_card_ratio=0.1,
        )
        mock_service.get_property_types.return_value = []

        try:
            # Test unified endpoint
            response = client.get("/game/test-game-id/state")
            assert response.status_code == 200
            data = response.json()

            # Check all required sections are present
            assert "turn" in data
            assert "human" in data
            assert "ai" in data
            assert "piles" in data
            assert "config" in data
            assert "actions" in data

            # Check turn state structure
            turn = data["turn"]
            assert "turn_idx" in turn
            assert "is_human_turn" in turn
            assert "selected_actions" in turn

            # Check human state structure
            human = data["human"]
            assert "hand" in human
            assert "properties" in human
            assert "cash" in human

            # Check AI state structure
            ai = data["ai"]
            assert "properties" in ai
            assert "cash" in ai
            assert "hand_count" in ai

            # Check piles structure
            piles = data["piles"]
            assert "deck" in piles
            assert "discard" in piles

            # Check config structure
            config = data["config"]
            assert "required_property_sets" in config

            # Check actions structure
            actions = data["actions"]
            assert isinstance(actions, list)

            # Verify service methods were called
            mock_service.get_enhanced_turn_state.assert_called_with("test-game-id")
            mock_service.get_human_state.assert_called_with("test-game-id")
            mock_service.get_bot_state.assert_called_with("test-game-id", show_hand=False)
            mock_service.get_public_pile_sizes.assert_called_with("test-game-id")
            mock_service.get_game_config.assert_called_with()
            mock_service.get_player_actions.assert_called_with("test-game-id")

        finally:
            app.dependency_overrides.clear()

    def test_unified_game_state_with_ai_hand(self, client):
        """Test the unified endpoint with show_ai_hand parameter"""
        # Create a fresh mock service
        mock_service = Mock(spec=GameService)

        # Mock the database dependency
        mock_db = Mock()
        app.dependency_overrides[get_db] = lambda: mock_db

        # Mock the game service dependency
        app.dependency_overrides[get_game_service] = lambda: mock_service

        # Set up required mocks
        mock_hand = Mock(spec=Hand)
        mock_hand.cards = []
        mock_properties = Mock(spec=Pile)
        mock_properties.cards = []
        mock_cash = Mock(spec=Pile)
        mock_cash.cards = []

        mock_human_state = Mock(spec=PlayerState)
        mock_human_state.hand = mock_hand
        mock_human_state.properties = mock_properties
        mock_human_state.cash = mock_cash
        mock_service.get_human_state.return_value = mock_human_state

        # Mock bot state
        mock_bot_properties = Mock(spec=Pile)
        mock_bot_properties.cards = []
        mock_bot_cash = Mock(spec=Pile)
        mock_bot_cash.cards = []

        mock_bot_state = Mock(spec=OpponentState)
        mock_bot_state.properties = mock_bot_properties
        mock_bot_state.cash = mock_bot_cash
        mock_service.get_bot_state.return_value = (mock_bot_state, 5, None)

        # Mock other required methods
        mock_service.get_player_actions.return_value = []
        mock_service.get_public_pile_sizes.return_value = Mock(deck=50, discard=0)
        mock_service.get_enhanced_turn_state.return_value = {
            "turn_idx": 0,
            "streak_idx": 0,
            "streaking_player_idx": 0,
            "acting_player_idx": 0,
            "is_human_turn": True,
            "human_player_index": 0,
            "cards_played_this_turn": 0,
            "max_cards_per_turn": 3,
            "remaining_cards": 3,
            "selected_actions": [],
            "game_over": False,
            "winner": None,
            "is_responding": False,
            "response_info": None,
        }

        # Mock game config
        mock_service.get_game_config.return_value = Mock(
            required_property_sets=3,
            max_consecutive_player_actions=3,
            cash_card_values=[1, 2, 3, 4, 5],
            rent_cards_per_property_type=2,
            deck_size_multiplier=1,
            get_total_deck_size=lambda: 100,
            initial_hand_size=5,
            new_cards_per_turn=2,
            card_to_special_card_ratio=0.1,
        )
        mock_service.get_property_types.return_value = []

        try:
            # Test unified endpoint with show_ai_hand=true
            response = client.get("/game/test-game-id/state?show_ai_hand=true")
            assert response.status_code == 200
            _ = response.json()

            # Verify service was called with show_hand=True
            mock_service.get_bot_state.assert_called_with("test-game-id", show_hand=True)

        finally:
            app.dependency_overrides.clear()
