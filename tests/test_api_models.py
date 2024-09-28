"""Tests for API model serialization functions"""

from unittest.mock import Mock

from app.api_models import GameConfigResponse, TurnStateResponse, serialize_game_config, serialize_turn_state
from game.action import PassAction
from game.cards import CashCard, PropertyTypeCard


class TestSerializeGameConfig:
    """Test game config serialization"""

    def test_serialize_game_config(self):
        """Test that game config is properly serialized"""
        # Mock game config object
        mock_config = Mock()
        mock_config.required_property_sets = 3
        mock_config.max_consecutive_player_actions = 3
        mock_config.cash_card_values = [1, 2, 3, 4, 5]
        mock_config.rent_cards_per_property_type = 2
        mock_config.deck_size_multiplier = 1
        mock_config.get_total_deck_size.return_value = 100
        mock_config.initial_hand_size = 5
        mock_config.new_cards_per_turn = 2
        mock_config.card_to_special_card_ratio = 0.1

        # Mock property types
        mock_prop_type1 = Mock()
        mock_prop_type1.name = "GREEN"
        mock_prop_type1.num_to_complete = 3
        mock_prop_type1.rent_progression = [2, 4, 7]
        mock_prop_type1.cash_value = 4

        mock_prop_type2 = Mock()
        mock_prop_type2.name = "PINK"
        mock_prop_type2.num_to_complete = 2
        mock_prop_type2.rent_progression = [1, 2, 4]
        mock_prop_type2.cash_value = 2

        property_types = [mock_prop_type1, mock_prop_type2]

        # Serialize
        result = serialize_game_config(mock_config, property_types)

        # Verify result
        assert isinstance(result, GameConfigResponse)
        assert result.required_property_sets == 3
        assert result.max_consecutive_player_actions == 3
        assert result.cash_card_values == [1, 2, 3, 4, 5]
        assert result.rent_cards_per_property_type == 2
        assert result.deck_size_multiplier == 1
        assert result.total_deck_size == 100
        assert result.initial_hand_size == 5
        assert result.new_cards_per_turn == 2
        assert result.card_to_special_card_ratio == 0.1
        assert result.required_property_sets_map == {"GREEN": 3, "PINK": 2}
        assert len(result.property_types) == 2
        assert result.property_types[0].name == "GREEN"
        assert result.property_types[0].num_to_complete == 3
        assert result.property_types[0].rent_progression == [2, 4, 7]
        assert result.property_types[0].cash_value == 4
        assert result.property_types[1].name == "PINK"
        assert result.property_types[1].num_to_complete == 2
        assert result.property_types[1].rent_progression == [1, 2, 4]
        assert result.property_types[1].cash_value == 2


class TestSerializeTurnState:
    """Test turn state serialization"""

    def test_serialize_turn_state_with_selected_actions(self):
        """Test turn state serialization with selected actions"""
        # Create mock selected action
        mock_action = Mock(spec=PassAction)
        mock_action.encode.return_value = 12345
        mock_action.plays_card = False
        mock_action.is_response = False

        mock_selected_action = Mock()
        mock_selected_action.turn_idx = 0
        mock_selected_action.streak_idx = 0
        mock_selected_action.player_idx = 0
        mock_selected_action.action = mock_action

        # Mock turn state data
        turn_state_data = {
            "turn_idx": 0,
            "streak_idx": 0,
            "streaking_player_idx": 0,
            "acting_player_idx": 0,
            "is_human_turn": True,
            "human_player_index": 0,
            "cards_played_this_turn": 0,
            "max_cards_per_turn": 3,
            "remaining_cards": 3,
            "selected_actions": [mock_selected_action],
            "game_over": False,
            "winner": None,
            "is_responding": False,
            "response_info": None,
        }

        # Serialize
        result = serialize_turn_state(turn_state_data)

        # Verify result
        assert isinstance(result, TurnStateResponse)
        assert result.turn_idx == 0
        assert result.streak_idx == 0
        assert result.is_human_turn is True
        assert result.game_over is False
        assert len(result.selected_actions) == 1
        assert result.selected_actions[0].turn_idx == 0
        assert result.selected_actions[0].action.id == 12345
        assert result.selected_actions[0].action.card is None
        assert result.selected_actions[0].action.is_response is False

    def test_serialize_turn_state_with_response_info(self):
        """Test turn state serialization with response info"""
        # Create mock cards
        mock_initiating_card = Mock(spec=CashCard)
        mock_initiating_card.name = "ONE"
        mock_initiating_card.cash_value = 1

        mock_response_card = Mock(spec=PropertyTypeCard)
        mock_response_card.name = "GREEN"
        mock_response_card.rent_progression = [1, 2, 3]
        mock_response_card.cash_value = 1

        # Mock response info
        response_info = {
            "initiating_card": mock_initiating_card,
            "initiating_player": "human",
            "responding_player": "bot",
            "response_cards_played": [mock_response_card],
        }

        # Mock turn state data
        turn_state_data = {
            "turn_idx": 1,
            "streak_idx": 0,
            "streaking_player_idx": 0,
            "acting_player_idx": 1,
            "is_human_turn": False,
            "human_player_index": 0,
            "cards_played_this_turn": 1,
            "max_cards_per_turn": 3,
            "remaining_cards": 2,
            "selected_actions": [],
            "game_over": False,
            "winner": None,
            "is_responding": True,
            "response_info": response_info,
        }

        # Serialize
        result = serialize_turn_state(turn_state_data)

        # Verify result
        assert isinstance(result, TurnStateResponse)
        assert result.is_responding is True
        assert result.response_info is not None
        assert result.response_info.initiating_player == "human"
        assert result.response_info.responding_player == "bot"
        # Note: Card serialization would be tested in separate card serialization tests

    def test_serialize_turn_state_empty_selected_actions(self):
        """Test turn state serialization with no selected actions"""
        turn_state_data = {
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

        # Serialize
        result = serialize_turn_state(turn_state_data)

        # Verify result
        assert isinstance(result, TurnStateResponse)
        assert result.turn_idx == 0
        assert len(result.selected_actions) == 0
        assert result.response_info is None
