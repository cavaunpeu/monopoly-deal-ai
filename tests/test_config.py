from db.utils import get_all_configs, get_config_by_name, upsert_game_config_types
from game.config import GameConfigType


# Test class
class TestConfigDatabase:
    """Test database operations for game configs"""

    def test_upsert_game_config_types_initial_insert(self, test_db_session):
        """Test inserting game config types for the first time"""
        # Initially, no configs should exist
        configs = get_all_configs(test_db_session)
        assert len(configs) == 0

        # Upsert the config types
        upsert_game_config_types(test_db_session)

        # Should now have all config types
        configs = get_all_configs(test_db_session)
        assert len(configs) == len(GameConfigType)

        # Verify TINY config
        tiny_config = get_config_by_name(test_db_session, "TINY")
        assert tiny_config is not None
        assert tiny_config.name == "TINY"
        assert tiny_config.cash_card_values == GameConfigType.TINY.value.cash_card_values
        assert tiny_config.rent_cards_per_property_type == GameConfigType.TINY.value.rent_cards_per_property_type
        assert tiny_config.required_property_sets == GameConfigType.TINY.value.required_property_sets
        assert tiny_config.deck_size_multiplier == GameConfigType.TINY.value.deck_size_multiplier
        assert tiny_config.initial_hand_size == GameConfigType.TINY.value.initial_hand_size
        assert tiny_config.new_cards_per_turn == GameConfigType.TINY.value.new_cards_per_turn
        assert tiny_config.max_consecutive_player_actions == GameConfigType.TINY.value.max_consecutive_player_actions

        # Verify SMALL config
        small_config = get_config_by_name(test_db_session, "SMALL")
        assert small_config is not None
        assert small_config.name == "SMALL"
        assert small_config.cash_card_values == GameConfigType.SMALL.value.cash_card_values
        assert small_config.rent_cards_per_property_type == GameConfigType.SMALL.value.rent_cards_per_property_type
        assert small_config.required_property_sets == GameConfigType.SMALL.value.required_property_sets
        assert small_config.deck_size_multiplier == GameConfigType.SMALL.value.deck_size_multiplier
        assert small_config.initial_hand_size == GameConfigType.SMALL.value.initial_hand_size
        assert small_config.new_cards_per_turn == GameConfigType.SMALL.value.new_cards_per_turn
        assert small_config.max_consecutive_player_actions == GameConfigType.SMALL.value.max_consecutive_player_actions

    def test_upsert_game_config_types_update_existing(self, test_db_session):
        """Test updating existing game config types"""
        # First, insert the configs
        upsert_game_config_types(test_db_session)

        # Verify initial state
        tiny_config = get_config_by_name(test_db_session, "TINY")
        assert tiny_config is not None
        assert tiny_config.cash_card_values == GameConfigType.TINY.value.cash_card_values

        # Manually update one config in the database
        tiny_config.cash_card_values = [999]  # Change the value
        test_db_session.commit()

        # Verify the change
        updated_config = get_config_by_name(test_db_session, "TINY")
        assert updated_config is not None
        assert updated_config.cash_card_values == [999]

        # Now upsert again - should restore original values
        upsert_game_config_types(test_db_session)

        # Verify it was restored
        restored_config = get_config_by_name(test_db_session, "TINY")
        assert restored_config is not None
        assert restored_config.cash_card_values == GameConfigType.TINY.value.cash_card_values

    def test_get_config_by_name(self, test_db_session):
        """Test getting config by name"""
        # Initially, no configs exist
        config = get_config_by_name(test_db_session, "TINY")
        assert config is None

        # Insert configs
        upsert_game_config_types(test_db_session)

        # Now should find the config
        config = get_config_by_name(test_db_session, "TINY")
        assert config is not None
        assert config.name == "TINY"

        # Non-existent config should return None
        config = get_config_by_name(test_db_session, "NONEXISTENT")
        assert config is None

    def test_get_all_configs(self, test_db_session):
        """Test getting all configs"""
        # Initially, no configs
        configs = get_all_configs(test_db_session)
        assert len(configs) == 0

        # Insert configs
        upsert_game_config_types(test_db_session)

        # Should have all config types
        configs = get_all_configs(test_db_session)
        assert len(configs) == len(GameConfigType)

        # Verify all expected configs are present
        config_names = {config.name for config in configs}
        expected_names = {config_type.name for config_type in GameConfigType}
        assert config_names == expected_names

    def test_config_data_matches_enum_values(self, test_db_session):
        """Test that database configs match the enum values exactly"""
        # Insert configs
        upsert_game_config_types(test_db_session)

        # Verify each config matches its enum value
        for config_type in GameConfigType:
            db_config = get_config_by_name(test_db_session, config_type.name)
            enum_config = config_type.value

            assert db_config is not None
            assert db_config.cash_card_values == enum_config.cash_card_values
            assert db_config.rent_cards_per_property_type == enum_config.rent_cards_per_property_type
            assert db_config.required_property_sets == enum_config.required_property_sets
            assert db_config.deck_size_multiplier == enum_config.deck_size_multiplier
            assert db_config.initial_hand_size == enum_config.initial_hand_size
            assert db_config.new_cards_per_turn == enum_config.new_cards_per_turn
            assert db_config.max_consecutive_player_actions == enum_config.max_consecutive_player_actions

    def test_config_timestamps(self, test_db_session):
        """Test that configs have proper timestamps"""
        # Insert configs
        upsert_game_config_types(test_db_session)

        # Verify timestamps are set
        configs = get_all_configs(test_db_session)
        for config in configs:
            # created_at should be set, updated_at should be None for new records
            assert config.created_at is not None
            assert config.updated_at is None
