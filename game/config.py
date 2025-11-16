from dataclasses import asdict, dataclass, field
from enum import Enum

from game.cards import PropertyTypeCard
from game.util import Serializable


@dataclass
class GameConfig(Serializable):
    """Configuration class defining game rules and parameters.

    This class contains all the configurable parameters that define how
    the Monopoly Deal game is played, including deck composition, win
    conditions, and turn mechanics.
    """

    cash_card_values: list[int] = field(default_factory=lambda: [1, 3])
    rent_cards_per_property_type: int = field(default=3)
    required_property_sets: int = field(default=3)
    deck_size_multiplier: int = field(default=10)
    initial_hand_size: int = field(default=5)
    new_cards_per_turn: int = field(default=2)
    max_consecutive_player_actions: int = field(default=3)
    card_to_special_card_ratio: int = field(default=3)

    def to_json(self) -> dict:
        """Convert the configuration to a JSON-serializable dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "GameConfig":
        """Create a GameConfig from a JSON dictionary.

        Args:
            data: Dictionary containing configuration parameters.

        Returns:
            GameConfig instance created from the data.
        """
        return cls(**data)

    def __post_init__(self):
        if self.required_property_sets > len(PropertyTypeCard):
            raise ValueError(
                f"You have {len(PropertyTypeCard)} property types, but you have set the required property sets to {self.required_property_sets}. Please increase the number of property types or decrease the required property sets."
            )

    def get_total_deck_size(self) -> int:
        """Calculate the total number of cards in a deck built with this configuration.

        Returns:
            Total number of cards in the deck.
        """
        from game.deck import Deck

        return len(Deck.build(game_config=self))


class GameConfigType(Enum):
    """Predefined game configuration types for different game sizes.

    These configurations provide different levels of game complexity.
    """

    TINY = GameConfig(
        cash_card_values=[1],
        rent_cards_per_property_type=0,
        required_property_sets=1,
        deck_size_multiplier=5,
        initial_hand_size=3,
        new_cards_per_turn=1,
        max_consecutive_player_actions=1,
        card_to_special_card_ratio=3,
    )
    SMALL = GameConfig(
        cash_card_values=[1, 3],
        rent_cards_per_property_type=1,
        required_property_sets=1,
        deck_size_multiplier=10,
        initial_hand_size=5,
        new_cards_per_turn=2,
        max_consecutive_player_actions=2,
        card_to_special_card_ratio=3,
    )
    MEDIUM = GameConfig(
        cash_card_values=[1, 3],
        rent_cards_per_property_type=1,
        required_property_sets=2,
        deck_size_multiplier=10,
        initial_hand_size=5,
        new_cards_per_turn=2,
        max_consecutive_player_actions=2,
        card_to_special_card_ratio=3,
    )
    LARGE = GameConfig(
        cash_card_values=[1, 3],
        rent_cards_per_property_type=1,
        required_property_sets=3,
        deck_size_multiplier=10,
        initial_hand_size=5,
        new_cards_per_turn=2,
        max_consecutive_player_actions=2,
        card_to_special_card_ratio=3,
    )
