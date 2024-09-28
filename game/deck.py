from game.cards import Card, CashCard, PropertyTypeCard, RentCard, SpecialCard
from game.config import GameConfig
from game.pile import BasePile


class Deck(BasePile):
    @classmethod
    def build(
        cls,
        *,
        game_config: GameConfig,
        random_seed: int = 0,
    ):
        cards = []

        # Add cash cards
        for ccv in game_config.cash_card_values:
            cards.append(CashCard(ccv))

        # Add property cards
        for ptc in PropertyTypeCard:
            cards.append(ptc)
            # Add rent cards
            for _ in range(game_config.rent_cards_per_property_type):
                cards.append(RentCard(ptc))

        # Multiply cards by multiplier to get final cards
        cards *= game_config.deck_size_multiplier

        # Add special cards
        for sc in SpecialCard:
            for _ in range(game_config.deck_size_multiplier // game_config.card_to_special_card_ratio):
                cards.append(sc)

        return Deck(cards, random_seed)

    def pick(self) -> Card:
        if not self.cards:
            raise ValueError("Deck is empty.")
        return self._random_rng.choice(self.cards)
