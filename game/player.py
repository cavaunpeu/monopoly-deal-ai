from game.pile import CashPile, Hand, PlayedCardsPile, PropertyPile


class Player:
    def __init__(self, index: int):
        self.index = index
        self.hand = Hand([])
        self.properties = PropertyPile([])
        self.cash = CashPile([])
        self.played_cards = PlayedCardsPile([])

    @property
    def name(self):
        return f"Player {self.index + 1}"

    @property
    def num_complete_property_sets(self):
        """Returns the number of complete property sets the player has."""
        return sum([count >= ptype.num_to_complete for ptype, count in self.properties.items()])

    def clone(self) -> "Player":
        new_player = self.__class__.__new__(self.__class__)
        new_player.index = self.index
        new_player.hand = self.hand.clone()
        new_player.properties = self.properties.clone()
        new_player.cash = self.cash.clone()
        new_player.played_cards = self.played_cards.clone()

        return new_player
