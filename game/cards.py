from dataclasses import dataclass
from enum import Enum


@dataclass
class PropertyTypeValue:
    rent_progression: tuple[int, ...]
    cash_value: int


################################################################################
# Define cards
################################################################################


class PropertyTypeCard(Enum):
    BROWN = PropertyTypeValue(rent_progression=(1, 2), cash_value=1)
    GREEN = PropertyTypeValue(rent_progression=(2, 4, 7), cash_value=4)
    PINK = PropertyTypeValue(rent_progression=(1, 2, 4), cash_value=2)

    @property
    def rent_progression(self):
        """Returns the rent progression for the property."""
        return self.value.rent_progression

    @property
    def cash_value(self):
        """Returns the cash value for the property."""
        return self.value.cash_value

    @property
    def num_to_complete(self):
        """Returns the number of cards needed to complete the set."""
        return len(self.value.rent_progression)


class CashCard(Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4

    @property
    def cash_value(self) -> int:
        """Get the cash value of this card."""
        return self.value


class RentCard(Enum):
    BROWN = PropertyTypeCard.BROWN
    GREEN = PropertyTypeCard.GREEN
    PINK = PropertyTypeCard.PINK


class SpecialCard(Enum):
    JUST_SAY_NO = "JUST_SAY_NO"

    def to_idx(self) -> int:
        """Convert this card to its integer index representation.

        Returns:
            Integer index for this card.
        """
        return CARD_TO_IDX[self]


################################################################################
# Define types
################################################################################

Card = CashCard | RentCard | PropertyTypeCard | SpecialCard


################################################################################
# Card to index, index to card mappings
################################################################################

CARD_TO_IDX = {}
CARD_TYPES = [PropertyTypeCard, RentCard, CashCard, SpecialCard]

for type in CARD_TYPES:
    for value in type:
        CARD_TO_IDX[value] = len(CARD_TO_IDX)

IDX_TO_CARD = {v: k for k, v in CARD_TO_IDX.items()}
