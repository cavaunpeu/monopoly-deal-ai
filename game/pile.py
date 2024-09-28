from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import random
from typing import Generic, Iterable, Self, TypeVar

import numpy as np

from game.cards import CARD_TO_IDX, IDX_TO_CARD, Card, CashCard, PropertyTypeCard


K = TypeVar("K", bound=Card)


class SafeCounter(Counter[K], Generic[K]):
    def __setitem__(self, key: K, value: int) -> None:
        if value < 0:
            raise ValueError(f"Negative value not allowed: {key} -> {value}")
        elif value == 0:
            self.pop(key, None)
        else:
            super().__setitem__(key, value)


@dataclass
class BasePile(SafeCounter[K], Generic[K]):
    initial: Iterable[K]
    random_seed: int = 0

    def __post_init__(self) -> None:
        super().__init__(self.initial)
        self._random_rng = random.Random(self.random_seed)

    def __repr__(self):
        return f"{self.__class__.__name__}({dict(self)})"

    @property
    def cards(self) -> list[K]:
        return list(self.elements())

    def contains(self, item: K) -> bool:
        return self[item] > 0

    def clone(self) -> Self:
        new_pile = self.__class__.__new__(self.__class__)
        new_pile.update(self)
        new_pile.initial = self.initial
        new_pile.random_seed = self.random_seed
        new_pile._random_rng = random.Random(self.random_seed)

        return new_pile

    def __eq__(self, other) -> bool:
        """Equality comparison that ignores the initial field."""
        if not isinstance(other, self.__class__):
            return False
        return dict(self) == dict(other) and self.random_seed == other.random_seed

    def add(self, item: K) -> None:
        self[item] += 1

    def remove(self, item: K) -> None:
        self[item] -= 1

    def __len__(self) -> int:
        return sum(self.values())

    def __iter__(self):
        return iter(self.elements())

    def encode(self) -> list[int]:
        encoding = np.zeros(len(CARD_TO_IDX), dtype=int)
        for card, idx in CARD_TO_IDX.items():
            encoding[idx] = self[card]
        return encoding.tolist()

    @classmethod
    def decode(cls, encoded: list[int]):
        cards: list[K] = []
        for idx, count in enumerate(encoded):
            if count > 0:
                card = IDX_TO_CARD[idx]
                cards.extend([card] * count)
        return cls(cards)


class CashPile(BasePile[CashCard | PropertyTypeCard]):
    @property
    def cash_total(self) -> int:
        return sum(card.cash_value for card in self.cards)


class PropertyPile(BasePile[PropertyTypeCard]):
    pass


class Hand(BasePile[Card]):
    pass


class Discard(BasePile[Card]):
    pass


class PlayedCardsPile(BasePile[Card]):
    pass


class Pile(Enum):
    HAND = "HAND"
    CASH = "CASH"
    PROPERTY = "PROPERTY"
    PLAYED_CARDS = "PLAYED_CARDS"
    DISCARD = "DISCARD"
    OPPONENT_CASH = "OPPONENT_CASH"
    DECK = "DECK"
