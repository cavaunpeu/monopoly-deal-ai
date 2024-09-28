import { CardModel, getCardKind, PropertyCardModel, CashCardModel, RentCardModel } from '@/types/cards';

export interface GroupedCard {
  card: CardModel;
  count: number;
  stableKey: string;
}

/**
 * Creates a unique key for a card (same logic as CardGrid)
 */
export function createCardKey(card: CardModel): string {
  const cardKind = getCardKind(card);
  if (!cardKind) return 'unknown';

  let key = `${cardKind}-${card.name}`;

  if (cardKind === 'PROPERTY') {
    key += `-${JSON.stringify((card as PropertyCardModel).rent_progression || [])}`;
  } else if (cardKind === 'CASH') {
    key += `-${(card as CashCardModel).value || ''}`;
  } else if (cardKind === 'RENT') {
    key += `-${(card as RentCardModel).property_name || ''}`;
  }

  return key;
}

/**
 * Groups cards by type and counts duplicates (reusable logic from CardGrid)
 */
export function groupCardsByType(cards: CardModel[]): GroupedCard[] {
  const groups = new Map<string, GroupedCard>();

  cards.forEach(card => {
    if (!card) return;

    const cardKind = getCardKind(card);
    if (!cardKind) return;

    const key = createCardKey(card);

    if (groups.has(key)) {
      groups.get(key)!.count++;
    } else {
      groups.set(key, {
        card,
        count: 1,
        stableKey: key
      });
    }
  });

  return Array.from(groups.values());
}

/**
 * Filters grouped cards to only include those that have valid actions
 */
export function filterGroupedCardsByValidActions(
  groupedCards: GroupedCard[],
  validCardKeys: Set<string>
): GroupedCard[] {
  return groupedCards.filter(group => validCardKeys.has(group.stableKey));
}

/**
 * Extracts valid card keys from actions
 */
export function extractValidCardKeysFromActions(actions: Array<{ card: CardModel | null }>): Set<string> {
  const validCardKeys = new Set<string>();

  actions.forEach(action => {
    if (!action.card) return;
    validCardKeys.add(createCardKey(action.card));
  });

  return validCardKeys;
}
