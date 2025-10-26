import { CardModel, getCardKind, PropertyCardModel, CashCardModel, RentCardModel } from '@/types/cards';

export interface GroupedCard {
  /** The card model */
  card: CardModel;
  /** Number of duplicate cards of this type */
  count: number;
  /** Stable unique key for the card type */
  stableKey: string;
}

/**
 * Creates a unique key for a card based on its type and properties.
 * Used for grouping identical cards together in the UI.
 *
 * @param card - The card to create a key for
 * @returns Unique string key for the card
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
 * Groups cards by type and counts duplicates for efficient UI rendering.
 * Combines identical cards into groups with counts to reduce visual clutter.
 *
 * @param cards - Array of cards to group
 * @returns Array of grouped cards with counts
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
