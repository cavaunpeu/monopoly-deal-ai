import { describe, it, expect } from 'vitest'
import {
  createCardKey,
  groupCardsByType,
  filterGroupedCardsByValidActions,
  extractValidCardKeysFromActions,
  type GroupedCard
} from './cardUtils'
import { createInvalidCard, createMixedCardArray, createIncompleteCard } from '../test/utils/mockTypes'
import {
  cashCard1,
  cashCard3,
  greenProperty,
  brownProperty,
  pinkProperty,
  greenRent,
  justSayNo,
  duplicateCashCards
} from '../test/fixtures/cards'
import { mockValidActions } from '../test/fixtures/gameState'

describe('cardUtils', () => {
  describe('createCardKey', () => {
    it('creates unique keys for different card types', () => {
      const cashKey = createCardKey(cashCard1)
      const propertyKey = createCardKey(greenProperty)
      const rentKey = createCardKey(greenRent)
      const specialKey = createCardKey(justSayNo)

      expect(cashKey).toBe('CASH-ONE-1')
      expect(propertyKey).toBe('PROPERTY-GREEN-[2,4,7]')
      expect(rentKey).toBe('RENT-GREEN-GREEN')
      expect(specialKey).toBe('SPECIAL-JUST_SAY_NO')
    })

    it('creates identical keys for identical cards', () => {
      const key1 = createCardKey(cashCard3)
      const key2 = createCardKey(cashCard3)
      const key3 = createCardKey({ ...cashCard3 })

      expect(key1).toBe(key2)
      expect(key2).toBe(key3)
      expect(key1).toBe('CASH-THREE-3')
    })

    it('handles different property cards with same name but different rent progressions', () => {
      const property1 = { ...greenProperty, rent_progression: [2, 4, 7] }
      const property2 = { ...greenProperty, rent_progression: [1, 3, 5] }

      const key1 = createCardKey(property1)
      const key2 = createCardKey(property2)

      expect(key1).not.toBe(key2)
      expect(key1).toBe('PROPERTY-GREEN-[2,4,7]')
      expect(key2).toBe('PROPERTY-GREEN-[1,3,5]')
    })

    it('handles cards with missing properties gracefully', () => {
      const incompleteCard = createIncompleteCard('CASH', 'INCOMPLETE') as import('@/types/cards').CardModel
      const key = createCardKey(incompleteCard)

      expect(key).toBe('CASH-INCOMPLETE-')
    })

    it('returns key with invalid kind for invalid card kinds', () => {
      const invalidCard = createInvalidCard('INVALID', 'TEST')
      const key = createCardKey(invalidCard)

      expect(key).toBe('INVALID-TEST')
    })
  })

  describe('groupCardsByType', () => {
    it('groups identical cards and counts them correctly', () => {
      const cards = [cashCard1, cashCard3, cashCard1, cashCard3, cashCard3]
      const grouped = groupCardsByType(cards)

      expect(grouped).toHaveLength(2)

      const oneDollarGroup = grouped.find(g => g.card.name === 'ONE')
      const threeDollarGroup = grouped.find(g => g.card.name === 'THREE')

      expect(oneDollarGroup?.count).toBe(2)
      expect(threeDollarGroup?.count).toBe(3)
    })

    it('handles empty array', () => {
      const grouped = groupCardsByType([])
      expect(grouped).toHaveLength(0)
    })

    it('handles array with null/undefined cards', () => {
      const cards = createMixedCardArray(cashCard1, null, cashCard3, undefined, cashCard1)
      const grouped = groupCardsByType(cards)

      expect(grouped).toHaveLength(2)
      expect(grouped.every(g => g.count > 0)).toBe(true)
    })

    it('creates stable keys for grouped cards', () => {
      const cards = [cashCard1, cashCard1]
      const grouped = groupCardsByType(cards)

      expect(grouped).toHaveLength(1)
      expect(grouped[0].stableKey).toBe('CASH-ONE-1')
    })

    it('groups complex card types correctly', () => {
      const cards = [greenProperty, brownProperty, greenProperty, pinkProperty]
      const grouped = groupCardsByType(cards)

      expect(grouped).toHaveLength(3)

      const greenGroup = grouped.find(g => g.card.name === 'GREEN')
      const brownGroup = grouped.find(g => g.card.name === 'BROWN')
      const pinkGroup = grouped.find(g => g.card.name === 'PINK')

      expect(greenGroup?.count).toBe(2)
      expect(brownGroup?.count).toBe(1)
      expect(pinkGroup?.count).toBe(1)
    })

    it('handles mixed card types', () => {
      const cards = [cashCard1, greenProperty, justSayNo, cashCard1, greenProperty]
      const grouped = groupCardsByType(cards)

      expect(grouped).toHaveLength(3)

      const cashGroup = grouped.find(g => g.card.kind === 'CASH')
      const propertyGroup = grouped.find(g => g.card.kind === 'PROPERTY')
      const specialGroup = grouped.find(g => g.card.kind === 'SPECIAL')

      expect(cashGroup?.count).toBe(2)
      expect(propertyGroup?.count).toBe(2)
      expect(specialGroup?.count).toBe(1)
    })
  })

  describe('extractValidCardKeysFromActions', () => {
    it('extracts card keys from actions with cards', () => {
      const actions = [
        { card: cashCard1 },
        { card: greenProperty },
        { card: null }
      ]

      const keys = extractValidCardKeysFromActions(actions)

      expect(keys.size).toBe(2)
      expect(keys.has('CASH-ONE-1')).toBe(true)
      expect(keys.has('PROPERTY-GREEN-[2,4,7]')).toBe(true)
    })

    it('handles actions with null cards', () => {
      const actions = [{ card: null }, { card: null }]
      const keys = extractValidCardKeysFromActions(actions)

      expect(keys.size).toBe(0)
    })

    it('handles empty actions array', () => {
      const keys = extractValidCardKeysFromActions([])
      expect(keys.size).toBe(0)
    })

    it('extracts keys from realistic game actions', () => {
      const keys = extractValidCardKeysFromActions(mockValidActions)

      expect(keys.size).toBe(4) // 4 actions with cards, 1 with null
      expect(keys.has('CASH-ONE-1')).toBe(true)
      expect(keys.has('CASH-THREE-3')).toBe(true)
      expect(keys.has('PROPERTY-GREEN-[2,4,7]')).toBe(true)
      expect(keys.has('SPECIAL-JUST_SAY_NO')).toBe(true)
    })

    it('deduplicates identical card keys', () => {
      const actions = [
        { card: cashCard1 },
        { card: cashCard1 },
        { card: { ...cashCard1 } } // Same card, different object reference
      ]

      const keys = extractValidCardKeysFromActions(actions)

      expect(keys.size).toBe(1)
      expect(keys.has('CASH-ONE-1')).toBe(true)
    })
  })

  describe('filterGroupedCardsByValidActions', () => {
    it('filters grouped cards to only include valid ones', () => {
      const groupedCards: GroupedCard[] = [
        { card: cashCard1, count: 2, stableKey: 'CASH-ONE-1' },
        { card: cashCard3, count: 1, stableKey: 'CASH-THREE-3' },
        { card: greenProperty, count: 1, stableKey: 'PROPERTY-GREEN-[2,4,7]' }
      ]

      const validKeys = new Set(['CASH-ONE-1', 'PROPERTY-GREEN-[2,4,7]'])
      const filtered = filterGroupedCardsByValidActions(groupedCards, validKeys)

      expect(filtered).toHaveLength(2)
      expect(filtered.find(f => f.card.name === 'ONE')).toBeDefined()
      expect(filtered.find(f => f.card.name === 'GREEN')).toBeDefined()
      expect(filtered.find(f => f.card.name === 'THREE')).toBeUndefined()
    })

    it('returns empty array when no cards match', () => {
      const groupedCards: GroupedCard[] = [
        { card: cashCard1, count: 2, stableKey: 'CASH-ONE-1' }
      ]

      const validKeys = new Set(['CASH-THREE-3'])
      const filtered = filterGroupedCardsByValidActions(groupedCards, validKeys)

      expect(filtered).toHaveLength(0)
    })

    it('handles empty grouped cards array', () => {
      const validKeys = new Set(['CASH-ONE-1'])
      const filtered = filterGroupedCardsByValidActions([], validKeys)

      expect(filtered).toHaveLength(0)
    })

    it('handles empty valid keys set', () => {
      const groupedCards: GroupedCard[] = [
        { card: cashCard1, count: 2, stableKey: 'CASH-ONE-1' }
      ]

      const filtered = filterGroupedCardsByValidActions(groupedCards, new Set())

      expect(filtered).toHaveLength(0)
    })

    it('preserves count and stableKey in filtered results', () => {
      const groupedCards: GroupedCard[] = [
        { card: cashCard1, count: 3, stableKey: 'CASH-ONE-1' }
      ]

      const validKeys = new Set(['CASH-ONE-1'])
      const filtered = filterGroupedCardsByValidActions(groupedCards, validKeys)

      expect(filtered).toHaveLength(1)
      expect(filtered[0].count).toBe(3)
      expect(filtered[0].stableKey).toBe('CASH-ONE-1')
    })
  })

  describe('integration tests', () => {
    it('works end-to-end with realistic game data', () => {
      // Simulate the ResponseActions component workflow
      const cards = [cashCard1, cashCard3, greenProperty, justSayNo, cashCard1, cashCard3]
      const actions = [
        { card: cashCard1 },
        { card: cashCard3 },
        { card: greenProperty },
        { card: justSayNo },
        { card: null } // Pass action
      ]

      // Step 1: Group cards by type
      const groupedCards = groupCardsByType(cards)
      expect(groupedCards).toHaveLength(4)

      // Step 2: Extract valid card keys from actions
      const validKeys = extractValidCardKeysFromActions(actions)
      expect(validKeys.size).toBe(4)

      // Step 3: Filter grouped cards by valid actions
      const filteredCards = filterGroupedCardsByValidActions(groupedCards, validKeys)
      expect(filteredCards).toHaveLength(4)

      // Verify all filtered cards have valid actions
      filteredCards.forEach(group => {
        expect(validKeys.has(group.stableKey)).toBe(true)
      })
    })

    it('handles duplicate cards correctly in realistic scenario', () => {
      const cards = duplicateCashCards // 3 x $3 cards
      const actions = [
        { card: cashCard3 },
        { card: cashCard3 }
      ]

      const groupedCards = groupCardsByType(cards)
      const validKeys = extractValidCardKeysFromActions(actions)
      const filteredCards = filterGroupedCardsByValidActions(groupedCards, validKeys)

      expect(groupedCards).toHaveLength(1)
      expect(groupedCards[0].count).toBe(3)
      expect(filteredCards).toHaveLength(1)
      expect(filteredCards[0].count).toBe(3)
    })
  })
})
