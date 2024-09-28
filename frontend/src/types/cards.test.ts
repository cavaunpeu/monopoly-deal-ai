/* eslint-disable @typescript-eslint/no-explicit-any */
import { describe, it, expect } from 'vitest'
import { getCardKind } from './cards'
import {
  cashCard1,
  cashCard3,
  greenProperty,
  brownProperty,
  greenRent,
  justSayNo,
  dealBreaker
} from '../test/fixtures/cards'

describe('types/cards', () => {
  describe('getCardKind', () => {
    it('returns correct kind for cash cards', () => {
      expect(getCardKind(cashCard1)).toBe('CASH')
      expect(getCardKind(cashCard3)).toBe('CASH')
    })

    it('returns correct kind for property cards', () => {
      expect(getCardKind(greenProperty)).toBe('PROPERTY')
      expect(getCardKind(brownProperty)).toBe('PROPERTY')
    })

    it('returns correct kind for rent cards', () => {
      expect(getCardKind(greenRent)).toBe('RENT')
    })

    it('returns correct kind for special cards', () => {
      expect(getCardKind(justSayNo)).toBe('SPECIAL')
      expect(getCardKind(dealBreaker)).toBe('SPECIAL')
    })

    it('handles cards with different object references but same data', () => {
      const cashCardCopy = { ...cashCard1 }
      const propertyCardCopy = { ...greenProperty }

      expect(getCardKind(cashCardCopy)).toBe('CASH')
      expect(getCardKind(propertyCardCopy)).toBe('PROPERTY')
    })

    it('works with cards that have additional properties', () => {
      const extendedCashCard = {
        ...cashCard1,
        extraProperty: 'test'
      }

      expect(getCardKind(extendedCashCard)).toBe('CASH')
    })

    it('handles edge cases gracefully', () => {
      // Test with minimal card objects
      const minimalCashCard = { kind: 'CASH', name: 'MINIMAL', value: 1 }
      const minimalPropertyCard = {
        kind: 'PROPERTY',
        name: 'MINIMAL',
        rent_progression: [1],
        value: 1
      }

      expect(getCardKind(minimalCashCard)).toBe('CASH')
      expect(getCardKind(minimalPropertyCard)).toBe('PROPERTY')
    })

    it('returns the exact string from the kind property', () => {
      // Ensure it returns the exact string, not a processed version
      expect(getCardKind(cashCard1)).toBe(cashCard1.kind)
      expect(getCardKind(greenProperty)).toBe(greenProperty.kind)
      expect(getCardKind(greenRent)).toBe(greenRent.kind)
      expect(getCardKind(justSayNo)).toBe(justSayNo.kind)
    })

    it('works with all card types in a mixed array', () => {
      const mixedCards = [cashCard1, greenProperty, greenRent, justSayNo]
      const kinds = mixedCards.map(getCardKind)

      expect(kinds).toEqual(['CASH', 'PROPERTY', 'RENT', 'SPECIAL'])
    })

    it('handles cards with undefined or null properties gracefully', () => {
      const cardWithUndefined = {
        kind: 'CASH',
        name: undefined,
        value: 1
      } as any

      const cardWithNull = {
        kind: 'PROPERTY',
        name: null,
        rent_progression: [1],
        value: 1
      } as any

      expect(getCardKind(cardWithUndefined)).toBe('CASH')
      expect(getCardKind(cardWithNull)).toBe('PROPERTY')
    })
  })

  describe('type safety', () => {
    it('ensures type guards work correctly', () => {
      const cards = [cashCard1, greenProperty, greenRent, justSayNo]

      cards.forEach(card => {
        const kind = getCardKind(card)
        expect(typeof kind).toBe('string')
        expect(kind).toMatch(/^(CASH|PROPERTY|RENT|SPECIAL)$/)
      })
    })

    it('handles union types correctly', () => {
      // Test that the function works with the CardModel union type
      const cashCard: typeof cashCard1 = cashCard1
      const propertyCard: typeof greenProperty = greenProperty
      const rentCard: typeof greenRent = greenRent
      const specialCard: typeof justSayNo = justSayNo

      expect(getCardKind(cashCard)).toBe('CASH')
      expect(getCardKind(propertyCard)).toBe('PROPERTY')
      expect(getCardKind(rentCard)).toBe('RENT')
      expect(getCardKind(specialCard)).toBe('SPECIAL')
    })
  })

  describe('performance and reliability', () => {
    it('handles large arrays of cards efficiently', () => {
      const largeCardArray = Array(1000).fill(null).map((_, i) => ({
        kind: 'CASH' as const,
        name: `CARD_${i}`,
        value: i % 10
      }))

      const start = performance.now()
      const kinds = largeCardArray.map(getCardKind)
      const end = performance.now()

      expect(kinds).toHaveLength(1000)
      expect(kinds.every(kind => kind === 'CASH')).toBe(true)
      expect(end - start).toBeLessThan(100) // Should be very fast
    })

    it('is consistent across multiple calls', () => {
      const card = cashCard1
      const results = Array(100).fill(null).map(() => getCardKind(card))

      expect(results.every(result => result === 'CASH')).toBe(true)
    })
  })
})
