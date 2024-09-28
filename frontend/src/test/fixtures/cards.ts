import { CardModel, CashCardModel, PropertyCardModel, RentCardModel, SpecialCardModel } from '@/types/cards'

// Cash Cards
export const cashCard1: CashCardModel = {
  kind: 'CASH',
  name: 'ONE',
  value: 1
}

export const cashCard3: CashCardModel = {
  kind: 'CASH',
  name: 'THREE',
  value: 3
}

export const cashCard5: CashCardModel = {
  kind: 'CASH',
  name: 'FIVE',
  value: 5
}

// Property Cards
export const greenProperty: PropertyCardModel = {
  kind: 'PROPERTY',
  name: 'GREEN',
  rent_progression: [2, 4, 7],
  value: 4
}

export const brownProperty: PropertyCardModel = {
  kind: 'PROPERTY',
  name: 'BROWN',
  rent_progression: [1, 2],
  value: 1
}

export const pinkProperty: PropertyCardModel = {
  kind: 'PROPERTY',
  name: 'PINK',
  rent_progression: [1, 2, 4],
  value: 2
}

// Rent Cards
export const greenRent: RentCardModel = {
  kind: 'RENT',
  name: 'GREEN',
  property_name: 'GREEN'
}

export const brownRent: RentCardModel = {
  kind: 'RENT',
  name: 'BROWN',
  property_name: 'BROWN'
}

// Special Cards
export const justSayNo: SpecialCardModel = {
  kind: 'SPECIAL',
  name: 'JUST_SAY_NO'
}

export const dealBreaker: SpecialCardModel = {
  kind: 'SPECIAL',
  name: 'DEAL_BREAKER'
}

// Card Collections
export const sampleCashCards: CashCardModel[] = [cashCard1, cashCard3, cashCard5, cashCard1, cashCard3]
export const samplePropertyCards: PropertyCardModel[] = [greenProperty, brownProperty, pinkProperty, greenProperty]
export const sampleRentCards: RentCardModel[] = [greenRent, brownRent]
export const sampleSpecialCards: SpecialCardModel[] = [justSayNo, dealBreaker]

export const allSampleCards: CardModel[] = [
  ...sampleCashCards,
  ...samplePropertyCards,
  ...sampleRentCards,
  ...sampleSpecialCards
]

// Duplicate cards for testing grouping
export const duplicateCashCards: CashCardModel[] = [cashCard3, cashCard3, cashCard3] // 3 x $3 cards
export const duplicatePropertyCards: PropertyCardModel[] = [greenProperty, greenProperty] // 2 x GREEN properties
