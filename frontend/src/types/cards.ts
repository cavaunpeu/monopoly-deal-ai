export interface CashCardModel {
  kind: "CASH";
  name: string;
  value: number;
}

export interface PropertyCardModel {
  kind: "PROPERTY";
  name: string;
  rent_progression: number[];
  value: number;
}

export interface RentCardModel {
  kind: "RENT";
  name: string;
  property_name: string;
}

export interface SpecialCardModel {
  kind: "SPECIAL";
  name: string;
}

export type CardModel = CashCardModel | PropertyCardModel | RentCardModel | SpecialCardModel;

// Helper function to get card kind
export function getCardKind(card: CardModel): string {
  return card.kind;
}

// Backend API types
export interface SerializedAction {
  id: number;
  is_response: boolean;
  card: CardModel | null;
  src: string | null;
  dst: string | null;
}

export interface HumanState {
  hand: CardModel[];
  properties: PropertyCardModel[];
  cash: (CashCardModel | PropertyCardModel)[];
}

export interface AiState {
  properties: PropertyCardModel[];
  cash: (CashCardModel | PropertyCardModel)[];
  hand_count: number;
  hand?: CardModel[];
}
