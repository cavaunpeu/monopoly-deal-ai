/** Represents a cash card with monetary value */
export interface CashCardModel {
  kind: "CASH";
  name: string;
  value: number;
}

/** Represents a property card with rent progression and cash value */
export interface PropertyCardModel {
  kind: "PROPERTY";
  name: string;
  rent_progression: number[];
  value: number;
}

/** Represents a rent card that charges rent for specific property types */
export interface RentCardModel {
  kind: "RENT";
  name: string;
  property_name: string;
}

/** Represents a special action card (e.g., Just Say No) */
export interface SpecialCardModel {
  kind: "SPECIAL";
  name: string;
}

/** Union type representing any card in the game */
export type CardModel = CashCardModel | PropertyCardModel | RentCardModel | SpecialCardModel;

// Helper function to get card kind
export function getCardKind(card: CardModel): string {
  return card.kind;
}

// Backend API types

/** Represents a serialized game action from the backend API */
export interface SerializedAction {
  id: number;
  is_response: boolean;
  card: CardModel | null;
  src: string | null;
  dst: string | null;
}

/** Represents the human player's game state */
export interface HumanState {
  hand: CardModel[];
  properties: PropertyCardModel[];
  cash: (CashCardModel | PropertyCardModel)[];
}

/** Represents the AI player's game state */
export interface AiState {
  properties: PropertyCardModel[];
  cash: (CashCardModel | PropertyCardModel)[];
  hand_count: number;
  hand?: CardModel[];
}
