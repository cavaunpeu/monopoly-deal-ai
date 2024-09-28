import { CardModel, HumanState, AiState, SerializedAction } from './cards';

export interface ResponseInfo {
  initiating_card: CardModel;
  initiating_player: 'human' | 'bot';
  responding_player: 'human' | 'bot';
  response_cards_played: CardModel[];
}

export interface SelectedActionEntry {
  turn_idx: number;
  streak_idx: number;
  player_idx: number;
  action: {
    id: number;
    card: CardModel | null;
    src: string;
    dst: string;
    is_response: boolean;
  };
}

export interface TurnState {
  turn_idx: number;
  streak_idx: number;
  streaking_player_idx: number;
  acting_player_idx: number;
  is_human_turn: boolean;
  human_player_index: number;
  cards_played_this_turn: number;
  max_cards_per_turn: number;
  remaining_cards: number;
  selected_actions: SelectedActionEntry[];
  game_over: boolean;
  winner: 'human' | 'bot' | 'tie' | null;
  is_responding: boolean;
  response_info: ResponseInfo | null;
}

export interface PropertyType {
  name: string;
  num_to_complete: number;
  rent_progression: number[];
  cash_value: number;
}

export interface GameConfig {
  required_property_sets: number;
  max_consecutive_player_actions: number;
  cash_card_values: number[];
  rent_cards_per_property_type: number;
  deck_size_multiplier: number;
  total_deck_size: number;
  initial_hand_size: number;
  new_cards_per_turn: number;
  card_to_special_card_ratio: number;
  required_property_sets_map: Record<string, number>;
  property_types: PropertyType[];
}

export interface PublicPileSizes {
  deck: number;
  discard: number;
}

// Re-export types from cards.ts for convenience
export type { HumanState, AiState, SerializedAction } from './cards';

export interface GameState {
  turn: TurnState;
  human: HumanState;
  ai: AiState;
  piles: PublicPileSizes;
  config: GameConfig;
  actions: SerializedAction[];
}
