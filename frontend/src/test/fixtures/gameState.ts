import { GameState, TurnState, ResponseInfo, SelectedActionEntry, GameConfig } from '@/types/game'
import { HumanState, AiState, SerializedAction } from '@/types/cards'
import { cashCard1, cashCard3, greenProperty, brownProperty, justSayNo, greenRent } from './cards'

// Game Configuration
export const mockGameConfig: GameConfig = {
  required_property_sets: 3,
  max_consecutive_player_actions: 3,
  cash_card_values: [1, 2, 3, 4, 5],
  rent_cards_per_property_type: 2,
  deck_size_multiplier: 1,
  total_deck_size: 100,
  initial_hand_size: 5,
  new_cards_per_turn: 2,
  card_to_special_card_ratio: 0.1,
  required_property_sets_map: {
    'GREEN': 3,
    'BROWN': 2,
    'PINK': 3
  },
  property_types: [
    {
      name: 'GREEN',
      num_to_complete: 3,
      rent_progression: [2, 4, 7],
      cash_value: 4
    },
    {
      name: 'BROWN',
      num_to_complete: 2,
      rent_progression: [1, 2],
      cash_value: 1
    }
  ]
}

// Human State
export const mockHumanState: HumanState = {
  hand: [cashCard1, cashCard3, greenProperty, justSayNo],
  properties: [greenProperty, brownProperty],
  cash: [cashCard1, cashCard3]
}

// AI State
export const mockAiState: AiState = {
  properties: [greenProperty, greenProperty],
  cash: [cashCard3, cashCard3],
  hand_count: 4,
  hand: [brownProperty, justSayNo, cashCard1, greenRent]
}

// Turn State
export const mockTurnState: TurnState = {
  turn_idx: 5,
  streak_idx: 2,
  streaking_player_idx: 0,
  acting_player_idx: 0,
  is_human_turn: true,
  human_player_index: 0,
  cards_played_this_turn: 1,
  max_cards_per_turn: 3,
  remaining_cards: 2,
  selected_actions: [],
  game_over: false,
  winner: null,
  is_responding: false,
  response_info: null
}

// Response Info for rent scenarios
export const mockResponseInfo: ResponseInfo = {
  initiating_card: greenRent,
  initiating_player: 'bot',
  responding_player: 'human',
  response_cards_played: []
}

// Selected Actions
export const mockSelectedActions: SelectedActionEntry[] = [
  {
    turn_idx: 4,
    streak_idx: 1,
    player_idx: 0,
    action: {
      id: 12345,
      card: greenProperty,
      src: 'HAND',
      dst: 'PROPERTY',
      is_response: false
    }
  },
  {
    turn_idx: 5,
    streak_idx: 2,
    player_idx: 1,
    action: {
      id: 12346,
      card: greenRent,
      src: 'HAND',
      dst: 'DISCARD',
      is_response: false
    }
  }
]

// Serialized Actions
export const mockValidActions: SerializedAction[] = [
  {
    id: 1001,
    is_response: false,
    card: cashCard1,
    src: 'HAND',
    dst: 'CASH'
  },
  {
    id: 1002,
    is_response: false,
    card: cashCard3,
    src: 'HAND',
    dst: 'CASH'
  },
  {
    id: 1003,
    is_response: false,
    card: greenProperty,
    src: 'HAND',
    dst: 'PROPERTY'
  },
  {
    id: 1004,
    is_response: true,
    card: justSayNo,
    src: 'HAND',
    dst: 'DISCARD'
  },
  {
    id: 1005,
    is_response: true,
    card: null,
    src: null,
    dst: null
  }
]

// Complete Game State
export const mockGameState: GameState = {
  turn: mockTurnState,
  human: mockHumanState,
  ai: mockAiState,
  piles: {
    deck: 45,
    discard: 12
  },
  config: mockGameConfig,
  actions: mockValidActions
}

// Response scenario game state
export const mockResponseGameState: GameState = {
  ...mockGameState,
  turn: {
    ...mockTurnState,
    is_responding: true,
    response_info: mockResponseInfo
  }
}

// Game over state
export const mockGameOverState: GameState = {
  ...mockGameState,
  turn: {
    ...mockTurnState,
    game_over: true,
    winner: 'human'
  }
}
