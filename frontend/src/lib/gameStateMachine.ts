import { GameState } from '@/types/game';
import { CardModel, SerializedAction, getCardKind, PropertyCardModel, CashCardModel } from '@/types/cards';

/** Types of game actions that can be performed */
export type GameActionType = 'PLAY_CARD' | 'PASS_TURN' | 'YIELD_RESPONSE';

/** Represents a game action with associated data */
export interface GameAction {
  type: GameActionType;
  card?: CardModel;
  action: SerializedAction;
}

/** Represents a transition between game states */
export interface GameStateTransition {
  from: GameState;
  to: GameState;
  action: GameAction;
  timestamp: number;
}

/**
 * Pure functions for game state transitions.
 * These are easily testable and don't have side effects.
 * Provides optimistic updates for the UI before server confirmation.
 */
export class GameStateMachine {
  /**
   * Apply a card play action to the game state optimistically.
   *
   * @param state - Current game state
   * @param card - Card being played
   * @param action - Serialized action details
   * @returns Updated game state with the card play applied
   */
  static playCard(state: GameState, card: CardModel, action: SerializedAction): GameState {
    const cardKind = getCardKind(card);

    // Remove card from hand
    const updatedHand = state.human.hand.filter((c: CardModel) =>
      !(getCardKind(c) === cardKind && c.name === card.name)
    );

    // Add card to appropriate destination
    let updatedProperties = [...state.human.properties];
    let updatedCash = [...state.human.cash];

    if (action.dst === "PROPERTY" && cardKind === "PROPERTY") {
      updatedProperties = [...updatedProperties, card as PropertyCardModel];
    } else if (action.dst === "CASH" && (cardKind === "CASH" || cardKind === "PROPERTY")) {
      updatedCash = [...updatedCash, card as CashCardModel | PropertyCardModel];
    }

    // Add to played card history
    const updatedSelectedActions = [...(state.turn.selected_actions || [])];
    if (action.dst === "DISCARD" || action.dst === "PROPERTY") {
      const newActionEntry = {
        turn_idx: state.turn.turn_idx,
        streak_idx: state.turn.streak_idx,
        player_idx: state.turn.human_player_index ?? 1,
        action: {
          id: action.id,
          card: card,
          src: action.src || "",
          dst: action.dst || "",
          is_response: action.is_response
        }
      };
      updatedSelectedActions.push(newActionEntry);
    }

    // Update turn state
    const updatedTurn = {
      ...state.turn,
      remaining_cards: Math.max(0, state.turn.remaining_cards - 1),
      cards_played_this_turn: state.turn.cards_played_this_turn + 1,
      selected_actions: updatedSelectedActions
    };

    return {
      ...state,
      human: {
        ...state.human,
        hand: updatedHand,
        properties: updatedProperties,
        cash: updatedCash
      },
      turn: updatedTurn
    };
  }

  /**
   * Apply a pass/yield action to the game state
   */
  static passTurn(state: GameState, action: SerializedAction): GameState {
    const updatedSelectedActions = [...(state.turn.selected_actions || [])];
    const newActionEntry = {
      turn_idx: state.turn.turn_idx,
      streak_idx: state.turn.streak_idx,
      player_idx: state.turn.human_player_index ?? 1,
      action: {
        id: action.id,
        card: null,
        src: "",
        dst: "",
        is_response: action.is_response
      }
    };
    updatedSelectedActions.push(newActionEntry);

    const updatedTurn = {
      ...state.turn,
      is_human_turn: false, // Optimistically switch to AI turn
      selected_actions: updatedSelectedActions
    };

    return {
      ...state,
      turn: updatedTurn
    };
  }

  /**
   * Apply any game action to the current state.
   *
   * @param state - Current game state
   * @param gameAction - Action to apply
   * @returns Updated game state with the action applied
   * @throws {Error} When the action type is unknown or invalid
   */
  static applyAction(state: GameState, gameAction: GameAction): GameState {
    switch (gameAction.type) {
      case 'PLAY_CARD':
        if (!gameAction.card) {
          throw new Error('Card is required for PLAY_CARD action');
        }
        return this.playCard(state, gameAction.card, gameAction.action);

      case 'PASS_TURN':
      case 'YIELD_RESPONSE':
        return this.passTurn(state, gameAction.action);

      default:
        throw new Error(`Unknown action type: ${(gameAction as GameAction).type}`);
    }
  }

  /**
   * Validate if an action can be applied to the current state.
   *
   * @param state - Current game state
   * @param gameAction - Action to validate
   * @returns True if the action can be applied, false otherwise
   */
  static canApplyAction(state: GameState, gameAction: GameAction): boolean {
    try {
      switch (gameAction.type) {
        case 'PLAY_CARD':
          if (!gameAction.card) return false;
          if (state.turn.remaining_cards <= 0) return false;
          if (!state.turn.is_human_turn) return false;
          return true;

        case 'PASS_TURN':
        case 'YIELD_RESPONSE':
          return state.turn.is_human_turn;

        default:
          return false;
      }
    } catch {
      return false;
    }
  }
}
