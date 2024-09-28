/* eslint-disable @typescript-eslint/no-explicit-any */
import { describe, it, expect } from 'vitest';
import { GameStateMachine, GameAction } from './gameStateMachine';
import { mockGameState } from '@/test/fixtures/gameState';
import { CashCardModel, PropertyCardModel } from '@/types/cards';

describe('GameStateMachine', () => {
  const mockCashCard: CashCardModel = { kind: 'CASH', name: 'ONE', value: 1 };
  const mockPropertyCard: PropertyCardModel = {
    kind: 'PROPERTY',
    name: 'GREEN',
    rent_progression: [2, 4, 7],
    value: 4
  };

  describe('playCard', () => {
    it('removes card from hand and adds to properties when dst is PROPERTY', () => {
      const gameState = mockGameState;
      const action = {
        id: 1,
        is_response: false,
        card: mockPropertyCard,
        src: 'HAND',
        dst: 'PROPERTY'
      };

      const result = GameStateMachine.playCard(gameState, mockPropertyCard, action);

      // Card should be removed from hand
      expect(result.human.hand).not.toContainEqual(mockPropertyCard);

      // Card should be added to properties
      expect(result.human.properties).toContainEqual(mockPropertyCard);

      // Turn state should be updated
      expect(result.turn.remaining_cards).toBe(gameState.turn.remaining_cards - 1);
      expect(result.turn.cards_played_this_turn).toBe(gameState.turn.cards_played_this_turn + 1);
    });

    it('removes card from hand and adds to cash when dst is CASH', () => {
      const gameState = mockGameState;
      const action = {
        id: 1,
        is_response: false,
        card: mockCashCard,
        src: 'HAND',
        dst: 'CASH'
      };

      const result = GameStateMachine.playCard(gameState, mockCashCard, action);

      // Card should be removed from hand
      expect(result.human.hand).not.toContainEqual(mockCashCard);

      // Card should be added to cash
      expect(result.human.cash).toContainEqual(mockCashCard);
    });

    it('adds action to selected_actions when dst is DISCARD', () => {
      const gameState = mockGameState;
      const action = {
        id: 1,
        is_response: false,
        card: mockCashCard,
        src: 'HAND',
        dst: 'DISCARD'
      };

      const result = GameStateMachine.playCard(gameState, mockCashCard, action);

      // Should add to selected actions
      expect(result.turn.selected_actions).toHaveLength(
        (gameState.turn.selected_actions?.length || 0) + 1
      );

      const lastAction = result.turn.selected_actions?.[result.turn.selected_actions.length - 1];
      expect(lastAction?.action.id).toBe(action.id);
      expect(lastAction?.action.card).toEqual(mockCashCard);
    });

    it('does not add to selected_actions when dst is CASH', () => {
      const gameState = mockGameState;
      const action = {
        id: 1,
        is_response: false,
        card: mockCashCard,
        src: 'HAND',
        dst: 'CASH'
      };

      const result = GameStateMachine.playCard(gameState, mockCashCard, action);

      // Should not add to selected actions for cash moves
      expect(result.turn.selected_actions).toHaveLength(
        gameState.turn.selected_actions?.length || 0
      );
    });
  });

  describe('passTurn', () => {
    it('switches turn to AI and adds pass action to history', () => {
      const gameState = mockGameState;
      const action = {
        id: 1,
        is_response: false,
        card: null,
        src: null,
        dst: null
      };

      const result = GameStateMachine.passTurn(gameState, action);

      // Should switch to AI turn
      expect(result.turn.is_human_turn).toBe(false);

      // Should add pass action to history
      expect(result.turn.selected_actions).toHaveLength(
        (gameState.turn.selected_actions?.length || 0) + 1
      );

      const lastAction = result.turn.selected_actions?.[result.turn.selected_actions.length - 1];
      expect(lastAction?.action.id).toBe(action.id);
      expect(lastAction?.action.card).toBeNull();
    });
  });

  describe('applyAction', () => {
    it('applies PLAY_CARD action correctly', () => {
      const gameState = mockGameState;
      const gameAction: GameAction = {
        type: 'PLAY_CARD',
        card: mockCashCard,
        action: {
          id: 1,
          is_response: false,
          card: mockCashCard,
          src: 'HAND',
          dst: 'CASH'
        }
      };

      const result = GameStateMachine.applyAction(gameState, gameAction);

      expect(result.human.hand).not.toContainEqual(mockCashCard);
      expect(result.human.cash).toContainEqual(mockCashCard);
    });

    it('applies PASS_TURN action correctly', () => {
      const gameState = mockGameState;
      const gameAction: GameAction = {
        type: 'PASS_TURN',
        action: {
          id: 1,
          is_response: false,
          card: null,
          src: null,
          dst: null
        }
      };

      const result = GameStateMachine.applyAction(gameState, gameAction);

      expect(result.turn.is_human_turn).toBe(false);
    });

    it('throws error for unknown action type', () => {
      const gameState = mockGameState;
      const gameAction = {
        type: 'UNKNOWN_ACTION' as any,
        action: {
          id: 1,
          is_response: false,
          card: null,
          src: null,
          dst: null
        }
      };

      expect(() => GameStateMachine.applyAction(gameState, gameAction)).toThrow(
        'Unknown action type: UNKNOWN_ACTION'
      );
    });

    it('throws error for PLAY_CARD without card', () => {
      const gameState = mockGameState;
      const gameAction: GameAction = {
        type: 'PLAY_CARD',
        action: {
          id: 1,
          is_response: false,
          card: mockCashCard,
          src: 'HAND',
          dst: 'CASH'
        }
      };

      expect(() => GameStateMachine.applyAction(gameState, gameAction)).toThrow(
        'Card is required for PLAY_CARD action'
      );
    });
  });

  describe('canApplyAction', () => {
    it('returns true for valid PLAY_CARD action', () => {
      const gameState = mockGameState;
      const gameAction: GameAction = {
        type: 'PLAY_CARD',
        card: mockCashCard,
        action: {
          id: 1,
          is_response: false,
          card: mockCashCard,
          src: 'HAND',
          dst: 'CASH'
        }
      };

      expect(GameStateMachine.canApplyAction(gameState, gameAction)).toBe(true);
    });

    it('returns false for PLAY_CARD when no remaining cards', () => {
      const gameState = { ...mockGameState, turn: { ...mockGameState.turn, remaining_cards: 0 } };
      const gameAction: GameAction = {
        type: 'PLAY_CARD',
        card: mockCashCard,
        action: {
          id: 1,
          is_response: false,
          card: mockCashCard,
          src: 'HAND',
          dst: 'CASH'
        }
      };

      expect(GameStateMachine.canApplyAction(gameState, gameAction)).toBe(false);
    });

    it('returns false for PLAY_CARD when not human turn', () => {
      const gameState = { ...mockGameState, turn: { ...mockGameState.turn, is_human_turn: false } };
      const gameAction: GameAction = {
        type: 'PLAY_CARD',
        card: mockCashCard,
        action: {
          id: 1,
          is_response: false,
          card: mockCashCard,
          src: 'HAND',
          dst: 'CASH'
        }
      };

      expect(GameStateMachine.canApplyAction(gameState, gameAction)).toBe(false);
    });

    it('returns true for valid PASS_TURN action', () => {
      const gameState = mockGameState;
      const gameAction: GameAction = {
        type: 'PASS_TURN',
        action: {
          id: 1,
          is_response: false,
          card: null,
          src: null,
          dst: null
        }
      };

      expect(GameStateMachine.canApplyAction(gameState, gameAction)).toBe(true);
    });

    it('returns false for PASS_TURN when not human turn', () => {
      const gameState = { ...mockGameState, turn: { ...mockGameState.turn, is_human_turn: false } };
      const gameAction: GameAction = {
        type: 'PASS_TURN',
        action: {
          id: 1,
          is_response: false,
          card: null,
          src: null,
          dst: null
        }
      };

      expect(GameStateMachine.canApplyAction(gameState, gameAction)).toBe(false);
    });

    it('returns false for unknown action type', () => {
      const gameState = mockGameState;
      const gameAction = {
        type: 'UNKNOWN_ACTION' as any,
        action: {
          id: 1,
          is_response: false,
          card: null,
          src: null,
          dst: null
        }
      };

      expect(GameStateMachine.canApplyAction(gameState, gameAction)).toBe(false);
    });
  });
});
