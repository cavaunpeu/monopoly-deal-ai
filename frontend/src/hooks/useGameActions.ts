import { useCallback, useState } from 'react';
import { GameState } from '@/types/game';
import { CardModel, SerializedAction } from '@/types/cards';
import { GameStateMachine, GameAction, GameStateTransition } from '@/lib/gameStateMachine';
import { logger } from '@/lib/logger';

interface UseGameActionsResult {
  isProcessing: boolean;
  playCard: (card: CardModel, action: SerializedAction, apiCall: () => Promise<void>) => Promise<void>;
  passTurn: (action: SerializedAction, apiCall: () => Promise<void>) => Promise<void>;
  yieldResponse: (action: SerializedAction, apiCall: () => Promise<void>) => Promise<void>;
  rollbackLastAction: () => void;
  getTransitionHistory: () => GameStateTransition[];
}

export function useGameActions(
  gameState: GameState | null,
  setGameState: (state: GameState | null) => void
): UseGameActionsResult {
  const [isProcessing, setIsProcessing] = useState(false);
  const [transitionHistory, setTransitionHistory] = useState<GameStateTransition[]>([]);

  const rollbackLastAction = useCallback(() => {
    if (transitionHistory.length === 0) {
      logger.warn('No actions to rollback');
      return;
    }

    const lastTransition = transitionHistory[transitionHistory.length - 1];
    setGameState(lastTransition.from);
    setTransitionHistory(prev => prev.slice(0, -1));

    logger.info('Rolled back action:', lastTransition.action.type);
  }, [transitionHistory, setGameState]);

  const executeAction = useCallback(async (
    gameAction: GameAction,
    apiCall: () => Promise<void>
  ) => {
    if (!gameState || isProcessing) {
      logger.warn('Cannot execute action: no game state or already processing');
      return;
    }

    // Validate action
    if (!GameStateMachine.canApplyAction(gameState, gameAction)) {
      logger.warn('Cannot apply action to current state:', gameAction);
      throw new Error(`Invalid action for current game state: ${gameAction.type}`);
    }

    setIsProcessing(true);

    try {
      // Store original state for rollback
      const originalState = gameState;

      // Apply optimistic update using state machine
      const optimisticState = GameStateMachine.applyAction(gameState, gameAction);

      // Create transition record
      const transition: GameStateTransition = {
        from: originalState,
        to: optimisticState,
        action: gameAction,
        timestamp: Date.now()
      };

      // Update state and history
      setGameState(optimisticState);
      setTransitionHistory(prev => [...prev, transition]);

      // Execute API call
      await apiCall();

      // Clear transition history on success (API call succeeded)
      setTransitionHistory([]);

    } catch (error) {
      logger.error('Action failed, rolling back:', error);
      rollbackLastAction();
      throw error;
    } finally {
      setIsProcessing(false);
    }
  }, [gameState, isProcessing, setGameState, rollbackLastAction]);

  const playCard = useCallback(async (
    card: CardModel,
    action: SerializedAction,
    apiCall: () => Promise<void>
  ) => {
    const gameAction: GameAction = {
      type: 'PLAY_CARD',
      card,
      action
    };
    await executeAction(gameAction, apiCall);
  }, [executeAction]);

  const passTurn = useCallback(async (
    action: SerializedAction,
    apiCall: () => Promise<void>
  ) => {
    const gameAction: GameAction = {
      type: 'PASS_TURN',
      action
    };
    await executeAction(gameAction, apiCall);
  }, [executeAction]);

  const yieldResponse = useCallback(async (
    action: SerializedAction,
    apiCall: () => Promise<void>
  ) => {
    const gameAction: GameAction = {
      type: 'YIELD_RESPONSE',
      action
    };
    await executeAction(gameAction, apiCall);
  }, [executeAction]);

  const getTransitionHistory = useCallback(() => {
    return [...transitionHistory];
  }, [transitionHistory]);

  return {
    isProcessing,
    playCard,
    passTurn,
    yieldResponse,
    rollbackLastAction,
    getTransitionHistory
  };
}
