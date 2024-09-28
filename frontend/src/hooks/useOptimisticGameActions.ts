import { useCallback, useState } from 'react';
import { GameState } from '@/types/game';
import { CardModel, SerializedAction, getCardKind, PropertyCardModel, CashCardModel } from '@/types/cards';
import { logger } from '@/lib/logger';

interface OptimisticUpdate {
  type: 'card_played' | 'turn_passed';
  action: SerializedAction;
  card?: CardModel;
  originalState: GameState;
}

interface UseOptimisticGameActionsResult {
  isProcessing: boolean;
  executeCardAction: (card: CardModel, action: SerializedAction, apiCall: () => Promise<void>) => Promise<void>;
  executePassAction: (action: SerializedAction, apiCall: () => Promise<void>) => Promise<void>;
  rollbackLastUpdate: () => void;
}

export function useOptimisticGameActions(
  gameState: GameState | null,
  setGameState: (state: GameState | null) => void
): UseOptimisticGameActionsResult {
  const [isProcessing, setIsProcessing] = useState(false);
  const [updateHistory, setUpdateHistory] = useState<OptimisticUpdate[]>([]);

  const applyCardPlayUpdate = useCallback((state: GameState, card: CardModel, action: SerializedAction): GameState => {
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
  }, []);

  const applyPassUpdate = useCallback((state: GameState, action: SerializedAction): GameState => {
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
  }, []);

  const rollbackLastUpdate = useCallback(() => {
    if (updateHistory.length === 0) return;

    const lastUpdate = updateHistory[updateHistory.length - 1];
    setGameState(lastUpdate.originalState);
    setUpdateHistory(prev => prev.slice(0, -1));
  }, [updateHistory, setGameState]);

  const executeCardAction = useCallback(async (
    card: CardModel,
    action: SerializedAction,
    apiCall: () => Promise<void>
  ) => {
    if (!gameState || isProcessing) return;

    setIsProcessing(true);

    try {
      // Store original state for rollback
      const originalState = gameState;
      const update: OptimisticUpdate = {
        type: 'card_played',
        action,
        card,
        originalState
      };

      // Apply optimistic update
      const optimisticState = applyCardPlayUpdate(gameState, card, action);
      setGameState(optimisticState);
      setUpdateHistory(prev => [...prev, update]);

      // Execute API call
      await apiCall();

      // Clear update history on success
      setUpdateHistory([]);

    } catch (error) {
      logger.error('Card action failed, rolling back:', error);
      rollbackLastUpdate();
      throw error;
    } finally {
      setIsProcessing(false);
    }
  }, [gameState, isProcessing, applyCardPlayUpdate, setGameState, rollbackLastUpdate]);

  const executePassAction = useCallback(async (
    action: SerializedAction,
    apiCall: () => Promise<void>
  ) => {
    if (!gameState || isProcessing) return;

    setIsProcessing(true);

    try {
      // Store original state for rollback
      const originalState = gameState;
      const update: OptimisticUpdate = {
        type: 'turn_passed',
        action,
        originalState
      };

      // Apply optimistic update
      const optimisticState = applyPassUpdate(gameState, action);
      setGameState(optimisticState);
      setUpdateHistory(prev => [...prev, update]);

      // Execute API call
      await apiCall();

      // Clear update history on success
      setUpdateHistory([]);

    } catch (error) {
      logger.error('Pass action failed, rolling back:', error);
      rollbackLastUpdate();
      throw error;
    } finally {
      setIsProcessing(false);
    }
  }, [gameState, isProcessing, applyPassUpdate, setGameState, rollbackLastUpdate]);

  return {
    isProcessing,
    executeCardAction,
    executePassAction,
    rollbackLastUpdate
  };
}
