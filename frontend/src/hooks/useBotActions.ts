import { useEffect, useState, useCallback } from 'react';
import { getAiAction, takeAction } from '@/lib/api';
import { logger } from '@/lib/logger';

interface UseBotActionsProps {
  gameId: string;
  isHumanTurn: boolean;
  isResponding: boolean;
  onStateUpdate: () => Promise<void>;
  onCardHistoryUpdate: (entry: {
    player: 'bot';
    card: unknown;
    actionType: 'card' | 'pass' | 'yield';
  }) => void;
}

export function useBotActions({
  gameId,
  isHumanTurn,
  onStateUpdate,
  onCardHistoryUpdate,
}: UseBotActionsProps) {
  const [botThinking, setBotThinking] = useState(false);

  const playBotAction = useCallback(async () => {
    if (isHumanTurn || botThinking) return;

    try {
      setBotThinking(true);

      // Add a delay for "AI is thinking" visual feedback
      await new Promise(resolve => setTimeout(resolve, 1500));

      // Get the AI's selected action
      const botAction = await getAiAction(gameId);
      logger.info('AI playing action:', botAction);

      // Add to card history - determine action type based on SerializedAction structure
      let actionType: 'card' | 'pass' | 'yield' = 'card';
      if (botAction.card !== null) {
        // Has a card - this is a GameAction
        actionType = 'card';
      } else if (botAction.is_response === true) {
        // No card + is_response=true = YieldAction
        actionType = 'yield';
      } else {
        // No card + is_response=false = PassAction
        actionType = 'pass';
      }

      // Add to history
      onCardHistoryUpdate({
        player: 'bot',
        card: botAction.card || null,
        actionType: actionType
      });

      // Execute the AI's action
      await takeAction(gameId, botAction.id);

      // Refresh game state to get the new turn state
      await onStateUpdate();

      setBotThinking(false);
    } catch (error) {
      logger.error('Error with AI action:', error);
      setBotThinking(false);
    }
  }, [gameId, isHumanTurn, botThinking, onStateUpdate, onCardHistoryUpdate]);

  useEffect(() => {
    if (!isHumanTurn && !botThinking) {
      playBotAction();
    }
  }, [isHumanTurn, playBotAction, botThinking]);

  return { botThinking };
}
