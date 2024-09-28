import { useState, useEffect, useCallback } from 'react';
import {
  getGameState,
  ApiError
} from '@/lib/api';
import { GameState } from '@/types/game';
import { logger } from '@/lib/logger';

interface UseGameStateReturn {
  gameState: GameState | null;
  isHumanTurn: boolean;
  isLoading: boolean;
  error: string | null;
  fetchGameState: () => Promise<void>;
}

export function useGameState(gameId: string): UseGameStateReturn {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [isLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchGameState = useCallback(async () => {
    if (!gameId) return;

    try {
      setError(null);

      const state = await getGameState(gameId);

      setGameState(state);
    } catch (err) {
      const errorMessage = err instanceof ApiError
        ? `API Error: ${err.message}`
        : `Unexpected error: ${err instanceof Error ? err.message : 'Unknown error'}`;
      setError(errorMessage);
      logger.error('Error fetching game state:', err);
    }
  }, [gameId]);

  useEffect(() => {
    fetchGameState();
  }, [fetchGameState]);

  const isHumanTurn = gameState?.turn?.is_human_turn ?? false;

  return {
    gameState,
    isHumanTurn,
    isLoading,
    error,
    fetchGameState,
  };
}
