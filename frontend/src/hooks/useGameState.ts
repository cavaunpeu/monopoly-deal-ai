import { useState, useEffect, useCallback } from 'react';
import {
  getGameState,
  ApiError
} from '@/lib/api';
import { GameState } from '@/types/game';
import { logger } from '@/lib/logger';

interface UseGameStateReturn {
  /** Current game state from the server */
  gameState: GameState | null;
  /** Whether it's currently the human player's turn */
  isHumanTurn: boolean;
  /** Whether the game state is currently loading */
  isLoading: boolean;
  /** Any error that occurred while fetching game state */
  error: string | null;
  /** Function to manually refresh the game state */
  fetchGameState: () => Promise<void>;
}

/**
 * Custom hook for managing game state with automatic fetching and error handling.
 * Provides a clean interface for components that need to access and refresh game state.
 *
 * @param gameId - Unique identifier for the game
 * @returns Object containing game state and related utilities
 */
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
