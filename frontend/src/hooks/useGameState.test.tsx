/* eslint-disable @typescript-eslint/no-explicit-any */
import { renderHook, waitFor } from '@testing-library/react';
import { useGameState } from './useGameState';
import { getGameState, ApiError } from '@/lib/api';
import { GameState } from '@/types/game';

// Mock the API module
vi.mock('@/lib/api', () => ({
  getGameState: vi.fn(),
  ApiError: class ApiError extends Error {
    constructor(message: string, public status: number, public endpoint: string) {
      super(message);
      this.name = 'ApiError';
    }
  }
}));

// Mock the logger
vi.mock('@/lib/logger', () => ({
  logger: {
    error: vi.fn()
  }
}));

describe('useGameState', () => {
  const mockGameId = 'test-game-123';

  const mockGameState: GameState = {
    turn: {
      turn_idx: 0,
      streak_idx: 0,
      streaking_player_idx: 0,
      acting_player_idx: 0,
      is_human_turn: true,
      human_player_index: 0,
      cards_played_this_turn: 0,
      max_cards_per_turn: 3,
      remaining_cards: 3,
      selected_actions: [],
      game_over: false,
      winner: null,
      is_responding: false,
      response_info: null
    },
    human: {
      hand: [],
      properties: [],
      cash: []
    },
    ai: {
      properties: [],
      cash: [],
      hand_count: 5,
      hand: undefined
    },
    piles: {
      deck: 20,
      discard: 0
    },
    config: {
      required_property_sets: 3,
      max_consecutive_player_actions: 3,
      cash_card_values: [1, 2, 3, 4, 5],
      rent_cards_per_property_type: 2,
      deck_size_multiplier: 1,
      total_deck_size: 30,
      initial_hand_size: 5,
      new_cards_per_turn: 2,
      card_to_special_card_ratio: 0.1,
      required_property_sets_map: {},
      property_types: []
    },
    actions: []
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('initial state', () => {
    it('returns initial state correctly', () => {
      const { result } = renderHook(() => useGameState(mockGameId));

      expect(result.current.gameState).toBeNull();
      expect(result.current.isHumanTurn).toBe(false);
      expect(result.current.isLoading).toBe(true);
      expect(result.current.error).toBeNull();
      expect(typeof result.current.fetchGameState).toBe('function');
    });

    it('handles empty gameId', () => {
      const { result } = renderHook(() => useGameState(''));

      expect(result.current.gameState).toBeNull();
      expect(result.current.isHumanTurn).toBe(false);
      expect(result.current.isLoading).toBe(true);
      expect(result.current.error).toBeNull();
    });
  });

  describe('successful data fetching', () => {
    it('fetches and sets game state successfully', async () => {
      vi.mocked(getGameState).mockResolvedValue(mockGameState);

      const { result } = renderHook(() => useGameState(mockGameId));

      await waitFor(() => {
        expect(result.current.gameState).toEqual(mockGameState);
      });

      expect(result.current.isHumanTurn).toBe(true);
      expect(result.current.error).toBeNull();
      expect(getGameState).toHaveBeenCalledWith(mockGameId);
    });

    it('updates isHumanTurn based on game state', async () => {
      const aiTurnState = {
        ...mockGameState,
        turn: {
          ...mockGameState.turn,
          is_human_turn: false
        }
      };

      vi.mocked(getGameState).mockResolvedValue(aiTurnState);

      const { result } = renderHook(() => useGameState(mockGameId));

      await waitFor(() => {
        expect(result.current.gameState).toEqual(aiTurnState);
      });

      expect(result.current.isHumanTurn).toBe(false);
    });

    it('handles null game state gracefully', async () => {
      vi.mocked(getGameState).mockResolvedValue(null as any);

      const { result } = renderHook(() => useGameState(mockGameId));

      await waitFor(() => {
        expect(result.current.gameState).toBeNull();
      });

      expect(result.current.isHumanTurn).toBe(false);
    });
  });

  describe('error handling', () => {
    it('handles API errors correctly', async () => {
      const apiError = new ApiError('Game not found', 404, '/game/test-game-123/state');
      vi.mocked(getGameState).mockRejectedValue(apiError);

      const { result } = renderHook(() => useGameState(mockGameId));

      await waitFor(() => {
        expect(result.current.error).toBe('API Error: Game not found');
      });

      expect(result.current.gameState).toBeNull();
      expect(result.current.isHumanTurn).toBe(false);
    });

    it('handles generic errors correctly', async () => {
      const genericError = new Error('Network error');
      vi.mocked(getGameState).mockRejectedValue(genericError);

      const { result } = renderHook(() => useGameState(mockGameId));

      await waitFor(() => {
        expect(result.current.error).toBe('Unexpected error: Network error');
      });

      expect(result.current.gameState).toBeNull();
    });

    it('handles unknown error types', async () => {
      vi.mocked(getGameState).mockRejectedValue('String error');

      const { result } = renderHook(() => useGameState(mockGameId));

      await waitFor(() => {
        expect(result.current.error).toBe('Unexpected error: Unknown error');
      });

      expect(result.current.gameState).toBeNull();
    });

    it('clears error on successful refetch', async () => {
      // First call fails
      vi.mocked(getGameState).mockRejectedValueOnce(new Error('Initial error'));

      const { result } = renderHook(() => useGameState(mockGameId));

      await waitFor(() => {
        expect(result.current.error).toBe('Unexpected error: Initial error');
      });

      // Second call succeeds
      vi.mocked(getGameState).mockResolvedValue(mockGameState);

      await result.current.fetchGameState();

      await waitFor(() => {
        expect(result.current.error).toBeNull();
        expect(result.current.gameState).toEqual(mockGameState);
      });
    });
  });

  describe('refetching', () => {
    it('allows manual refetching of game state', async () => {
      vi.mocked(getGameState).mockResolvedValue(mockGameState);

      const { result } = renderHook(() => useGameState(mockGameId));

      await waitFor(() => {
        expect(result.current.gameState).toEqual(mockGameState);
      });

      // Clear the mock to verify it's called again
      vi.mocked(getGameState).mockClear();

      // Manually refetch
      await result.current.fetchGameState();

      expect(getGameState).toHaveBeenCalledWith(mockGameId);
    });

    it('handles refetch errors', async () => {
      vi.mocked(getGameState).mockResolvedValueOnce(mockGameState);

      const { result } = renderHook(() => useGameState(mockGameId));

      await waitFor(() => {
        expect(result.current.gameState).toEqual(mockGameState);
      });

      // Second call fails
      const refetchError = new Error('Refetch failed');
      vi.mocked(getGameState).mockRejectedValue(refetchError);

      await result.current.fetchGameState();

      await waitFor(() => {
        expect(result.current.error).toBe('Unexpected error: Refetch failed');
      });

      // Game state should remain from previous successful fetch
      expect(result.current.gameState).toEqual(mockGameState);
    });
  });

  describe('gameId changes', () => {
    it('refetches when gameId changes', async () => {
      const { rerender } = renderHook(
        ({ gameId }) => useGameState(gameId),
        { initialProps: { gameId: 'game-1' } }
      );

      vi.mocked(getGameState).mockResolvedValue(mockGameState);

      await waitFor(() => {
        expect(getGameState).toHaveBeenCalledWith('game-1');
      });

      // Change gameId
      rerender({ gameId: 'game-2' });

      await waitFor(() => {
        expect(getGameState).toHaveBeenCalledWith('game-2');
      });

      expect(getGameState).toHaveBeenCalledTimes(2);
    });

    it('handles empty gameId on rerender', async () => {
      const { rerender } = renderHook(
        ({ gameId }) => useGameState(gameId),
        { initialProps: { gameId: mockGameId } }
      );

      vi.mocked(getGameState).mockResolvedValue(mockGameState);

      await waitFor(() => {
        expect(getGameState).toHaveBeenCalledWith(mockGameId);
      });

      // Change to empty gameId
      rerender({ gameId: '' });

      // Should not call API with empty gameId
      expect(getGameState).toHaveBeenCalledTimes(1);
    });
  });

  describe('edge cases', () => {
    it('handles undefined game state turn', async () => {
      const stateWithoutTurn = {
        ...mockGameState,
        turn: undefined as any
      };

      vi.mocked(getGameState).mockResolvedValue(stateWithoutTurn);

      const { result } = renderHook(() => useGameState(mockGameId));

      await waitFor(() => {
        expect(result.current.gameState).toEqual(stateWithoutTurn);
      });

      expect(result.current.isHumanTurn).toBe(false);
    });

    it('handles malformed game state', async () => {
      const malformedState = {
        turn: {
          is_human_turn: true
        }
        // Missing other required fields
      } as any;

      vi.mocked(getGameState).mockResolvedValue(malformedState);

      const { result } = renderHook(() => useGameState(mockGameId));

      await waitFor(() => {
        expect(result.current.gameState).toEqual(malformedState);
      });

      expect(result.current.isHumanTurn).toBe(true);
    });
  });

  describe('performance', () => {
    it('memoizes fetchGameState function', () => {
      const { result, rerender } = renderHook(() => useGameState(mockGameId));

      const firstFetchFunction = result.current.fetchGameState;

      rerender();

      const secondFetchFunction = result.current.fetchGameState;

      expect(firstFetchFunction).toBe(secondFetchFunction);
    });

    it('handles rapid successive calls', async () => {
      vi.mocked(getGameState).mockResolvedValue(mockGameState);

      const { result } = renderHook(() => useGameState(mockGameId));

      // Make multiple rapid calls
      const promises = [
        result.current.fetchGameState(),
        result.current.fetchGameState(),
        result.current.fetchGameState()
      ];

      await Promise.all(promises);

      await waitFor(() => {
        expect(result.current.gameState).toEqual(mockGameState);
      });

      // Should have been called multiple times
      expect(getGameState).toHaveBeenCalledTimes(4); // 1 initial + 3 manual calls
    });
  });
});
