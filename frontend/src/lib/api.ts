import { config } from './config';
import { HumanState, AiState, SerializedAction, GameConfig, TurnState, PublicPileSizes, GameState } from '@/types/game';

/**
 * Custom error class for API-related errors.
 * Extends the base Error class with additional HTTP status and endpoint information.
 */
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public endpoint: string
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/**
 * Retry logic with exponential backoff for API requests.
 * Automatically retries failed requests with increasing delays, but skips retries for client errors (4xx).
 *
 * @param fn - Function to retry
 * @param maxRetries - Maximum number of retry attempts (default: 3)
 * @param baseDelay - Base delay in milliseconds for exponential backoff (default: 1000)
 * @returns Promise that resolves with the function result or rejects after max retries
 */
async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  baseDelay: number = 1000
): Promise<T> {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      if (attempt === maxRetries - 1) {
        throw error; // Last attempt failed
      }

      // Don't retry on client errors (4xx)
      if (error instanceof ApiError && error.status >= 400 && error.status < 500) {
        throw error;
      }

      // Exponential backoff: 1s, 2s, 4s
      const delay = baseDelay * Math.pow(2, attempt);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  throw new Error('Max retries exceeded');
}

/**
 * Generic API request handler with error handling and retry logic.
 * Wraps fetch requests with automatic retries and standardized error handling.
 *
 * @param endpoint - API endpoint path (relative to base URL)
 * @param options - Optional fetch request options
 * @returns Promise that resolves with the parsed JSON response
 * @throws {ApiError} When the API request fails
 */
async function apiRequest<T>(endpoint: string, options?: RequestInit): Promise<T> {
  return retryWithBackoff(async () => {
    const response = await fetch(`${config.apiUrl}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
    });

    if (!response.ok) {
      throw new ApiError(
        `HTTP ${response.status}: ${response.statusText}`,
        response.status,
        endpoint
      );
    }

    return await response.json();
  });
}

export async function getHumanState(gameId: string): Promise<HumanState> {
  return apiRequest<HumanState>(`/game/${gameId}/human_state`);
}

export async function getAiState(gameId: string, showHand: boolean = false): Promise<AiState> {
  const params = showHand ? '?show_hand=true' : '';
  return apiRequest<AiState>(`/game/${gameId}/ai_state${params}`);
}

export async function getAvailableActions(gameId: string): Promise<SerializedAction[]> {
  const data = await apiRequest<{ actions: SerializedAction[] }>(`/game/${gameId}/actions`);
  return data.actions;
}

export async function getPublicPileSizes(gameId: string): Promise<PublicPileSizes> {
  return apiRequest<PublicPileSizes>(`/game/${gameId}/public_pile_sizes`);
}

/**
 * Creates a new game instance on the server.
 *
 * @returns Promise that resolves with the new game ID
 * @throws {ApiError} When game creation fails
 */
export async function createGame(): Promise<string> {
  const data = await apiRequest<{ game_id: string }>('/game/', { method: 'POST' });
  return data.game_id;
}

export async function getGameConfig(gameId: string): Promise<GameConfig> {
  return apiRequest<GameConfig>(`/game/${gameId}/config`);
}

export async function getDefaultGameConfig(): Promise<GameConfig> {
  return apiRequest<GameConfig>('/game/config');
}

export async function getTurnState(gameId: string): Promise<TurnState> {
  return apiRequest<TurnState>(`/game/${gameId}/turn_state`);
}

/**
 * Fetches the complete game state for a specific game.
 *
 * @param gameId - Unique identifier for the game
 * @param showAiHand - Whether to include AI's hand in the response (for debugging)
 * @returns Promise that resolves with the complete game state
 * @throws {ApiError} When the game is not found or request fails
 */
export async function getGameState(gameId: string, showAiHand: boolean = false): Promise<GameState> {
  const params = showAiHand ? '?show_ai_hand=true' : '';
  return apiRequest<GameState>(`/game/${gameId}/state${params}`);
}

/**
 * Executes a player action in the game.
 *
 * @param gameId - Unique identifier for the game
 * @param actionId - ID of the action to execute
 * @returns Promise that resolves when the action is processed
 * @throws {ApiError} When the action is invalid or game is not found
 */
export async function takeAction(gameId: string, actionId: number): Promise<void> {
  await apiRequest(`/game/${gameId}/step`, {
    method: 'POST',
    body: JSON.stringify({ action_id: actionId }),
  });
}

export async function getAiAction(gameId: string): Promise<SerializedAction> {
  return apiRequest<SerializedAction>(`/game/${gameId}/ai_action`);
}


export async function getSelectionInfo(gameId: string): Promise<Record<string, unknown>> {
  return apiRequest<Record<string, unknown>>(`/game/${gameId}/selection_info`);
}

export async function getModels(): Promise<{ models: Record<string, { name: string; description: string; checkpoint_path: string }> }> {
  return apiRequest<{ models: Record<string, { name: string; description: string; checkpoint_path: string }> }>('/game/models');
}