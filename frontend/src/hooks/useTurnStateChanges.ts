import { useMemo } from 'react';
import { GameState } from '@/types/game';

export function useTurnState(gameState: GameState | null) {
  return useMemo(() => {
    if (!gameState) return null;

    return {
      isHumanTurn: gameState.turn.is_human_turn,
      isResponding: gameState.turn.is_responding,
      turnIdx: gameState.turn.turn_idx,
      streakIdx: gameState.turn.streak_idx,
      actingPlayerIdx: gameState.turn.acting_player_idx,
      streakingPlayerIdx: gameState.turn.streaking_player_idx,
    };
  }, [gameState]);
}

export function useAiActionDependencies(gameState: GameState | null, botSpeed: number, gameId: string) {
  return useMemo(() => [
    gameState,
    gameId,
    botSpeed,
  ], [gameState, gameId, botSpeed]);
}

export function useSelectionInfoDependencies(gameState: GameState | null, showSelectionInfo: boolean, gameId: string) {
  const turnState = useTurnState(gameState);

  return useMemo(() => [
    turnState?.isHumanTurn,
    showSelectionInfo,
    gameId,
  ], [turnState, showSelectionInfo, gameId]);
}
