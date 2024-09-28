import { useState, useCallback } from 'react';
import { CardModel } from '@/types/cards';

interface CardHistoryEntry {
  player: 'human' | 'bot';
  card: CardModel | null;
  turn: number;
  actionType: 'card' | 'pass' | 'yield';
}

export function useCardHistory() {
  const [cardHistory, setCardHistory] = useState<CardHistoryEntry[]>([]);

  const addToHistory = useCallback((entry: Omit<CardHistoryEntry, 'turn'>) => {
    setCardHistory(prev => [...prev, {
      ...entry,
      turn: prev.length, // Use the current history length as a sequence number
    }]);
  }, []);

  const clearHistory = useCallback(() => {
    setCardHistory([]);
  }, []);

  return {
    cardHistory,
    addToHistory,
    clearHistory,
  };
}
