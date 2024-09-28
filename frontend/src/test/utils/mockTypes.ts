import React from 'react';
import { CardModel } from '@/types/cards';

// Proper types for test mocks
export interface MockSelectProps {
  children: React.ReactNode;
  value?: string;
  onValueChange?: (value: string) => void;
  className?: string;
  [key: string]: unknown;
}

export interface MockTooltipProps {
  children: React.ReactNode;
  asChild?: boolean;
  className?: string;
  [key: string]: unknown;
}

// Utility to create invalid cards for testing error handling
export function createInvalidCard(kind: string, name: string): CardModel {
  return { kind: kind as CardModel['kind'], name } as CardModel;
}

// Utility to create cards with missing properties for testing
export function createIncompleteCard(kind: string, name: string): Partial<CardModel> {
  return { kind: kind as CardModel['kind'], name };
}

// Utility to create arrays with null/undefined for testing
export function createMixedCardArray(...cards: (CardModel | null | undefined)[]): (CardModel | null | undefined)[] {
  return cards;
}
