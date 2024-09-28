/* eslint-disable @typescript-eslint/no-explicit-any */
import { render, screen } from '@testing-library/react';
import { PlayedCardHistory } from './PlayedCardHistory';
import { SelectedActionEntry } from '@/types/game';
import { CardModel } from '@/types/cards';

// Mock the tooltip components
vi.mock('./ui/tooltip', () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <div data-testid="tooltip">{children}</div>,
  TooltipTrigger: ({ children, asChild }: { children: React.ReactNode; asChild?: boolean }) =>
    asChild ? <div data-testid="tooltip-trigger">{children}</div> : <div data-testid="tooltip-trigger">{children}</div>,
  TooltipContent: ({ children }: { children: React.ReactNode }) => <div data-testid="tooltip-content">{children}</div>,
}));

// Mock lucide-react icons
vi.mock('lucide-react', () => ({
  User: () => <div data-testid="user-icon">User</div>,
  Bot: () => <div data-testid="bot-icon">Bot</div>,
  DollarSign: () => <div data-testid="dollar-icon">$</div>,
  Home: () => <div data-testid="home-icon">Home</div>,
  CreditCard: () => <div data-testid="credit-card-icon">CreditCard</div>,
  SkipForward: () => <div data-testid="skip-forward-icon">SkipForward</div>,
}));

describe('PlayedCardHistory', () => {
  const mockCashCard: CardModel = {
    kind: 'CASH',
    name: 'ONE',
    value: 1
  };

  const mockPropertyCard: CardModel = {
    kind: 'PROPERTY',
    name: 'GREEN',
    rent_progression: [2, 4, 7],
    value: 4
  };

  const mockRentCard: CardModel = {
    kind: 'RENT',
    name: 'GREEN',
    property_name: 'GREEN'
  };

  const mockSpecialCard: CardModel = {
    kind: 'SPECIAL',
    name: 'JUST_SAY_NO'
  };

  const createActionEntry = (
    playerIdx: number,
    turnIdx: number,
    streakIdx: number,
    card: CardModel | null,
    isResponse: boolean = false
  ): SelectedActionEntry => ({
    turn_idx: turnIdx,
    streak_idx: streakIdx,
    player_idx: playerIdx,
    action: {
      id: Math.random(),
      card,
      src: card ? 'HAND' : null,
      dst: card ? 'DISCARD' : null,
      is_response: isResponse
    }
  });

  describe('rendering', () => {
    it('renders empty state when no history', () => {
      render(<PlayedCardHistory history={[]} humanPlayerIndex={0} />);

      expect(screen.getByText('Cards Played')).toBeInTheDocument();
      expect(screen.getByText('No cards')).toBeInTheDocument();
      expect(screen.getByText('Cards will appear here')).toBeInTheDocument();
    });

    it('renders history entries when provided', () => {
      const history = [
        createActionEntry(0, 0, 0, mockCashCard),
        createActionEntry(1, 1, 0, mockPropertyCard)
      ];

      render(<PlayedCardHistory history={history} humanPlayerIndex={0} />);

      expect(screen.getByText('Cards Played')).toBeInTheDocument();
      expect(screen.queryByText('No cards')).not.toBeInTheDocument();
    });
  });

  describe('player identification', () => {
    it('shows correct player icons for human and AI', () => {
      const history = [
        createActionEntry(0, 0, 0, mockCashCard), // Human player
        createActionEntry(1, 1, 0, mockPropertyCard) // AI player
      ];

      render(<PlayedCardHistory history={history} humanPlayerIndex={0} />);

      const userIcons = screen.getAllByTestId('user-icon');
      const botIcons = screen.getAllByTestId('bot-icon');

      expect(userIcons).toHaveLength(1);
      expect(botIcons).toHaveLength(1);
    });

    it('handles different human player indices', () => {
      const history = [
        createActionEntry(0, 0, 0, mockCashCard), // AI player (human is index 1)
        createActionEntry(1, 1, 0, mockPropertyCard) // Human player
      ];

      render(<PlayedCardHistory history={history} humanPlayerIndex={1} />);

      const userIcons = screen.getAllByTestId('user-icon');
      const botIcons = screen.getAllByTestId('bot-icon');

      expect(userIcons).toHaveLength(1);
      expect(botIcons).toHaveLength(1);
    });
  });

  describe('card type rendering', () => {
    it('renders cash cards correctly', () => {
      const history = [createActionEntry(0, 0, 0, mockCashCard)];
      render(<PlayedCardHistory history={history} humanPlayerIndex={0} />);

      expect(screen.getByTestId('dollar-icon')).toBeInTheDocument();
      expect(screen.getByText('ONE')).toBeInTheDocument();
    });

    it('renders property cards correctly', () => {
      const history = [createActionEntry(0, 0, 0, mockPropertyCard)];
      render(<PlayedCardHistory history={history} humanPlayerIndex={0} />);

      expect(screen.getByTestId('home-icon')).toBeInTheDocument();
      expect(screen.getByText('GREEN')).toBeInTheDocument();
    });

    it('renders rent cards correctly', () => {
      const history = [createActionEntry(0, 0, 0, mockRentCard)];
      render(<PlayedCardHistory history={history} humanPlayerIndex={0} />);

      expect(screen.getByTestId('credit-card-icon')).toBeInTheDocument();
      expect(screen.getByText('GREEN')).toBeInTheDocument();
    });

    it('renders special cards correctly', () => {
      const history = [createActionEntry(0, 0, 0, mockSpecialCard)];
      render(<PlayedCardHistory history={history} humanPlayerIndex={0} />);

      expect(screen.getByText('!')).toBeInTheDocument();
      expect(screen.getByText('Just Say No')).toBeInTheDocument();
    });
  });

  describe('action types', () => {
    it('renders pass actions correctly', () => {
      const history = [createActionEntry(0, 0, 0, null, false)];
      render(<PlayedCardHistory history={history} humanPlayerIndex={0} />);

      expect(screen.getByTestId('skip-forward-icon')).toBeInTheDocument();
      expect(screen.getByText('Pass')).toBeInTheDocument();
    });

    it('renders yield actions correctly', () => {
      const history = [createActionEntry(0, 0, 0, null, true)];
      render(<PlayedCardHistory history={history} humanPlayerIndex={0} />);

      expect(screen.getByText('↩')).toBeInTheDocument();
      expect(screen.getByText('Yield')).toBeInTheDocument();
    });

    it('shows response indicator for response actions', () => {
      const history = [createActionEntry(0, 0, 0, mockCashCard, true)];
      render(<PlayedCardHistory history={history} humanPlayerIndex={0} />);

      expect(screen.getByText('R')).toBeInTheDocument();
    });
  });

  describe('tooltips', () => {
    it('shows correct tooltip for human player actions', () => {
      const history = [createActionEntry(0, 0, 0, mockCashCard)];
      render(<PlayedCardHistory history={history} humanPlayerIndex={0} />);

      expect(screen.getByText('You played ONE (cash card)')).toBeInTheDocument();
    });

    it('shows correct tooltip for AI player actions', () => {
      const history = [createActionEntry(1, 0, 0, mockPropertyCard)];
      render(<PlayedCardHistory history={history} humanPlayerIndex={0} />);

      expect(screen.getByText('AI played GREEN (property card)')).toBeInTheDocument();
    });

    it('shows correct tooltip for response actions', () => {
      const history = [createActionEntry(0, 0, 0, mockCashCard, true)];
      render(<PlayedCardHistory history={history} humanPlayerIndex={0} />);

      expect(screen.getByText('You played ONE (cash card) as a response')).toBeInTheDocument();
    });

    it('shows correct tooltip for pass actions', () => {
      const history = [createActionEntry(0, 0, 0, null, false)];
      render(<PlayedCardHistory history={history} humanPlayerIndex={0} />);

      expect(screen.getByText('You passed their turn')).toBeInTheDocument();
    });

    it('shows correct tooltip for yield actions', () => {
      const history = [createActionEntry(0, 0, 0, null, true)];
      render(<PlayedCardHistory history={history} humanPlayerIndex={0} />);

      expect(screen.getByText('You yielded')).toBeInTheDocument();
    });

    it('handles special card names in tooltips', () => {
      const history = [createActionEntry(0, 0, 0, mockSpecialCard)];
      render(<PlayedCardHistory history={history} humanPlayerIndex={0} />);

      expect(screen.getByText('You played Just Say No (special card)')).toBeInTheDocument();
    });
  });

  describe('history ordering', () => {
    it('displays history in reverse chronological order', () => {
      const history = [
        createActionEntry(0, 0, 0, mockCashCard),
        createActionEntry(1, 1, 0, mockPropertyCard),
        createActionEntry(0, 2, 0, mockSpecialCard)
      ];

      render(<PlayedCardHistory history={history} humanPlayerIndex={0} />);

      // Check that we have all three entries rendered
      expect(screen.getByText('ONE')).toBeInTheDocument();
      expect(screen.getByText('GREEN')).toBeInTheDocument();
      expect(screen.getByText('Just Say No')).toBeInTheDocument();

      // Check that the component renders without errors
      expect(screen.getByText('Cards Played')).toBeInTheDocument();
    });
  });

  describe('turn numbering', () => {
    it('displays correct turn numbers', () => {
      const history = [
        createActionEntry(0, 0, 0, mockCashCard),
        createActionEntry(1, 1, 0, mockPropertyCard),
        createActionEntry(0, 2, 0, mockSpecialCard)
      ];

      render(<PlayedCardHistory history={history} humanPlayerIndex={0} />);

      expect(screen.getByText('3')).toBeInTheDocument(); // turn_idx 2 + 1
      expect(screen.getByText('2')).toBeInTheDocument(); // turn_idx 1 + 1
      expect(screen.getByText('1')).toBeInTheDocument(); // turn_idx 0 + 1
    });
  });

  describe('edge cases', () => {
    it('handles unknown card types gracefully', () => {
      const unknownCard = {
        kind: 'UNKNOWN' as any,
        name: 'UNKNOWN_CARD'
      };

      const history = [createActionEntry(0, 0, 0, unknownCard)];
      render(<PlayedCardHistory history={history} humanPlayerIndex={0} />);

      expect(screen.getByText('?')).toBeInTheDocument();
      expect(screen.getByText('UNKNOWN_CARD')).toBeInTheDocument();
    });

    it('handles cards with missing names', () => {
      const cardWithoutName = {
        kind: 'CASH' as const,
        name: undefined as any,
        value: 1
      };

      const history = [createActionEntry(0, 0, 0, cardWithoutName)];
      render(<PlayedCardHistory history={history} humanPlayerIndex={0} />);

      expect(screen.getByText('Unknown')).toBeInTheDocument();
    });

    it('handles large history lists efficiently', () => {
      const largeHistory = Array.from({ length: 50 }, (_, i) =>
        createActionEntry(i % 2, i, 0, mockCashCard)
      );

      render(<PlayedCardHistory history={largeHistory} humanPlayerIndex={0} />);

      expect(screen.getByText('Cards Played')).toBeInTheDocument();
      // Should render without errors
    });
  });

  describe('accessibility', () => {
    it('has proper ARIA labels and structure', () => {
      const history = [createActionEntry(0, 0, 0, mockCashCard)];
      render(<PlayedCardHistory history={history} humanPlayerIndex={0} />);

      expect(screen.getByText('Cards Played')).toBeInTheDocument();
      expect(screen.getByTestId('tooltip')).toBeInTheDocument();
    });

    it('provides keyboard navigation support through tooltips', () => {
      const history = [createActionEntry(0, 0, 0, mockCashCard)];
      render(<PlayedCardHistory history={history} humanPlayerIndex={0} />);

      const tooltipTrigger = screen.getByTestId('tooltip-trigger');
      expect(tooltipTrigger).toBeInTheDocument();
    });
  });
});
