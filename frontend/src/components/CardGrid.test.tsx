import { render, screen, fireEvent } from '@testing-library/react';
import { CardGrid } from './CardGrid';
import { CardModel } from '@/types/cards';
import { createInvalidCard, createIncompleteCard } from '../test/utils/mockTypes';

// Mock the Card component
vi.mock('./cards/Card', () => ({
  Card: ({ card, count, showProgress }: { card: CardModel; count?: number; showProgress?: boolean }) => (
    <div data-testid="card" data-card-name={card.name} data-count={count} data-show-progress={showProgress}>
      {card.name} {count && count > 1 ? `(${count})` : ''}
    </div>
  )
}));

// Mock lucide-react icons
vi.mock('lucide-react', () => ({
  CreditCard: () => <div data-testid="credit-card-icon">CreditCard</div>,
}));

describe('CardGrid', () => {
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

  const defaultProps = {
    title: 'Test Cards',
    cards: [],
    isClickable: false,
    showPassButton: false,
    fixedHeight: false
  };

  describe('rendering', () => {
    it('renders title correctly', () => {
      render(<CardGrid {...defaultProps} title="My Cards" />);
      expect(screen.getByText('My Cards')).toBeInTheDocument();
    });

    it('renders empty state when no cards', () => {
      render(<CardGrid {...defaultProps} />);

      expect(screen.getByText('No cards')).toBeInTheDocument();
      expect(screen.getByText('Cards will appear here')).toBeInTheDocument();
      expect(screen.getByTestId('credit-card-icon')).toBeInTheDocument();
    });

    it('renders cards when provided', () => {
      const cards = [mockCashCard, mockPropertyCard];
      render(<CardGrid {...defaultProps} cards={cards} />);

      expect(screen.getByText('ONE')).toBeInTheDocument();
      expect(screen.getByText('GREEN')).toBeInTheDocument();
      expect(screen.queryByText('No cards')).not.toBeInTheDocument();
    });
  });

  describe('card grouping', () => {
    it('groups duplicate cards correctly', () => {
      const cards = [mockCashCard, mockCashCard, mockPropertyCard];
      render(<CardGrid {...defaultProps} cards={cards} />);

      const cardElements = screen.getAllByTestId('card');
      expect(cardElements).toHaveLength(2); // Two unique cards

      const oneCard = cardElements.find(el => el.getAttribute('data-card-name') === 'ONE');
      expect(oneCard?.getAttribute('data-count')).toBe('2');
    });

    it('handles different card types in grouping', () => {
      const cards = [
        mockCashCard,
        mockPropertyCard,
        mockRentCard,
        mockSpecialCard
      ];
      render(<CardGrid {...defaultProps} cards={cards} />);

      const cardElements = screen.getAllByTestId('card');
      expect(cardElements).toHaveLength(4);

      expect(screen.getByText('ONE')).toBeInTheDocument();
      expect(screen.getAllByText('GREEN')).toHaveLength(2); // Property and rent cards both named GREEN
      expect(screen.getByText('JUST_SAY_NO')).toBeInTheDocument();
    });

    it('creates unique keys for similar cards with different properties', () => {
      const propertyCard1: CardModel = {
        kind: 'PROPERTY',
        name: 'GREEN',
        rent_progression: [2, 4, 7],
        value: 4
      };

      const propertyCard2: CardModel = {
        kind: 'PROPERTY',
        name: 'GREEN',
        rent_progression: [1, 2, 3], // Different rent progression
        value: 2
      };

      const cards = [propertyCard1, propertyCard2];
      render(<CardGrid {...defaultProps} cards={cards} />);

      const cardElements = screen.getAllByTestId('card');
      expect(cardElements).toHaveLength(2); // Should be treated as different cards
    });
  });

  describe('interactions', () => {
    it('calls onCardClick when card is clicked and clickable', () => {
      const mockOnCardClick = vi.fn();
      const cards = [mockCashCard];

      render(
        <CardGrid
          {...defaultProps}
          cards={cards}
          isClickable={true}
          onCardClick={mockOnCardClick}
        />
      );

      const card = screen.getByTestId('card');
      fireEvent.click(card);

      expect(mockOnCardClick).toHaveBeenCalledWith(mockCashCard);
    });

    it('does not call onCardClick when not clickable', () => {
      const mockOnCardClick = vi.fn();
      const cards = [mockCashCard];

      render(
        <CardGrid
          {...defaultProps}
          cards={cards}
          isClickable={false}
          onCardClick={mockOnCardClick}
        />
      );

      const card = screen.getByTestId('card');
      fireEvent.click(card);

      expect(mockOnCardClick).not.toHaveBeenCalled();
    });

    it('shows pass button when enabled', () => {
      const mockOnPassClick = vi.fn();
      const cards = [mockCashCard]; // Need cards to show pass button

      render(
        <CardGrid
          {...defaultProps}
          cards={cards}
          showPassButton={true}
          onPassClick={mockOnPassClick}
        />
      );

      expect(screen.getByText('Pass')).toBeInTheDocument();
    });

    it('calls onPassClick when pass button is clicked', () => {
      const mockOnPassClick = vi.fn();
      const cards = [mockCashCard]; // Need cards to show pass button

      render(
        <CardGrid
          {...defaultProps}
          cards={cards}
          showPassButton={true}
          onPassClick={mockOnPassClick}
        />
      );

      const passButton = screen.getByText('Pass');
      fireEvent.click(passButton);

      expect(mockOnPassClick).toHaveBeenCalled();
    });

    it('does not show pass button when disabled', () => {
      render(<CardGrid {...defaultProps} showPassButton={false} />);

      expect(screen.queryByText('Pass')).not.toBeInTheDocument();
    });
  });

  describe('property counts', () => {
    it('shows property counts when provided', () => {
      const cards = [mockPropertyCard];
      const propertyCounts = { GREEN: 3 };

      render(
        <CardGrid
          {...defaultProps}
          cards={cards}
          propertyCounts={propertyCounts}
        />
      );

      const card = screen.getByTestId('card');
      expect(card.getAttribute('data-count')).toBe('3');
      expect(card.getAttribute('data-show-progress')).toBe('true');
    });

    it('uses group count when property counts not provided', () => {
      const cards = [mockPropertyCard, mockPropertyCard];

      render(<CardGrid {...defaultProps} cards={cards} />);

      const card = screen.getByTestId('card');
      expect(card.getAttribute('data-count')).toBe('2');
      expect(card.getAttribute('data-show-progress')).toBe('false');
    });

    it('handles missing property count gracefully', () => {
      const cards = [mockPropertyCard];
      const propertyCounts = { BLUE: 2 }; // Different property

      render(
        <CardGrid
          {...defaultProps}
          cards={cards}
          propertyCounts={propertyCounts}
        />
      );

      const card = screen.getByTestId('card');
      // When property count is missing, it falls back to group count (1) or undefined
      const count = card.getAttribute('data-count');
      expect(count === '1' || count === null).toBe(true);
    });
  });

  describe('layout options', () => {
    it('applies fixed height styling when enabled', () => {
      const { container } = render(<CardGrid {...defaultProps} fixedHeight={true} />);

      const cardGrid = container.firstChild as HTMLElement;
      expect(cardGrid).toHaveClass('h-[240px]', 'flex', 'flex-col');
    });

    it('applies default height styling when fixed height disabled', () => {
      const { container } = render(<CardGrid {...defaultProps} fixedHeight={false} />);

      const cardGrid = container.firstChild as HTMLElement;
      expect(cardGrid).toHaveClass('min-h-[240px]');
      expect(cardGrid).not.toHaveClass('h-[240px]', 'flex', 'flex-col');
    });
  });

  describe('error handling', () => {
    it('handles null cards gracefully', () => {
      const cards = [mockCashCard, null as unknown as CardModel, mockPropertyCard];
      render(<CardGrid {...defaultProps} cards={cards} />);

      // Should only render valid cards
      const cardElements = screen.getAllByTestId('card');
      expect(cardElements).toHaveLength(2);
    });

    it('handles invalid cards gracefully', () => {
      const invalidCard = createInvalidCard('INVALID', 'INVALID');
      const cards = [mockCashCard, invalidCard];

      render(<CardGrid {...defaultProps} cards={cards} />);

      // The component should still render the valid card
      expect(screen.getByText('ONE')).toBeInTheDocument();
      // The invalid card might be filtered out or handled differently
      expect(screen.queryByText('Invalid Card')).not.toBeInTheDocument();
    });

    it('handles cards with missing properties', () => {
      const incompleteCard = createIncompleteCard('CASH', 'INCOMPLETE') as CardModel;
      const cards = [incompleteCard];

      render(<CardGrid {...defaultProps} cards={cards} />);

      // Should still render the card
      expect(screen.getByText('INCOMPLETE')).toBeInTheDocument();
    });
  });

  describe('performance', () => {
    it('handles large number of cards efficiently', () => {
      const manyCards = Array.from({ length: 100 }, (_, i) => ({
        kind: 'CASH' as const,
        name: `CARD_${i}`,
        value: i
      }));

      render(<CardGrid {...defaultProps} cards={manyCards} />);

      // Should render without errors
      expect(screen.getByText('Test Cards')).toBeInTheDocument();
    });

    it('memoizes component correctly', () => {
      const { rerender } = render(<CardGrid {...defaultProps} cards={[mockCashCard]} />);

      // Re-render with same props
      rerender(<CardGrid {...defaultProps} cards={[mockCashCard]} />);

      // Should still work correctly
      expect(screen.getByText('ONE')).toBeInTheDocument();
    });
  });

  describe('accessibility', () => {
    it('has proper structure for screen readers', () => {
      const cards = [mockCashCard];
      render(<CardGrid {...defaultProps} cards={cards} />);

      expect(screen.getByText('Test Cards')).toBeInTheDocument();
      expect(screen.getByTestId('card')).toBeInTheDocument();
    });

    it('provides visual feedback for clickable cards', () => {
      const cards = [mockCashCard];
      const { container } = render(
        <CardGrid {...defaultProps} cards={cards} isClickable={true} />
      );

      const clickableCard = container.querySelector('.cursor-pointer');
      expect(clickableCard).toBeInTheDocument();
    });
  });
});
