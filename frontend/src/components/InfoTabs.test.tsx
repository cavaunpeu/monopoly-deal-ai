import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { InfoTabs } from './InfoTabs';
import { getDefaultGameConfig, getModels } from '@/lib/api';
import { GameConfig } from '@/types/game';
import React from 'react';
import { MockSelectProps } from '../test/utils/mockTypes';

// Mock the API functions
vi.mock('@/lib/api', () => ({
  getDefaultGameConfig: vi.fn(),
  getModels: vi.fn(),
}));

// Mock the content config
vi.mock('@/config/content.json', () => ({
  default: {
    gameRules: {
      title: "Monopoly Deal Rules",
      content: [
        "🎯 **Objective**: Complete the required number of property sets to win!",
        "",
        "🃏 **Card Types**:",
        "• **Property Cards** - Add these cards to your property pile",
        "• **Cash Cards** - Add these cards to your cash pile",
        "",
        "⚡ **Gameplay**:",
        "• Play up to `{{max_consecutive_player_actions}}` cards per turn",
        "• Start with `{{initial_hand_size}}` cards in hand",
        "• Cash values: `{{cash_card_values}}`",
        "",
        "🏆 **Win**: Complete `{{required_property_sets}}` property sets first!"
      ]
    }
  }
}));

// Mock Radix UI components
vi.mock('@radix-ui/react-select', () => ({
  Root: ({ children, value, onValueChange }: MockSelectProps) => (
    <div data-testid="select-root" data-value={value} data-on-value-change={onValueChange}>
      {children}
    </div>
  ),
  Trigger: ({ children, className, ...props }: MockSelectProps) => (
    <button data-testid="select-trigger" className={className} {...props}>
      {children}
    </button>
  ),
  Value: ({ className }: { className?: string }) => <span data-testid="select-value" className={className} />,
  Icon: ({ children }: { children: React.ReactNode }) => <span data-testid="select-icon">{children}</span>,
  Portal: ({ children }: { children: React.ReactNode }) => <div data-testid="select-portal">{children}</div>,
  Content: ({ children, className, ...props }: MockSelectProps) => (
    <div data-testid="select-content" className={className} {...props}>
      {children}
    </div>
  ),
  Viewport: ({ children, className }: { children: React.ReactNode; className?: string }) => (
    <div data-testid="select-viewport" className={className}>
      {children}
    </div>
  ),
  Item: ({ children, value, className, ...props }: MockSelectProps) => (
    <div data-testid="select-item" data-value={value} className={className} {...props}>
      {children}
    </div>
  ),
  ItemText: ({ children }: { children: React.ReactNode }) => <span data-testid="select-item-text">{children}</span>,
}));

// Mock tooltip components
vi.mock('@/components/ui/tooltip', () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <div data-testid="tooltip">{children}</div>,
  TooltipTrigger: ({ children, asChild }: { children: React.ReactNode; asChild?: boolean }) =>
    asChild ? children : <div data-testid="tooltip-trigger">{children}</div>,
  TooltipContent: ({ children, className }: { children: React.ReactNode; className?: string }) => (
    <div data-testid="tooltip-content" className={className}>
      {children}
    </div>
  ),
}));

const mockGameConfig: GameConfig = {
  required_property_sets: 3,
  max_consecutive_player_actions: 3,
  cash_card_values: [1, 2, 3, 4, 5],
  rent_cards_per_property_type: 2,
  deck_size_multiplier: 1,
  total_deck_size: 30,
  initial_hand_size: 5,
  new_cards_per_turn: 2,
  card_to_special_card_ratio: 0.1,
  required_property_sets_map: { 'GREEN': 3, 'BLUE': 3 },
  property_types: [
    { name: 'GREEN', num_to_complete: 3, rent_progression: [2, 4, 7], cash_value: 4 },
    { name: 'BLUE', num_to_complete: 3, rent_progression: [3, 8], cash_value: 4 }
  ]
};

const mockModels = {
  'model-1': {
    name: 'Model 1',
    description: 'First model description',
    checkpoint_path: '/path/to/model1'
  },
  'model-2': {
    name: 'Model 2',
    description: 'Second model description',
    checkpoint_path: '/path/to/model2'
  }
};

const defaultProps = {
  onNewGame: vi.fn(),
  isCreatingGame: false,
  botSpeed: 3,
  onBotSpeedChange: vi.fn(),
  hasActiveGame: false,
  showSelectionInfo: false,
  onShowSelectionInfoChange: vi.fn(),
};

describe('InfoTabs', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getModels).mockResolvedValue({ models: mockModels });
    vi.mocked(getDefaultGameConfig).mockResolvedValue(mockGameConfig);
  });

  describe('rendering', () => {
    it('renders all control buttons', () => {
      render(<InfoTabs {...defaultProps} />);

      expect(screen.getByText('About This Work')).toBeInTheDocument();
      expect(screen.getByText('Game Rules')).toBeInTheDocument();
      expect(screen.getByText('AI Speed:')).toBeInTheDocument();
      expect(screen.getByText('Show AI Hand')).toBeInTheDocument();
      expect(screen.getByText('New Game')).toBeInTheDocument();
    });

    it('renders AI speed buttons with correct values', () => {
      render(<InfoTabs {...defaultProps} />);

      expect(screen.getByText('1s')).toBeInTheDocument();
      expect(screen.getByText('3s')).toBeInTheDocument();
      expect(screen.getByText('10s')).toBeInTheDocument();
      expect(screen.getByText('30s')).toBeInTheDocument();
    });

    it('shows correct active AI speed', () => {
      render(<InfoTabs {...defaultProps} botSpeed={10} />);

      const speedButton = screen.getByText('10s');
      expect(speedButton).toHaveClass('bg-blue-100', 'text-blue-700');
    });

    it('disables new game button when creating game', () => {
      render(<InfoTabs {...defaultProps} isCreatingGame={true} />);

      const newGameButton = screen.getByText('New Game');
      expect(newGameButton).toBeDisabled();
    });
  });

  describe('AI speed controls', () => {
    it('calls onBotSpeedChange when speed button is clicked', () => {
      render(<InfoTabs {...defaultProps} />);

      fireEvent.click(screen.getByText('10s'));

      expect(defaultProps.onBotSpeedChange).toHaveBeenCalledWith(10);
    });

    it('highlights current speed', () => {
      render(<InfoTabs {...defaultProps} botSpeed={1} />);

      const speedButton = screen.getByText('1s');
      expect(speedButton).toHaveClass('bg-blue-100', 'text-blue-700');
    });
  });

  describe('new game button', () => {
    it('calls onNewGame when clicked', () => {
      render(<InfoTabs {...defaultProps} />);

      fireEvent.click(screen.getByText('New Game'));

      expect(defaultProps.onNewGame).toHaveBeenCalled();
    });

    it('shows loading state when creating game', () => {
      render(<InfoTabs {...defaultProps} isCreatingGame={true} />);

      const newGameButton = screen.getByText('New Game');
      expect(newGameButton).toBeDisabled();

      // Check for spinning icon
      const refreshIcon = newGameButton.querySelector('svg');
      expect(refreshIcon).toHaveClass('animate-spin');
    });
  });

  describe('model selection', () => {
    it('loads models on mount', async () => {
      render(<InfoTabs {...defaultProps} />);

      await waitFor(() => {
        expect(getModels).toHaveBeenCalled();
      });
    });

    it('shows model selector', () => {
      render(<InfoTabs {...defaultProps} />);

      expect(screen.getByTestId('select-root')).toBeInTheDocument();
      expect(screen.getByTestId('select-trigger')).toBeInTheDocument();
    });
  });

  describe('tab functionality', () => {
    it('opens about tab when clicked', async () => {
      render(<InfoTabs {...defaultProps} />);

      // Find the button specifically (not the heading that appears after clicking)
      const aboutButton = screen.getAllByText('About This Work')[0];
      fireEvent.click(aboutButton);

      await waitFor(() => {
        expect(screen.getByText(/This work presents a modified version of the card game/)).toBeInTheDocument();
      });
    });

    it('opens game rules tab when clicked', async () => {
      render(<InfoTabs {...defaultProps} />);

      // Find the button specifically (not the heading that appears after clicking)
      const gameRulesButton = screen.getAllByText('Game Rules')[0];
      fireEvent.click(gameRulesButton);

      await waitFor(() => {
        expect(getDefaultGameConfig).toHaveBeenCalled();
      });
    });
  });

  describe('error handling', () => {
    it('handles models API error gracefully', async () => {
      vi.mocked(getModels).mockRejectedValue(new Error('API Error'));

      render(<InfoTabs {...defaultProps} />);

      await waitFor(() => {
        // Component should still render without crashing
        expect(screen.getByText('New Game')).toBeInTheDocument();
      });
    });

    it('handles game config API error gracefully', async () => {
      vi.mocked(getDefaultGameConfig).mockRejectedValue(new Error('API Error'));

      render(<InfoTabs {...defaultProps} />);

      const gameRulesButton = screen.getAllByText('Game Rules')[0];
      fireEvent.click(gameRulesButton);

      await waitFor(() => {
        // Should still render the tab content
        expect(screen.getAllByText('Game Rules')).toHaveLength(2); // Button + heading
      });
    });
  });

  describe('accessibility', () => {
    it('has proper button roles and labels', () => {
      render(<InfoTabs {...defaultProps} />);

      const newGameButton = screen.getByRole('button', { name: 'New Game' });
      expect(newGameButton).toBeInTheDocument();
    });

    it('has proper focus management', () => {
      render(<InfoTabs {...defaultProps} />);

      const aboutButton = screen.getAllByText('About This Work')[0];
      aboutButton.focus();
      expect(aboutButton).toHaveFocus();
    });
  });

  describe('performance', () => {
    it('only fetches game config when game tab is opened', async () => {
      render(<InfoTabs {...defaultProps} />);

      // Should not fetch game config initially
      expect(getDefaultGameConfig).not.toHaveBeenCalled();

      // Open game tab
      const gameRulesButton = screen.getAllByText('Game Rules')[0];
      fireEvent.click(gameRulesButton);

      await waitFor(() => {
        expect(getDefaultGameConfig).toHaveBeenCalledTimes(1);
      });
    });
  });
});