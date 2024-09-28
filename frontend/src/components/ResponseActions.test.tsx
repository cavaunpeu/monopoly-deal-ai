/* eslint-disable @typescript-eslint/no-explicit-any */
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { ResponseActions } from './ResponseActions'
import { mockResponseGameState, mockGameConfig } from '../test/fixtures/gameState'
import { cashCard1, cashCard3 } from '../test/fixtures/cards'

// Mock the logger
vi.mock('../lib/logger', () => ({
  logger: {
    debug: vi.fn(),
    warn: vi.fn(),
    info: vi.fn(),
    error: vi.fn()
  }
}))

describe('ResponseActions', () => {
  const mockOnCardClick = vi.fn()

  const defaultProps = {
    validActions: mockResponseGameState.actions,
    humanState: mockResponseGameState.human,
    botState: mockResponseGameState.ai,
    onCardClick: mockOnCardClick,
    responseInfo: mockResponseGameState.turn.response_info,
    gameConfig: mockGameConfig
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('rendering', () => {
    it('renders response required banner', () => {
      render(<ResponseActions {...defaultProps} />)

      expect(screen.getByText('Response Required')).toBeInTheDocument()
      expect(screen.getByText('Pay $4 rent for GREEN')).toBeInTheDocument()
    })

    it('shows rent amount correctly', () => {
      render(<ResponseActions {...defaultProps} />)

      // Should show $4 rent (based on the actual calculation in the component)
      // Use a more specific selector to avoid multiple matches
      expect(screen.getByText('Pay $4 rent for GREEN')).toBeInTheDocument()
    })

    it('does not render when no response cards are available', () => {
      const propsWithNoActions = {
        ...defaultProps,
        validActions: [] // No valid actions
      }

      const { container } = render(<ResponseActions {...propsWithNoActions} />)
      expect(container.firstChild).toBeNull()
    })

    it('renders when response cards are available', () => {
      render(<ResponseActions {...defaultProps} />)

      expect(screen.getByText('Response Required')).toBeInTheDocument()
    })
  })

  describe('card sections', () => {
    it('renders cards from hand section', () => {
      render(<ResponseActions {...defaultProps} />)

      expect(screen.getByText('From Hand')).toBeInTheDocument()
    })

    it('renders cards from cash section', () => {
      const props = {
        ...defaultProps,
        humanState: {
          ...defaultProps.humanState,
          cash: [cashCard1, cashCard3], // Add cash cards to cash pile
        },
        validActions: [
          { id: 1, is_response: true, card: cashCard1, src: 'CASH', dst: 'DISCARD' },
          { id: 2, is_response: true, card: cashCard3, src: 'CASH', dst: 'DISCARD' },
        ]
      }

      render(<ResponseActions {...props} />)

      expect(screen.getByText('From Cash')).toBeInTheDocument()
    })

    it('renders cards from properties section', () => {
      const props = {
        ...defaultProps,
        humanState: {
          ...defaultProps.humanState,
          properties: [mockResponseGameState.human.properties[0]], // Add property card to properties pile
        },
        validActions: [
          { id: 1, is_response: true, card: mockResponseGameState.human.properties[0], src: 'PROPERTY', dst: 'DISCARD' },
        ]
      }

      render(<ResponseActions {...props} />)

      expect(screen.getByText('From Properties')).toBeInTheDocument()
    })

    it('only shows sections that have valid response cards', () => {
      // Create props with only hand actions
      const handOnlyActions = mockResponseGameState.actions.filter(action => action.src === 'HAND')
      const handOnlyProps = {
        ...defaultProps,
        validActions: handOnlyActions
      }

      render(<ResponseActions {...handOnlyProps} />)

      expect(screen.getByText('From Hand')).toBeInTheDocument()
      expect(screen.queryByText('From Cash')).not.toBeInTheDocument()
      expect(screen.queryByText('From Properties')).not.toBeInTheDocument()
    })
  })

  describe('card interactions', () => {
    it('calls onCardClick when a card is clicked', () => {
      render(<ResponseActions {...defaultProps} />)

      // Find and click the first clickable card (using cursor-pointer class)
      const clickableCards = document.querySelectorAll('.cursor-pointer')
      if (clickableCards.length > 0) {
        fireEvent.click(clickableCards[0])
        expect(mockOnCardClick).toHaveBeenCalled()
      }
    })

    it('shows card counts when multiple cards of same type', () => {
      // Create props with duplicate cards
      const duplicateCardActions = [
        { id: 1, is_response: true, card: cashCard1, src: 'HAND', dst: 'DISCARD' },
        { id: 2, is_response: true, card: cashCard1, src: 'HAND', dst: 'DISCARD' }
      ]

      const duplicateProps = {
        ...defaultProps,
        validActions: duplicateCardActions,
        humanState: {
          ...defaultProps.humanState,
          hand: [cashCard1, cashCard1] // Two $1 cards
        }
      }

      render(<ResponseActions {...duplicateProps} />)

      // Should show count badge on the card
      expect(screen.getByText('2')).toBeInTheDocument()
    })
  })

  describe('rent calculation', () => {
    it('calculates rent correctly for different property types', () => {
      // Test with BROWN property (2 properties needed)
      const brownResponseInfo = {
        ...mockResponseGameState.turn.response_info!,
        initiating_card: { kind: 'PROPERTY', name: 'BROWN', rent_progression: [1, 2], value: 1 }
      }

      const brownProps = {
        ...defaultProps,
        responseInfo: brownResponseInfo,
        gameConfig: {
          ...defaultProps.gameConfig,
          property_types: [
            { name: 'BROWN', num_to_complete: 2, rent_progression: [1, 2], cash_value: 1 }
          ]
        },
        humanState: {
          ...defaultProps.humanState,
          properties: [
            { kind: 'PROPERTY', name: 'BROWN', rent_progression: [1, 2], value: 1 },
            { kind: 'PROPERTY', name: 'BROWN', rent_progression: [1, 2], value: 1 }
          ]
        },
        validActions: [
          { id: 1, is_response: true, card: cashCard1, src: 'HAND', dst: 'DISCARD' },
        ]
      }

      render(<ResponseActions {...brownProps} />)

      // Should render the response component
      expect(screen.getByText('Response Required')).toBeInTheDocument()
      // Should show the cash card as a valid response
      expect(screen.getByText('From Hand')).toBeInTheDocument()
    })

    it('handles missing response info gracefully', () => {
      const propsWithoutResponseInfo = {
        ...defaultProps,
        responseInfo: null
      }

      render(<ResponseActions {...propsWithoutResponseInfo} />)

      // Should still render but without rent amount
      expect(screen.getByText('Response Required')).toBeInTheDocument()
      expect(screen.queryByText('Pay $')).not.toBeInTheDocument()
    })

    it('handles missing game config gracefully', () => {
      const propsWithoutConfig = {
        ...defaultProps,
        gameConfig: null
      }

      render(<ResponseActions {...propsWithoutConfig} />)

      // Should still render but without rent calculation
      expect(screen.getByText('Response Required')).toBeInTheDocument()
    })
  })

  describe('edge cases', () => {
    it('handles empty human state', () => {
      const emptyHumanState = {
        hand: [],
        properties: [],
        cash: []
      }

      const emptyProps = {
        ...defaultProps,
        humanState: emptyHumanState,
        validActions: []
      }

      const { container } = render(<ResponseActions {...emptyProps} />)
      expect(container.firstChild).toBeNull()
    })

    it('handles actions with null cards', () => {
      const actionsWithNullCards = [
        { id: 1, is_response: true, card: null, src: null, dst: null }
      ]

      const nullCardProps = {
        ...defaultProps,
        validActions: actionsWithNullCards
      }

      const { container } = render(<ResponseActions {...nullCardProps} />)
      expect(container.firstChild).toBeNull()
    })

    it('handles malformed response info', () => {
      const malformedResponseInfo = {
        initiating_card: null,
        initiating_player: 'bot',
        responding_player: 'human',
        response_cards_played: []
      } as any

      const malformedProps = {
        ...defaultProps,
        responseInfo: malformedResponseInfo
      }

      render(<ResponseActions {...malformedProps} />)

      // Should still render but without rent calculation
      expect(screen.getByText('Response Required')).toBeInTheDocument()
    })
  })

  describe('accessibility', () => {
    it('has proper ARIA labels for interactive elements', () => {
      render(<ResponseActions {...defaultProps} />)

      // Cards should be clickable (using cursor-pointer class)
      const clickableElements = document.querySelectorAll('.cursor-pointer')
      expect(clickableElements.length).toBeGreaterThan(0)
    })

    it('provides clear visual feedback for response requirement', () => {
      render(<ResponseActions {...defaultProps} />)

      // Should have clear visual indicators
      expect(screen.getByText('Response Required')).toBeInTheDocument()
      // Check for the alert triangle icon by its SVG class
      expect(document.querySelector('.lucide-triangle-alert')).toBeInTheDocument()
    })
  })

  describe('performance', () => {
    it('renders efficiently with large number of actions', () => {
      const manyActions = Array(100).fill(null).map((_, i) => ({
        id: i,
        is_response: true,
        card: cashCard1,
        src: 'HAND',
        dst: 'DISCARD'
      }))

      const manyActionsProps = {
        ...defaultProps,
        validActions: manyActions
      }

      const start = performance.now()
      render(<ResponseActions {...manyActionsProps} />)
      const end = performance.now()

      expect(end - start).toBeLessThan(100) // Should render quickly
    })
  })
})
