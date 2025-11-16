import { useEffect, useState, useCallback } from "react";
import {
    getGameState,
    takeAction,
    getAiAction,
    getSelectionInfo,
    ApiError,
} from "@/lib/api";
import { CardGrid } from "./CardGrid";
import { GameOverview } from "./GameOverview";
import { PlayedCardHistory } from "./PlayedCardHistory";
import { ResponseActions } from "./ResponseActions";
import { SelectionInfoPanel } from "./SelectionInfoPanel";
import { GameNotFound } from "./GameNotFound";
import { CardModel, SerializedAction, getCardKind, PropertyCardModel } from "@/types/cards";
import { GameState } from "@/types/game";
import { logger } from "@/lib/logger";
import { useGameActions } from "@/hooks/useGameActions";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
} from "./ui/dialog";

type Props = {
    /** Unique identifier for the current game */
    gameId: string;
    /** Delay in seconds for bot actions (affects AI response time) */
    botSpeed: number;
    /** Whether to show selection information panel */
    showSelectionInfo: boolean;
    /** Callback to start a new game */
    onNewGame: () => void;
    /** Whether a new game is currently being created */
    isCreatingGame: boolean;
};

/**
 * Main game board component that orchestrates the entire game interface.
 * Manages game state, handles player and AI actions, and renders all game UI components.
 *
 * @param props - Component props
 * @returns JSX element containing the complete game interface
 */
export function GameBoard({ gameId, botSpeed, showSelectionInfo }: Props) {
    const [gameState, setGameState] = useState<GameState | null>(null);
    const [botThinking, setBotThinking] = useState(false);
    const [showJustSayNoDialog, setShowJustSayNoDialog] = useState(false);
    const [selectionInfo, setSelectionInfo] = useState<Record<string, unknown> | null>(null);
    const [gameNotFound, setGameNotFound] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Derived state from single source of truth
    const isHumanTurn = gameState?.turn.is_human_turn ?? false;
    const isGameOver = gameState?.turn.game_over ?? false;

    // Use the new game actions hook for clean optimistic updates
    const { isProcessing: isProcessingAction, playCard, passTurn } = useGameActions(gameState, setGameState);

    // Stable dependencies for effects - derived from gameState
    const isAiTurn = gameState && !gameState.turn.is_human_turn;
    const shouldShowSelectionInfo = showSelectionInfo && isAiTurn && !botThinking;

    /**
     * Fetches the current game state from the API and updates local state.
     * Handles error cases including game not found (404) and other API errors.
     */
    const fetchGameState = useCallback(async () => {
        if (!gameId) return;

        try {
            setGameNotFound(false);
            setError(null);
            const state = await getGameState(gameId, showSelectionInfo);

            setGameState(state);
        } catch (error) {
            if (error instanceof ApiError && error.status === 404) {
                setGameNotFound(true);
                logger.error('Game not found:', gameId);
            } else {
                const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
                setError(errorMessage);
                logger.error('Error fetching game state:', error);
            }
        }
    }, [gameId, showSelectionInfo]);

    const handleRetry = useCallback(() => {
        setError(null);
        fetchGameState();
    }, [fetchGameState]);

    // Fetch game state when gameId changes
    useEffect(() => {
        fetchGameState();
    }, [fetchGameState]);

    // Fetch selection info when AI is thinking and selection info is enabled
    useEffect(() => {
        if (shouldShowSelectionInfo) {
            const fetchSelectionInfo = async () => {
                try {
                    const info = await getSelectionInfo(gameId);
                    setSelectionInfo(info);
                } catch (error) {
                    logger.error('Error fetching selection info:', error);
                    setSelectionInfo(null);
                }
            };
            fetchSelectionInfo();
        } else if (gameState && gameState.turn.is_human_turn) {
            // Clear selection info when it's human's turn
            setSelectionInfo(null);
        }
    }, [shouldShowSelectionInfo, gameId, gameState]);

    // Auto-play AI actions when it's the AI's turn
    useEffect(() => {
        if (isAiTurn && !botThinking && !isGameOver) {
            const playBotAction = async () => {
                try {
                    setBotThinking(true);

                    // Add a delay for "AI is thinking" visual feedback
                    await new Promise(resolve => setTimeout(resolve, botSpeed * 1000));

                    // Get the AI's selected action
                    const botAction = await getAiAction(gameId);
                    logger.info('AI playing action:', botAction);

                    // Execute the AI's action
                    await takeAction(gameId, botAction.id);

                    // Refresh game state to get the new turn state
                    await fetchGameState();

                    setBotThinking(false);
                } catch (error) {
                    const errorMessage = error instanceof Error ? error.message : 'AI action failed';
                    setError(errorMessage);
                    logger.error('Error with AI action:', error);
                    setBotThinking(false);
                }
            };

            playBotAction();
        }
    }, [isAiTurn, botThinking, botSpeed, gameId, fetchGameState, isGameOver]);


    const handleCardClick = async (card: CardModel) => {
        if (!isHumanTurn || isProcessingAction || !gameState) return;

        // Check if the player has reached their turn limit
        if (gameState.turn.remaining_cards <= 0) {
            logger.info('Turn limit reached, cannot play more cards');
            return;
        }

        // Check if it's a Just Say No card
        const cardKind = getCardKind(card);
        if (cardKind === "SPECIAL" && card.name === "JUST_SAY_NO") {
            // Only show "Not Available" dialog if it's NOT a response turn
            if (!gameState.turn.is_responding) {
                setShowJustSayNoDialog(true);
                return;
            }
        }

        try {
            logger.debug('Available actions:', gameState.actions);

            // Find matching action for this card
            const matchingAction = gameState.actions.find((action: SerializedAction) => {
                if (!action.card) return false;

                // Match card kind and name
                const actionCardKind = getCardKind(action.card);
                const cardMatches = actionCardKind === cardKind && action.card.name === card.name;

                // Match source pile - can be from HAND, PROPERTY, or CASH depending on the action
                const srcMatches = action.src === "HAND" || action.src === "PROPERTY" || action.src === "CASH";

                return cardMatches && srcMatches;
            });

            if (matchingAction) {
                logger.info('Playing card:', card, 'with action ID:', matchingAction.id);

                // Perform optimistic update
                await playCard(card, matchingAction, async () => {
                    await takeAction(gameId, matchingAction.id);
                    await fetchGameState();
                });
            } else {
                logger.warn('No matching action found for card:', card);
            }
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to play card';
            setError(errorMessage);
            logger.error('Error playing card:', error);
        }
    };

    const handlePassClick = useCallback(async () => {
        if (!isHumanTurn || isProcessingAction || !gameState) return;

        try {
            // Find pass action (has card: null)
            const passAction = gameState.actions.find((action: SerializedAction) => action.card === null);

            if (passAction) {
                logger.info('Passing turn with action ID:', passAction.id);

                // Perform optimistic update
                await passTurn(passAction, async () => {
                    await takeAction(gameId, passAction.id);
                    const newState = await getGameState(gameId, showSelectionInfo);
                    setGameState(newState);
                });
            } else {
                logger.warn('No pass action available');
            }
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to pass turn';
            setError(errorMessage);
            logger.error('Error passing:', error);
        }
    }, [isHumanTurn, isProcessingAction, gameState, gameId, showSelectionInfo, passTurn]);

    // Auto-yield for human when they have only a yield action during a response
    useEffect(() => {
        if (isHumanTurn && gameState?.turn.is_responding && gameState.actions.length === 1 && gameState.actions[0].card === null && gameState.actions[0].is_response) {
            // This is a yield action - automatically execute it
            logger.info('Human has no valid response actions, auto-yielding');
            handlePassClick();
        }
    }, [isHumanTurn, gameState?.turn.is_responding, gameState?.actions, handlePassClick]);

    if (gameNotFound) {
        return <GameNotFound gameId={gameId} />;
    }

    if (error) {
        return (
            <div className="space-y-4">
                <div className="border border-red-200 rounded-lg p-8 bg-red-50 text-center shadow-lg">
                    <h2 className="text-2xl font-bold mb-4 text-red-800">Something went wrong</h2>
                    <p className="text-red-700 mb-6">{error}</p>
                    <button
                        onClick={handleRetry}
                        className="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded transition-colors"
                    >
                        Try Again
                    </button>
                </div>
            </div>
        );
    }

    if (!gameState) return <div>Loading...</div>;

    // Get AI hand count from the API response
    const botHandCount = gameState.ai.hand_count || 0;

    // Convert properties to counts for display
    const humanProperties = gameState.human.properties.reduce((acc: Record<string, number>, card: PropertyCardModel) => {
        acc[card.name] = (acc[card.name] || 0) + 1;
        return acc;
    }, {});

    const botProperties = gameState.ai.properties.reduce((acc: Record<string, number>, card: PropertyCardModel) => {
        acc[card.name] = (acc[card.name] || 0) + 1;
        return acc;
    }, {});

    // Calculate completed sets
    const getCompletedSets = (properties: Record<string, number>) => {
        return gameState.config.property_types.filter((propType) => {
            const count = properties[propType.name] || 0;
            return count >= propType.num_to_complete;
        }).length;
    };

    const humanCompleted = getCompletedSets(humanProperties);
    const botCompleted = getCompletedSets(botProperties);

    // Calculate cash totals
    const getCashTotal = (cashCards: CardModel[]) => {
        return cashCards.reduce((total, card) => {
            return total + ('value' in card ? card.value : 0);
        }, 0);
    };

    const humanCashTotal = getCashTotal(gameState.human.cash);
    const botCashTotal = getCashTotal(gameState.ai.cash);

    // Game over banner component
    const GameOverBanner = () => {
        if (!isGameOver) return null;

        return (
            <div className="relative z-10 mb-4">
                <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg p-4 shadow-lg border-2 border-white">
                    <div className="flex items-center justify-center space-x-4">
                        <h2 className="text-2xl font-bold">Game Over!</h2>
                        {gameState.turn.winner === 'human' && (
                            <div className="flex items-center space-x-2 bg-green-500/20 rounded-full px-4 py-2">
                                <span className="text-xl">🎉</span>
                                <span className="font-semibold">You Won!</span>
                                <span className="text-xl">🎉</span>
                            </div>
                        )}
                        {gameState.turn.winner === 'bot' && (
                            <div className="flex items-center space-x-2 bg-red-500/20 rounded-full px-4 py-2">
                                <span className="text-xl">🤖</span>
                                <span className="font-semibold">AI Won</span>
                                <span className="text-xl">💀</span>
                            </div>
                        )}
                        {gameState.turn.winner === 'tie' && (
                            <div className="flex items-center space-x-2 bg-gray-500/20 rounded-full px-4 py-2">
                                <span className="text-xl">🤝</span>
                                <span className="font-semibold">It&apos;s a Tie!</span>
                                <span className="text-xl">🤝</span>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div className={`space-y-4 ${isGameOver ? 'relative' : ''}`}>
            {/* Game Over Banner */}
            <GameOverBanner />

            {/* Subtle overlay when game is over */}
            {isGameOver && (
                <div className="absolute inset-0 bg-black/5 pointer-events-none z-0 rounded-lg"></div>
            )}

            {/* AI Hand Banner */}
            <SelectionInfoPanel
                selectionInfo={selectionInfo}
                isVisible={showSelectionInfo && selectionInfo !== null}
            />

            <GameOverview
                isHumanTurn={isHumanTurn}
                botThinking={botThinking}
                cardsPlayedThisTurn={gameState.turn.cards_played_this_turn}
                maxCardsPerTurn={gameState.turn.max_cards_per_turn}
                remainingCards={gameState.turn.remaining_cards}
                humanCompleted={humanCompleted}
                botCompleted={botCompleted}
                requiredSets={gameState.config.required_property_sets}
                humanCashTotal={humanCashTotal}
                botCashTotal={botCashTotal}
                isResponding={gameState.turn.is_responding}
                responseInfo={gameState.turn.response_info}
                deckCount={gameState.piles.deck || 0}
            />

            {/* Response Actions - only show when human is responding */}
            {gameState.turn.is_responding && isHumanTurn && !isGameOver && (
                <ResponseActions
                    validActions={gameState.actions}
                    humanState={gameState.human}
                    botState={gameState.ai}
                    onCardClick={handleCardClick}
                    responseInfo={gameState.turn.response_info}
                    gameConfig={gameState.config}
                />
            )}

            <PlayedCardHistory history={gameState.turn.selected_actions || []} humanPlayerIndex={gameState.turn.human_player_index ?? 1} />

            {/* Two-column layout: Human (left) vs AI (right) */}
            <div className="grid grid-cols-2 gap-6">
                {/* Human Section */}
                <div className="space-y-4">
                    <CardGrid
                        title="Your Hand"
                        cards={gameState.human.hand}
                        isClickable={isHumanTurn && !isProcessingAction && !isGameOver}
                        onCardClick={handleCardClick}
                        onPassClick={handlePassClick}
                        showPassButton={isHumanTurn && !isProcessingAction && !isGameOver}
                        fixedHeight={true}
                    />
                    <CardGrid
                        title="Your Properties"
                        cards={gameState.human.properties}
                        propertyCounts={humanProperties}
                        fixedHeight={true}
                    />
                    <CardGrid
                        title="Your Cash"
                        cards={gameState.human.cash}
                        fixedHeight={true}
                    />
                </div>

                {/* AI Section */}
                <div className="space-y-4">
                    {showSelectionInfo ? (
                        <CardGrid
                            title="AI Hand"
                            cards={gameState.ai.hand || []}
                            isClickable={false}
                            onCardClick={handleCardClick}
                            onPassClick={handlePassClick}
                            showPassButton={true}
                            fixedHeight={true}
                        />
                    ) : (
                        <div className="border border-gray-200 rounded-lg bg-white h-[240px] flex flex-col">
                            <div className="p-4 pb-0 flex-shrink-0">
                                <h2 className="text-sm font-medium mb-4 text-gray-900">AI Hand</h2>
                            </div>
                            <div className="px-4 pb-4 flex-1 flex items-start">
                                <div className="flex gap-2 overflow-x-auto overflow-y-hidden">
                                    {/* Show a blank card with the count only when NOT showing selection info */}
                                    {!showSelectionInfo && (
                                        <div className="relative snap-start" style={{ padding: '8px' }}>
                                            <div className="w-[120px] h-[160px] bg-gray-100 border border-gray-300 rounded-lg flex items-center justify-center relative">
                                                <div className="text-gray-400 text-lg">?</div>
                                                {/* Blue counter in top right - inside the card */}
                                                <div className="absolute top-1 right-1 text-white text-[10px] font-bold rounded-full w-7 h-7 flex items-center justify-center border-2 border-white shadow-sm z-10 bg-blue-500">
                                                    <span className="leading-none">{botHandCount}</span>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}
                    <CardGrid
                        title="AI Properties"
                        cards={gameState.ai.properties}
                        propertyCounts={botProperties}
                        fixedHeight={true}
                    />
                    <CardGrid
                        title="AI Cash"
                        cards={gameState.ai.cash}
                        fixedHeight={true}
                    />
                </div>
            </div>


            {/* Just Say No Dialog */}
            <Dialog open={showJustSayNoDialog} onOpenChange={setShowJustSayNoDialog}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Just Say No Not Available</DialogTitle>
                        <DialogDescription>
                            Just Say No cards can only be played as a response to certain actions (like Rent cards) that require a response.
                            They are not available during your normal turn. You&apos;ll be able to use this card when the AI plays an action that requires a response from you.
                        </DialogDescription>
                    </DialogHeader>
                </DialogContent>
            </Dialog>
        </div>
    );
}