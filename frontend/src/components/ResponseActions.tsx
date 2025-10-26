import React from 'react';
import { CardModel, SerializedAction, AiState } from '../types/cards';
import { Card } from './cards/Card';
import { AlertTriangle } from 'lucide-react';
import { GameConfig, ResponseInfo } from '../types/game';
import { logger } from '../lib/logger';
import { groupCardsByType, filterGroupedCardsByValidActions, extractValidCardKeysFromActions, GroupedCard } from '../lib/cardUtils';

type Props = {
    /** Array of valid response actions available to the player */
    validActions: SerializedAction[];
    /** Human player's current game state */
    humanState: {
        hand: CardModel[];
        properties: CardModel[];
        cash: CardModel[];
    };
    /** AI player's current game state */
    botState: AiState;
    /** Callback when a card is clicked for response */
    onCardClick: (card: CardModel) => void;
    /** Information about the action requiring a response */
    responseInfo: ResponseInfo | null;
    /** Game configuration for calculating rent amounts */
    gameConfig: GameConfig | null;
};

/**
 * Component that displays available response actions when the player needs to respond to an action.
 * Shows valid response cards and calculates rent amounts for rent card responses.
 *
 * @param props - Component props
 * @returns JSX element containing the response actions interface
 */
function ResponseActionsComponent({ validActions, humanState, botState, onCardClick, responseInfo, gameConfig }: Props) {
    // Calculate rent amount based on the initiating card and responding player's properties
    const getRentAmount = () => {
        logger.debug('getRentAmount called');
        if (!responseInfo?.initiating_card || !gameConfig) {
            logger.debug('getRentAmount: Missing initiating_card or gameConfig, returning null');
            return null;
        }

        const rentCard = responseInfo.initiating_card;
        logger.debug('getRentAmount: rentCard:', rentCard);

        if (rentCard.kind !== 'RENT') {
            logger.debug('getRentAmount: rentCard is not of kind RENT, returning null');
            return null;
        }

        const propertyName = rentCard.property_name;
        logger.debug('getRentAmount: propertyName:', propertyName);

        // Find the property type to get rent progression
        const propertyType = gameConfig.property_types.find(
            pt => pt.name === propertyName
        );
        logger.debug('getRentAmount: propertyType:', propertyType);

        if (!propertyType) {
            logger.debug('getRentAmount: propertyType not found, returning null');
            return null;
        }

        // Get the responding player's properties
        const respondingPlayer = responseInfo.responding_player;
        logger.debug('getRentAmount: respondingPlayer:', respondingPlayer);

        // For rent calculation, we need to check the properties of the player who OWNS the properties
        // (the one being charged rent), not the one responding
        // If human is responding, check AI's properties (AI is being charged)
        // If AI is responding, check human's properties (human is being charged)
        const properties = respondingPlayer === 'human' ? botState.properties : humanState.properties;
        logger.debug('getRentAmount: properties to check (owner of properties):', properties);

        // Count how many properties of this type the responding player has
        const propertyCount = properties.filter(prop => prop.name === propertyName).length;
        logger.debug('getRentAmount: propertyCount for', propertyName, ':', propertyCount);

        // Calculate rent based on property count and rent progression
        const maxRentIndex = Math.min(propertyCount, propertyType.rent_progression.length) - 1;
        logger.debug('getRentAmount: maxRentIndex:', maxRentIndex);

        const calculatedRent = maxRentIndex >= 0 ? propertyType.rent_progression[maxRentIndex] : 0;
        logger.debug('getRentAmount: Calculated rent:', calculatedRent);
        return calculatedRent;
    };

    const rentAmount = getRentAmount();

    // Debug logging
    logger.debug('ResponseActions - responseInfo:', responseInfo);
    logger.debug('ResponseActions - calculated rentAmount:', rentAmount);

    // Get response data for a specific pile using reusable utility functions
    const getPileResponseData = (pileType: 'HAND' | 'PROPERTY' | 'CASH'): GroupedCard[] => {
        const actions = validActions.filter(action => action.src === pileType);
        const cards = pileType === 'HAND' ? humanState.hand :
                     pileType === 'PROPERTY' ? humanState.properties :
                     humanState.cash;

        // Extract valid card keys from actions
        const validCardKeys = extractValidCardKeysFromActions(actions);

        // Group all cards in the pile by type
        const groupedCards = groupCardsByType(cards);

        // Filter to only include cards that have valid actions
        return filterGroupedCardsByValidActions(groupedCards, validCardKeys);
    };

    const handData = getPileResponseData('HAND');
    const propertyData = getPileResponseData('PROPERTY');
    const cashData = getPileResponseData('CASH');

    const hasAnyResponseCards = handData.length > 0 || propertyData.length > 0 || cashData.length > 0;

    // Reusable component for rendering a pile section
    const PileSection = ({ title, data }: { title: string; data: GroupedCard[] }) => {
        if (data.length === 0) return null;

        return (
            <div>
                <div className="text-xs font-medium text-orange-800 mb-1">{title}</div>
                <div className="flex gap-2 overflow-x-auto pb-1">
                    {data.map(({ card, count, stableKey }) => (
                        <div key={stableKey} className="flex-shrink-0">
                            <div
                                className="cursor-pointer hover:scale-105 transition-transform"
                                onClick={() => onCardClick(card)}
                                style={{ transform: 'scale(0.8)', transformOrigin: 'top' }}
                            >
                                <Card card={card} count={count > 1 ? count : undefined} />
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        );
    };

    if (!hasAnyResponseCards) {
        return null;
    }

    return (
        <div className="bg-orange-50 border border-orange-200 rounded-lg p-3 mb-3">
            {/* Compact Response Header */}
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                    <div className="p-1 rounded-full bg-orange-500 text-white">
                        <AlertTriangle className="w-3 h-3" />
                    </div>
                    <div>
                        <h3 className="font-semibold text-orange-900 text-sm">Response Required</h3>
                        {rentAmount !== null && (
                            <p className="text-xs text-orange-700">
                                Pay ${rentAmount} rent for {(responseInfo?.initiating_card as CardModel & { property_name?: string })?.property_name || 'property'}
                            </p>
                        )}
                    </div>
                </div>
                            {rentAmount !== null && (
                                <div className="flex items-center space-x-1 bg-orange-100 px-2 py-1 rounded">
                                    <span className="text-sm font-bold text-orange-800">${rentAmount}</span>
                                </div>
                            )}
            </div>

            {/* Response Options - Grouped by pile */}
            <div className="space-y-2">
                <PileSection title="From Hand" data={handData} />
                <PileSection title="From Cash" data={cashData} />
                <PileSection title="From Properties" data={propertyData} />
            </div>
        </div>
    );
}

export const ResponseActions = React.memo(ResponseActionsComponent);
