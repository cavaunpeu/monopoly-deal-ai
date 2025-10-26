import React from "react";
import { CardModel, PropertyCardModel, CashCardModel, RentCardModel, getCardKind } from "@/types/cards";
import { Card } from "./cards/Card";
import { CreditCard } from "lucide-react";

type Props = {
    /** Title to display above the card grid */
    title: string;
    /** Array of cards to display */
    cards: CardModel[];
    /** Whether cards should be clickable */
    isClickable?: boolean;
    /** Callback when a card is clicked */
    onCardClick?: (card: CardModel) => void;
    /** Callback when the pass button is clicked */
    onPassClick?: () => void;
    /** Whether to show a pass button */
    showPassButton?: boolean;
    /** Property counts for display purposes */
    propertyCounts?: { [key: string]: number };
    /** Whether to use a fixed height for the grid */
    fixedHeight?: boolean;
};

/**
 * Component that displays a grid of cards with grouping and interaction capabilities.
 * Groups identical cards together and shows counts to reduce visual clutter.
 *
 * @param props - Component props
 * @returns JSX element containing the card grid
 */
function CardGridComponent({ title, cards, isClickable = false, onCardClick, onPassClick, showPassButton = false, propertyCounts, fixedHeight = false }: Props) {

    // Group duplicate cards and count them
    const groupedCards = () => {
        if (!cards || cards.length === 0) return [];

        const groups = new Map<string, { card: CardModel; count: number; indices: number[]; stableKey: string }>();

        cards.forEach((card, index) => {
            if (!card) return;

            const cardKind = getCardKind(card);
            if (!cardKind) return;

            // Create a unique key for the card
            let key = `${cardKind}-${card.name}`;

            // Add type-specific properties to make the key unique
            if (cardKind === 'PROPERTY') {
                key += `-${JSON.stringify((card as PropertyCardModel).rent_progression || [])}`;
            } else if (cardKind === 'CASH') {
                key += `-${(card as CashCardModel).value || ''}`;
            } else if (cardKind === 'RENT') {
                key += `-${(card as RentCardModel).property_name || ''}`;
            }

            if (groups.has(key)) {
                const group = groups.get(key)!;
                group.count++;
                group.indices.push(index);
            } else {
                groups.set(key, {
                    card,
                    count: 1,
                    indices: [index],
                    stableKey: key // Add stable key for consistent rendering
                });
            }
        });

        return Array.from(groups.values());
    };

    const handleCardClick = (group: { card: CardModel; count: number; indices: number[]; stableKey: string }) => {
        if (isClickable && onCardClick) {
            // Use the first index for the click
            onCardClick(group.card);
        }
    };

    return (
        <div className={`border border-gray-200 rounded-lg bg-white ${fixedHeight ? 'h-[240px] flex flex-col' : 'min-h-[240px]'}`}>
            <div className="p-4 pb-0 flex-shrink-0">
                <h2 className="text-sm font-medium mb-4 text-gray-900">{title}</h2>
            </div>
            <div className={`px-4 pb-4 ${fixedHeight ? 'flex-1 flex items-start' : ''}`}>
                {!cards || cards.length === 0 ? (
                    <div className={`flex flex-col items-center justify-center py-8 ${fixedHeight ? 'w-full h-full' : 'min-h-[160px]'}`}>
                        <div className="bg-gray-50 rounded-full p-3 mb-3">
                            <CreditCard className="w-6 h-6 text-gray-400" />
                        </div>
                        <p className="text-gray-500 text-sm font-medium">No cards</p>
                        <p className="text-gray-400 text-xs mt-1">Cards will appear here</p>
                    </div>
                ) : (
                    <div className="flex gap-2 overflow-x-auto overflow-y-hidden">
                    {groupedCards().map((group, groupIndex) => {
                        const { card, stableKey } = group;

                        const cardKind = getCardKind(card);
                        if (!card || !cardKind) {
                            return (
                                <div key={groupIndex} className="bg-red-50 text-red-700 rounded-lg p-3 border border-red-200 min-w-[120px] h-[160px] flex items-center justify-center snap-start">
                                    <div className="text-center text-xs">Invalid Card</div>
                                </div>
                            );
                        }

                        return (
                            <div
                                key={stableKey}
                                className={`snap-start relative ${isClickable ? 'cursor-pointer hover:scale-105 transition-transform' : ''}`}
                                onClick={() => handleCardClick(group)}
                                style={{
                                    transformOrigin: 'center',
                                    padding: '8px' // Consistent padding to prevent size changes
                                }}
                            >
                                <Card
                                    card={card}
                                    count={cardKind === 'PROPERTY' && propertyCounts ? propertyCounts[card.name] : group.count > 1 ? group.count : undefined}
                                    showProgress={cardKind === 'PROPERTY' && propertyCounts !== undefined}
                                />
                            </div>
                        );
                    })}
                    {showPassButton && (
                        <div
                            className="snap-start relative"
                            style={{
                                transformOrigin: 'center',
                                padding: '8px' // Consistent padding to prevent size changes
                            }}
                        >
                            <button
                                onClick={onPassClick}
                                className="bg-white text-gray-700 rounded-lg p-3 border border-gray-200 w-[120px] h-[160px] flex flex-col justify-between transition-colors hover:bg-gray-50"
                            >
                                <div className="text-[10px] font-medium text-gray-500 uppercase tracking-wide py-1">Action</div>
                                <div className="text-center flex-1 flex flex-col items-center justify-center">
                                    <div className="text-lg font-bold">Pass</div>
                                </div>
                                <div></div>
                            </button>
                        </div>
                    )}
                    </div>
                )}
            </div>
        </div>
    );
}

export const CardGrid = React.memo(CardGridComponent);