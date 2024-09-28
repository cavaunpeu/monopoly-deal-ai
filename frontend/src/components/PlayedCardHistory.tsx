import { SelectedActionEntry } from "@/types/game";
import { getCardKind } from "@/types/cards";
import { User, Bot, DollarSign, Home, CreditCard, SkipForward } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "./ui/tooltip";

type Props = {
    history: SelectedActionEntry[];
    humanPlayerIndex: number;
};

export function PlayedCardHistory({ history, humanPlayerIndex }: Props) {
    const getActionIcon = (entry: SelectedActionEntry) => {
        // Determine action type based on the action properties
        if (entry.action.card === null) {
            // No card - this is a pass or yield action
            if (entry.action.is_response) {
                return <span className="text-xs font-bold">↩</span>; // Yield
            } else {
                return <SkipForward className="w-3 h-3" />; // Pass
            }
        }

        // Has a card - show card icon
        const cardKind = getCardKind(entry.action.card);
        switch (cardKind) {
            case 'CASH':
                return <DollarSign className="w-3 h-3" />;
            case 'PROPERTY':
                return <Home className="w-3 h-3" />;
            case 'RENT':
                return <CreditCard className="w-3 h-3" />;
            case 'SPECIAL':
                return <span className="text-xs font-bold">!</span>;
            default:
                return <span className="text-xs">?</span>;
        }
    };

    const getPlayerIcon = (playerIdx: number) => {
        return playerIdx === humanPlayerIndex ? (
            <User className="w-3 h-3 text-blue-500" />
        ) : (
            <Bot className="w-3 h-3 text-gray-500" />
        );
    };

    const getActionColor = (entry: SelectedActionEntry) => {
        if (entry.action.card === null) {
            // No card - this is a pass or yield action
            if (entry.action.is_response) {
                return 'bg-orange-100 border-orange-200 text-orange-800'; // Yield
            } else {
                return 'bg-gray-100 border-gray-200 text-gray-800'; // Pass
            }
        }

        // Has a card - show card color
        const cardKind = getCardKind(entry.action.card);
        switch (cardKind) {
            case 'CASH':
                return 'bg-green-100 border-green-200 text-green-800';
            case 'PROPERTY':
                return 'bg-blue-100 border-blue-200 text-blue-800';
            case 'RENT':
                return 'bg-red-100 border-red-200 text-red-800';
            case 'SPECIAL':
                return 'bg-purple-100 border-purple-200 text-purple-800';
            default:
                return 'bg-gray-100 border-gray-200 text-gray-800';
        }
    };

    const getTooltipContent = (entry: SelectedActionEntry) => {
        const playerName = entry.player_idx === humanPlayerIndex ? 'You' : 'AI';

        if (entry.action.card === null) {
            if (entry.action.is_response) {
                return `${playerName} yielded`;
            } else {
                return `${playerName} passed their turn`;
            }
        }

        const cardName = entry.action.card.name === "JUST_SAY_NO" ? "Just Say No" : entry.action.card.name?.replace(/_/g, " ") || "Unknown Card";
        const cardKind = getCardKind(entry.action.card);
        const cardType = cardKind?.toLowerCase() || "unknown";
        const responseText = entry.action.is_response ? " as a response" : "";

        return `${playerName} played ${cardName} (${cardType} card)${responseText}`;
    };

    const getActionType = (entry: SelectedActionEntry) => {
        if (entry.action.card === null) {
            if (entry.action.is_response) {
                return 'Yield';
            } else {
                return 'Pass';
            }
        }

        if (entry.action.card.name === "JUST_SAY_NO") {
            return "Just Say No";
        }

        return entry.action.card.name || 'Unknown';
    };

    return (
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
            <h3 className="text-sm font-medium text-gray-900 mb-3">Cards Played</h3>
            <div className="flex gap-2 overflow-x-auto overflow-y-hidden pb-2 h-[100px] whitespace-nowrap">
                {history.length === 0 ? (
                    <div className="flex flex-col items-center justify-center w-full h-full">
                        <div className="bg-gray-50 rounded-full p-3 mb-3">
                            <CreditCard className="w-6 h-6 text-gray-400" />
                        </div>
                        <p className="text-gray-500 text-sm font-medium">No cards</p>
                        <p className="text-gray-400 text-xs mt-1">Cards will appear here</p>
                    </div>
                ) : (
                    [...history].reverse().map((entry, index) => (
                        <Tooltip key={`${entry.player_idx}-${entry.turn_idx}-${entry.streak_idx}-${entry.action.card?.name || 'null'}-${index}`}>
                            <TooltipTrigger asChild>
                                <div className="flex-shrink-0 bg-white rounded-lg border border-gray-200 p-1.5 w-[80px] h-[80px] flex flex-col cursor-help relative">
                                    <div className="flex items-center justify-between mb-1">
                                        {getPlayerIcon(entry.player_idx)}
                                        <span className="text-xs text-gray-500 font-mono">{entry.turn_idx + 1}</span>
                                    </div>
                                    <div className={`w-6 h-8 rounded border flex items-center justify-center mx-auto mb-1 relative ${getActionColor(entry)}`}>
                                        {getActionIcon(entry)}
                                        {/* Response indicator - overlay on the card */}
                                        {entry.action.is_response && (
                                            <div className="absolute -top-1 -right-1 bg-orange-500 text-white text-[8px] font-bold rounded-full w-4 h-4 flex items-center justify-center z-10">
                                                R
                                            </div>
                                        )}
                                    </div>
                                    <div className="text-[10px] text-center text-gray-600 truncate leading-tight">
                                        {getActionType(entry)}
                                    </div>
                                </div>
                            </TooltipTrigger>
                            <TooltipContent>
                                <p>{getTooltipContent(entry)}</p>
                            </TooltipContent>
                        </Tooltip>
                    ))
                )}
            </div>
        </div>
    );
}
