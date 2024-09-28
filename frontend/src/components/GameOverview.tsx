import { ResponseInfo } from "@/types/game";
import { User, Bot, Home, DollarSign, Clock, AlertTriangle, Layers } from "lucide-react";

type Props = {
    isHumanTurn: boolean;
    botThinking: boolean;
    cardsPlayedThisTurn: number;
    maxCardsPerTurn: number;
    remainingCards: number;
    humanCompleted: number;
    botCompleted: number;
    requiredSets: number;
    humanCashTotal: number;
    botCashTotal: number;
    isResponding: boolean;
    responseInfo: ResponseInfo | null;
    deckCount: number;
};

export function GameOverview({
    isHumanTurn,
    botThinking,
    cardsPlayedThisTurn,
    maxCardsPerTurn,
    remainingCards,
    humanCompleted,
    botCompleted,
    requiredSets,
    humanCashTotal,
    botCashTotal,
    isResponding,
    deckCount,
}: Props) {

    return (
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl p-5 mb-6 shadow-sm">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-6">
                {/* Turn Status */}
                <div className="flex items-center space-x-3">
                    <div className={`p-2.5 rounded-full ${isResponding ? 'bg-orange-500 text-white' : isHumanTurn ? 'bg-blue-500 text-white' : 'bg-gray-400 text-white'}`}>
                        {isResponding ? <AlertTriangle className="w-4 h-4" /> : isHumanTurn ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                    </div>
                    <div className="flex flex-col justify-center">
                        <div className="font-semibold text-gray-900 text-base leading-tight">
                            {isResponding ? "Response Required" : isHumanTurn ? "Your Turn" : botThinking ? "AI Thinking..." : "AI's Turn"}
                        </div>
                        <div className="h-[32px] flex items-start">
                            {botThinking ? (
                                <div className="flex items-center mt-1">
                                    <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-blue-500"></div>
                                    <span className="text-xs text-gray-500 ml-2">Processing...</span>
                                </div>
                            ) : null}
                        </div>
                    </div>
                </div>

                {/* Deck - moved next to turn status */}
                <div className="flex items-center space-x-3">
                    <div className="p-2 rounded-full bg-gray-100 text-gray-600">
                        <Layers className="w-4 h-4" />
                    </div>
                    <div>
                        <div className="font-semibold text-gray-900 text-sm">Deck</div>
                        <div className="text-xs text-gray-600">
                            <span className="text-gray-500">{deckCount}</span>
                        </div>
                    </div>
                </div>

                {/* Actions This Turn */}
                <div className="flex items-center space-x-3">
                    <div className="p-2 rounded-full bg-purple-100 text-purple-600">
                        <Clock className="w-4 h-4" />
                    </div>
                    <div>
                        <div className="font-semibold text-gray-900 text-sm">
                            {isResponding ? "Response Turn" : "Actions This Turn"}
                        </div>
                        <div className="text-xs text-gray-600">
                            {isResponding ? (
                                <span className="text-orange-600">Choose response</span>
                            ) : (
                                <>
                                    <span className="font-medium text-purple-600">{cardsPlayedThisTurn}/{maxCardsPerTurn}</span>
                                    {remainingCards > 0 && (
                                        <span className="text-gray-500 ml-1">({remainingCards} left)</span>
                                    )}
                                </>
                            )}
                        </div>
                    </div>
                </div>

                {/* Cash Totals */}
                <div className="flex items-center space-x-3">
                    <div className="p-2 rounded-full bg-emerald-100 text-emerald-600">
                        <DollarSign className="w-4 h-4" />
                    </div>
                    <div>
                        <div className="font-semibold text-gray-900 text-sm">Cash</div>
                        <div className="text-xs text-gray-600">
                            <div className="flex items-center space-x-3">
                                <span className="flex items-center">
                                    <User className="w-3 h-3 text-emerald-500 mr-1" />
                                    <span className="font-medium text-emerald-600">${humanCashTotal}</span>
                                </span>
                                <span className="text-gray-400">•</span>
                                <span className="flex items-center">
                                    <Bot className="w-3 h-3 text-gray-500 mr-1" />
                                    <span className="text-gray-500">${botCashTotal}</span>
                                </span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Property Sets - inline on the right */}
                <div className="flex items-center space-x-3">
                    <div className="p-2 rounded-full bg-green-100 text-green-600">
                        <Home className="w-4 h-4" />
                    </div>
                    <div>
                        <div className="font-semibold text-gray-900 text-sm">Property Sets</div>
                        <div className="text-xs text-gray-600">
                            <div className="flex items-center space-x-3">
                                <span className="flex items-center">
                                    <User className="w-3 h-3 text-blue-500 mr-1" />
                                    <span className="font-medium text-blue-600">{humanCompleted}/{requiredSets}</span>
                                </span>
                                <span className="text-gray-400">•</span>
                                <span className="flex items-center">
                                    <Bot className="w-3 h-3 text-gray-500 mr-1" />
                                    <span className="text-gray-500">{botCompleted}/{requiredSets}</span>
                                </span>
                            </div>
                        </div>
                    </div>
                </div>

            </div>
        </div>
    );
}
