import React from 'react';
import { GameConfig } from '@/types/game';

interface GameRulesProps {
  gameConfig?: GameConfig;
}

export function GameRules({ gameConfig }: GameRulesProps) {
  const maxActions = gameConfig?.max_consecutive_player_actions || 2;
  const newCards = gameConfig?.new_cards_per_turn || 2;
  const initialHand = gameConfig?.initial_hand_size || 5;
  const totalDeck = gameConfig?.total_deck_size || 113;
  const cashValues = gameConfig?.cash_card_values?.join(', ') || '1, 3';
  const requiredSets = gameConfig?.required_property_sets || 2;

  return (
    <div className="space-y-8">
      {/* Objective */}
      <section>
        <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center gap-2">
          <span className="text-xl">🎯</span>
          Objective
        </h3>
        <p className="text-gray-700 leading-relaxed">
          Complete the required number of property sets to win.
        </p>
      </section>

      {/* Video */}
      <section>
        <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center gap-2">
          <span className="text-xl">📺</span>
          How to Play Video
        </h3>
        <div className="relative w-full max-w-2xl mx-auto">
          <div className="relative" style={{ paddingBottom: '56.25%', height: 0 }}>
            <iframe
              src="https://www.youtube.com/embed/ynb8fhRzREY"
              title="How to Play Monopoly Deal"
              frameBorder="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
              className="absolute top-0 left-0 w-full h-full rounded-lg shadow-sm"
            />
          </div>
        </div>
      </section>

      {/* Card Types */}
      <section>
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <span className="text-xl">🃏</span>
          Card Types
        </h3>
        <div className="space-y-3">
          <div className="flex items-start gap-3">
            <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
            <div>
              <span className="font-medium text-gray-900">Property Cards</span>
              <span className="text-gray-600"> - Add these cards to your property pile to generate rent and progress towards winning</span>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></div>
            <div>
              <span className="font-medium text-gray-900">Cash Cards</span>
              <span className="text-gray-600"> - Add these cards to your cash pile to pay opponents when charged rent</span>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-2 h-2 bg-red-500 rounded-full mt-2 flex-shrink-0"></div>
            <div>
              <span className="font-medium text-gray-900">Rent Cards</span>
              <span className="text-gray-600"> - Charge opponents rent</span>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-2 h-2 bg-purple-500 rounded-full mt-2 flex-shrink-0"></div>
            <div>
              <span className="font-medium text-gray-900">Special Cards</span>
              <span className="text-gray-600"> - Block actions (e.g. being charged rent) with &apos;Just Say No&apos;</span>
            </div>
          </div>
        </div>
      </section>

      {/* Gameplay */}
      <section>
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <span className="text-xl">⚡</span>
          Gameplay
        </h3>
        <div className="space-y-3">
          <div className="flex items-start gap-3">
            <div className="w-2 h-2 bg-gray-400 rounded-full mt-2 flex-shrink-0"></div>
            <p className="text-gray-700 leading-relaxed">
              Play up to <span className="font-semibold text-blue-600">{maxActions}</span> cards per turn, then draw <span className="font-semibold text-blue-600">{newCards}</span> new ones
            </p>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-2 h-2 bg-gray-400 rounded-full mt-2 flex-shrink-0"></div>
            <p className="text-gray-700 leading-relaxed">
              Start with <span className="font-semibold text-blue-600">{initialHand}</span> cards in hand from a deck of <span className="font-semibold text-blue-600">{totalDeck}</span> total cards
            </p>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-2 h-2 bg-gray-400 rounded-full mt-2 flex-shrink-0"></div>
            <p className="text-gray-700 leading-relaxed">
              When charged rent, respond with cash, property, or &apos;Just Say No&apos;
            </p>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-2 h-2 bg-gray-400 rounded-full mt-2 flex-shrink-0"></div>
            <p className="text-gray-700 leading-relaxed">
              If you can&apos;t respond, the system yields automatically
            </p>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-2 h-2 bg-gray-400 rounded-full mt-2 flex-shrink-0"></div>
            <p className="text-gray-700 leading-relaxed">
              Property cards used to pay rent go to your cash pile
            </p>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-2 h-2 bg-gray-400 rounded-full mt-2 flex-shrink-0"></div>
            <p className="text-gray-700 leading-relaxed">
              Cash values: <span className="font-semibold text-blue-600">{cashValues}</span>
            </p>
          </div>
        </div>
      </section>

      {/* Win Condition */}
      <section>
        <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center gap-2">
          <span className="text-xl">🏆</span>
          Win
        </h3>
        <p className="text-gray-700 leading-relaxed">
          Complete <span className="font-semibold text-blue-600">{requiredSets}</span> property sets first.
        </p>
      </section>
    </div>
  );
}
