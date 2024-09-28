'use client';

import { useEffect, useState, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { GameBoard } from "@/components/GameBoard";
import { InfoTabs } from "@/components/InfoTabs";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { createGame, ApiError } from "@/lib/api";
import { logger } from "@/lib/logger";

function HomeContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [gameId, setGameId] = useState<string | null>(null);
  const [isCreatingGame, setIsCreatingGame] = useState(false);
  const [botSpeed, setBotSpeed] = useState(3); // Default to 3 seconds
  const [showSelectionInfo, setShowSelectionInfo] = useState(false);

  const startNewGame = async () => {
    if (isCreatingGame) {
      // Prevent multiple simultaneous game creation requests
      return;
    }

    setIsCreatingGame(true);
    try {
      const newGameId = await createGame();
      // Update URL first, then let the useEffect handle setting gameId
      router.push(`/?game=${newGameId}`);
    } catch (error) {
      logger.error('Error creating game:', error);
      if (error instanceof ApiError) {
        logger.error('API Error:', error.message);
      }
    } finally {
      setIsCreatingGame(false);
    }
  };

  useEffect(() => {
    // Check if there's a game ID in the URL
    const urlGameId = searchParams.get('game');

    if (urlGameId && urlGameId !== gameId) {
      // Use the game ID from URL only if it's different from current state
      setGameId(urlGameId);
    } else if (!urlGameId && !gameId) {
      // Create a new game if no game ID in URL and no current game
      createGame()
        .then((newGameId) => {
          setGameId(newGameId);
          // Update URL with the new game ID
          router.push(`/?game=${newGameId}`);
        })
        .catch((error) => {
          logger.error('Error creating initial game:', error);
          if (error instanceof ApiError) {
            logger.error('API Error:', error.message);
          }
        });
    }
  }, [searchParams, gameId, router]);

  if (!gameId) return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-lg text-gray-600">Creating game...</div>
    </div>
  );

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {/* Title Section */}
          <div className="text-center mb-8">
            <h1 className="text-3xl font-semibold text-gray-900 mb-2">
              Monopoly Deal vs AI
            </h1>
            <div className="w-16 h-0.5 bg-gray-300 mx-auto mb-6"></div>
          </div>

          <InfoTabs
            onNewGame={startNewGame}
            isCreatingGame={isCreatingGame}
            botSpeed={botSpeed}
            onBotSpeedChange={setBotSpeed}
            hasActiveGame={!!gameId}
            showSelectionInfo={showSelectionInfo}
            onShowSelectionInfoChange={setShowSelectionInfo}
          />
          {gameId && (
            <ErrorBoundary>
              <GameBoard
                gameId={gameId}
                botSpeed={botSpeed}
                showSelectionInfo={showSelectionInfo}
                onNewGame={startNewGame}
                isCreatingGame={isCreatingGame}
              />
            </ErrorBoundary>
          )}
        </div>
      </div>
    </ErrorBoundary>
  );
}

export default function Home() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-lg text-gray-600">Loading...</div>
    </div>}>
      <HomeContent />
    </Suspense>
  );
}