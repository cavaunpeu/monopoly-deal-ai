'use client';

import { useState, useEffect } from 'react';
import { BookOpen, X, RefreshCw, Clock, ChevronDown, Bot, Eye, FileText } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { getDefaultGameConfig, getModels } from '@/lib/api';
import { GameConfig } from '@/types/game';
import { GameRules } from '@/components/GameRules';
import * as Select from '@radix-ui/react-select';

// Type for the API model types
type ModelType = {
  name: string;
  description: string;
  checkpoint_path: string;
};

type TabType = 'about' | 'game' | null;

type Props = {
  /** Callback to start a new game */
  onNewGame: () => void;
  /** Whether a new game is currently being created */
  isCreatingGame: boolean;
  /** Current bot speed setting */
  botSpeed: number;
  /** Callback to change bot speed */
  onBotSpeedChange: (speed: number) => void;
  /** Whether there is an active game */
  hasActiveGame: boolean;
  /** Whether to show selection information */
  showSelectionInfo: boolean;
  /** Callback to toggle selection information display */
  onShowSelectionInfoChange: (enabled: boolean) => void;
};

/**
 * Component that provides information tabs and game controls.
 * Includes game rules, model selection, bot speed controls, and new game functionality.
 *
 * @param props - Component props
 * @returns JSX element containing the info tabs interface
 */
export function InfoTabs({ onNewGame, isCreatingGame, botSpeed, onBotSpeedChange, hasActiveGame, showSelectionInfo, onShowSelectionInfoChange }: Props) {
  const [activeTab, setActiveTab] = useState<TabType>(null);
  const [showModelChangeModal, setShowModelChangeModal] = useState(false);
  const [gameConfig, setGameConfig] = useState<GameConfig | null>(null);
  const [currentModelType, setCurrentModelType] = useState<string>('');
  const [models, setModels] = useState<Record<string, ModelType>>({});

  const closeTab = () => setActiveTab(null);

  // Fetch models from API on component mount
  useEffect(() => {
    getModels()
      .then(data => {
        setModels(data.models);
        // Set the first model as the default
        const firstModelKey = Object.keys(data.models)[0];
        if (firstModelKey) {
          setCurrentModelType(firstModelKey);
        }
      })
      .catch(console.error);
  }, []);

  // Fetch game config when game tab is opened
  useEffect(() => {
    if (activeTab === 'game' && !gameConfig) {
      // Always use default config since game config is the same for all games
      getDefaultGameConfig()
        .then(setGameConfig)
        .catch(console.error);
    }
  }, [activeTab, gameConfig]);

  const modelOptions = Object.entries(models).map(([key]) => ({
    value: key,
    label: key
  }));

  const [pendingModelType, setPendingModelType] = useState<string | null>(null);

  const handleModelTypeSelect = (modelType: string) => {
    if (hasActiveGame) {
      setPendingModelType(modelType);
      setShowModelChangeModal(true);
    } else {
      // No active game, can change model directly
      setCurrentModelType(modelType);
    }
  };

  const confirmModelChange = () => {
    setShowModelChangeModal(false);
    if (pendingModelType) {
      setCurrentModelType(pendingModelType);
      setPendingModelType(null);
    }
    onNewGame(); // Start a new game with the new model
  };

  const cancelModelChange = () => {
    setShowModelChangeModal(false);
    setPendingModelType(null);
  };

  return (
    <>
      {/* Tab Buttons */}
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setActiveTab('about')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            activeTab === 'about'
              ? 'bg-blue-100 text-blue-700 border border-blue-200'
              : 'bg-white text-gray-600 border border-gray-200 hover:bg-gray-50'
          }`}
        >
          <FileText className="w-4 h-4" />
          About This Work
        </button>
        <button
          onClick={() => setActiveTab('game')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            activeTab === 'game'
              ? 'bg-blue-100 text-blue-700 border border-blue-200'
              : 'bg-white text-gray-600 border border-gray-200 hover:bg-gray-50'
          }`}
        >
          <BookOpen className="w-4 h-4" />
          Game Rules
        </button>

        {/* AI Speed Slider */}
        <div className="flex items-center gap-3 px-4 py-2 rounded-lg bg-white border border-gray-200">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-gray-600" />
            <span className="text-sm font-medium text-gray-700">AI Speed:</span>
          </div>
          <div className="flex gap-2">
            {[1, 3, 10, 30].map((speed) => (
              <button
                key={speed}
                onClick={() => onBotSpeedChange(speed)}
                className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                  botSpeed === speed
                    ? 'bg-blue-100 text-blue-700 border border-blue-200'
                    : 'bg-gray-100 text-gray-600 border border-gray-200 hover:bg-gray-200'
                }`}
              >
                {speed}s
              </button>
            ))}
          </div>
        </div>

        {/* Model Type Select */}
        <Select.Root value={currentModelType} onValueChange={handleModelTypeSelect}>
          <Tooltip>
            <TooltipTrigger asChild>
              <Select.Trigger className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors bg-white text-gray-600 border border-gray-200 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 min-w-[140px]">
                <Bot className="w-4 h-4 flex-shrink-0" />
                <Select.Value className="flex-1 whitespace-nowrap text-left" />
                <Select.Icon className="flex-shrink-0">
                  <ChevronDown className="w-4 h-4" />
                </Select.Icon>
              </Select.Trigger>
            </TooltipTrigger>
            <TooltipContent sideOffset={5} className="max-w-xs">
              {models[currentModelType]?.description || `No description available for ${currentModelType}`}
            </TooltipContent>
          </Tooltip>

          <Select.Portal>
            <Select.Content
              className="bg-white border border-gray-200 rounded-lg shadow-lg z-50 min-w-[160px]"
              position="popper"
              sideOffset={4}
              align="start"
              side="bottom"
              avoidCollisions={true}
              collisionPadding={8}
            >
              <Select.Viewport className="p-1">
                {modelOptions.map((option) => (
                  <Select.Item
                    key={option.value}
                    value={option.value}
                    className="px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 rounded cursor-pointer focus:outline-none focus:bg-gray-50 whitespace-nowrap"
                  >
                    <Select.ItemText>{option.label}</Select.ItemText>
                  </Select.Item>
                ))}
              </Select.Viewport>
            </Select.Content>
          </Select.Portal>
        </Select.Root>

        {/* Show AI Hand Toggle */}
        <div className="flex items-center gap-3 px-4 py-2">
          <div className="flex items-center gap-2">
            <Eye className="w-4 h-4 text-gray-600" />
            <span className="text-sm font-medium text-gray-700">Show AI Hand</span>
          </div>
          <button
            onClick={() => onShowSelectionInfoChange(!showSelectionInfo)}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 ${
              showSelectionInfo ? 'bg-purple-600' : 'bg-gray-200'
            }`}
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                showSelectionInfo ? 'translate-x-6' : 'translate-x-1'
              }`}
            />
          </button>
        </div>

        <button
          onClick={onNewGame}
          disabled={isCreatingGame}
          className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors bg-green-50 text-green-700 border border-green-200 hover:bg-green-100 hover:text-green-800 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <RefreshCw className={`w-4 h-4 ${isCreatingGame ? 'animate-spin' : ''}`} />
          New Game
        </button>
      </div>

      {/* Tab Content */}
      {activeTab && (
        <div className="bg-white border border-gray-200 rounded-lg p-6 mb-6 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">
              {activeTab === 'game' ? 'Game Rules' : 'About This Work'}
            </h2>
            <button
              onClick={closeTab}
              className="p-1 hover:bg-gray-100 rounded-full transition-colors"
            >
              <X className="w-4 h-4 text-gray-500" />
            </button>
          </div>

          {activeTab === 'game' && (
            <div className="text-sm">
              <GameRules gameConfig={gameConfig || undefined} />
            </div>
          )}

          {activeTab === 'about' && (
            <div className="space-y-4 text-sm text-gray-700">
                <div className="space-y-3">
                  <p>
                    This work presents a modified version of the card game{' '}
                    <a
                      href="https://en.wikipedia.org/wiki/Monopoly_Deal"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:text-blue-800 underline"
                    >
                      Monopoly Deal
                    </a>
                  . It serves as a platform for ongoing independent research on systems and algorithms for sequential decision-making under imperfect information, focusing on classical and modern approaches from game theory and reinforcement learning.
                  </p>
                  <p>
                    This work is authored by{' '}
                    <a
                      href="https://willwolf.io/about/"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:text-blue-800 underline"
                    >
                      Will Wolf
                    </a>
                    . The code can be found{' '}
                    <a
                      href="https://github.com/cavaunpeu/monopoly-deal-ai"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:text-blue-800 underline"
                    >
                      here
                    </a>
                    .
                  </p>
                </div>

                <div className="space-y-3">
                  <h3 className="font-semibold text-gray-900">Publications</h3>
                  <ul className="space-y-1">
                    <li>• <a href="https://arxiv.org/abs/2510.25080" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-800 underline"><strong>Monopoly Deal: A Benchmark Environment for Bounded One-Sided Response Games</strong></a></li>
                  </ul>
                </div>
            </div>
          )}
        </div>
      )}

      {/* Model Change Modal */}
      {showModelChangeModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Change AI Model
            </h3>
            <p className="text-gray-600 mb-6">
              You&apos;re currently in the middle of a game. Changing the AI model will start a new game. Are you sure you want to continue?
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={cancelModelChange}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={confirmModelChange}
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors"
              >
                Start New Game
              </button>
            </div>
          </div>
        </div>
      )}

    </>
  );
}
