'use client';

import { useState } from 'react';
import { Copy, Check } from 'lucide-react';

type Props = {
  gameId: string;
};

export function GameNotFound({ gameId }: Props) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(gameId);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
      <div className="max-w-md w-full text-center">
        {/* Error */}
        <div className="mb-8">
          <div className="text-6xl font-bold text-gray-300 mb-2">404</div>
          <h1 className="text-2xl font-semibold text-gray-900 mb-2">Game Not Found</h1>
        </div>

        {/* Game ID */}
        <div className="mb-8">
          <div className="bg-white border-2 border-dashed border-gray-300 rounded-lg p-4 mb-3">
            <p className="text-lg font-mono text-gray-800 break-all">{gameId}</p>
          </div>
          <button
            onClick={handleCopy}
            className="text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1 mx-auto"
          >
            {copied ? (
              <>
                <Check className="w-4 h-4" />
                <span>Copied!</span>
              </>
            ) : (
              <>
                <Copy className="w-4 h-4" />
                <span>Copy ID</span>
              </>
            )}
          </button>
        </div>

        {/* Instructions */}
        <div className="text-gray-600">
          <p className="mb-2">This game might be deleted or expired.</p>
          <p className="text-sm">Use the &quot;New Game&quot; button in the top right to start fresh.</p>
        </div>
      </div>
    </div>
  );
}


