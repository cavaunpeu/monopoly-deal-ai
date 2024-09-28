import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';

interface SelectionInfo {
  policy?: Record<string, number>;
  state_key?: string;
  update_count?: number;
  model_type?: string;
}

interface SelectionInfoPanelProps {
  selectionInfo: SelectionInfo | null;
  isVisible: boolean;
}

export function SelectionInfoPanel({ selectionInfo, isVisible }: SelectionInfoPanelProps) {
  const [copied, setCopied] = useState(false);

  if (!isVisible || !selectionInfo) {
    return null;
  }

  const handleCopyStateKey = async () => {
    if (selectionInfo.state_key) {
      await navigator.clipboard.writeText(selectionInfo.state_key);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="w-full bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 border border-slate-200/60 rounded-xl p-4 mb-4 shadow-lg backdrop-blur-sm">
      {/* Header with improved styling */}
      <div className="mb-3">
        <h3 className="text-sm font-semibold text-slate-800">Action Selection Info</h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {/* State Key with improved styling */}
        {selectionInfo.state_key && (
          <div className="bg-white/80 backdrop-blur-sm rounded-lg p-3 border border-slate-200/50 shadow-sm hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-emerald-400 rounded-full"></div>
                <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Game State Key</span>
              </div>
              <button
                onClick={handleCopyStateKey}
                className="flex items-center gap-1 px-2 py-1 text-xs text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded transition-colors"
                title="Copy state key"
              >
                {copied ? (
                  <>
                    <Check className="w-3 h-3" />
                    <span>Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3 h-3" />
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>
            <code className="text-slate-700 font-mono break-all text-xs leading-relaxed bg-slate-50 px-2 py-1 rounded border block">
              {selectionInfo.state_key}
            </code>
          </div>
        )}

        {/* Update Count with improved styling */}
        {selectionInfo.update_count !== undefined && (
          <div className="bg-white/80 backdrop-blur-sm rounded-lg p-3 border border-slate-200/50 shadow-sm hover:shadow-md transition-shadow">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
              <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Training Updates</span>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold text-blue-600">
                {selectionInfo.update_count}
              </span>
              <span className="text-xs text-slate-500">updates</span>
            </div>
          </div>
        )}

        {/* Policy with improved styling */}
        {selectionInfo.policy && Object.keys(selectionInfo.policy).length > 0 && (
          <div className="bg-white/80 backdrop-blur-sm rounded-lg p-3 border border-slate-200/50 shadow-sm hover:shadow-md transition-shadow md:col-span-2 lg:col-span-1">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
              <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Action Probabilities</span>
            </div>
            <div className="space-y-1.5">
              {Object.entries(selectionInfo.policy).map(([action, probability]) => (
                <div key={action} className="flex items-center justify-between group">
                  <span className="text-xs font-medium text-slate-700 bg-slate-100 px-2 py-1 rounded-md group-hover:bg-slate-200 transition-colors">
                    {action.replace(/_/g, ' ').toLowerCase()}
                  </span>
                  <div className="flex items-center gap-2">
                    <div className="w-16 bg-slate-200 rounded-full h-1 overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-purple-400 to-pink-400 rounded-full transition-all duration-300"
                        style={{ width: `${probability * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-xs font-semibold text-purple-600 min-w-[3rem] text-right">
                      {(probability * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
