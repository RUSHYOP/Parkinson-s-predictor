'use client';

import { useState, useCallback } from 'react';
import { VoiceFeatures, FEATURE_RANGES, getDefaultFeatures } from '@/lib/types';
import { Info, RotateCcw } from 'lucide-react';

interface ManualInputProps {
  onSubmit: (features: VoiceFeatures) => void;
  isProcessing: boolean;
}

export default function ManualInput({ onSubmit, isProcessing }: ManualInputProps) {
  const [features, setFeatures] = useState<VoiceFeatures>(getDefaultFeatures());
  const [showTooltip, setShowTooltip] = useState<string | null>(null);

  const handleSliderChange = useCallback((key: keyof VoiceFeatures, value: number) => {
    setFeatures(prev => ({
      ...prev,
      [key]: value,
    }));
  }, []);

  const handleReset = useCallback(() => {
    setFeatures(getDefaultFeatures());
  }, []);

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(features);
  }, [features, onSubmit]);

  const formatValue = (value: number, step: number): string => {
    if (step >= 1) return value.toFixed(0);
    if (step >= 0.1) return value.toFixed(1);
    if (step >= 0.01) return value.toFixed(2);
    if (step >= 0.001) return value.toFixed(3);
    if (step >= 0.0001) return value.toFixed(4);
    return value.toFixed(5);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-gray-700">Voice Biomarker Values</h3>
        <button
          type="button"
          onClick={handleReset}
          className="flex items-center space-x-1 text-sm text-gray-500 hover:text-gray-700"
        >
          <RotateCcw className="w-4 h-4" />
          <span>Reset to Defaults</span>
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {FEATURE_RANGES.map((range) => (
          <div key={range.key} className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="flex items-center space-x-1 text-sm font-medium text-gray-700">
                <span>{range.name}</span>
                <button
                  type="button"
                  className="text-gray-400 hover:text-gray-600"
                  onMouseEnter={() => setShowTooltip(range.key)}
                  onMouseLeave={() => setShowTooltip(null)}
                >
                  <Info className="w-4 h-4" />
                </button>
              </label>
              <span className="text-sm font-mono text-gray-500">
                {formatValue(features[range.key], range.step)} {range.unit}
              </span>
            </div>
            
            {showTooltip === range.key && (
              <div className="absolute z-10 bg-gray-800 text-white text-xs rounded px-2 py-1 max-w-xs">
                {range.description}
              </div>
            )}
            
            <input
              type="range"
              min={range.min}
              max={range.max}
              step={range.step}
              value={features[range.key]}
              onChange={(e) => handleSliderChange(range.key, parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
            />
            
            <div className="flex justify-between text-xs text-gray-400">
              <span>{range.min}</span>
              <span>{range.max}</span>
            </div>
          </div>
        ))}
      </div>

      <button
        type="submit"
        disabled={isProcessing}
        className="w-full py-3 px-4 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-semibold rounded-lg transition-colors"
      >
        {isProcessing ? 'Analyzing...' : 'Assess Risk'}
      </button>

      <div className="text-sm text-gray-500 bg-gray-50 p-4 rounded-lg">
        <p className="font-medium mb-2">📊 About these biomarkers:</p>
        <ul className="list-disc list-inside space-y-1">
          <li><strong>Jitter</strong> - Frequency variation in voice (higher in PD)</li>
          <li><strong>Shimmer</strong> - Amplitude variation in voice (higher in PD)</li>
          <li><strong>NHR/HNR</strong> - Noise vs harmonics ratio (voice clarity)</li>
          <li><strong>RPDE, DFA, PPE</strong> - Nonlinear dynamics of voice signal</li>
        </ul>
      </div>
    </form>
  );
}
