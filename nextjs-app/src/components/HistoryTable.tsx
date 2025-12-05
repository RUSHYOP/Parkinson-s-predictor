'use client';

import { useState, useEffect, useCallback } from 'react';
import { PredictionHistory, getRiskColor, getRiskBgColor } from '@/lib/types';
import { Trash2, RefreshCw, Download, ChevronDown, ChevronUp } from 'lucide-react';

interface HistoryTableProps {
  onSelectPrediction?: (prediction: PredictionHistory) => void;
}

export default function HistoryTable({ onSelectPrediction }: HistoryTableProps) {
  const [history, setHistory] = useState<PredictionHistory[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [stats, setStats] = useState<{
    total: number;
    avgRisk: number;
    highRiskCount: number;
  } | null>(null);

  const fetchHistory = useCallback(async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/history');
      if (response.ok) {
        const data = await response.json();
        setHistory(data.predictions || []);
        setStats(data.stats || null);
      }
    } catch (error) {
      console.error('Failed to fetch history:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this prediction?')) return;
    
    try {
      const response = await fetch(`/api/history/${id}`, { method: 'DELETE' });
      if (response.ok) {
        fetchHistory();
      }
    } catch (error) {
      console.error('Failed to delete:', error);
    }
  };

  const handleClearAll = async () => {
    if (!confirm('Are you sure you want to clear all history? This cannot be undone.')) return;
    
    try {
      const response = await fetch('/api/history', { method: 'DELETE' });
      if (response.ok) {
        setHistory([]);
        setStats(null);
      }
    } catch (error) {
      console.error('Failed to clear history:', error);
    }
  };

  const handleExport = () => {
    const csv = [
      ['Timestamp', 'Input Type', 'Risk Score', 'Risk Level', 'Risk %', 'Confidence Lower', 'Confidence Upper'].join(','),
      ...history.map(p => [
        new Date(p.timestamp).toISOString(),
        p.inputType,
        p.riskScore.toFixed(4),
        p.riskLevel,
        p.riskPercentage.toFixed(2),
        p.confidenceLower.toFixed(2),
        p.confidenceUpper.toFixed(2),
      ].join(','))
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `parkinsons-predictions-${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (isLoading) {
    return (
      <div className="text-center py-12">
        <RefreshCw className="w-8 h-8 mx-auto animate-spin text-gray-400" />
        <p className="text-gray-500 mt-2">Loading history...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Stats Summary */}
      {stats && stats.total > 0 && (
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-blue-50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-blue-600">{stats.total}</div>
            <div className="text-sm text-blue-600/70">Total Predictions</div>
          </div>
          <div className="bg-gray-50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-gray-600">{stats.avgRisk.toFixed(1)}%</div>
            <div className="text-sm text-gray-500">Average Risk</div>
          </div>
          <div className="bg-red-50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-red-600">{stats.highRiskCount}</div>
            <div className="text-sm text-red-600/70">High Risk</div>
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex justify-between items-center">
        <button
          onClick={fetchHistory}
          className="flex items-center space-x-1 text-sm text-gray-500 hover:text-gray-700"
        >
          <RefreshCw className="w-4 h-4" />
          <span>Refresh</span>
        </button>
        
        <div className="flex space-x-2">
          {history.length > 0 && (
            <>
              <button
                onClick={handleExport}
                className="flex items-center space-x-1 text-sm text-blue-500 hover:text-blue-700"
              >
                <Download className="w-4 h-4" />
                <span>Export CSV</span>
              </button>
              <button
                onClick={handleClearAll}
                className="flex items-center space-x-1 text-sm text-red-500 hover:text-red-700"
              >
                <Trash2 className="w-4 h-4" />
                <span>Clear All</span>
              </button>
            </>
          )}
        </div>
      </div>

      {/* History List */}
      {history.length === 0 ? (
        <div className="text-center py-12 text-gray-400">
          <p>No prediction history yet.</p>
          <p className="text-sm">Complete a voice analysis or manual assessment to see results here.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {history.map((prediction) => (
            <div
              key={prediction.id}
              className={`border rounded-lg overflow-hidden ${
                expandedId === prediction.id ? 'border-blue-300' : 'border-gray-200'
              }`}
            >
              <div
                className={`p-4 cursor-pointer hover:bg-gray-50 flex items-center justify-between ${
                  getRiskBgColor(prediction.riskLevel)
                }`}
                onClick={() => {
                  setExpandedId(expandedId === prediction.id ? null : prediction.id);
                  onSelectPrediction?.(prediction);
                }}
              >
                <div className="flex items-center space-x-4">
                  <div className={`text-2xl font-bold ${getRiskColor(prediction.riskLevel)}`}>
                    {prediction.riskPercentage.toFixed(1)}%
                  </div>
                  <div>
                    <div className="font-medium text-gray-700">{prediction.riskLevel} Risk</div>
                    <div className="text-sm text-gray-500">
                      {new Date(prediction.timestamp).toLocaleString()} • {prediction.inputType}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(prediction.id);
                    }}
                    className="p-1 text-gray-400 hover:text-red-500"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                  {expandedId === prediction.id ? (
                    <ChevronUp className="w-5 h-5 text-gray-400" />
                  ) : (
                    <ChevronDown className="w-5 h-5 text-gray-400" />
                  )}
                </div>
              </div>
              
              {expandedId === prediction.id && (
                <div className="p-4 bg-white border-t border-gray-200">
                  <h4 className="font-medium text-gray-700 mb-2">Feature Values</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                    {Object.entries(prediction.features).map(([key, value]) => (
                      <div key={key} className="bg-gray-50 p-2 rounded">
                        <div className="text-xs text-gray-500">{key}</div>
                        <div className="font-mono">{typeof value === 'number' ? value.toFixed(4) : value}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
