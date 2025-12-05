'use client';

import { PredictionResult, getRiskColor, getRiskBgColor } from '@/lib/types';
import { AlertTriangle, CheckCircle, AlertCircle, TrendingUp } from 'lucide-react';

interface RiskDisplayProps {
  result: PredictionResult | null;
}

export default function RiskDisplay({ result }: RiskDisplayProps) {
  if (!result) {
    return (
      <div className="text-center py-12 text-gray-400">
        <TrendingUp className="w-16 h-16 mx-auto mb-4 opacity-50" />
        <p>Complete a voice analysis or enter manual values to see risk assessment</p>
      </div>
    );
  }

  const riskColor = getRiskColor(result.riskLevel);
  const riskBgColor = getRiskBgColor(result.riskLevel);

  const RiskIcon = () => {
    switch (result.riskLevel) {
      case 'Low':
        return <CheckCircle className="w-12 h-12 text-green-500" />;
      case 'Moderate':
        return <AlertCircle className="w-12 h-12 text-yellow-500" />;
      case 'High':
        return <AlertTriangle className="w-12 h-12 text-red-500" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Main Risk Display */}
      <div className={`${riskBgColor} rounded-xl p-6 text-center`}>
        <div className="flex justify-center mb-4">
          <RiskIcon />
        </div>
        
        <h3 className={`text-3xl font-bold ${riskColor} mb-2`}>
          {result.riskLevel} Risk
        </h3>
        
        <div className="text-5xl font-bold text-gray-800 mb-2">
          {result.riskPercentage.toFixed(1)}%
        </div>
        
        <p className="text-sm text-gray-600">
          Confidence interval: {result.confidenceLower.toFixed(1)}% - {result.confidenceUpper.toFixed(1)}%
        </p>
      </div>

      {/* Risk Gauge */}
      <div className="relative h-8 bg-gradient-to-r from-green-400 via-yellow-400 to-red-500 rounded-full overflow-hidden">
        <div
          className="absolute top-0 h-full w-1 bg-black"
          style={{ left: `${Math.min(result.riskPercentage, 100)}%` }}
        >
          <div className="absolute -top-2 left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-b-4 border-transparent border-b-black" />
        </div>
        <div className="absolute inset-0 flex justify-between items-center px-4 text-xs font-semibold text-white">
          <span>Low</span>
          <span>Moderate</span>
          <span>High</span>
        </div>
      </div>

      {/* Interpretation */}
      <div className="bg-gray-50 rounded-lg p-4 space-y-3">
        <h4 className="font-semibold text-gray-700">What does this mean?</h4>
        
        {result.riskLevel === 'Low' && (
          <p className="text-sm text-gray-600">
            The voice biomarkers analyzed show patterns typically associated with healthy individuals. 
            However, this is a screening tool and not a diagnostic test. If you have concerns about 
            Parkinson&apos;s disease, please consult a healthcare professional.
          </p>
        )}
        
        {result.riskLevel === 'Moderate' && (
          <p className="text-sm text-gray-600">
            The voice biomarkers show some patterns that may warrant further evaluation. This does not 
            mean you have Parkinson&apos;s disease. Many factors can affect voice characteristics. 
            Consider consulting a neurologist for a comprehensive evaluation if you have other symptoms.
          </p>
        )}
        
        {result.riskLevel === 'High' && (
          <p className="text-sm text-gray-600">
            The voice biomarkers show patterns that are often associated with Parkinson&apos;s disease. 
            <strong> This is not a diagnosis.</strong> Please consult a neurologist for proper clinical 
            evaluation. Early intervention can significantly improve quality of life.
          </p>
        )}
      </div>

      {/* Feature Summary */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="font-semibold text-gray-700 mb-3">Analysis Details</h4>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="text-gray-500">Input Type:</div>
          <div className="font-medium capitalize">{result.inputType}</div>
          
          <div className="text-gray-500">Analyzed At:</div>
          <div className="font-medium">{new Date(result.timestamp).toLocaleString()}</div>
          
          <div className="text-gray-500">Raw Score:</div>
          <div className="font-medium">{result.riskScore.toFixed(4)}</div>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="text-xs text-gray-400 bg-gray-50 p-3 rounded border border-gray-200">
        <strong>⚠️ Important Disclaimer:</strong> This tool is for educational and research purposes only. 
        It is not a medical device and should not be used for diagnosis. Always consult qualified 
        healthcare professionals for medical advice, diagnosis, or treatment.
      </div>
    </div>
  );
}
