'use client';

import { useState, useCallback, useEffect } from 'react';
import { Mic, Edit3, History, Info, Brain } from 'lucide-react';
import VoiceAnalyzer from '@/components/VoiceAnalyzer';
import ManualInput from '@/components/ManualInput';
import RiskDisplay from '@/components/RiskDisplay';
import HistoryTable from '@/components/HistoryTable';
import { VoiceFeatures, PredictionResult } from '@/lib/types';
import { predict, loadModel, getModelInfo } from '@/lib/predictor';

type TabType = 'voice' | 'manual' | 'history' | 'about';

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabType>('voice');
  const [isProcessing, setIsProcessing] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelInfo, setModelInfo] = useState<{ accuracy: number; f1: number } | null>(null);

  // Load model on mount
  useEffect(() => {
    loadModel()
      .then(() => {
        setModelLoaded(true);
        return getModelInfo();
      })
      .then(info => {
        if (info) {
          setModelInfo({ accuracy: info.accuracy * 100, f1: info.f1_score * 100 });
        }
      })
      .catch(err => console.error('Failed to load model:', err));
  }, []);

  const handlePrediction = useCallback(async (features: VoiceFeatures, inputType: 'voice' | 'manual') => {
    setIsProcessing(true);
    try {
      const result = await predict(features);
      
      const predictionResult: PredictionResult = {
        riskScore: result.probability,
        riskLevel: result.riskLevel,
        riskPercentage: result.riskPercentage,
        confidenceLower: result.confidenceLower,
        confidenceUpper: result.confidenceUpper,
        features,
        inputType,
        timestamp: new Date(),
      };
      
      setPrediction(predictionResult);
      
      // Save to history
      try {
        await fetch('/api/history', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(predictionResult),
        });
      } catch (err) {
        console.error('Failed to save to history:', err);
      }
    } catch (error) {
      console.error('Prediction error:', error);
      alert('Failed to make prediction. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  }, []);

  const handleVoiceFeatures = useCallback((features: Record<string, number>) => {
    handlePrediction(features as unknown as VoiceFeatures, 'voice');
  }, [handlePrediction]);

  const handleManualSubmit = useCallback((features: VoiceFeatures) => {
    handlePrediction(features, 'manual');
  }, [handlePrediction]);

  const tabs = [
    { id: 'voice' as TabType, label: 'Voice Analysis', icon: Mic },
    { id: 'manual' as TabType, label: 'Manual Input', icon: Edit3 },
    { id: 'history' as TabType, label: 'History', icon: History },
    { id: 'about' as TabType, label: 'About', icon: Info },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-6xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Brain className="w-10 h-10 text-indigo-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-800">
                  Parkinson&apos;s Disease Risk Assessment
                </h1>
                <p className="text-sm text-gray-500">
                  AI-powered voice biomarker analysis
                </p>
              </div>
            </div>
            
            {modelLoaded && modelInfo && (
              <div className="text-right text-sm">
                <div className="text-green-600 font-medium">✓ Model Loaded</div>
                <div className="text-gray-500">
                  Accuracy: {modelInfo.accuracy.toFixed(1)}% | F1: {modelInfo.f1.toFixed(1)}%
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 py-8">
        {/* Tabs */}
        <div className="flex space-x-1 bg-white rounded-xl p-1 shadow-sm mb-6">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 flex items-center justify-center space-x-2 py-3 px-4 rounded-lg font-medium transition-colors ${
                activeTab === tab.id
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              <tab.icon className="w-5 h-5" />
              <span className="hidden sm:inline">{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Panel - Input */}
          <div className="bg-white rounded-xl shadow-sm p-6">
            {activeTab === 'voice' && (
              <>
                <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center space-x-2">
                  <Mic className="w-6 h-6 text-indigo-600" />
                  <span>Voice Analysis</span>
                </h2>
                <VoiceAnalyzer 
                  onFeaturesExtracted={handleVoiceFeatures}
                  isProcessing={isProcessing}
                />
              </>
            )}
            
            {activeTab === 'manual' && (
              <>
                <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center space-x-2">
                  <Edit3 className="w-6 h-6 text-indigo-600" />
                  <span>Manual Input</span>
                </h2>
                <ManualInput 
                  onSubmit={handleManualSubmit}
                  isProcessing={isProcessing}
                />
              </>
            )}
            
            {activeTab === 'history' && (
              <>
                <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center space-x-2">
                  <History className="w-6 h-6 text-indigo-600" />
                  <span>Prediction History</span>
                </h2>
                <HistoryTable />
              </>
            )}
            
            {activeTab === 'about' && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold text-gray-800 flex items-center space-x-2">
                  <Info className="w-6 h-6 text-indigo-600" />
                  <span>About This Tool</span>
                </h2>
                
                <div className="prose prose-sm max-w-none">
                  <h3 className="text-lg font-semibold">How It Works</h3>
                  <p className="text-gray-600">
                    This tool analyzes voice biomarkers to assess Parkinson&apos;s disease risk. 
                    It extracts 16 clinical features from voice recordings, including jitter, shimmer, 
                    and nonlinear dynamics measures, which have been shown to correlate with 
                    Parkinson&apos;s disease in research studies.
                  </p>
                  
                  <h3 className="text-lg font-semibold mt-4">Voice Biomarkers</h3>
                  <ul className="text-gray-600 space-y-1">
                    <li><strong>Jitter</strong> - Measures frequency variation in voice</li>
                    <li><strong>Shimmer</strong> - Measures amplitude variation in voice</li>
                    <li><strong>NHR/HNR</strong> - Noise-to-Harmonics ratios (voice clarity)</li>
                    <li><strong>RPDE</strong> - Recurrence Period Density Entropy</li>
                    <li><strong>DFA</strong> - Detrended Fluctuation Analysis</li>
                    <li><strong>PPE</strong> - Pitch Period Entropy</li>
                  </ul>
                  
                  <h3 className="text-lg font-semibold mt-4">Data Sources</h3>
                  <p className="text-gray-600">
                    The model was trained on combined datasets from UCI Machine Learning Repository, 
                    including Oxford Parkinson&apos;s Disease Detection Dataset and 
                    Parkinson&apos;s Telemonitoring Dataset.
                  </p>
                  
                  <h3 className="text-lg font-semibold mt-4">Technology</h3>
                  <ul className="text-gray-600 space-y-1">
                    <li>XGBoost gradient boosting classifier</li>
                    <li>Parselmouth (Praat) for voice feature extraction</li>
                    <li>Client-side inference using model exported to JSON</li>
                  </ul>
                </div>
                
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                  <h4 className="font-semibold text-yellow-800">⚠️ Important Disclaimer</h4>
                  <p className="text-sm text-yellow-700 mt-1">
                    This tool is for educational and research purposes only. It is NOT a medical 
                    diagnostic device. The results should not be used to diagnose, treat, or 
                    prevent any disease. Always consult qualified healthcare professionals for 
                    medical advice.
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Right Panel - Results */}
          <div className="bg-white rounded-xl shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Risk Assessment Result</h2>
            <RiskDisplay result={prediction} />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-auto">
        <div className="max-w-6xl mx-auto px-4 py-6 text-center text-sm text-gray-500">
          <p>Parkinson&apos;s Disease Risk Assessment Tool</p>
          <p className="mt-1">Built with Next.js, XGBoost, and ❤️</p>
        </div>
      </footer>
    </div>
  );
}
