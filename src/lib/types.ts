// Type definitions for Parkinson's Disease Predictor

export interface VoiceFeatures {
  'Jitter(%)': number;
  'Jitter(Abs)': number;
  'Jitter:RAP': number;
  'Jitter:PPQ5': number;
  'Jitter:DDP': number;
  'Shimmer': number;
  'Shimmer(dB)': number;
  'Shimmer:APQ3': number;
  'Shimmer:APQ5': number;
  'Shimmer:APQ11': number;
  'Shimmer:DDA': number;
  'NHR': number;
  'HNR': number;
  'RPDE': number;
  'DFA': number;
  'PPE': number;
}

export interface ScalerParams {
  mean: number[];
  scale: number[];
  feature_names: string[];
}

export interface ModelMeta {
  model_type: string;
  n_features: number;
  feature_names: string[];
  feature_order: string[];
  accuracy: number;
  f1_score: number;
  roc_auc: number;
  best_params: Record<string, number>;
  converted_at: string;
}

export interface PredictionResult {
  riskScore: number;
  riskLevel: 'Low' | 'Moderate' | 'High';
  riskPercentage: number;
  confidenceLower: number;
  confidenceUpper: number;
  features: VoiceFeatures;
  inputType: 'voice' | 'manual';
  timestamp: Date;
}

export interface PredictionHistory {
  id: string;
  timestamp: Date;
  inputType: 'voice' | 'manual';
  features: VoiceFeatures;
  riskScore: number;
  riskLevel: 'Low' | 'Moderate' | 'High';
  riskPercentage: number;
  confidenceLower: number;
  confidenceUpper: number;
  audioFilename?: string;
  audioDuration?: number;
  notes?: string;
}

export interface FeatureRange {
  name: string;
  key: keyof VoiceFeatures;
  min: number;
  max: number;
  step: number;
  defaultValue: number;
  description: string;
  unit: string;
}

// Feature ranges based on the Parkinson's dataset statistics
export const FEATURE_RANGES: FeatureRange[] = [
  { name: 'Jitter (%)', key: 'Jitter(%)', min: 0, max: 3, step: 0.001, defaultValue: 0.5, description: 'Frequency variation percentage', unit: '%' },
  { name: 'Jitter (Abs)', key: 'Jitter(Abs)', min: 0, max: 0.0003, step: 0.00001, defaultValue: 0.00003, description: 'Absolute jitter in seconds', unit: 's' },
  { name: 'Jitter RAP', key: 'Jitter:RAP', min: 0, max: 2, step: 0.001, defaultValue: 0.3, description: 'Relative Average Perturbation', unit: '' },
  { name: 'Jitter PPQ5', key: 'Jitter:PPQ5', min: 0, max: 2, step: 0.001, defaultValue: 0.3, description: 'Five-point Period Perturbation Quotient', unit: '' },
  { name: 'Jitter DDP', key: 'Jitter:DDP', min: 0, max: 5, step: 0.001, defaultValue: 0.8, description: 'Average absolute difference of differences', unit: '' },
  { name: 'Shimmer', key: 'Shimmer', min: 0, max: 0.2, step: 0.001, defaultValue: 0.03, description: 'Amplitude variation', unit: '' },
  { name: 'Shimmer (dB)', key: 'Shimmer(dB)', min: 0, max: 2, step: 0.01, defaultValue: 0.3, description: 'Shimmer in decibels', unit: 'dB' },
  { name: 'Shimmer APQ3', key: 'Shimmer:APQ3', min: 0, max: 0.1, step: 0.001, defaultValue: 0.015, description: 'Three-point Amplitude Perturbation Quotient', unit: '' },
  { name: 'Shimmer APQ5', key: 'Shimmer:APQ5', min: 0, max: 0.15, step: 0.001, defaultValue: 0.02, description: 'Five-point Amplitude Perturbation Quotient', unit: '' },
  { name: 'Shimmer APQ11', key: 'Shimmer:APQ11', min: 0, max: 0.15, step: 0.001, defaultValue: 0.025, description: '11-point Amplitude Perturbation Quotient', unit: '' },
  { name: 'Shimmer DDA', key: 'Shimmer:DDA', min: 0, max: 0.3, step: 0.001, defaultValue: 0.045, description: 'Average absolute difference between consecutive differences', unit: '' },
  { name: 'NHR', key: 'NHR', min: 0, max: 0.4, step: 0.001, defaultValue: 0.03, description: 'Noise-to-Harmonics Ratio', unit: '' },
  { name: 'HNR', key: 'HNR', min: 0, max: 40, step: 0.1, defaultValue: 22, description: 'Harmonics-to-Noise Ratio', unit: 'dB' },
  { name: 'RPDE', key: 'RPDE', min: 0, max: 1, step: 0.001, defaultValue: 0.5, description: 'Recurrence Period Density Entropy', unit: '' },
  { name: 'DFA', key: 'DFA', min: 0.5, max: 1, step: 0.001, defaultValue: 0.7, description: 'Detrended Fluctuation Analysis', unit: '' },
  { name: 'PPE', key: 'PPE', min: 0, max: 0.6, step: 0.001, defaultValue: 0.2, description: 'Pitch Period Entropy', unit: '' },
];

export function getDefaultFeatures(): VoiceFeatures {
  const features: Partial<VoiceFeatures> = {};
  for (const range of FEATURE_RANGES) {
    features[range.key] = range.defaultValue;
  }
  return features as VoiceFeatures;
}

export function getRiskLevel(score: number): 'Low' | 'Moderate' | 'High' {
  if (score < 0.4) return 'Low';
  if (score < 0.7) return 'Moderate';
  return 'High';
}

export function getRiskColor(level: 'Low' | 'Moderate' | 'High'): string {
  switch (level) {
    case 'Low': return 'text-green-600';
    case 'Moderate': return 'text-yellow-600';
    case 'High': return 'text-red-600';
  }
}

export function getRiskBgColor(level: 'Low' | 'Moderate' | 'High'): string {
  switch (level) {
    case 'Low': return 'bg-green-100';
    case 'Moderate': return 'bg-yellow-100';
    case 'High': return 'bg-red-100';
  }
}
