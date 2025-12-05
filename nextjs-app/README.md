# Parkinson's Disease Risk Assessment - Next.js

An AI-powered voice biomarker analysis tool for Parkinson's disease risk assessment, built with Next.js and deployed on Vercel.

## Features

- 🎤 **Voice Analysis**: Record or upload audio for real-time voice biomarker extraction
- ✏️ **Manual Input**: Enter clinical measurements manually for risk assessment  
- 📊 **Risk Prediction**: XGBoost-based ML model running client-side for instant predictions
- 📈 **History Tracking**: View and export your prediction history
- 🔒 **Privacy-First**: Audio processing on server, predictions run in your browser

## Tech Stack

- **Frontend**: Next.js 14+ with App Router, React, Tailwind CSS
- **ML Inference**: XGBoost model exported to JSON, running client-side
- **Voice Analysis**: Python serverless functions using Parselmouth (Praat) 
- **Deployment**: Vercel with Python runtime support

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the app.

### Local Development Notes

- Voice analysis uses simulated features in development
- Deploy to Vercel for real Python-based voice feature extraction
- The XGBoost model runs entirely in the browser

## Deployment to Vercel

1. Push this repository to GitHub
2. Import the project in [Vercel](https://vercel.com)
3. Vercel will automatically detect Next.js and Python functions
4. Deploy!

## Project Structure

```
nextjs-app/
├── api/
│   ├── extract-features.py    # Python serverless function for voice analysis
│   └── requirements.txt       # Python dependencies
├── public/
│   └── models/
│       ├── parkinsons_model.json  # XGBoost model
│       ├── scaler.json            # Feature scaler
│       └── model_meta.json        # Model metadata
├── src/
│   ├── app/
│   │   ├── api/              # Next.js API routes
│   │   ├── page.tsx          # Main page
│   │   └── layout.tsx        # Root layout
│   ├── components/
│   │   ├── VoiceAnalyzer.tsx # Audio recording/upload
│   │   ├── ManualInput.tsx   # Feature sliders
│   │   ├── RiskDisplay.tsx   # Prediction results
│   │   └── HistoryTable.tsx  # Prediction history
│   └── lib/
│       ├── types.ts          # TypeScript types
│       └── predictor.ts      # Client-side ML inference
└── vercel.json               # Vercel configuration
```

## Voice Biomarkers

The model analyzes 16 voice biomarkers:

| Feature | Description |
|---------|-------------|
| Jitter (5 types) | Frequency variation measures |
| Shimmer (6 types) | Amplitude variation measures |
| NHR | Noise-to-Harmonics Ratio |
| HNR | Harmonics-to-Noise Ratio |
| RPDE | Recurrence Period Density Entropy |
| DFA | Detrended Fluctuation Analysis |
| PPE | Pitch Period Entropy |

## Model Performance

- **Accuracy**: 74.4%
- **F1 Score**: 75.5%
- **ROC AUC**: 82.4%

Trained on combined UCI Parkinson's datasets (6,000+ samples).

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is NOT a medical diagnostic device. The results should not be used to diagnose, treat, or prevent any disease. Always consult qualified healthcare professionals for medical advice.

## License

MIT
