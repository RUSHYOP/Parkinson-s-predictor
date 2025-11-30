---
title: Parkinson's Disease Risk Assessment
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: AI-powered voice biomarker analysis for Parkinson's screening
---

# Parkinson's Disease Risk Assessment

An AI-powered tool that analyzes voice biomarkers to assess Parkinson's disease risk.

## Features

- 🎤 **Voice Analysis**: Upload or record voice samples for automatic biomarker extraction
- 📊 **ML Prediction**: XGBoost model trained on 6,000+ clinical voice samples
- 🔍 **Explainability**: SHAP values show which features influence predictions
- 📈 **History Tracking**: SQLite database stores prediction history

## Voice Biomarkers Analyzed

| Biomarker | Description |
|-----------|-------------|
| Jitter | Frequency variation (vocal tremor) |
| Shimmer | Amplitude variation (vocal weakness) |
| HNR | Harmonics-to-noise ratio (voice clarity) |
| RPDE, DFA, PPE | Nonlinear dynamics measures |

## Data Sources

- UCI Parkinson's Voice Dataset (195 samples)
- Oxford Parkinson's Telemonitoring Dataset (5,875 samples)

## ⚠️ Disclaimer

This is a screening tool for educational purposes only. It is NOT a medical diagnostic tool. 
Please consult a neurologist for proper medical evaluation.

## Technology

- Gradio for UI
- XGBoost with Optuna hyperparameter tuning
- Praat/Parselmouth for voice feature extraction
- SHAP for model explainability
