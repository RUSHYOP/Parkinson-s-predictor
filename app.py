"""
Parkinson's Disease Risk Assessment Application

A modern ML-powered tool for assessing Parkinson's disease risk using:
1. Voice analysis with audio biomarker extraction
2. Manual measurement input
3. Session history tracking

Built with real clinical datasets (UCI + Oxford) and XGBoost ML pipeline.
"""

import gradio as gr
import numpy as np
import pandas as pd
import os
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import warnings
import threading

warnings.filterwarnings('ignore')

# Import our modules
from src.data_loader import DataLoader
from src.voice_analyzer import VoiceAnalyzer, PARSELMOUTH_AVAILABLE, LIBROSA_AVAILABLE
from src.model_trainer import ModelTrainer, SHAP_AVAILABLE
from src.database import DatabaseManager


class ParkinsonsApp:
    """Main application class for Parkinson's Disease Risk Assessment."""
    
    def __init__(self):
        """Initialize the application components."""
        print("\n" + "=" * 60)
        print("🧠 Parkinson's Disease Risk Assessment System")
        print("=" * 60)
        
        self.data_dir = "data"
        self.models_dir = "models"
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize components
        self.db = DatabaseManager(db_path=os.path.join(self.data_dir, "history.db"))
        self.voice_analyzer = VoiceAnalyzer() if (PARSELMOUTH_AVAILABLE or LIBROSA_AVAILABLE) else None
        self.model_trainer = ModelTrainer(data_dir=self.data_dir, models_dir=self.models_dir)
        
        # Training status
        self.is_trained = False
        self.training_in_progress = False
        
        # Try to load existing model
        if self.model_trainer.model_exists():
            self.model_trainer.load_model()
            self.is_trained = True
    
    def get_model_status(self) -> Tuple[str, str, str]:
        """Get model status for display."""
        if self.is_trained and self.model_trainer.metrics:
            m = self.model_trainer.metrics
            return (
                f"✓ {m.get('model_type', 'Model').upper()} Ready",
                f"{m.get('test_accuracy', 0):.1%}",
                f"{m.get('test_roc_auc', 0):.1%}"
            )
        return ("○ Not Trained", "—", "—")
    
    def start_training(self) -> Tuple[str, str, str]:
        """Start model training."""
        if self.training_in_progress:
            return ("⏳ Training...", "—", "—")
        
        if not self.is_trained:
            self.training_in_progress = True
            try:
                self.model_trainer.train(n_trials=30)
                self.is_trained = True
            finally:
                self.training_in_progress = False
        
        return self.get_model_status()
    
    def predict_from_voice(self, audio_file) -> Tuple[str, str, str, Optional[str]]:
        """Analyze voice recording for Parkinson's risk."""
        if audio_file is None:
            return ("—", "Upload an audio file to begin", "", None)
        
        if not self.is_trained:
            return ("—", "Please train the model first", "", None)
        
        if self.voice_analyzer is None:
            return ("—", "Voice analysis unavailable", "", None)
        
        try:
            # Extract features and predict
            features = self.voice_analyzer.extract_features(audio_file)
            feature_dict = features.to_dict()
            result = self.model_trainer.predict_with_explanation(features.to_array())
            shap_plot = self.model_trainer.get_shap_plot(features.to_array())
            
            # Save to database
            self.db.save_prediction(
                input_type='voice',
                features=feature_dict,
                risk_score=result['probability'],
                risk_level=result['risk_level'],
                confidence_interval=result['confidence_interval'],
                shap_values=result.get('feature_contributions'),
                audio_filename=os.path.basename(audio_file) if audio_file else None,
                audio_duration=features.duration_seconds
            )
            
            # Format output
            risk_pct = f"{result['risk_percentage']:.0f}%"
            ci = f"{result['confidence_interval'][0]*100:.0f}–{result['confidence_interval'][1]*100:.0f}%"
            
            details = f"""**Duration:** {features.duration_seconds:.1f}s  •  **Method:** {features.extraction_method}

| Biomarker | Value | | Biomarker | Value |
|:--|--:|:--|:--|--:|
| Jitter | {feature_dict['Jitter(%)']:.3f}% | | Shimmer | {feature_dict['Shimmer']:.4f} |
| HNR | {feature_dict['HNR']:.1f} dB | | NHR | {feature_dict['NHR']:.4f} |
| RPDE | {feature_dict['RPDE']:.3f} | | DFA | {feature_dict['DFA']:.3f} |
| PPE | {feature_dict['PPE']:.3f} | | | |"""
            
            return (risk_pct, f"Risk Level: **{result['risk_level']}**  •  Confidence: {ci}", details, shap_plot)
            
        except Exception as e:
            return ("—", f"Error: {str(e)}", "", None)
    
    def predict_from_manual(self, jitter, jitter_abs, shimmer, shimmer_db, nhr, hnr, rpde, dfa, ppe) -> Tuple[str, str, Optional[str]]:
        """Predict from manual input."""
        if not self.is_trained:
            return ("—", "Please train the model first", None)
        
        try:
            features = np.array([
                jitter, jitter_abs, jitter/3, jitter/2.5, jitter,
                shimmer, shimmer_db, shimmer/3, shimmer/4, shimmer/5, shimmer,
                nhr, hnr, rpde, dfa, ppe
            ])
            
            result = self.model_trainer.predict_with_explanation(features)
            shap_plot = self.model_trainer.get_shap_plot(features)
            
            # Save to database
            feature_names = self.model_trainer.feature_names
            self.db.save_prediction(
                input_type='manual',
                features=dict(zip(feature_names, features)),
                risk_score=result['probability'],
                risk_level=result['risk_level'],
                confidence_interval=result['confidence_interval'],
                shap_values=result.get('feature_contributions')
            )
            
            risk_pct = f"{result['risk_percentage']:.0f}%"
            ci = f"{result['confidence_interval'][0]*100:.0f}–{result['confidence_interval'][1]*100:.0f}%"
            
            return (risk_pct, f"Risk Level: **{result['risk_level']}**  •  Confidence: {ci}", shap_plot)
            
        except Exception as e:
            return ("—", f"Error: {str(e)}", None)
    
    def get_history(self) -> pd.DataFrame:
        """Get prediction history."""
        history = self.db.get_history(limit=50)
        if not history:
            return pd.DataFrame(columns=['Date', 'Type', 'Risk', 'Score'])
        
        return pd.DataFrame([{
            'Date': h['timestamp'][:16].replace('T', ' ') if h['timestamp'] else '',
            'Type': '🎤' if h['input_type'] == 'voice' else '✏️',
            'Risk': h['risk_level'],
            'Score': f"{h['risk_percentage']:.0f}%"
        } for h in history])
    
    def get_stats(self) -> str:
        """Get statistics summary."""
        s = self.db.get_statistics()
        if s['total_predictions'] == 0:
            return "No predictions yet"
        return f"**{s['total_predictions']}** total  •  🎤 {s['voice_analyses']}  •  ✏️ {s['symptom_assessments']}  •  Avg: **{s['average_risk_score']*100:.0f}%**"
    
    def clear_history(self) -> Tuple[pd.DataFrame, str]:
        """Clear all history."""
        self.db.clear_all()
        return self.get_history(), "History cleared"


def create_app():
    """Create the Gradio application with modern UI."""
    
    app = ParkinsonsApp()
    status, acc, auc = app.get_model_status()
    
    with gr.Blocks(
        title="Parkinson's Risk Assessment",
        fill_height=True
    ) as interface:
        
        # Header
        gr.HTML("""
        <div class="header-section">
            <h1 style="font-size: 1.6rem; font-weight: 600; color: #1e293b; margin: 0;">
                Parkinson's Risk Assessment
            </h1>
            <p style="color: #64748b; margin-top: 0.4rem; font-size: 0.9rem;">
                AI-powered voice biomarker analysis
            </p>
        </div>
        """)
        
        # Status Bar
        with gr.Row():
            with gr.Column(scale=1, min_width=100):
                gr.HTML('<p class="stat-label">STATUS</p>')
                model_status = gr.Markdown(f"**{status}**")
            with gr.Column(scale=1, min_width=100):
                gr.HTML('<p class="stat-label">ACCURACY</p>')
                model_acc = gr.Markdown(f"**{acc}**")
            with gr.Column(scale=1, min_width=100):
                gr.HTML('<p class="stat-label">ROC AUC</p>')
                model_auc = gr.Markdown(f"**{auc}**")
            with gr.Column(scale=1, min_width=120):
                train_btn = gr.Button("Train Model", variant="secondary", size="sm")
        
        gr.HTML('<hr style="border: none; border-top: 1px solid #e2e8f0; margin: 1rem 0;">')
        
        # Main Content
        with gr.Tabs():
            
            # Voice Tab
            with gr.TabItem("🎤 Voice", id="voice"):
                gr.HTML("""
                <div class="info-box">
                    <p style="margin: 0; color: #475569; font-size: 0.875rem;">
                        Record or upload a sustained <strong>"ahhh"</strong> sound (3–10 seconds).
                        The AI extracts jitter, shimmer, and harmonic features.
                    </p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(
                            label="Voice Recording",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                        analyze_btn = gr.Button("Analyze", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.HTML('<p class="stat-label">RISK SCORE</p>')
                        voice_score = gr.Markdown("—", elem_classes=["score-big"])
                        voice_summary = gr.Markdown("Upload audio to begin")
                
                with gr.Accordion("📊 Details", open=False):
                    voice_details = gr.Markdown("")
                    voice_shap = gr.Image(show_label=False, height=300)
                
                analyze_btn.click(
                    fn=app.predict_from_voice,
                    inputs=[audio_input],
                    outputs=[voice_score, voice_summary, voice_details, voice_shap]
                )
            
            # Manual Tab
            with gr.TabItem("✏️ Manual", id="manual"):
                gr.HTML("""
                <div class="info-box">
                    <p style="margin: 0; color: #475569; font-size: 0.875rem;">
                        Enter voice measurements from clinical equipment.
                    </p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column():
                        jitter = gr.Slider(0, 3, value=0.5, step=0.01, label="Jitter (%)", info="Frequency variation")
                        jitter_abs = gr.Slider(0, 0.0005, value=0.00003, step=0.00001, label="Jitter (Abs)")
                        shimmer = gr.Slider(0, 0.3, value=0.03, step=0.005, label="Shimmer", info="Amplitude variation")
                        shimmer_db = gr.Slider(0, 1.5, value=0.3, step=0.05, label="Shimmer (dB)")
                    
                    with gr.Column():
                        nhr = gr.Slider(0, 0.3, value=0.02, step=0.005, label="NHR", info="Noise-to-harmonics")
                        hnr = gr.Slider(5, 35, value=22, step=0.5, label="HNR (dB)", info="Harmonics-to-noise")
                        rpde = gr.Slider(0.2, 0.8, value=0.5, step=0.01, label="RPDE")
                        dfa = gr.Slider(0.5, 0.9, value=0.7, step=0.01, label="DFA")
                        ppe = gr.Slider(0, 0.4, value=0.2, step=0.01, label="PPE")
                
                assess_btn = gr.Button("Assess Risk", variant="primary", size="lg")
                
                gr.HTML('<p class="stat-label" style="margin-top: 1rem;">RESULT</p>')
                with gr.Row():
                    manual_score = gr.Markdown("—", elem_classes=["score-big"])
                    manual_summary = gr.Markdown("Adjust values and assess")
                
                manual_shap = gr.Image(show_label=False, height=280)
                
                assess_btn.click(
                    fn=app.predict_from_manual,
                    inputs=[jitter, jitter_abs, shimmer, shimmer_db, nhr, hnr, rpde, dfa, ppe],
                    outputs=[manual_score, manual_summary, manual_shap]
                )
            
            # History Tab
            with gr.TabItem("📊 History", id="history"):
                with gr.Row():
                    stats_display = gr.Markdown(app.get_stats())
                with gr.Row():
                    refresh_btn = gr.Button("↻ Refresh", size="sm")
                    clear_btn = gr.Button("Clear All", variant="stop", size="sm")
                
                history_table = gr.DataFrame(value=app.get_history(), interactive=False)
                clear_msg = gr.Markdown("")
                
                refresh_btn.click(fn=app.get_history, outputs=history_table)
                refresh_btn.click(fn=app.get_stats, outputs=stats_display)
                clear_btn.click(fn=app.clear_history, outputs=[history_table, clear_msg])
            
            # About Tab
            with gr.TabItem("ℹ️ About", id="about"):
                gr.Markdown("""
### Voice Biomarkers

| Biomarker | What it measures | Parkinson's effect |
|:--|:--|:--|
| **Jitter** | Pitch instability | ↑ Increased |
| **Shimmer** | Volume instability | ↑ Increased |
| **HNR** | Voice clarity | ↓ Decreased |
| **RPDE, DFA, PPE** | Signal dynamics | Altered |

### Data & Model

- **6,000+** voice samples from UCI & Oxford datasets
- **XGBoost** classifier with Optuna tuning
- **SHAP** values for explainability

---
                """)
                
                gr.HTML("""
                <div class="disclaimer-box">
                    <strong>⚠️ Disclaimer</strong><br>
                    This is a screening tool for educational purposes. 
                    It cannot diagnose Parkinson's disease. Consult a neurologist for medical evaluation.
                </div>
                """)
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 1rem 0; margin-top: 1rem; border-top: 1px solid #e2e8f0;">
            <p style="color: #94a3b8; font-size: 0.75rem; margin: 0;">
                ⚠️ Screening tool only — not a medical diagnosis
            </p>
        </div>
        """)
        
        train_btn.click(fn=app.start_training, outputs=[model_status, model_acc, model_auc])
    
    return interface


if __name__ == "__main__":
    interface = create_app()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
