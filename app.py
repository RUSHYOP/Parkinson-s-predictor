import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import warnings

warnings.filterwarnings('ignore')


class ParkinsonsPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'age', 'tremor_at_rest', 'bradykinesia', 'rigidity',
            'postural_instability', 'speech_problems', 'writing_changes',
            'sleep_disorders', 'smell_loss', 'depression_anxiety',
            'family_history', 'head_trauma', 'pesticide_exposure'
        ]
        # Try to load existing model, if not available, train a new one
        if not self.load_model():
            self.train_model()

    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic data for demonstration purposes"""
        np.random.seed(42)
        data = []

        for _ in range(n_samples):
            # Generate features with realistic correlations
            age = np.random.normal(65, 15)
            age = max(30, min(90, age))  # Clamp between 30-90

            # Higher risk features for older individuals
            age_factor = (age - 30) / 60  # Normalize age factor

            tremor = np.random.choice([0, 1], p=[0.7 - 0.2 * age_factor, 0.3 + 0.2 * age_factor])
            bradykinesia = np.random.choice([0, 1], p=[0.75 - 0.2 * age_factor, 0.25 + 0.2 * age_factor])
            rigidity = np.random.choice([0, 1], p=[0.8 - 0.15 * age_factor, 0.2 + 0.15 * age_factor])
            postural = np.random.choice([0, 1], p=[0.85 - 0.15 * age_factor, 0.15 + 0.15 * age_factor])
            speech = np.random.choice([0, 1], p=[0.8, 0.2])
            writing = np.random.choice([0, 1], p=[0.75, 0.25])
            sleep = np.random.choice([0, 1], p=[0.6, 0.4])
            smell = np.random.choice([0, 1], p=[0.7, 0.3])
            depression = np.random.choice([0, 1], p=[0.7, 0.3])
            family_hist = np.random.choice([0, 1], p=[0.9, 0.1])
            head_trauma = np.random.choice([0, 1], p=[0.85, 0.15])
            pesticide = np.random.choice([0, 1], p=[0.9, 0.1])

            # Calculate target based on risk factors
            risk_score = (tremor * 0.25 + bradykinesia * 0.25 + rigidity * 0.2 +
                          postural * 0.15 + speech * 0.1 + writing * 0.1 +
                          sleep * 0.05 + smell * 0.1 + depression * 0.05 +
                          family_hist * 0.15 + head_trauma * 0.05 + pesticide * 0.05 +
                          age_factor * 0.2)

            # Add some noise and create binary target
            risk_score += np.random.normal(0, 0.1)
            target = 1 if risk_score > 0.4 else 0

            data.append([
                age, tremor, bradykinesia, rigidity, postural, speech, writing,
                sleep, smell, depression, family_hist, head_trauma, pesticide, target
            ])

        columns = self.feature_names + ['target']
        return pd.DataFrame(data, columns=columns)

    def save_model(self):
        """Save the trained model and scaler using joblib"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)

            # Save model and scaler
            joblib.dump(self.model, 'models/parkinsons_model.pkl')
            joblib.dump(self.scaler, 'models/parkinsons_scaler.pkl')
            print("Model and scaler saved successfully!")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self):
        """Load pre-trained model and scaler using joblib"""
        try:
            if os.path.exists('models/parkinsons_model.pkl') and os.path.exists('models/parkinsons_scaler.pkl'):
                self.model = joblib.load('models/parkinsons_model.pkl')
                self.scaler = joblib.load('models/parkinsons_scaler.pkl')
                print("Pre-trained model loaded successfully!")
                return True
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def train_model(self):
        """Train the machine learning model"""
        # Generate synthetic dataset
        df = self.generate_synthetic_data()

        X = df[self.feature_names]
        y = df['target']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        # Print detailed evaluation
        print(f"Model Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))

        # Save the trained model and scaler
        self.save_model()

    def predict_risk(self, age, tremor, bradykinesia, rigidity, postural,
                     speech, writing, sleep, smell, depression, family, trauma, pesticide):
        """Predict Parkinson's risk based on input symptoms"""

        # Prepare input data
        input_data = np.array([[
            age, tremor, bradykinesia, rigidity, postural, speech, writing,
            sleep, smell, depression, family, trauma, pesticide
        ]])

        # Scale the input
        input_scaled = self.scaler.transform(input_data)

        # Get prediction and probability
        prediction = self.model.predict(input_scaled)[0]
        probability = self.model.predict_proba(input_scaled)[0]

        risk_percentage = probability[1] * 100

        # Generate detailed response
        if risk_percentage < 20:
            risk_level = "Low"
            recommendation = "Your symptoms suggest a low risk for Parkinson's disease. Continue maintaining a healthy lifestyle."
        elif risk_percentage < 50:
            risk_level = "Moderate"
            recommendation = "Your symptoms suggest moderate risk. Consider consulting with a neurologist for further evaluation."
        else:
            risk_level = "High"
            recommendation = "Your symptoms suggest higher risk. It's recommended to consult with a neurologist as soon as possible for proper evaluation."

        # Feature importance for explanation
        feature_importance = self.model.feature_importances_
        input_values = [age, tremor, bradykinesia, rigidity, postural, speech, writing,
                        sleep, smell, depression, family, trauma, pesticide]

        important_factors = []
        for i, (feature, importance, value) in enumerate(zip(self.feature_names, feature_importance, input_values)):
            if importance > 0.05 and value > 0:  # Only show important factors that are present
                important_factors.append(f"• {feature.replace('_', ' ').title()}")

        explanation = "Key factors contributing to this assessment:\n" + "\n".join(important_factors[:5])

        return {
            "Risk Level": risk_level,
            "Risk Percentage": f"{risk_percentage:.1f}%",
            "Recommendation": recommendation,
            "Explanation": explanation,
            "Disclaimer": "⚠️ This is an AI-based assessment tool for educational purposes only. It is NOT a substitute for professional medical diagnosis. Please consult with a qualified healthcare provider for proper medical evaluation."
        }


# Initialize the predictor
predictor = ParkinsonsPredictor()


def chatbot_interface(age, tremor, bradykinesia, rigidity, postural, speech, writing,
                      sleep, smell, depression, family, trauma, pesticide):
    """Main chatbot interface function"""
    try:
        result = predictor.predict_risk(
            age, tremor, bradykinesia, rigidity, postural, speech, writing,
            sleep, smell, depression, family, trauma, pesticide
        )

        response = f"""
# Parkinson's Disease Risk Assessment

**Risk Level:** {result['Risk Level']}
**Risk Percentage:** {result['Risk Percentage']}

## Recommendation
{result['Recommendation']}

## Analysis
{result['Explanation']}

---
{result['Disclaimer']}
        """

        return response

    except Exception as e:
        return f"An error occurred during assessment: {str(e)}"


# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Parkinson's Disease Risk Assessment", theme=gr.themes.Soft()) as iface:
        gr.Markdown(
            """
            # 🧠 Parkinson's Disease Risk Assessment Chatbot

            This AI-powered tool helps assess your risk of Parkinson's disease based on common symptoms and risk factors.
            Please answer the questions below honestly for the most accurate assessment.

            **Important:** This tool is for educational purposes only and should not replace professional medical advice.
            """
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Basic Information")
                age = gr.Slider(
                    minimum=30, maximum=90, value=65, step=1,
                    label="Age", info="Your current age"
                )

                gr.Markdown("### Motor Symptoms")
                tremor = gr.Checkbox(
                    label="Tremor at Rest",
                    info="Do you experience shaking when your muscles are relaxed?"
                )
                bradykinesia = gr.Checkbox(
                    label="Bradykinesia (Slowness)",
                    info="Have you noticed slowness in your movements?"
                )
                rigidity = gr.Checkbox(
                    label="Muscle Rigidity",
                    info="Do you experience muscle stiffness or rigidity?"
                )
                postural = gr.Checkbox(
                    label="Postural Instability",
                    info="Do you have trouble with balance or posture?"
                )

            with gr.Column():
                gr.Markdown("### Non-Motor Symptoms")
                speech = gr.Checkbox(
                    label="Speech Problems",
                    info="Have you noticed changes in your speech (softer, slurred)?"
                )
                writing = gr.Checkbox(
                    label="Writing Changes",
                    info="Has your handwriting become smaller or more difficult?"
                )
                sleep = gr.Checkbox(
                    label="Sleep Disorders",
                    info="Do you experience sleep disturbances or REM sleep behavior?"
                )
                smell = gr.Checkbox(
                    label="Loss of Smell",
                    info="Have you experienced a reduced sense of smell?"
                )
                depression = gr.Checkbox(
                    label="Depression/Anxiety",
                    info="Have you experienced depression or anxiety symptoms?"
                )

                gr.Markdown("### Risk Factors")
                family = gr.Checkbox(
                    label="Family History",
                    info="Do you have family members with Parkinson's disease?"
                )
                trauma = gr.Checkbox(
                    label="Head Trauma History",
                    info="Have you experienced significant head injuries?"
                )
                pesticide = gr.Checkbox(
                    label="Pesticide Exposure",
                    info="Have you been exposed to pesticides or herbicides?"
                )

        with gr.Row():
            assess_btn = gr.Button("🔍 Assess Risk", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ Clear All", variant="secondary")

        output = gr.Markdown()

        # Event handlers
        assess_btn.click(
            fn=chatbot_interface,
            inputs=[age, tremor, bradykinesia, rigidity, postural, speech, writing,
                    sleep, smell, depression, family, trauma, pesticide],
            outputs=output
        )

        def clear_all():
            return (65, False, False, False, False, False, False,
                    False, False, False, False, False, False, "")

        clear_btn.click(
            fn=clear_all,
            outputs=[age, tremor, bradykinesia, rigidity, postural, speech, writing,
                     sleep, smell, depression, family, trauma, pesticide, output]
        )

        gr.Markdown(
            """
            ---
            ### About This Tool
            This chatbot uses machine learning to analyze symptom patterns associated with Parkinson's disease risk.
            The model is trained on symptom correlations and provides educational insights only.

            **Always consult with healthcare professionals for proper medical evaluation and diagnosis.**
            """
        )

    return iface


# Launch the application
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="localhost",
        server_port=7860,
        share=True,
        debug=True
    )