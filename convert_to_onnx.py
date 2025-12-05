"""
Convert XGBoost model to ONNX format for browser-based inference.
Uses simpler approach compatible with newer XGBoost versions.
"""

import joblib
import numpy as np
import json
import os

def convert_model():
    """Convert XGBoost model to ONNX format using native xgboost ubj export."""
    
    # Load the trained model
    model_path = "models/best_model.pkl"
    scaler_path = "models/scaler.pkl"
    metrics_path = "models/metrics.json"
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    feature_names = metrics.get('feature_names', [])
    n_features = len(feature_names)
    
    print(f"Model type: {type(model).__name__}")
    print(f"Number of features: {n_features}")
    print(f"Feature names: {feature_names}")
    
    # Create output directory
    output_dir = "nextjs-app/public/models"
    os.makedirs(output_dir, exist_ok=True)
    
    # For XGBoost, save as JSON format which can be loaded in JS
    # XGBoost JSON format is well-documented and parseable
    booster = model.get_booster()
    
    json_path = os.path.join(output_dir, "parkinsons_model.json")
    booster.save_model(json_path)
    print(f"✅ XGBoost model saved as JSON to {json_path}")
    
    # Also save in ubj format (Universal Binary JSON) which is more compact
    ubj_path = os.path.join(output_dir, "parkinsons_model.ubj")
    booster.save_model(ubj_path)
    print(f"✅ XGBoost model saved as UBJ to {ubj_path}")
    
    # Save scaler parameters as JSON for client-side normalization
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": feature_names
    }
    
    scaler_json_path = os.path.join(output_dir, "scaler.json")
    with open(scaler_json_path, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    print(f"✅ Scaler parameters saved to {scaler_json_path}")
    
    # Save model metadata
    model_meta = {
        "model_type": "xgboost",
        "n_features": n_features,
        "feature_names": feature_names,
        "feature_order": feature_names,  # Important: order matters!
        "accuracy": metrics.get('test_accuracy'),
        "f1_score": metrics.get('test_f1'),
        "roc_auc": metrics.get('test_roc_auc'),
        "best_params": metrics.get('best_params'),
        "converted_at": str(np.datetime64('now'))
    }
    
    meta_path = os.path.join(output_dir, "model_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(model_meta, f, indent=2)
    print(f"✅ Model metadata saved to {meta_path}")
    
    # Verify by running a test prediction
    print("\nVerifying model...")
    test_input = np.random.randn(1, n_features)
    test_scaled = scaler.transform(test_input)
    prediction = model.predict_proba(test_scaled)
    print(f"✅ Test prediction successful!")
    print(f"   Input shape: {test_input.shape}")
    print(f"   Prediction proba: {prediction}")
    
    # Create a simple test case for JS verification
    test_case = {
        "input": test_input.tolist()[0],
        "scaled_input": test_scaled.tolist()[0],
        "expected_proba": prediction.tolist()[0]
    }
    
    test_path = os.path.join(output_dir, "test_case.json")
    with open(test_path, 'w') as f:
        json.dump(test_case, f, indent=2)
    print(f"✅ Test case saved to {test_path}")
    
    return json_path

if __name__ == "__main__":
    convert_model()
