"""
Voice feature extraction API for Parkinson's Disease Predictor.
Uses librosa for audio analysis - approximates clinical voice biomarkers.
"""

from http.server import BaseHTTPRequestHandler
import json
import tempfile
import os
import numpy as np
import librosa


def extract_features(audio_path: str) -> dict:
    """Extract voice features using librosa."""
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    
    if len(y) < sr * 0.5:  # Less than 0.5 seconds
        raise ValueError("Audio too short. Please provide at least 0.5 seconds of audio.")
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    
    # RMS energy for shimmer approximation
    rms = librosa.feature.rms(y=y)[0]
    
    # Pitch tracking for jitter approximation
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=600)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)
    
    # Calculate jitter approximations
    if len(pitch_values) > 10:
        pitch_array = np.array(pitch_values)
        periods = 1 / pitch_array
        
        # Jitter (local) - average absolute difference between consecutive periods
        period_diffs = np.abs(np.diff(periods))
        jitter_abs = float(np.mean(period_diffs))
        jitter_percent = float(jitter_abs / np.mean(periods) * 100) if np.mean(periods) > 0 else 0.5
        
        # RAP - Relative Average Perturbation (3-point)
        if len(periods) >= 3:
            rap_vals = []
            for i in range(1, len(periods) - 1):
                avg_neighbor = (periods[i-1] + periods[i] + periods[i+1]) / 3
                rap_vals.append(abs(periods[i] - avg_neighbor))
            jitter_rap = float(np.mean(rap_vals) / np.mean(periods) * 100) if len(rap_vals) > 0 else 0.3
        else:
            jitter_rap = jitter_percent * 0.6
        
        # PPQ5 - Five-point Period Perturbation Quotient
        if len(periods) >= 5:
            ppq_vals = []
            for i in range(2, len(periods) - 2):
                avg_neighbor = np.mean(periods[i-2:i+3])
                ppq_vals.append(abs(periods[i] - avg_neighbor))
            jitter_ppq5 = float(np.mean(ppq_vals) / np.mean(periods) * 100) if len(ppq_vals) > 0 else 0.3
        else:
            jitter_ppq5 = jitter_percent * 0.6
        
        jitter_ddp = jitter_rap * 3  # DDP = 3 * RAP
    else:
        jitter_percent = 0.5
        jitter_abs = 0.00003
        jitter_rap = 0.3
        jitter_ppq5 = 0.3
        jitter_ddp = 0.9
    
    # Calculate shimmer approximations
    if len(rms) > 10:
        rms_nonzero = rms[rms > 0]
        if len(rms_nonzero) > 10:
            amp_diffs = np.abs(np.diff(rms_nonzero))
            shimmer = float(np.mean(amp_diffs) / np.mean(rms_nonzero))
            shimmer_db = float(20 * np.log10(np.max(rms_nonzero) / (np.min(rms_nonzero) + 1e-10)))
            
            # APQ3 - Three-point Amplitude Perturbation Quotient
            if len(rms_nonzero) >= 3:
                apq3_vals = []
                for i in range(1, len(rms_nonzero) - 1):
                    avg_neighbor = (rms_nonzero[i-1] + rms_nonzero[i] + rms_nonzero[i+1]) / 3
                    apq3_vals.append(abs(rms_nonzero[i] - avg_neighbor))
                shimmer_apq3 = float(np.mean(apq3_vals) / np.mean(rms_nonzero))
            else:
                shimmer_apq3 = shimmer * 0.5
            
            # APQ5
            if len(rms_nonzero) >= 5:
                apq5_vals = []
                for i in range(2, len(rms_nonzero) - 2):
                    avg_neighbor = np.mean(rms_nonzero[i-2:i+3])
                    apq5_vals.append(abs(rms_nonzero[i] - avg_neighbor))
                shimmer_apq5 = float(np.mean(apq5_vals) / np.mean(rms_nonzero))
            else:
                shimmer_apq5 = shimmer * 0.65
            
            # APQ11
            if len(rms_nonzero) >= 11:
                apq11_vals = []
                for i in range(5, len(rms_nonzero) - 5):
                    avg_neighbor = np.mean(rms_nonzero[i-5:i+6])
                    apq11_vals.append(abs(rms_nonzero[i] - avg_neighbor))
                shimmer_apq11 = float(np.mean(apq11_vals) / np.mean(rms_nonzero))
            else:
                shimmer_apq11 = shimmer * 0.8
            
            shimmer_dda = shimmer_apq3 * 3
        else:
            shimmer = 0.03
            shimmer_db = 0.3
            shimmer_apq3 = 0.015
            shimmer_apq5 = 0.02
            shimmer_apq11 = 0.025
            shimmer_dda = 0.045
    else:
        shimmer = 0.03
        shimmer_db = 0.3
        shimmer_apq3 = 0.015
        shimmer_apq5 = 0.02
        shimmer_apq11 = 0.025
        shimmer_dda = 0.045
    
    # HNR approximation from spectral flatness
    mean_flatness = float(np.mean(spectral_flatness))
    hnr = float(-10 * np.log10(mean_flatness + 1e-10))
    hnr = np.clip(hnr, 0, 40)
    
    # NHR (inverse relationship with HNR)
    nhr = float(1 / (10 ** (hnr / 10))) if hnr > 0 else 0.03
    nhr = np.clip(nhr, 0, 0.4)
    
    # RPDE approximation - entropy of signal
    hist, _ = np.histogram(y, bins=50, density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    rpde = float(np.clip(entropy / 15, 0.3, 0.7))
    
    # DFA approximation - scaling exponent
    dfa = float(np.clip(np.std(zcr) * 5 + 0.6, 0.5, 0.9))
    
    # PPE approximation - pitch entropy
    if len(pitch_values) > 10:
        pitch_hist, _ = np.histogram(pitch_values, bins=30, density=True)
        pitch_hist = pitch_hist[pitch_hist > 0]
        pitch_entropy = -np.sum(pitch_hist * np.log2(pitch_hist + 1e-10))
        ppe = float(np.clip(pitch_entropy / 12, 0.1, 0.5))
    else:
        ppe = 0.2
    
    # Ensure all values are valid
    def safe_value(v, default=0.0, min_val=None, max_val=None):
        if v is None or np.isnan(v) or np.isinf(v):
            return default
        result = float(v)
        if min_val is not None:
            result = max(result, min_val)
        if max_val is not None:
            result = min(result, max_val)
        return result
    
    return {
        'Jitter(%)': safe_value(jitter_percent, 0.5, 0, 3),
        'Jitter(Abs)': safe_value(jitter_abs, 0.00003, 0, 0.0003),
        'Jitter:RAP': safe_value(jitter_rap, 0.3, 0, 2),
        'Jitter:PPQ5': safe_value(jitter_ppq5, 0.3, 0, 2),
        'Jitter:DDP': safe_value(jitter_ddp, 0.9, 0, 5),
        'Shimmer': safe_value(shimmer, 0.03, 0, 0.2),
        'Shimmer(dB)': safe_value(shimmer_db, 0.3, 0, 2),
        'Shimmer:APQ3': safe_value(shimmer_apq3, 0.015, 0, 0.1),
        'Shimmer:APQ5': safe_value(shimmer_apq5, 0.02, 0, 0.15),
        'Shimmer:APQ11': safe_value(shimmer_apq11, 0.025, 0, 0.15),
        'Shimmer:DDA': safe_value(shimmer_dda, 0.045, 0, 0.3),
        'NHR': safe_value(nhr, 0.03, 0, 0.4),
        'HNR': safe_value(hnr, 22, 0, 40),
        'RPDE': safe_value(rpde, 0.5, 0.3, 0.7),
        'DFA': safe_value(dfa, 0.7, 0.5, 0.9),
        'PPE': safe_value(ppe, 0.2, 0.1, 0.5),
    }


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Get content length
            content_length = int(self.headers.get('Content-Length', 0))
            
            # Read the body
            body = self.rfile.read(content_length)
            
            # Parse multipart form data
            content_type = self.headers.get('Content-Type', '')
            
            if 'multipart/form-data' in content_type:
                # Extract boundary
                boundary = content_type.split('boundary=')[1].encode()
                parts = body.split(b'--' + boundary)
                
                audio_data = None
                for part in parts:
                    if b'name="audio"' in part:
                        # Find the start of file data (after headers)
                        header_end = part.find(b'\r\n\r\n')
                        if header_end != -1:
                            audio_data = part[header_end + 4:]
                            # Remove trailing boundary markers
                            if audio_data.endswith(b'\r\n'):
                                audio_data = audio_data[:-2]
                            if audio_data.endswith(b'--'):
                                audio_data = audio_data[:-2]
                            if audio_data.endswith(b'\r\n'):
                                audio_data = audio_data[:-2]
                
                if audio_data is None:
                    self.send_error_response(400, "No audio file found in request")
                    return
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
                    f.write(audio_data)
                    temp_path = f.name
                
                try:
                    # Extract features
                    features = extract_features(temp_path)
                    
                    # Send success response
                    response = {
                        "features": features,
                        "extraction_method": "librosa",
                    }
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())
                    
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            else:
                self.send_error_response(400, "Expected multipart/form-data")
                
        except Exception as e:
            self.send_error_response(500, str(e))
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def send_error_response(self, code: int, message: str):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode())
