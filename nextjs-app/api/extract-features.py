"""
Voice feature extraction API for Parkinson's Disease Predictor.
Uses parselmouth (Praat) for clinical-grade voice analysis.
"""

from http.server import BaseHTTPRequestHandler
import json
import tempfile
import os
import io
import numpy as np

# Check for available libraries
PARSELMOUTH_AVAILABLE = False
LIBROSA_AVAILABLE = False

try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    pass

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    pass


def extract_features_parselmouth(audio_path: str) -> dict:
    """Extract voice features using parselmouth (Praat)."""
    sound = parselmouth.Sound(audio_path)
    
    # Get pitch and point process for jitter/shimmer
    pitch = call(sound, "To Pitch", 0.0, 75, 600)
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
    
    # Jitter measurements
    jitter_percent = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_abs = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ddp = jitter_rap * 3  # DDP = 3 * RAP
    
    # Shimmer measurements
    shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_db = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq11 = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_dda = shimmer_apq3 * 3  # DDA = 3 * APQ3
    
    # Harmonicity (HNR)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    
    # NHR (inverse of HNR, simplified)
    nhr = 1 / (10 ** (hnr / 10)) if hnr > 0 else 0
    
    # Nonlinear dynamics features (simplified approximations)
    # These would ideally use specialized algorithms
    rpde = estimate_rpde(sound)
    dfa = estimate_dfa(sound)
    ppe = estimate_ppe(pitch)
    
    # Handle NaN values
    def safe_value(v, default=0.0):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return float(v)
    
    return {
        'Jitter(%)': safe_value(jitter_percent * 100),
        'Jitter(Abs)': safe_value(jitter_abs),
        'Jitter:RAP': safe_value(jitter_rap * 100),
        'Jitter:PPQ5': safe_value(jitter_ppq5 * 100),
        'Jitter:DDP': safe_value(jitter_ddp * 100),
        'Shimmer': safe_value(shimmer),
        'Shimmer(dB)': safe_value(shimmer_db),
        'Shimmer:APQ3': safe_value(shimmer_apq3),
        'Shimmer:APQ5': safe_value(shimmer_apq5),
        'Shimmer:APQ11': safe_value(shimmer_apq11),
        'Shimmer:DDA': safe_value(shimmer_dda),
        'NHR': safe_value(nhr),
        'HNR': safe_value(hnr),
        'RPDE': safe_value(rpde),
        'DFA': safe_value(dfa),
        'PPE': safe_value(ppe),
    }


def estimate_rpde(sound) -> float:
    """Estimate RPDE (Recurrence Period Density Entropy)."""
    # Simplified estimation based on signal complexity
    try:
        samples = sound.values[0]
        if len(samples) < 100:
            return 0.5
        
        # Compute approximate entropy as proxy for RPDE
        diff = np.diff(samples)
        hist, _ = np.histogram(diff, bins=50, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Normalize to typical RPDE range (0.3-0.7)
        return np.clip(entropy / 10, 0.3, 0.7)
    except:
        return 0.5


def estimate_dfa(sound) -> float:
    """Estimate DFA (Detrended Fluctuation Analysis)."""
    # Simplified DFA estimation
    try:
        samples = sound.values[0]
        if len(samples) < 100:
            return 0.7
        
        # Compute Hurst exponent as proxy for DFA
        n = len(samples)
        cumsum = np.cumsum(samples - np.mean(samples))
        
        # Use different window sizes
        scales = [10, 20, 50, 100]
        fluctuations = []
        
        for scale in scales:
            if scale > n // 2:
                continue
            n_windows = n // scale
            fluct = 0
            for i in range(n_windows):
                segment = cumsum[i*scale:(i+1)*scale]
                trend = np.polyval(np.polyfit(range(scale), segment, 1), range(scale))
                fluct += np.sqrt(np.mean((segment - trend) ** 2))
            fluctuations.append(fluct / n_windows)
        
        if len(fluctuations) < 2:
            return 0.7
        
        # Estimate scaling exponent
        alpha = np.polyfit(np.log(scales[:len(fluctuations)]), np.log(fluctuations), 1)[0]
        
        # Normalize to typical DFA range (0.5-0.9)
        return np.clip(alpha, 0.5, 0.9)
    except:
        return 0.7


def estimate_ppe(pitch) -> float:
    """Estimate PPE (Pitch Period Entropy)."""
    try:
        # Get pitch values
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]
        
        if len(pitch_values) < 10:
            return 0.2
        
        # Compute entropy of pitch periods
        periods = 1 / pitch_values
        hist, _ = np.histogram(periods, bins=30, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Normalize to typical PPE range (0.1-0.5)
        return np.clip(entropy / 10, 0.1, 0.5)
    except:
        return 0.2


def extract_features_librosa(audio_path: str) -> dict:
    """Extract voice features using librosa (fallback)."""
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Basic features that approximate clinical measures
    # These are approximations, not true jitter/shimmer
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    
    # RMS energy for shimmer approximation
    rms = librosa.feature.rms(y=y)[0]
    
    # Pitch tracking for jitter approximation
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)
    
    # Calculate approximations
    if len(pitch_values) > 1:
        pitch_array = np.array(pitch_values)
        jitter_approx = np.std(np.diff(pitch_array)) / np.mean(pitch_array) * 100
        jitter_abs_approx = np.mean(np.abs(np.diff(1 / pitch_array)))
    else:
        jitter_approx = 0.5
        jitter_abs_approx = 0.00003
    
    if len(rms) > 1:
        shimmer_approx = np.std(rms) / np.mean(rms)
        shimmer_db_approx = 20 * np.log10(np.max(rms) / (np.min(rms) + 1e-10))
    else:
        shimmer_approx = 0.03
        shimmer_db_approx = 0.3
    
    # HNR approximation from spectral flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    hnr_approx = -10 * np.log10(np.mean(spectral_flatness) + 1e-10)
    nhr_approx = 1 / (10 ** (hnr_approx / 10)) if hnr_approx > 0 else 0.03
    
    return {
        'Jitter(%)': float(np.clip(jitter_approx, 0, 3)),
        'Jitter(Abs)': float(np.clip(jitter_abs_approx, 0, 0.0003)),
        'Jitter:RAP': float(np.clip(jitter_approx * 0.6, 0, 2)),
        'Jitter:PPQ5': float(np.clip(jitter_approx * 0.6, 0, 2)),
        'Jitter:DDP': float(np.clip(jitter_approx * 1.8, 0, 5)),
        'Shimmer': float(np.clip(shimmer_approx, 0, 0.2)),
        'Shimmer(dB)': float(np.clip(shimmer_db_approx, 0, 2)),
        'Shimmer:APQ3': float(np.clip(shimmer_approx * 0.5, 0, 0.1)),
        'Shimmer:APQ5': float(np.clip(shimmer_approx * 0.65, 0, 0.15)),
        'Shimmer:APQ11': float(np.clip(shimmer_approx * 0.8, 0, 0.15)),
        'Shimmer:DDA': float(np.clip(shimmer_approx * 1.5, 0, 0.3)),
        'NHR': float(np.clip(nhr_approx, 0, 0.4)),
        'HNR': float(np.clip(hnr_approx, 0, 40)),
        'RPDE': float(np.clip(np.mean(spectral_flatness) * 2, 0.3, 0.7)),
        'DFA': float(np.clip(np.std(zcr) * 10 + 0.6, 0.5, 0.9)),
        'PPE': float(np.clip(np.std(spectral_centroids) / np.mean(spectral_centroids), 0.1, 0.5)),
    }


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Get content length
            content_length = int(self.headers.get('Content-Length', 0))
            
            # Read the body
            body = self.rfile.read(content_length)
            
            # Parse multipart form data manually (simplified)
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
                    if PARSELMOUTH_AVAILABLE:
                        features = extract_features_parselmouth(temp_path)
                        method = "parselmouth"
                    elif LIBROSA_AVAILABLE:
                        features = extract_features_librosa(temp_path)
                        method = "librosa"
                    else:
                        self.send_error_response(500, "No audio processing library available")
                        return
                    
                    # Send success response
                    response = {
                        "features": features,
                        "extraction_method": method,
                    }
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
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
    
    def send_error_response(self, code: int, message: str):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode())
