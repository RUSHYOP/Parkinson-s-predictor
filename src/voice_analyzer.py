"""
Voice Analyzer Module for Parkinson's Disease Predictor

Extracts clinical biomarkers from audio files for Parkinson's detection.
Uses parselmouth (Praat wrapper) as primary extractor with librosa fallback.
Supports chunked processing for longer audio files.
"""

import os
import numpy as np
import warnings
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings('ignore')

# Check for available audio libraries
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
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    pass

try:
    import soundfile as sf
except ImportError:
    sf = None


@dataclass
class VoiceFeatures:
    """Container for extracted voice features."""
    # Jitter features (frequency perturbation)
    jitter_percent: float = 0.0
    jitter_abs: float = 0.0
    jitter_rap: float = 0.0
    jitter_ppq5: float = 0.0
    jitter_ddp: float = 0.0
    
    # Shimmer features (amplitude perturbation)
    shimmer: float = 0.0
    shimmer_db: float = 0.0
    shimmer_apq3: float = 0.0
    shimmer_apq5: float = 0.0
    shimmer_apq11: float = 0.0
    shimmer_dda: float = 0.0
    
    # Noise features
    nhr: float = 0.0  # Noise-to-Harmonics Ratio
    hnr: float = 0.0  # Harmonics-to-Noise Ratio
    
    # Nonlinear dynamics features
    rpde: float = 0.0  # Recurrence Period Density Entropy
    dfa: float = 0.0   # Detrended Fluctuation Analysis
    ppe: float = 0.0   # Pitch Period Entropy
    
    # Additional librosa features (fallback)
    mfcc_mean: Optional[np.ndarray] = None
    spectral_centroid: float = 0.0
    spectral_rolloff: float = 0.0
    zero_crossing_rate: float = 0.0
    
    # Metadata
    extraction_method: str = "unknown"
    duration_seconds: float = 0.0
    sample_rate: int = 0
    num_chunks: int = 1
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary matching dataset feature names."""
        return {
            'Jitter(%)': self.jitter_percent,
            'Jitter(Abs)': self.jitter_abs,
            'Jitter:RAP': self.jitter_rap,
            'Jitter:PPQ5': self.jitter_ppq5,
            'Jitter:DDP': self.jitter_ddp,
            'Shimmer': self.shimmer,
            'Shimmer(dB)': self.shimmer_db,
            'Shimmer:APQ3': self.shimmer_apq3,
            'Shimmer:APQ5': self.shimmer_apq5,
            'Shimmer:APQ11': self.shimmer_apq11,
            'Shimmer:DDA': self.shimmer_dda,
            'NHR': self.nhr,
            'HNR': self.hnr,
            'RPDE': self.rpde,
            'DFA': self.dfa,
            'PPE': self.ppe
        }
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model prediction."""
        return np.array(list(self.to_dict().values()))


class VoiceAnalyzer:
    """
    Extracts Parkinson's-related biomarkers from voice recordings.
    
    Primary extraction uses parselmouth (Python-Praat) for clinical-grade
    jitter, shimmer, and harmonic analysis. Falls back to librosa-based
    approximations if parselmouth is unavailable.
    
    Supports chunked processing for long audio files (10-second chunks).
    """
    
    CHUNK_DURATION = 10.0  # seconds
    MIN_DURATION = 0.5     # minimum audio duration in seconds
    TARGET_SR = 22050      # target sample rate
    
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac'}
    
    def __init__(self, use_fallback: bool = True):
        """
        Initialize VoiceAnalyzer.
        
        Args:
            use_fallback: If True, use librosa fallback when parselmouth unavailable
        """
        self.use_fallback = use_fallback
        self.extraction_method = self._detect_extraction_method()
        
        print(f"🎤 VoiceAnalyzer initialized")
        print(f"   - Parselmouth available: {PARSELMOUTH_AVAILABLE}")
        print(f"   - Librosa available: {LIBROSA_AVAILABLE}")
        print(f"   - Extraction method: {self.extraction_method}")
    
    def _detect_extraction_method(self) -> str:
        """Detect which extraction method to use."""
        if PARSELMOUTH_AVAILABLE:
            return "parselmouth"
        elif LIBROSA_AVAILABLE and self.use_fallback:
            return "librosa_fallback"
        else:
            return "none"
    
    def is_available(self) -> bool:
        """Check if voice analysis is available."""
        return self.extraction_method != "none"
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file to numpy array.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        path = Path(audio_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {path.suffix}")
        
        # Try librosa first (handles most formats)
        if LIBROSA_AVAILABLE:
            try:
                audio, sr = librosa.load(audio_path, sr=self.TARGET_SR, mono=True)
                return audio, sr
            except Exception as e:
                print(f"⚠️ Librosa failed to load: {e}")
        
        # Fallback to soundfile
        if sf is not None:
            try:
                audio, sr = sf.read(audio_path)
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)  # Convert to mono
                return audio, sr
            except Exception as e:
                print(f"⚠️ Soundfile failed to load: {e}")
        
        raise RuntimeError(f"Could not load audio file: {audio_path}")
    
    def _chunk_audio(self, audio: np.ndarray, sr: int) -> List[np.ndarray]:
        """
        Split audio into chunks for processing.
        
        Args:
            audio: Audio data as numpy array
            sr: Sample rate
            
        Returns:
            List of audio chunks
        """
        chunk_samples = int(self.CHUNK_DURATION * sr)
        total_samples = len(audio)
        
        if total_samples <= chunk_samples:
            return [audio]
        
        chunks = []
        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            chunk = audio[start:end]
            # Only include chunks longer than minimum duration
            if len(chunk) >= int(self.MIN_DURATION * sr):
                chunks.append(chunk)
        
        return chunks
    
    def _extract_parselmouth_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract features using parselmouth (Praat).
        
        Args:
            audio: Audio data
            sr: Sample rate
            
        Returns:
            Dictionary of extracted features
        """
        # Create Praat Sound object
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        
        # Extract pitch
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        
        # Extract point process for jitter/shimmer
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        
        features = {}
        
        # Jitter measurements
        try:
            features['jitter_percent'] = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
            features['jitter_abs'] = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
            features['jitter_rap'] = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            features['jitter_ppq5'] = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            features['jitter_ddp'] = features['jitter_rap'] * 3  # DDP = 3 * RAP
        except:
            features.update({
                'jitter_percent': 0.0, 'jitter_abs': 0.0, 'jitter_rap': 0.0,
                'jitter_ppq5': 0.0, 'jitter_ddp': 0.0
            })
        
        # Shimmer measurements
        try:
            features['shimmer'] = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features['shimmer_db'] = call([sound, point_process], "Get shimmer (local, dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features['shimmer_apq3'] = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features['shimmer_apq5'] = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features['shimmer_apq11'] = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features['shimmer_dda'] = features['shimmer_apq3'] * 3  # DDA = 3 * APQ3
        except:
            features.update({
                'shimmer': 0.0, 'shimmer_db': 0.0, 'shimmer_apq3': 0.0,
                'shimmer_apq5': 0.0, 'shimmer_apq11': 0.0, 'shimmer_dda': 0.0
            })
        
        # Harmonics-to-Noise Ratio
        try:
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            features['hnr'] = call(harmonicity, "Get mean", 0, 0)
            features['nhr'] = 1 / (10 ** (features['hnr'] / 10)) if features['hnr'] > 0 else 0.5
        except:
            features['hnr'] = 20.0
            features['nhr'] = 0.01
        
        # Nonlinear dynamics (approximations)
        # These require specialized libraries, using approximations
        features['rpde'] = self._estimate_rpde(audio, sr)
        features['dfa'] = self._estimate_dfa(audio, sr)
        features['ppe'] = self._estimate_ppe(audio, sr)
        
        return features
    
    def _extract_librosa_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract features using librosa (fallback mode).
        
        This provides approximations of clinical features using
        spectral analysis when parselmouth is unavailable.
        
        Args:
            audio: Audio data
            sr: Sample rate
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Approximate jitter using zero-crossing rate variability
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_std = np.std(zcr)
        features['jitter_percent'] = min(zcr_std * 100, 5.0)  # Cap at 5%
        features['jitter_abs'] = zcr_std * 0.0001
        features['jitter_rap'] = features['jitter_percent'] / 300
        features['jitter_ppq5'] = features['jitter_percent'] / 250
        features['jitter_ddp'] = features['jitter_rap'] * 3
        
        # Approximate shimmer using RMS energy variability
        rms = librosa.feature.rms(y=audio)[0]
        rms_std = np.std(rms) / (np.mean(rms) + 1e-10)
        features['shimmer'] = min(rms_std, 0.5)
        features['shimmer_db'] = 20 * np.log10(1 + features['shimmer'])
        features['shimmer_apq3'] = features['shimmer'] / 3
        features['shimmer_apq5'] = features['shimmer'] / 4
        features['shimmer_apq11'] = features['shimmer'] / 5
        features['shimmer_dda'] = features['shimmer_apq3'] * 3
        
        # Spectral features for HNR approximation
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=audio))
        
        # HNR approximation from spectral flatness (lower flatness = more harmonic)
        features['hnr'] = max(0, 30 * (1 - spectral_flatness))
        features['nhr'] = spectral_flatness
        
        # Nonlinear dynamics approximations
        features['rpde'] = self._estimate_rpde(audio, sr)
        features['dfa'] = self._estimate_dfa(audio, sr)
        features['ppe'] = self._estimate_ppe(audio, sr)
        
        # Store additional librosa features
        features['spectral_centroid'] = spectral_centroid
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        features['zero_crossing_rate'] = np.mean(zcr)
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        
        return features
    
    def _estimate_rpde(self, audio: np.ndarray, sr: int) -> float:
        """
        Estimate Recurrence Period Density Entropy.
        
        Uses simplified entropy calculation as approximation.
        """
        try:
            # Downsample for efficiency
            if len(audio) > sr:
                audio = audio[::int(len(audio) / sr)]
            
            # Calculate histogram entropy
            hist, _ = np.histogram(audio, bins=50, density=True)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log2(hist + 1e-10)) / np.log2(len(hist))
            
            return min(max(entropy, 0.2), 0.8)  # Typical RPDE range
        except:
            return 0.5
    
    def _estimate_dfa(self, audio: np.ndarray, sr: int) -> float:
        """
        Estimate Detrended Fluctuation Analysis scaling exponent.
        
        Uses simplified variance scaling as approximation.
        """
        try:
            # Downsample for efficiency
            if len(audio) > sr * 2:
                audio = audio[::int(len(audio) / (sr * 2))]
            
            # Calculate cumulative sum
            cumsum = np.cumsum(audio - np.mean(audio))
            
            # Calculate scaling at different window sizes
            scales = [10, 20, 50, 100, 200]
            fluctuations = []
            
            for scale in scales:
                if scale >= len(cumsum):
                    break
                n_segments = len(cumsum) // scale
                if n_segments < 2:
                    break
                    
                fluct = []
                for i in range(n_segments):
                    segment = cumsum[i*scale:(i+1)*scale]
                    trend = np.polyval(np.polyfit(range(scale), segment, 1), range(scale))
                    fluct.append(np.sqrt(np.mean((segment - trend) ** 2)))
                
                fluctuations.append(np.mean(fluct))
            
            if len(fluctuations) >= 2:
                # Estimate scaling exponent
                log_scales = np.log(scales[:len(fluctuations)])
                log_fluct = np.log(np.array(fluctuations) + 1e-10)
                alpha = np.polyfit(log_scales, log_fluct, 1)[0]
                return min(max(alpha, 0.5), 1.0)
            
            return 0.7
        except:
            return 0.7
    
    def _estimate_ppe(self, audio: np.ndarray, sr: int) -> float:
        """
        Estimate Pitch Period Entropy.
        
        Uses fundamental frequency variability as approximation.
        """
        try:
            if LIBROSA_AVAILABLE:
                # Extract pitch using librosa
                pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
                pitch_values = []
                
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                
                if len(pitch_values) > 10:
                    # Calculate entropy of pitch distribution
                    hist, _ = np.histogram(pitch_values, bins=20, density=True)
                    hist = hist[hist > 0]
                    entropy = -np.sum(hist * np.log2(hist + 1e-10)) / np.log2(len(hist))
                    return min(max(entropy, 0.0), 0.5)
            
            return 0.2
        except:
            return 0.2
    
    def _aggregate_chunk_features(self, chunk_features: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate features from multiple chunks.
        
        Args:
            chunk_features: List of feature dictionaries from each chunk
            
        Returns:
            Aggregated features (mean values)
        """
        if len(chunk_features) == 1:
            return chunk_features[0]
        
        aggregated = {}
        keys = chunk_features[0].keys()
        
        for key in keys:
            values = [cf.get(key, 0) for cf in chunk_features if isinstance(cf.get(key), (int, float))]
            if values:
                aggregated[key] = np.mean(values)
        
        return aggregated
    
    def extract_features(self, audio_path: str) -> VoiceFeatures:
        """
        Extract voice features from an audio file.
        
        Processes audio in chunks if longer than CHUNK_DURATION.
        Uses parselmouth if available, otherwise falls back to librosa.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            VoiceFeatures object with extracted features
        """
        if not self.is_available():
            raise RuntimeError("No audio processing library available. Install parselmouth or librosa.")
        
        print(f"\n🔊 Analyzing audio: {Path(audio_path).name}")
        
        # Load audio
        audio, sr = self._load_audio(audio_path)
        duration = len(audio) / sr
        
        print(f"   - Duration: {duration:.2f} seconds")
        print(f"   - Sample rate: {sr} Hz")
        
        if duration < self.MIN_DURATION:
            raise ValueError(f"Audio too short ({duration:.2f}s). Minimum: {self.MIN_DURATION}s")
        
        # Chunk audio if needed
        chunks = self._chunk_audio(audio, sr)
        print(f"   - Processing {len(chunks)} chunk(s)")
        
        # Extract features from each chunk
        chunk_features = []
        for i, chunk in enumerate(chunks):
            if self.extraction_method == "parselmouth":
                features = self._extract_parselmouth_features(chunk, sr)
            else:
                features = self._extract_librosa_features(chunk, sr)
            chunk_features.append(features)
        
        # Aggregate chunk features
        aggregated = self._aggregate_chunk_features(chunk_features)
        
        # Create VoiceFeatures object
        voice_features = VoiceFeatures(
            jitter_percent=aggregated.get('jitter_percent', 0),
            jitter_abs=aggregated.get('jitter_abs', 0),
            jitter_rap=aggregated.get('jitter_rap', 0),
            jitter_ppq5=aggregated.get('jitter_ppq5', 0),
            jitter_ddp=aggregated.get('jitter_ddp', 0),
            shimmer=aggregated.get('shimmer', 0),
            shimmer_db=aggregated.get('shimmer_db', 0),
            shimmer_apq3=aggregated.get('shimmer_apq3', 0),
            shimmer_apq5=aggregated.get('shimmer_apq5', 0),
            shimmer_apq11=aggregated.get('shimmer_apq11', 0),
            shimmer_dda=aggregated.get('shimmer_dda', 0),
            nhr=aggregated.get('nhr', 0),
            hnr=aggregated.get('hnr', 0),
            rpde=aggregated.get('rpde', 0),
            dfa=aggregated.get('dfa', 0),
            ppe=aggregated.get('ppe', 0),
            extraction_method=self.extraction_method,
            duration_seconds=duration,
            sample_rate=sr,
            num_chunks=len(chunks)
        )
        
        # Add additional librosa features if available
        if 'mfcc_mean' in aggregated:
            voice_features.mfcc_mean = aggregated['mfcc_mean']
        if 'spectral_centroid' in aggregated:
            voice_features.spectral_centroid = aggregated['spectral_centroid']
        if 'spectral_rolloff' in aggregated:
            voice_features.spectral_rolloff = aggregated['spectral_rolloff']
        if 'zero_crossing_rate' in aggregated:
            voice_features.zero_crossing_rate = aggregated['zero_crossing_rate']
        
        print(f"   ✅ Feature extraction complete using {self.extraction_method}")
        
        return voice_features
    
    def extract_features_from_array(self, audio: np.ndarray, sr: int) -> VoiceFeatures:
        """
        Extract features from audio array directly.
        
        Args:
            audio: Audio data as numpy array
            sr: Sample rate
            
        Returns:
            VoiceFeatures object
        """
        if not self.is_available():
            raise RuntimeError("No audio processing library available.")
        
        duration = len(audio) / sr
        chunks = self._chunk_audio(audio, sr)
        
        chunk_features = []
        for chunk in chunks:
            if self.extraction_method == "parselmouth":
                features = self._extract_parselmouth_features(chunk, sr)
            else:
                features = self._extract_librosa_features(chunk, sr)
            chunk_features.append(features)
        
        aggregated = self._aggregate_chunk_features(chunk_features)
        
        return VoiceFeatures(
            jitter_percent=aggregated.get('jitter_percent', 0),
            jitter_abs=aggregated.get('jitter_abs', 0),
            jitter_rap=aggregated.get('jitter_rap', 0),
            jitter_ppq5=aggregated.get('jitter_ppq5', 0),
            jitter_ddp=aggregated.get('jitter_ddp', 0),
            shimmer=aggregated.get('shimmer', 0),
            shimmer_db=aggregated.get('shimmer_db', 0),
            shimmer_apq3=aggregated.get('shimmer_apq3', 0),
            shimmer_apq5=aggregated.get('shimmer_apq5', 0),
            shimmer_apq11=aggregated.get('shimmer_apq11', 0),
            shimmer_dda=aggregated.get('shimmer_dda', 0),
            nhr=aggregated.get('nhr', 0),
            hnr=aggregated.get('hnr', 0),
            rpde=aggregated.get('rpde', 0),
            dfa=aggregated.get('dfa', 0),
            ppe=aggregated.get('ppe', 0),
            extraction_method=self.extraction_method,
            duration_seconds=duration,
            sample_rate=sr,
            num_chunks=len(chunks)
        )
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all voice features."""
        return {
            'Jitter(%)': 'Frequency variation between consecutive pitch periods (%)',
            'Jitter(Abs)': 'Absolute jitter in seconds',
            'Jitter:RAP': 'Relative Average Perturbation',
            'Jitter:PPQ5': 'Five-point Period Perturbation Quotient',
            'Jitter:DDP': 'Difference of Differences of Periods',
            'Shimmer': 'Amplitude variation between consecutive periods',
            'Shimmer(dB)': 'Shimmer in decibels',
            'Shimmer:APQ3': 'Three-point Amplitude Perturbation Quotient',
            'Shimmer:APQ5': 'Five-point Amplitude Perturbation Quotient',
            'Shimmer:APQ11': 'Eleven-point Amplitude Perturbation Quotient',
            'Shimmer:DDA': 'Difference of Differences of Amplitudes',
            'NHR': 'Noise-to-Harmonics Ratio (voice breathiness)',
            'HNR': 'Harmonics-to-Noise Ratio (voice clarity)',
            'RPDE': 'Recurrence Period Density Entropy (voice stability)',
            'DFA': 'Detrended Fluctuation Analysis (signal self-similarity)',
            'PPE': 'Pitch Period Entropy (pitch regularity)'
        }


if __name__ == "__main__":
    # Test the voice analyzer
    print("Testing VoiceAnalyzer...")
    analyzer = VoiceAnalyzer()
    print(f"\nAvailable: {analyzer.is_available()}")
    print(f"Method: {analyzer.extraction_method}")
    
    print("\nFeature descriptions:")
    for name, desc in analyzer.get_feature_descriptions().items():
        print(f"  {name}: {desc}")
