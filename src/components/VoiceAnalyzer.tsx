'use client';

import { useState, useRef, useCallback } from 'react';
import { Mic, Square, Upload, Loader2, AlertCircle, CheckCircle } from 'lucide-react';

interface VoiceAnalyzerProps {
  onFeaturesExtracted: (features: Record<string, number>) => void;
  isProcessing: boolean;
}

export default function VoiceAnalyzer({ onFeaturesExtracted, isProcessing }: VoiceAnalyzerProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<'idle' | 'recording' | 'processing' | 'done' | 'error'>('idle');
  const [recordingDuration, setRecordingDuration] = useState(0);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const startRecording = useCallback(async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          channelCount: 1,
          sampleRate: 22050,
          echoCancellation: true,
          noiseSuppression: true,
        } 
      });
      
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/mp4'
      });
      
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: mediaRecorder.mimeType });
        setAudioBlob(audioBlob);
        setAudioUrl(URL.createObjectURL(audioBlob));
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start(100);
      setIsRecording(true);
      setStatus('recording');
      setRecordingDuration(0);
      
      // Start duration timer
      timerRef.current = setInterval(() => {
        setRecordingDuration(prev => prev + 1);
      }, 1000);
      
    } catch (err) {
      console.error('Error starting recording:', err);
      setError('Could not access microphone. Please ensure microphone permissions are granted.');
      setStatus('error');
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setStatus('idle');
      
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  }, [isRecording]);

  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    setError(null);
    
    // Check file type
    const validTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/ogg', 'audio/flac', 'audio/webm', 'audio/mp4'];
    if (!validTypes.some(type => file.type.includes(type.split('/')[1]))) {
      setError('Please upload a valid audio file (WAV, MP3, OGG, FLAC, or WebM)');
      setStatus('error');
      return;
    }
    
    // Check file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB');
      setStatus('error');
      return;
    }
    
    setAudioBlob(file);
    setAudioUrl(URL.createObjectURL(file));
    setStatus('idle');
  }, []);

  const analyzeAudio = useCallback(async () => {
    if (!audioBlob) {
      setError('No audio to analyze. Please record or upload audio first.');
      return;
    }
    
    setStatus('processing');
    setError(null);
    
    try {
      // Create form data
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');
      
      // Send to Python API for feature extraction
      const response = await fetch('/api/extract-features', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Failed to extract features: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.features) {
        onFeaturesExtracted(data.features);
        setStatus('done');
      } else {
        throw new Error('No features returned from analysis');
      }
    } catch (err) {
      console.error('Error analyzing audio:', err);
      setError(err instanceof Error ? err.message : 'Failed to analyze audio');
      setStatus('error');
    }
  }, [audioBlob, onFeaturesExtracted]);

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="space-y-6">
      {/* Recording Controls */}
      <div className="flex flex-col items-center space-y-4">
        <div className="flex items-center space-x-4">
          {!isRecording ? (
            <button
              onClick={startRecording}
              disabled={isProcessing}
              className="flex items-center justify-center w-20 h-20 rounded-full bg-red-500 hover:bg-red-600 disabled:bg-gray-400 text-white transition-colors shadow-lg"
              title="Start Recording"
            >
              <Mic className="w-8 h-8" />
            </button>
          ) : (
            <button
              onClick={stopRecording}
              className="flex items-center justify-center w-20 h-20 rounded-full bg-gray-700 hover:bg-gray-800 text-white transition-colors shadow-lg animate-pulse"
              title="Stop Recording"
            >
              <Square className="w-8 h-8" />
            </button>
          )}
          
          <div className="text-center">
            <span className="text-sm text-gray-500">or</span>
          </div>
          
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isRecording || isProcessing}
            className="flex items-center justify-center w-20 h-20 rounded-full bg-blue-500 hover:bg-blue-600 disabled:bg-gray-400 text-white transition-colors shadow-lg"
            title="Upload Audio File"
          >
            <Upload className="w-8 h-8" />
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            onChange={handleFileUpload}
            className="hidden"
          />
        </div>
        
        {isRecording && (
          <div className="flex items-center space-x-2 text-red-500">
            <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />
            <span className="font-mono text-lg">{formatDuration(recordingDuration)}</span>
          </div>
        )}
      </div>

      {/* Audio Player */}
      {audioUrl && (
        <div className="bg-gray-50 rounded-lg p-4">
          <p className="text-sm text-gray-600 mb-2">Recorded/Uploaded Audio:</p>
          <audio controls src={audioUrl} className="w-full" />
        </div>
      )}

      {/* Analyze Button */}
      {audioBlob && !isRecording && (
        <button
          onClick={analyzeAudio}
          disabled={isProcessing || status === 'processing'}
          className="w-full py-3 px-4 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white font-semibold rounded-lg transition-colors flex items-center justify-center space-x-2"
        >
          {status === 'processing' ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Extracting Voice Features...</span>
            </>
          ) : (
            <>
              <CheckCircle className="w-5 h-5" />
              <span>Analyze Voice Recording</span>
            </>
          )}
        </button>
      )}

      {/* Status Messages */}
      {error && (
        <div className="flex items-center space-x-2 text-red-600 bg-red-50 p-3 rounded-lg">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <span className="text-sm">{error}</span>
        </div>
      )}
      
      {status === 'done' && (
        <div className="flex items-center space-x-2 text-green-600 bg-green-50 p-3 rounded-lg">
          <CheckCircle className="w-5 h-5 flex-shrink-0" />
          <span className="text-sm">Voice features extracted successfully!</span>
        </div>
      )}

      {/* Instructions */}
      <div className="text-sm text-gray-500 space-y-1">
        <p>💡 <strong>Tips for best results:</strong></p>
        <ul className="list-disc list-inside space-y-1 ml-4">
          <li>Record for at least 5 seconds (10+ seconds recommended)</li>
          <li>Speak clearly with sustained vowel sounds (e.g., &quot;ahhh&quot;)</li>
          <li>Use a quiet environment to minimize background noise</li>
          <li>Keep consistent distance from the microphone</li>
        </ul>
      </div>
    </div>
  );
}
