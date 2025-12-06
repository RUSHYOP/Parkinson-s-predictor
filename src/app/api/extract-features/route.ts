import { NextRequest, NextResponse } from 'next/server';

// This is a placeholder route that will be replaced by the Python API in Vercel
// For local development, you can run the Python server separately

export async function POST(request: NextRequest) {
  try {
    // In production on Vercel, this endpoint is handled by /api/extract-features.py
    // For local development, return a message about running the Python server
    
    const formData = await request.formData();
    const audioFile = formData.get('audio');
    
    if (!audioFile) {
      return NextResponse.json(
        { error: 'No audio file provided' },
        { status: 400 }
      );
    }

    // For development without Python backend, return simulated features
    // based on typical healthy voice characteristics
    const simulatedFeatures = {
      'Jitter(%)': 0.4 + Math.random() * 0.2,
      'Jitter(Abs)': 0.00002 + Math.random() * 0.00002,
      'Jitter:RAP': 0.2 + Math.random() * 0.1,
      'Jitter:PPQ5': 0.2 + Math.random() * 0.1,
      'Jitter:DDP': 0.6 + Math.random() * 0.3,
      'Shimmer': 0.02 + Math.random() * 0.02,
      'Shimmer(dB)': 0.2 + Math.random() * 0.15,
      'Shimmer:APQ3': 0.01 + Math.random() * 0.01,
      'Shimmer:APQ5': 0.015 + Math.random() * 0.01,
      'Shimmer:APQ11': 0.02 + Math.random() * 0.01,
      'Shimmer:DDA': 0.03 + Math.random() * 0.02,
      'NHR': 0.02 + Math.random() * 0.02,
      'HNR': 20 + Math.random() * 5,
      'RPDE': 0.4 + Math.random() * 0.2,
      'DFA': 0.65 + Math.random() * 0.1,
      'PPE': 0.15 + Math.random() * 0.1,
    };

    return NextResponse.json({
      features: simulatedFeatures,
      extraction_method: 'simulated',
      note: 'This is simulated data for development. Deploy to Vercel for real Python-based voice analysis.',
    });

  } catch (error) {
    console.error('Feature extraction error:', error);
    return NextResponse.json(
      { error: 'Failed to process audio file' },
      { status: 500 }
    );
  }
}

// Configure route segment for handling large files
export const dynamic = 'force-dynamic';
export const maxDuration = 10;

