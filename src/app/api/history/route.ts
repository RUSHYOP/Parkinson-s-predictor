// Simple in-memory history store for development
// In production, this would use Vercel Postgres or similar

import { NextRequest, NextResponse } from 'next/server';
import { PredictionHistory } from '@/lib/types';

// In-memory store (resets on each cold start)
// For production, use a database
let historyStore: PredictionHistory[] = [];

export async function GET() {
  // Calculate stats
  const stats = historyStore.length > 0 ? {
    total: historyStore.length,
    avgRisk: historyStore.reduce((acc, p) => acc + p.riskPercentage, 0) / historyStore.length,
    highRiskCount: historyStore.filter(p => p.riskLevel === 'High').length,
  } : null;

  return NextResponse.json({
    predictions: historyStore.sort((a, b) => 
      new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    ),
    stats,
  });
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    const prediction: PredictionHistory = {
      id: `pred_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(body.timestamp || Date.now()),
      inputType: body.inputType,
      features: body.features,
      riskScore: body.riskScore,
      riskLevel: body.riskLevel,
      riskPercentage: body.riskPercentage,
      confidenceLower: body.confidenceLower,
      confidenceUpper: body.confidenceUpper,
      audioFilename: body.audioFilename,
      audioDuration: body.audioDuration,
      notes: body.notes,
    };
    
    historyStore.unshift(prediction);
    
    // Keep only last 100 predictions
    if (historyStore.length > 100) {
      historyStore = historyStore.slice(0, 100);
    }
    
    return NextResponse.json({ success: true, prediction });
  } catch (error) {
    console.error('Failed to save prediction:', error);
    return NextResponse.json(
      { error: 'Failed to save prediction' },
      { status: 500 }
    );
  }
}

export async function DELETE() {
  historyStore = [];
  return NextResponse.json({ success: true });
}
