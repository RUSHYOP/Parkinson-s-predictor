import { NextRequest, NextResponse } from 'next/server';

// This would connect to the same store as the main history route
// For now, we'll return a simple response

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  
  // In a real implementation, this would delete from database
  console.log('Deleting prediction:', id);
  
  return NextResponse.json({ success: true, deletedId: id });
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  
  // In a real implementation, this would fetch from database
  return NextResponse.json({ 
    error: 'Prediction not found',
    id 
  }, { status: 404 });
}
