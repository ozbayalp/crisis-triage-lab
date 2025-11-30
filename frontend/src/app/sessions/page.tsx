'use client';

import { useState, useEffect, useCallback } from 'react';
import { TriageAnalytics, TriageEvent } from '@/lib/types';

/**
 * Sessions Page
 * 
 * Displays session history and management for the CrisisTriage AI system.
 * Shows recent triage events grouped by session with summary statistics.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface SessionSummary {
  sessionId: string;
  eventCount: number;
  firstEvent: Date;
  lastEvent: Date;
  riskLevels: string[];
  avgUrgency: number;
  maxUrgency: number;
}

export default function SessionsPage() {
  const [events, setEvents] = useState<TriageEvent[]>([]);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [selectedSession, setSelectedSession] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch recent events
  const fetchEvents = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/api/analytics/recent?limit=500`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch events: ${response.statusText}`);
      }
      
      const data: TriageEvent[] = await response.json();
      setEvents(data);
      
      // Group events by session
      const sessionMap = new Map<string, TriageEvent[]>();
      for (const event of data) {
        const existing = sessionMap.get(event.session_id) || [];
        existing.push(event);
        sessionMap.set(event.session_id, existing);
      }
      
      // Build session summaries
      const summaries: SessionSummary[] = [];
      sessionMap.forEach((sessionEvents, sessionId) => {
        const urgencies = sessionEvents.map(e => e.urgency_score);
        summaries.push({
          sessionId,
          eventCount: sessionEvents.length,
          firstEvent: new Date(Math.min(...sessionEvents.map(e => new Date(e.timestamp).getTime()))),
          lastEvent: new Date(Math.max(...sessionEvents.map(e => new Date(e.timestamp).getTime()))),
          riskLevels: Array.from(new Set(sessionEvents.map(e => e.risk_level))),
          avgUrgency: urgencies.reduce((a, b) => a + b, 0) / urgencies.length,
          maxUrgency: Math.max(...urgencies),
        });
      });
      
      // Sort by most recent
      summaries.sort((a, b) => b.lastEvent.getTime() - a.lastEvent.getTime());
      setSessions(summaries);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch sessions');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchEvents();
  }, [fetchEvents]);

  // Get events for selected session
  const selectedSessionEvents = selectedSession
    ? events.filter(e => e.session_id === selectedSession)
    : [];

  // Risk level colors
  const riskColors: Record<string, string> = {
    low: 'bg-green-500',
    medium: 'bg-yellow-500',
    high: 'bg-orange-500',
    imminent: 'bg-red-500',
  };

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 style={{ fontSize: '32px', fontWeight: 600, letterSpacing: '-0.01em', marginBottom: '8px' }}>Session History</h1>
        <p style={{ color: 'var(--text-secondary)' }}>
          View and analyze past triage sessions. Data is ephemeral and stored in memory only.
        </p>
      </div>

      {/* Error Display */}
      {error && (
        <div 
          className="mb-6 p-4 rounded-lg"
          style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--danger)', color: 'var(--danger)' }}
        >
          {error}
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="flex items-center justify-center py-12">
          <div 
            className="animate-spin rounded-full h-8 w-8 border-b-2"
            style={{ borderColor: 'var(--accent)' }}
          />
          <span className="ml-3" style={{ color: 'var(--text-tertiary)' }}>Loading sessions...</span>
        </div>
      )}

      {/* Empty State */}
      {!loading && sessions.length === 0 && (
        <div className="text-center py-12">
          <h2 className="text-xl font-semibold mb-2">No Sessions Yet</h2>
          <p style={{ color: 'var(--text-tertiary)', marginBottom: '24px' }}>
            Start a triage session from the Live Triage page to see history here.
          </p>
          <a
            href="/"
            className="inline-flex items-center px-4 py-2 rounded-lg transition-colors"
            style={{ background: 'var(--accent)', color: 'var(--bg-primary)' }}
          >
            Go to Live Triage
          </a>
        </div>
      )}

      {/* Sessions Grid */}
      {!loading && sessions.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Sessions List */}
          <div 
            className="lg:col-span-1 rounded-lg p-6"
            style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-primary)' }}
          >
            <div className="flex items-center justify-between mb-4">
              <h2 style={{ fontSize: '16px', fontWeight: 600 }}>Sessions ({sessions.length})</h2>
              <div className="flex items-center gap-2">
                <button
                  onClick={fetchEvents}
                  className="px-3 py-1 text-sm rounded-lg transition-colors"
                  style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)', border: '1px solid var(--border-primary)' }}
                >
                  Refresh
                </button>
                <button
                  onClick={() => {
                    if (confirm('Clear all session data?')) {
                      setSessions([]);
                      setEvents([]);
                      setSelectedSession(null);
                    }
                  }}
                  className="px-3 py-1 text-sm rounded-lg transition-colors"
                  style={{ background: 'var(--bg-tertiary)', color: 'var(--danger)', border: '1px solid var(--border-primary)' }}
                >
                  Clear All
                </button>
              </div>
            </div>
            
            <div className="space-y-3 max-h-[500px] overflow-y-auto">
              {sessions.map((session) => (
                <button
                  key={session.sessionId}
                  onClick={() => setSelectedSession(session.sessionId)}
                  className="w-full text-left p-4 rounded-lg transition-colors"
                  style={{
                    background: selectedSession === session.sessionId ? 'var(--accent-subtle)' : 'var(--bg-tertiary)',
                    border: `1px solid ${selectedSession === session.sessionId ? 'var(--accent)' : 'var(--border-primary)'}`
                  }}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-mono text-sm">
                      {session.sessionId}
                    </span>
                    <span className="text-xs" style={{ color: 'var(--text-tertiary)' }}>
                      {session.eventCount} events
                    </span>
                  </div>
                  
                  {/* Risk level badges */}
                  <div className="flex gap-1 mb-2">
                    {session.riskLevels.map((risk) => (
                      <span
                        key={risk}
                        className={`px-2 py-0.5 rounded text-xs text-white ${riskColors[risk] || 'bg-gray-500'}`}
                      >
                        {risk}
                      </span>
                    ))}
                  </div>
                  
                  <div className="flex items-center justify-between text-xs" style={{ color: 'var(--text-tertiary)' }}>
                    <span>Max urgency: {session.maxUrgency}</span>
                    <span>{session.lastEvent.toLocaleTimeString()}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Session Details */}
          <div className="lg:col-span-2">
            {selectedSession ? (
              <div 
                className="rounded-lg p-6"
                style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-primary)' }}
              >
                <div className="flex items-center justify-between mb-6">
                  <h2 style={{ fontSize: '16px', fontWeight: 600 }}>
                    Session: <span className="font-mono text-blue-400">{selectedSession}</span>
                  </h2>
                  <button
                    onClick={() => setSelectedSession(null)}
                    className="text-sm text-gray-400 hover:text-white transition-colors"
                  >
                    Close
                  </button>
                </div>

                {/* Session Stats */}
                {sessions.find(s => s.sessionId === selectedSession) && (
                  <div className="grid grid-cols-3 gap-4 mb-6">
                    <div className="p-4 rounded-lg" style={{ background: 'var(--bg-tertiary)' }}>
                      <div className="text-2xl font-bold">
                        {selectedSessionEvents.length}
                      </div>
                      <div className="text-sm" style={{ color: 'var(--text-tertiary)' }}>Total Events</div>
                    </div>
                    <div className="p-4 rounded-lg" style={{ background: 'var(--bg-tertiary)' }}>
                      <div className="text-2xl font-bold">
                        {Math.round(sessions.find(s => s.sessionId === selectedSession)!.avgUrgency)}
                      </div>
                      <div className="text-sm" style={{ color: 'var(--text-tertiary)' }}>Avg Urgency</div>
                    </div>
                    <div className="p-4 rounded-lg" style={{ background: 'var(--bg-tertiary)' }}>
                      <div className="text-2xl font-bold">
                        {sessions.find(s => s.sessionId === selectedSession)!.maxUrgency}
                      </div>
                      <div className="text-sm" style={{ color: 'var(--text-tertiary)' }}>Max Urgency</div>
                    </div>
                  </div>
                )}

                {/* Events Timeline */}
                <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-tertiary)' }}>Event Timeline</h3>
                <div className="space-y-3 max-h-[400px] overflow-y-auto">
                  {selectedSessionEvents
                    .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
                    .map((event, idx) => (
                      <div
                        key={idx}
                        className="p-4 rounded-lg"
                        style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--border-primary)' }}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <span
                              className={`px-2 py-0.5 rounded text-xs text-white ${riskColors[event.risk_level] || 'bg-gray-500'}`}
                            >
                              {event.risk_level}
                            </span>
                            <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                              {event.emotional_state}
                            </span>
                          </div>
                          <span className="text-xs" style={{ color: 'var(--text-tertiary)' }}>
                            {new Date(event.timestamp).toLocaleString()}
                          </span>
                        </div>
                        
                        <div className="grid grid-cols-3 gap-4 text-sm">
                          <div>
                            <span style={{ color: 'var(--text-tertiary)' }}>Urgency:</span>{' '}
                            <span className="font-medium">{event.urgency_score}</span>
                          </div>
                          <div>
                            <span style={{ color: 'var(--text-tertiary)' }}>Confidence:</span>{' '}
                            <span className="font-medium">{(event.confidence * 100).toFixed(0)}%</span>
                          </div>
                          <div>
                            <span style={{ color: 'var(--text-tertiary)' }}>Modality:</span>{' '}
                            <span className="font-medium">{event.modality}</span>
                          </div>
                        </div>
                        
                        {event.text_snippet && (
                          <div 
                            className="mt-2 p-2 rounded text-sm italic"
                            style={{ background: 'var(--bg-secondary)', color: 'var(--text-secondary)' }}
                          >
                            "{event.text_snippet}"
                          </div>
                        )}
                      </div>
                    ))}
                </div>
              </div>
            ) : (
              <div 
                className="rounded-lg p-12 text-center"
                style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-primary)' }}
              >
                <p style={{ color: 'var(--text-tertiary)' }}>Select a session to view details</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Footer Note */}
      <div 
        className="mt-8 p-4 rounded-lg"
        style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-primary)' }}
      >
        <p className="text-sm" style={{ color: 'var(--text-tertiary)' }}>
          <strong style={{ color: 'var(--text-secondary)' }}>Note:</strong> Session data is stored in memory only and will be 
          cleared when the backend restarts. This is intentional for privacy. Session IDs shown here are 
          truncated (first 8 characters) for display purposes.
        </p>
      </div>
    </div>
  );
}
