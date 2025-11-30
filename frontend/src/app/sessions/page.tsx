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
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Session History</h1>
        <p className="text-gray-400">
          View and analyze past triage sessions. Data is ephemeral and stored in memory only.
        </p>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 rounded-lg bg-red-900/30 border border-red-700 text-red-400">
          {error}
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
          <span className="ml-3 text-gray-400">Loading sessions...</span>
        </div>
      )}

      {/* Empty State */}
      {!loading && sessions.length === 0 && (
        <div className="text-center py-12">
          <div className="text-6xl mb-4">📋</div>
          <h2 className="text-xl font-semibold text-white mb-2">No Sessions Yet</h2>
          <p className="text-gray-400 mb-6">
            Start a triage session from the Live Triage page to see history here.
          </p>
          <a
            href="/"
            className="inline-flex items-center px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white transition-colors"
          >
            Go to Live Triage
          </a>
        </div>
      )}

      {/* Sessions Grid */}
      {!loading && sessions.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Sessions List */}
          <div className="lg:col-span-1 space-y-4">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-white">Sessions ({sessions.length})</h2>
              <button
                onClick={fetchEvents}
                className="text-sm text-blue-400 hover:text-blue-300 transition-colors"
              >
                Refresh
              </button>
            </div>
            
            <div className="space-y-3 max-h-[600px] overflow-y-auto pr-2">
              {sessions.map((session) => (
                <button
                  key={session.sessionId}
                  onClick={() => setSelectedSession(session.sessionId)}
                  className={`w-full text-left p-4 rounded-lg border transition-colors ${
                    selectedSession === session.sessionId
                      ? 'bg-blue-900/30 border-blue-600'
                      : 'bg-gray-800/30 border-gray-700 hover:border-gray-600'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-mono text-sm text-white">
                      {session.sessionId}
                    </span>
                    <span className="text-xs text-gray-500">
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
                  
                  <div className="flex items-center justify-between text-xs text-gray-400">
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
              <div className="rounded-lg border border-gray-700 bg-gray-800/30 p-6">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-lg font-semibold text-white">
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
                    <div className="p-4 rounded-lg bg-gray-900/50">
                      <div className="text-2xl font-bold text-white">
                        {selectedSessionEvents.length}
                      </div>
                      <div className="text-sm text-gray-400">Total Events</div>
                    </div>
                    <div className="p-4 rounded-lg bg-gray-900/50">
                      <div className="text-2xl font-bold text-white">
                        {Math.round(sessions.find(s => s.sessionId === selectedSession)!.avgUrgency)}
                      </div>
                      <div className="text-sm text-gray-400">Avg Urgency</div>
                    </div>
                    <div className="p-4 rounded-lg bg-gray-900/50">
                      <div className="text-2xl font-bold text-white">
                        {sessions.find(s => s.sessionId === selectedSession)!.maxUrgency}
                      </div>
                      <div className="text-sm text-gray-400">Max Urgency</div>
                    </div>
                  </div>
                )}

                {/* Events Timeline */}
                <h3 className="text-sm font-medium text-gray-400 mb-4">Event Timeline</h3>
                <div className="space-y-3 max-h-[400px] overflow-y-auto">
                  {selectedSessionEvents
                    .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
                    .map((event, idx) => (
                      <div
                        key={idx}
                        className="p-4 rounded-lg bg-gray-900/50 border border-gray-700"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <span
                              className={`px-2 py-0.5 rounded text-xs text-white ${riskColors[event.risk_level] || 'bg-gray-500'}`}
                            >
                              {event.risk_level}
                            </span>
                            <span className="text-sm text-gray-300">
                              {event.emotional_state}
                            </span>
                          </div>
                          <span className="text-xs text-gray-500">
                            {new Date(event.timestamp).toLocaleString()}
                          </span>
                        </div>
                        
                        <div className="grid grid-cols-3 gap-4 text-sm">
                          <div>
                            <span className="text-gray-500">Urgency:</span>{' '}
                            <span className="text-white font-medium">{event.urgency_score}</span>
                          </div>
                          <div>
                            <span className="text-gray-500">Confidence:</span>{' '}
                            <span className="text-white font-medium">{(event.confidence * 100).toFixed(0)}%</span>
                          </div>
                          <div>
                            <span className="text-gray-500">Modality:</span>{' '}
                            <span className="text-white font-medium">{event.modality}</span>
                          </div>
                        </div>
                        
                        {event.text_snippet && (
                          <div className="mt-2 p-2 rounded bg-gray-800 text-sm text-gray-300 italic">
                            "{event.text_snippet}"
                          </div>
                        )}
                      </div>
                    ))}
                </div>
              </div>
            ) : (
              <div className="rounded-lg border border-gray-700 bg-gray-800/30 p-12 text-center">
                <div className="text-4xl mb-4">👈</div>
                <p className="text-gray-400">Select a session to view details</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Footer Note */}
      <div className="mt-8 p-4 rounded-lg bg-gray-800/30 border border-gray-700">
        <p className="text-sm text-gray-500">
          <strong className="text-gray-400">Note:</strong> Session data is stored in memory only and will be 
          cleared when the backend restarts. This is intentional for privacy. Session IDs shown here are 
          truncated (first 8 characters) for display purposes.
        </p>
      </div>
    </div>
  );
}
