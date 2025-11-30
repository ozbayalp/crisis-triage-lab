'use client';

import { useState, useCallback, FormEvent, useEffect } from 'react';
import { useTriageSocket } from '@/lib/hooks/useTriageSocket';
import { useAudioStream } from '@/lib/hooks/useAudioStream';
import {
  ConnectionStatus,
  TriageSnapshot,
  TriageHistoryEntry,
  TranscriptSegment,
  RISK_LEVEL_COLORS,
  EMOTIONAL_STATE_LABELS,
  ACTION_LABELS,
  RiskLevel,
} from '@/lib/types';

/**
 * CrisisTriage AI - Live Triage Dashboard
 *
 * Real-time dashboard for triage simulation and research.
 * Connects to backend via WebSocket for streaming triage updates.
 *
 * Features:
 * - Create/manage sessions
 * - Send text messages for triage
 * - Simulate audio events (for testing)
 * - Display real-time triage results
 * - View triage history
 */

// =============================================================================
// Helper Components
// =============================================================================

/** Connection status badge */
function ConnectionBadge({ status }: { status: ConnectionStatus }) {
  const statusConfig: Record<ConnectionStatus, { color: string; label: string }> = {
    disconnected: { color: 'var(--text-tertiary)', label: 'Disconnected' },
    connecting: { color: 'var(--warning)', label: 'Connecting...' },
    connected: { color: 'var(--success)', label: 'Connected' },
    error: { color: 'var(--danger)', label: 'Error' },
  };

  const { color, label } = statusConfig[status];

  return (
    <div className="flex items-center gap-2">
      <div 
        className={`h-2 w-2 rounded-full ${status === 'connecting' ? 'animate-pulse' : ''}`} 
        style={{ background: color }}
      />
      <span className="text-sm" style={{ color: 'var(--text-tertiary)' }}>{label}</span>
    </div>
  );
}

/** Risk level badge with color */
function RiskBadge({ level }: { level: RiskLevel }) {
  const bgColor = RISK_LEVEL_COLORS[level];
  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${bgColor} text-white capitalize`}
    >
      {level}
    </span>
  );
}

/** Urgency score bar visualization */
function UrgencyBar({ score }: { score: number }) {
  // Color based on score
  let barColor = 'var(--success)';
  if (score >= 70) barColor = 'var(--danger)';
  else if (score >= 45) barColor = 'var(--warning)';
  else if (score >= 25) barColor = '#EAB308';

  return (
    <div className="w-full">
      <div className="flex justify-between text-xs mb-1">
        <span style={{ color: 'var(--text-tertiary)' }}>Urgency</span>
        <span className="font-mono font-bold">{score}</span>
      </div>
      <div className="h-2 rounded-full overflow-hidden" style={{ background: 'var(--bg-tertiary)' }}>
        <div
          className="h-full transition-all duration-300"
          style={{ width: `${score}%`, background: barColor }}
        />
      </div>
    </div>
  );
}

/** Current triage result display */
function CurrentTriagePanel({ triage }: { triage: TriageSnapshot | null }) {
  if (!triage) {
    return (
      <div 
        className="rounded-lg p-6"
        style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-primary)' }}
      >
        <div className="text-center" style={{ color: 'var(--text-tertiary)' }}>
          <p className="text-lg mb-2">No triage results yet</p>
          <p className="text-sm">Send a message to begin triage analysis</p>
        </div>
      </div>
    );
  }

  return (
    <div 
      className="rounded-lg p-6 space-y-6"
      style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-primary)' }}
    >
      {/* Main metrics grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Risk Level */}
        <div className="space-y-2">
          <div className="text-xs uppercase tracking-wide" style={{ color: 'var(--text-tertiary)', letterSpacing: '0.04em' }}>Risk Level</div>
          <RiskBadge level={triage.risk_level} />
        </div>

        {/* Emotional State */}
        <div className="space-y-2">
          <div className="text-xs uppercase tracking-wide" style={{ color: 'var(--text-tertiary)', letterSpacing: '0.04em' }}>Emotion</div>
          <div className="text-lg font-semibold">
            {EMOTIONAL_STATE_LABELS[triage.emotional_state]}
          </div>
        </div>

        {/* Recommended Action */}
        <div className="space-y-2">
          <div className="text-xs uppercase tracking-wide" style={{ color: 'var(--text-tertiary)', letterSpacing: '0.04em' }}>Action</div>
          <div className="text-sm font-medium" style={{ color: 'var(--info)' }}>
            {ACTION_LABELS[triage.recommended_action]}
          </div>
        </div>

        {/* Confidence */}
        <div className="space-y-2">
          <div className="text-xs uppercase tracking-wide" style={{ color: 'var(--text-tertiary)', letterSpacing: '0.04em' }}>Confidence</div>
          <div className="text-lg font-semibold">
            {Math.round(triage.confidence * 100)}%
          </div>
        </div>
      </div>

      {/* Urgency bar */}
      <UrgencyBar score={triage.urgency_score} />

      {/* Explanation */}
      {triage.explanation?.summary && (
        <div className="pt-4" style={{ borderTop: '1px solid var(--border-primary)' }}>
          <div className="text-xs uppercase tracking-wide mb-2" style={{ color: 'var(--text-tertiary)', letterSpacing: '0.04em' }}>
            Explanation
          </div>
          <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>{triage.explanation.summary}</p>
        </div>
      )}

      {/* Processing time (debug info) */}
      {triage.processing_time_ms && (
        <div className="text-xs text-right" style={{ color: 'var(--text-tertiary)' }}>
          Processed in {triage.processing_time_ms.toFixed(1)}ms
        </div>
      )}
    </div>
  );
}

/** Triage scoring history list - shows latest first */
function HistoryPanel({ 
  history 
}: { 
  history: TriageHistoryEntry[]; 
}) {
  if (history.length === 0) {
    return (
      <div className="text-center py-8" style={{ color: 'var(--text-tertiary)' }}>
        <p>No scoring history yet</p>
      </div>
    );
  }

  // Sort by timestamp descending (latest first)
  const sortedHistory = [...history].sort((a, b) => b.timestamp_ms - a.timestamp_ms);

  return (
    <div className="space-y-2 max-h-64 overflow-y-auto">
      {sortedHistory.map((entry) => (
        <div
          key={entry.id}
          className="flex items-center justify-between p-3 rounded-lg"
          style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--border-primary)' }}
        >
          <div className="flex items-center gap-3">
            <RiskBadge level={entry.risk_level} />
            <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
              {EMOTIONAL_STATE_LABELS[entry.emotional_state]}
            </span>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-sm font-mono" style={{ color: 'var(--text-tertiary)' }}>
              Score: {entry.urgency_score}
            </span>
            <span className="text-xs" style={{ color: 'var(--text-tertiary)' }}>
              {new Date(entry.timestamp_ms).toLocaleTimeString()}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

// =============================================================================
// Main Dashboard Component
// =============================================================================

export default function DashboardHome() {
  // Session state
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionName, setSessionName] = useState<string>('');
  const [showSessionModal, setShowSessionModal] = useState(false);
  const [sessionNameInput, setSessionNameInput] = useState('');
  const [messageInput, setMessageInput] = useState('');

  // Generate a new session ID
  const generateSessionId = useCallback(() => {
    return crypto.randomUUID();
  }, []);

  // Persisted history state
  const [persistedHistory, setPersistedHistory] = useState<TriageHistoryEntry[]>([]);

  // Load persisted history on mount
  useEffect(() => {
    const saved = localStorage.getItem('triageHistory');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setPersistedHistory(parsed);
      } catch (e) {
        console.error('Failed to parse saved history:', e);
      }
    }
    const savedSession = localStorage.getItem('sessionName');
    const savedSessionId = localStorage.getItem('sessionId');
    if (savedSession && savedSessionId) {
      setSessionName(savedSession);
      setSessionId(savedSessionId);
    }
  }, []);

  // WebSocket hook
  const {
    connectionStatus,
    currentTriage,
    triageHistory,
    transcripts,
    alerts,
    lastError,
    sendText,
    sendSimulatedAudio,
    sendAudioChunk,
    connect,
    disconnect,
    clearHistory,
  } = useTriageSocket(sessionId, {
    onConnect: () => console.log('Dashboard: Connected to session'),
    onDisconnect: () => console.log('Dashboard: Disconnected from session'),
    onError: (err) => console.error('Dashboard: Error:', err),
    onTriageResult: (result) => console.log('Dashboard: New triage result:', result.risk_level),
    onAlert: (alert) => console.warn('Dashboard: Alert:', alert.message),
  });

  // Merge persisted history with current session history and save
  useEffect(() => {
    if (triageHistory.length > 0) {
      const merged = [...persistedHistory];
      triageHistory.forEach(item => {
        if (!merged.find(m => m.id === item.id)) {
          merged.push(item);
        }
      });
      setPersistedHistory(merged);
      localStorage.setItem('triageHistory', JSON.stringify(merged));
    }
  }, [triageHistory]);

  // Save session info to localStorage
  useEffect(() => {
    if (sessionId && sessionName) {
      localStorage.setItem('sessionId', sessionId);
      localStorage.setItem('sessionName', sessionName);
    }
  }, [sessionId, sessionName]);

  // Audio stream hook
  const {
    isRecording,
    hasPermission,
    error: audioError,
    audioLevel,
    startRecording,
    stopRecording,
    requestPermission,
  } = useAudioStream({
    onAudioChunk: (chunk) => {
      // Send audio chunk to backend via WebSocket
      sendAudioChunk(chunk);
    },
    onError: (err) => console.error('Audio error:', err.message),
  });

  // Stop recording when disconnected
  useEffect(() => {
    if (connectionStatus !== 'connected' && isRecording) {
      console.log('[Page] Stopping recording because connectionStatus is:', connectionStatus);
      stopRecording();
    }
  }, [connectionStatus, isRecording, stopRecording]);

  // Show session name modal
  const handleShowSessionModal = useCallback(() => {
    setSessionNameInput('');
    setShowSessionModal(true);
  }, []);

  // Confirm session start with name
  const handleConfirmSession = useCallback(() => {
    const newSessionId = generateSessionId();
    setSessionId(newSessionId);
    setSessionName(sessionNameInput.trim() || `Session ${new Date().toLocaleTimeString()}`);
    setShowSessionModal(false);
    setTimeout(() => connect(), 0);
  }, [generateSessionId, sessionNameInput, connect]);

  // End current session
  const handleEndSession = useCallback(() => {
    disconnect();
    setSessionId(null);
    setSessionName('');
    clearHistory();
  }, [disconnect, clearHistory]);

  // Clear all scoring history
  const handleClearAll = useCallback(() => {
    clearHistory();
    setPersistedHistory([]);
    localStorage.removeItem('triageHistory');
  }, [clearHistory]);

  // Send text message
  const handleSendMessage = useCallback(
    (e: FormEvent) => {
      e.preventDefault();
      if (!messageInput.trim()) return;

      sendText(messageInput.trim());
      setMessageInput('');
    },
    [messageInput, sendText]
  );

  // Send simulated audio
  const handleSimulateAudio = useCallback(() => {
    sendSimulatedAudio();
  }, [sendSimulatedAudio]);

  const isConnected = connectionStatus === 'connected';
  const isConnecting = connectionStatus === 'connecting';

  return (
    <div className="max-w-7xl mx-auto">
      {/* SAFETY DISCLAIMER */}
      <div 
        className="mb-8 p-4 rounded-lg"
        style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--warning)' }}
      >
        <div className="flex items-start gap-3">
          <div className="text-sm">
            <p className="font-bold mb-1" style={{ color: 'var(--warning)' }}>RESEARCH SIMULATION ONLY</p>
            <p style={{ color: 'var(--text-secondary)' }}>
              This is an internal lab tool for AI research — NOT a medical device. 
              Do NOT use for real crisis intervention. 
              If you or someone you know is in crisis, please call your local emergency services 
              or a crisis hotline (e.g., 988 in the U.S.).
            </p>
          </div>
        </div>
      </div>

      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 style={{ fontSize: '32px', fontWeight: 600, letterSpacing: '-0.01em' }}>CrisisTriage Lab</h1>
            <p style={{ color: 'var(--text-secondary)', marginTop: '4px' }}>
              Real-time triage simulation{' '}
              <span style={{ color: 'var(--warning)', fontSize: '13px' }}>(research-only)</span>
            </p>
          </div>
          <div className="flex items-center gap-4">
            <ConnectionBadge status={connectionStatus} />
            {!sessionId ? (
              <button
                onClick={handleShowSessionModal}
                className="rounded-lg px-4 py-2 text-sm font-medium bg-blue-600 hover:bg-blue-700 text-white transition-colors"
              >
                Start Session
              </button>
            ) : (
              <button
                onClick={handleEndSession}
                disabled={isConnecting}
                className="rounded-lg px-4 py-2 text-sm font-medium bg-red-600 hover:bg-red-700 text-white transition-colors disabled:opacity-50"
              >
                End Session
              </button>
            )}
          </div>
        </div>

        {/* Session name modal */}
        {showSessionModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
            <div 
              className="rounded-lg p-6 w-full max-w-md"
              style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-primary)' }}
            >
              <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '16px' }}>New Session</h3>
              <input
                type="text"
                placeholder="Enter session name (optional)"
                value={sessionNameInput}
                onChange={(e) => setSessionNameInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleConfirmSession()}
                className="w-full px-3 py-2 rounded-lg mb-4"
                style={{ 
                  background: 'var(--bg-secondary)', 
                  border: '1px solid var(--border-primary)',
                  color: 'var(--text-primary)'
                }}
                autoFocus
              />
              <div className="flex gap-3 justify-end">
                <button
                  onClick={() => setShowSessionModal(false)}
                  className="px-4 py-2 rounded-lg text-sm"
                  style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
                >
                  Cancel
                </button>
                <button
                  onClick={handleConfirmSession}
                  className="px-4 py-2 rounded-lg text-sm font-medium"
                  style={{ background: 'var(--accent)', color: 'var(--bg-primary)' }}
                >
                  Start
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Session display */}
        {sessionId && (
          <div className="mt-4 text-sm" style={{ color: 'var(--text-tertiary)' }}>
            Session: <strong style={{ color: 'var(--text-primary)' }}>{sessionName}</strong>
            <code className="ml-2 px-2 py-1 rounded text-xs" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-tertiary)' }}>
              {sessionId.slice(0, 8)}...
            </code>
          </div>
        )}

        {/* Error display */}
        {lastError && (
          <div className="mt-4 p-3 rounded-lg bg-red-900/30 border border-red-700 text-red-400 text-sm">
            Error: {lastError}
          </div>
        )}

        {/* Alerts */}
        {alerts.length > 0 && (
          <div className="mt-4 space-y-2">
            {alerts.slice(0, 3).map((alert, idx) => (
              <div
                key={idx}
                className={`p-3 rounded-lg border text-sm ${
                  alert.level === 'critical'
                    ? 'bg-red-900/50 border-red-600 text-red-300'
                    : alert.level === 'high'
                    ? 'bg-orange-900/50 border-orange-600 text-orange-300'
                    : 'bg-yellow-900/50 border-yellow-600 text-yellow-300'
                }`}
              >
                <strong className="uppercase">{alert.level}:</strong> {alert.message}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Panel: Input & Session */}
        <div className="space-y-6">
          {/* Microphone Section */}
          <div 
            className="rounded-lg p-6"
            style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-primary)' }}
          >
            <h2 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>Microphone Input</h2>
            
            {/* Audio error display */}
            {audioError && (
              <div 
                className="mb-4 p-3 rounded-lg text-sm"
                style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--danger)', color: 'var(--danger)' }}
              >
                {audioError}
              </div>
            )}

            {/* Recording controls */}
            <div className="flex items-center gap-4">
              {!isRecording ? (
                <button
                  onClick={startRecording}
                  disabled={!isConnected || hasPermission === false}
                  className="flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium bg-green-600 hover:bg-green-700 text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <span className="w-3 h-3 rounded-full bg-white" />
                  Start Microphone
                </button>
              ) : (
                <button
                  onClick={stopRecording}
                  className="flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium bg-red-600 hover:bg-red-700 text-white transition-colors"
                >
                  <span className="w-3 h-3 rounded-full bg-white animate-pulse" />
                  Stop Recording
                </button>
              )}

              {/* Audio level indicator */}
              {isRecording && (
                <div className="flex items-center gap-2">
                  <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-green-500 transition-all duration-75"
                      style={{ width: `${audioLevel * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-400">Level</span>
                </div>
              )}
            </div>

            {/* Recording status */}
            {isRecording && (
              <div className="mt-4 flex items-center gap-2 text-sm" style={{ color: 'var(--success)' }}>
                <div className="w-2 h-2 rounded-full animate-pulse" style={{ background: 'var(--danger)' }} />
                Recording... Audio is being streamed to the backend.
              </div>
            )}

            {/* Permission status */}
            {hasPermission === false && (
              <div className="mt-4 text-sm" style={{ color: 'var(--text-tertiary)' }}>
                Microphone access denied. 
                <button 
                  onClick={requestPermission}
                  className="ml-2 underline"
                  style={{ color: 'var(--info)' }}
                >
                  Request permission
                </button>
              </div>
            )}

            {!isConnected && (
              <p className="mt-4 text-sm" style={{ color: 'var(--text-tertiary)' }}>
                Start a session to enable microphone input.
              </p>
            )}

            <p className="mt-4 text-xs" style={{ color: 'var(--text-tertiary)' }}>
              Audio is processed locally. No data is sent to external APIs.
            </p>
          </div>

          {/* Live Transcript */}
          {transcripts.length > 0 && (
            <div 
              className="rounded-lg p-6"
              style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-primary)' }}
            >
              <h2 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>Live Transcript</h2>
              <div className="max-h-32 overflow-y-auto space-y-2">
                {transcripts.slice(-5).map((t, idx) => (
                  <p key={idx} className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                    {t.text}
                  </p>
                ))}
              </div>
            </div>
          )}

          {/* Message Input */}
          <div 
            className="rounded-lg p-6"
            style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-primary)' }}
          >
            <h2 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>Text Input</h2>
            <form onSubmit={handleSendMessage} className="space-y-4">
              <textarea
                value={messageInput}
                onChange={(e) => setMessageInput(e.target.value)}
                placeholder={
                  isConnected
                    ? 'Type a message to analyze...'
                    : 'Start a session to send messages'
                }
                disabled={!isConnected}
                rows={4}
                className="w-full px-4 py-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed resize-none"
                style={{ 
                  background: 'var(--bg-tertiary)', 
                  border: '1px solid var(--border-primary)',
                  color: 'var(--text-primary)'
                }}
              />
              <div className="flex gap-3">
                <button
                  type="submit"
                  disabled={!isConnected || !messageInput.trim()}
                  className="flex-1 rounded-lg px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{ background: 'var(--accent)', color: 'var(--bg-primary)' }}
                >
                  Send for Triage
                </button>
                <button
                  type="button"
                  onClick={handleSimulateAudio}
                  disabled={!isConnected}
                  className="rounded-lg px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)', border: '1px solid var(--border-primary)' }}
                  title="Send simulated audio data for testing"
                >
                  Simulate Audio
                </button>
              </div>
            </form>
          </div>

          {/* Quick Test Messages */}
          {isConnected && (
            <div 
              className="rounded-lg p-6"
              style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-primary)' }}
            >
              <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-tertiary)' }}>Quick Test Messages</h3>
              <div className="flex flex-wrap gap-2">
                {[
                  "I've been feeling overwhelmed lately",
                  "I'm not sure what to do anymore",
                  "I feel hopeless and trapped",
                  "Thank you for listening",
                ].map((msg, idx) => (
                  <button
                    key={idx}
                    onClick={() => sendText(msg)}
                    className="text-xs px-3 py-1.5 rounded-full transition-colors"
                    style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)', border: '1px solid var(--border-primary)' }}
                  >
                    {msg.length > 30 ? msg.slice(0, 30) + '...' : msg}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* History Panel */}
          <div 
            className="rounded-lg p-6"
            style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-primary)' }}
          >
            <div className="flex items-center justify-between mb-4">
              <h2 style={{ fontSize: '16px', fontWeight: 600 }}>Scoring History</h2>
              <div className="flex items-center gap-3">
                {persistedHistory.length > 0 && (
                  <span style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>
                    {persistedHistory.length} results
                  </span>
                )}
                {persistedHistory.length > 0 && (
                  <button
                    onClick={handleClearAll}
                    className="px-2 py-1 rounded text-xs transition-colors"
                    style={{ 
                      background: 'var(--bg-tertiary)', 
                      color: 'var(--danger)',
                      border: '1px solid var(--border-primary)'
                    }}
                  >
                    Clear All
                  </button>
                )}
              </div>
            </div>
            <HistoryPanel history={persistedHistory} />
          </div>
        </div>

        {/* Right Panel: Live Triage Output */}
        <div className="space-y-6">
          {/* Current Triage Result */}
          <div>
            <h2 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>Latest Triage Result</h2>
            <CurrentTriagePanel triage={currentTriage} />
          </div>

          {/* Feature Attribution (Explanation Details) */}
          {currentTriage?.explanation && (
            <div 
              className="rounded-lg p-6"
              style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-primary)' }}
            >
              <h2 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>Feature Attribution</h2>
              <div className="grid grid-cols-2 gap-6">
                {/* Text Features */}
                <div>
                  <h3 
                    className="text-xs uppercase tracking-wide mb-3"
                    style={{ color: 'var(--text-tertiary)', letterSpacing: '0.04em' }}
                  >
                    Text Features
                  </h3>
                  {currentTriage.explanation.text_features?.length > 0 ? (
                    <ul className="space-y-2">
                      {currentTriage.explanation.text_features.slice(0, 5).map((f, idx) => (
                        <li key={idx} className="text-sm">
                          <span style={{ color: 'var(--text-secondary)' }}>{f.feature_name || 'keyword'}</span>
                          {f.value && (
                            <span className="ml-2" style={{ color: 'var(--text-tertiary)' }}>({String(f.value)})</span>
                          )}
                          <span className="ml-2 font-mono text-xs" style={{ color: 'var(--info)' }}>
                            +{(f.contribution * 100).toFixed(0)}%
                          </span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-sm" style={{ color: 'var(--text-tertiary)' }}>No text features</p>
                  )}
                </div>

                {/* Prosody Features */}
                <div>
                  <h3 
                    className="text-xs uppercase tracking-wide mb-3"
                    style={{ color: 'var(--text-tertiary)', letterSpacing: '0.04em' }}
                  >
                    Voice Features
                  </h3>
                  {currentTriage.explanation.prosody_features?.length > 0 ? (
                    <ul className="space-y-2">
                      {currentTriage.explanation.prosody_features.slice(0, 5).map((f, idx) => (
                        <li key={idx} className="text-sm">
                          <span className="text-gray-300">
                            {f.feature_name?.replace(/_/g, ' ') || 'feature'}
                          </span>
                          {f.value && (
                            <span className="text-gray-500 ml-2">
                              ({typeof f.value === 'number' ? f.value.toFixed(2) : f.value})
                            </span>
                          )}
                          <span className="text-purple-400 ml-2 font-mono text-xs">
                            +{(f.contribution * 100).toFixed(0)}%
                          </span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-sm text-gray-500">No voice features (text-only)</p>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Connection Info */}
          {!sessionId && (
            <div className="rounded-lg border border-dashed border-gray-700 p-8 text-center">
              <div className="text-gray-500 space-y-2">
                <p className="text-lg">No Active Session</p>
                <p className="text-sm">Click &ldquo;Start Session&rdquo; to begin triage simulation</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
