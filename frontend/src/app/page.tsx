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
    disconnected: { color: 'bg-gray-500', label: 'Disconnected' },
    connecting: { color: 'bg-yellow-500 animate-pulse', label: 'Connecting...' },
    connected: { color: 'bg-green-500', label: 'Connected' },
    error: { color: 'bg-red-500', label: 'Error' },
  };

  const { color, label } = statusConfig[status];

  return (
    <div className="flex items-center gap-2">
      <div className={`h-2 w-2 rounded-full ${color}`} />
      <span className="text-sm text-gray-400">{label}</span>
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
  let barColor = 'bg-green-500';
  if (score >= 70) barColor = 'bg-red-500';
  else if (score >= 45) barColor = 'bg-orange-500';
  else if (score >= 25) barColor = 'bg-yellow-500';

  return (
    <div className="w-full">
      <div className="flex justify-between text-xs mb-1">
        <span className="text-gray-400">Urgency</span>
        <span className="font-mono font-bold text-white">{score}</span>
      </div>
      <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full ${barColor} transition-all duration-300`}
          style={{ width: `${score}%` }}
        />
      </div>
    </div>
  );
}

/** Current triage result display */
function CurrentTriagePanel({ triage }: { triage: TriageSnapshot | null }) {
  if (!triage) {
    return (
      <div className="rounded-lg border border-gray-700 bg-gray-800/30 p-6">
        <div className="text-center text-gray-500">
          <p className="text-lg mb-2">No triage results yet</p>
          <p className="text-sm">Send a message to begin triage analysis</p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800/30 p-6 space-y-6">
      {/* Main metrics grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Risk Level */}
        <div className="space-y-2">
          <div className="text-xs text-gray-400 uppercase tracking-wide">Risk Level</div>
          <RiskBadge level={triage.risk_level} />
        </div>

        {/* Emotional State */}
        <div className="space-y-2">
          <div className="text-xs text-gray-400 uppercase tracking-wide">Emotion</div>
          <div className="text-lg font-semibold text-white">
            {EMOTIONAL_STATE_LABELS[triage.emotional_state]}
          </div>
        </div>

        {/* Recommended Action */}
        <div className="space-y-2">
          <div className="text-xs text-gray-400 uppercase tracking-wide">Action</div>
          <div className="text-sm font-medium text-blue-400">
            {ACTION_LABELS[triage.recommended_action]}
          </div>
        </div>

        {/* Confidence */}
        <div className="space-y-2">
          <div className="text-xs text-gray-400 uppercase tracking-wide">Confidence</div>
          <div className="text-lg font-semibold text-white">
            {Math.round(triage.confidence * 100)}%
          </div>
        </div>
      </div>

      {/* Urgency bar */}
      <UrgencyBar score={triage.urgency_score} />

      {/* Explanation */}
      {triage.explanation?.summary && (
        <div className="pt-4 border-t border-gray-700">
          <div className="text-xs text-gray-400 uppercase tracking-wide mb-2">
            Explanation
          </div>
          <p className="text-sm text-gray-300">{triage.explanation.summary}</p>
        </div>
      )}

      {/* Processing time (debug info) */}
      {triage.processing_time_ms && (
        <div className="text-xs text-gray-500 text-right">
          Processed in {triage.processing_time_ms.toFixed(1)}ms
        </div>
      )}
    </div>
  );
}

/** Triage history list */
function HistoryPanel({ history }: { history: TriageHistoryEntry[] }) {
  if (history.length === 0) {
    return (
      <div className="text-center text-gray-500 py-8">
        <p>No history yet</p>
      </div>
    );
  }

  return (
    <div className="space-y-2 max-h-64 overflow-y-auto">
      {history.map((entry) => (
        <div
          key={entry.id}
          className="flex items-center justify-between p-3 rounded-lg bg-gray-800/50 border border-gray-700/50"
        >
          <div className="flex items-center gap-3">
            <RiskBadge level={entry.risk_level} />
            <span className="text-sm text-gray-300">
              {EMOTIONAL_STATE_LABELS[entry.emotional_state]}
            </span>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-sm font-mono text-gray-400">
              Score: {entry.urgency_score}
            </span>
            <span className="text-xs text-gray-500">
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
  const [messageInput, setMessageInput] = useState('');

  // Generate a new session ID
  const generateSessionId = useCallback(() => {
    return crypto.randomUUID();
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
      stopRecording();
    }
  }, [connectionStatus, isRecording, stopRecording]);

  // Start a new session
  const handleStartSession = useCallback(() => {
    const newSessionId = generateSessionId();
    setSessionId(newSessionId);
    // Note: connection happens after state update, in useEffect via autoConnect
    // For manual control, we connect explicitly after setting ID
    setTimeout(() => connect(), 0);
  }, [generateSessionId, connect]);

  // End current session
  const handleEndSession = useCallback(() => {
    disconnect();
    setSessionId(null);
    clearHistory();
  }, [disconnect, clearHistory]);

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
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      {/* SAFETY DISCLAIMER */}
      <div className="mb-6 p-4 rounded-lg border-2 border-yellow-600 bg-yellow-900/30 text-yellow-200">
        <div className="flex items-start gap-3">
          <span className="text-2xl">⚠️</span>
          <div className="text-sm">
            <p className="font-bold mb-1">RESEARCH SIMULATION ONLY</p>
            <p>
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
            <h1 className="text-3xl font-bold text-white">CrisisTriage Lab</h1>
            <p className="text-gray-400 mt-1">
              Real-time triage simulation{' '}
              <span className="text-yellow-500 text-sm">(research-only)</span>
            </p>
          </div>
          <div className="flex items-center gap-4">
            <ConnectionBadge status={connectionStatus} />
            {!sessionId ? (
              <button
                onClick={handleStartSession}
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

        {/* Session ID display */}
        {sessionId && (
          <div className="mt-4 text-sm text-gray-500">
            Session: <code className="text-gray-400 bg-gray-800 px-2 py-1 rounded">{sessionId}</code>
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
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Panel: Input & Session */}
        <div className="space-y-6">
          {/* Microphone Section */}
          <div className="rounded-lg border border-gray-700 bg-gray-800/30 p-6">
            <h2 className="text-lg font-semibold text-white mb-4">Microphone Input</h2>
            
            {/* Audio error display */}
            {audioError && (
              <div className="mb-4 p-3 rounded-lg bg-red-900/30 border border-red-700 text-red-400 text-sm">
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
              <div className="mt-4 flex items-center gap-2 text-green-400 text-sm">
                <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                Recording... Audio is being streamed to the backend.
              </div>
            )}

            {/* Permission status */}
            {hasPermission === false && (
              <div className="mt-4 text-sm text-gray-400">
                Microphone access denied. 
                <button 
                  onClick={requestPermission}
                  className="ml-2 text-blue-400 hover:text-blue-300 underline"
                >
                  Request permission
                </button>
              </div>
            )}

            {!isConnected && (
              <p className="mt-4 text-sm text-gray-500">
                Start a session to enable microphone input.
              </p>
            )}

            <p className="mt-4 text-xs text-gray-500">
              Audio is processed locally. No data is sent to external APIs.
            </p>
          </div>

          {/* Live Transcript */}
          {transcripts.length > 0 && (
            <div className="rounded-lg border border-gray-700 bg-gray-800/30 p-6">
              <h2 className="text-lg font-semibold text-white mb-4">Live Transcript</h2>
              <div className="max-h-32 overflow-y-auto space-y-2">
                {transcripts.slice(-5).map((t, idx) => (
                  <p key={idx} className="text-sm text-gray-300">
                    {t.text}
                  </p>
                ))}
              </div>
            </div>
          )}

          {/* Message Input */}
          <div className="rounded-lg border border-gray-700 bg-gray-800/30 p-6">
            <h2 className="text-lg font-semibold text-white mb-4">Text Input</h2>
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
                className="w-full px-4 py-3 rounded-lg bg-gray-900 border border-gray-700 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed resize-none"
              />
              <div className="flex gap-3">
                <button
                  type="submit"
                  disabled={!isConnected || !messageInput.trim()}
                  className="flex-1 rounded-lg px-4 py-2 text-sm font-medium bg-blue-600 hover:bg-blue-700 text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Send for Triage
                </button>
                <button
                  type="button"
                  onClick={handleSimulateAudio}
                  disabled={!isConnected}
                  className="rounded-lg px-4 py-2 text-sm font-medium bg-gray-700 hover:bg-gray-600 text-gray-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  title="Send simulated audio data for testing"
                >
                  Simulate Audio
                </button>
              </div>
            </form>
          </div>

          {/* Quick Test Messages */}
          {isConnected && (
            <div className="rounded-lg border border-gray-700 bg-gray-800/30 p-6">
              <h3 className="text-sm font-medium text-gray-400 mb-3">Quick Test Messages</h3>
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
                    className="text-xs px-3 py-1.5 rounded-full bg-gray-700 hover:bg-gray-600 text-gray-300 transition-colors"
                  >
                    {msg.length > 30 ? msg.slice(0, 30) + '...' : msg}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* History Panel */}
          <div className="rounded-lg border border-gray-700 bg-gray-800/30 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-white">History</h2>
              {triageHistory.length > 0 && (
                <span className="text-xs text-gray-500">{triageHistory.length} results</span>
              )}
            </div>
            <HistoryPanel history={triageHistory} />
          </div>
        </div>

        {/* Right Panel: Live Triage Output */}
        <div className="space-y-6">
          {/* Current Triage Result */}
          <div>
            <h2 className="text-lg font-semibold text-white mb-4">Latest Triage Result</h2>
            <CurrentTriagePanel triage={currentTriage} />
          </div>

          {/* Feature Attribution (Explanation Details) */}
          {currentTriage?.explanation && (
            <div className="rounded-lg border border-gray-700 bg-gray-800/30 p-6">
              <h2 className="text-lg font-semibold text-white mb-4">Feature Attribution</h2>
              <div className="grid grid-cols-2 gap-6">
                {/* Text Features */}
                <div>
                  <h3 className="text-xs text-gray-400 uppercase tracking-wide mb-3">
                    Text Features
                  </h3>
                  {currentTriage.explanation.text_features?.length > 0 ? (
                    <ul className="space-y-2">
                      {currentTriage.explanation.text_features.slice(0, 5).map((f, idx) => (
                        <li key={idx} className="text-sm">
                          <span className="text-gray-300">{f.feature_name || 'keyword'}</span>
                          {f.value && (
                            <span className="text-gray-500 ml-2">({String(f.value)})</span>
                          )}
                          <span className="text-blue-400 ml-2 font-mono text-xs">
                            +{(f.contribution * 100).toFixed(0)}%
                          </span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-sm text-gray-500">No text features</p>
                  )}
                </div>

                {/* Prosody Features */}
                <div>
                  <h3 className="text-xs text-gray-400 uppercase tracking-wide mb-3">
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
