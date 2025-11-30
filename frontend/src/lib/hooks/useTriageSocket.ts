/**
 * CrisisTriage AI - WebSocket Hook for Triage Sessions
 *
 * Manages WebSocket connection lifecycle and message handling for real-time
 * triage streaming. Provides a clean interface for the dashboard to:
 * - Connect/disconnect from sessions
 * - Send text messages for triage
 * - Simulate audio events (for testing)
 * - Receive and parse server messages
 *
 * Usage:
 *   const {
 *     connectionStatus,
 *     currentTriage,
 *     triageHistory,
 *     alerts,
 *     sendText,
 *     sendSimulatedAudio,
 *     connect,
 *     disconnect,
 *   } = useTriageSocket(sessionId);
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { getSessionWsUrl } from '../config';
import type {
  ConnectionStatus,
  TriageSnapshot,
  TriageHistoryEntry,
  TranscriptSegment,
  WSServerMessage,
  WSClientTextMessage,
  WSClientControlMessage,
  WSAlertMessage,
} from '../types';

// =============================================================================
// Types
// =============================================================================

export interface UseTriageSocketOptions {
  /** Auto-connect when sessionId changes (default: false) */
  autoConnect?: boolean;
  /** Maximum history entries to retain (default: 100) */
  maxHistoryEntries?: number;
  /** Callback when connection opens */
  onConnect?: () => void;
  /** Callback when connection closes */
  onDisconnect?: () => void;
  /** Callback on error */
  onError?: (error: string) => void;
  /** Callback on new triage result */
  onTriageResult?: (result: TriageSnapshot) => void;
  /** Callback on alert */
  onAlert?: (alert: WSAlertMessage) => void;
}

export interface UseTriageSocketReturn {
  /** Current connection status */
  connectionStatus: ConnectionStatus;
  /** Latest triage result */
  currentTriage: TriageSnapshot | null;
  /** History of triage results */
  triageHistory: TriageHistoryEntry[];
  /** Transcript segments */
  transcripts: TranscriptSegment[];
  /** Active alerts */
  alerts: WSAlertMessage[];
  /** Last error message */
  lastError: string | null;
  /** Send a text message for triage */
  sendText: (text: string) => void;
  /** Send simulated audio data (for testing) */
  sendSimulatedAudio: () => void;
  /** Send real audio chunk (binary PCM data) */
  sendAudioChunk: (chunk: ArrayBuffer) => void;
  /** Send a control message (pause/resume/end) */
  sendControl: (action: 'pause' | 'resume' | 'end') => void;
  /** Manually connect to the session */
  connect: () => void;
  /** Manually disconnect from the session */
  disconnect: () => void;
  /** Clear all history and alerts */
  clearHistory: () => void;
}

// =============================================================================
// Helper Functions
// =============================================================================

function generateHistoryId(): string {
  return `triage_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

function triageToHistoryEntry(triage: TriageSnapshot): TriageHistoryEntry {
  return {
    id: generateHistoryId(),
    timestamp_ms: triage.timestamp_ms,
    urgency_score: triage.urgency_score,
    risk_level: triage.risk_level,
    emotional_state: triage.emotional_state,
    recommended_action: triage.recommended_action,
    confidence: triage.confidence,
    explanation_summary: triage.explanation?.summary || '',
  };
}

// =============================================================================
// Hook Implementation
// =============================================================================

export function useTriageSocket(
  sessionId: string | null,
  options: UseTriageSocketOptions = {}
): UseTriageSocketReturn {
  const {
    autoConnect = false,
    maxHistoryEntries = 100,
    onConnect,
    onDisconnect,
    onError,
    onTriageResult,
    onAlert,
  } = options;

  // Connection state
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [lastError, setLastError] = useState<string | null>(null);

  // Data state
  const [currentTriage, setCurrentTriage] = useState<TriageSnapshot | null>(null);
  const [triageHistory, setTriageHistory] = useState<TriageHistoryEntry[]>([]);
  const [transcripts, setTranscripts] = useState<TranscriptSegment[]>([]);
  const [alerts, setAlerts] = useState<WSAlertMessage[]>([]);

  // Refs for WebSocket and callbacks
  const wsRef = useRef<WebSocket | null>(null);
  const sessionIdRef = useRef<string | null>(sessionId);

  // Keep sessionId ref in sync
  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  // ---------------------------------------------------------------------------
  // Message Parsing
  // ---------------------------------------------------------------------------

  const handleMessage = useCallback(
    (event: MessageEvent) => {
      try {
        const message: WSServerMessage = JSON.parse(event.data);

        switch (message.type) {
          case 'connected':
            console.log('[WS] Connected:', message.message);
            setConnectionStatus('connected');
            setLastError(null);
            onConnect?.();
            break;

          case 'triage': {
            const triage = message.data;
            setCurrentTriage(triage);

            // Add to history (with limit)
            setTriageHistory((prev: TriageHistoryEntry[]) => {
              const entry = triageToHistoryEntry(triage);
              const updated = [entry, ...prev];
              return updated.slice(0, maxHistoryEntries);
            });

            onTriageResult?.(triage);
            break;
          }

          case 'transcript':
            setTranscripts((prev: TranscriptSegment[]) => [...prev, message.data]);
            break;

          case 'alert':
            setAlerts((prev: WSAlertMessage[]) => [message, ...prev]);
            onAlert?.(message);
            break;

          case 'status':
            console.log('[WS] Status:', message.message);
            break;

          case 'error':
            console.error('[WS] Server error:', message.message);
            setLastError(message.message);
            onError?.(message.message);
            break;

          default:
            console.warn('[WS] Unknown message type:', message);
        }
      } catch (err) {
        console.error('[WS] Failed to parse message:', err);
      }
    },
    [maxHistoryEntries, onConnect, onTriageResult, onAlert, onError]
  );

  // ---------------------------------------------------------------------------
  // Connection Management
  // ---------------------------------------------------------------------------

  const connect = useCallback(() => {
    if (!sessionIdRef.current) {
      console.warn('[WS] Cannot connect: no session ID');
      return;
    }

    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    setConnectionStatus('connecting');
    setLastError(null);

    const wsUrl = getSessionWsUrl(sessionIdRef.current);
    console.log('[WS] Connecting to:', wsUrl);

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('[WS] Connection opened');
      // Note: We don't set 'connected' here - we wait for the 'connected' message from server
    };

    ws.onmessage = handleMessage;

    ws.onerror = (event) => {
      console.error('[WS] WebSocket error:', event);
      setConnectionStatus('error');
      setLastError('WebSocket connection error');
      onError?.('WebSocket connection error');
    };

    ws.onclose = (event) => {
      console.log('[WS] Connection closed:', event.code, event.reason);
      setConnectionStatus('disconnected');
      wsRef.current = null;
      onDisconnect?.();
    };

    wsRef.current = ws;
  }, [handleMessage, onDisconnect, onError]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      // Send end control message before closing
      if (wsRef.current.readyState === WebSocket.OPEN) {
        const controlMsg: WSClientControlMessage = { type: 'control', action: 'end' };
        wsRef.current.send(JSON.stringify(controlMsg));
      }
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnectionStatus('disconnected');
  }, []);

  // ---------------------------------------------------------------------------
  // Message Sending
  // ---------------------------------------------------------------------------

  const sendText = useCallback((text: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.warn('[WS] Cannot send: not connected');
      setLastError('Not connected to server');
      return;
    }

    const message: WSClientTextMessage = {
      type: 'text',
      data: { text },
    };

    wsRef.current.send(JSON.stringify(message));
    console.log('[WS] Sent text message:', text.slice(0, 50) + (text.length > 50 ? '...' : ''));
  }, []);

  const sendSimulatedAudio = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.warn('[WS] Cannot send: not connected');
      setLastError('Not connected to server');
      return;
    }

    // Generate fake audio data (random bytes simulating PCM audio)
    // In real implementation, this would be actual audio data
    const fakeAudioLength = 16000 * 2; // 1 second at 16kHz, 16-bit
    const fakeAudio = new Uint8Array(fakeAudioLength);
    for (let i = 0; i < fakeAudioLength; i++) {
      fakeAudio[i] = Math.floor(Math.random() * 256);
    }

    wsRef.current.send(fakeAudio.buffer);
    console.log('[WS] Sent simulated audio:', fakeAudioLength, 'bytes');
  }, []);

  const sendAudioChunk = useCallback((chunk: ArrayBuffer) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.log('[WS] Cannot send audio: not connected, state:', wsRef.current?.readyState);
      return;
    }

    // Send binary audio data directly
    wsRef.current.send(chunk);
    console.log('[WS] Sent audio chunk:', chunk.byteLength, 'bytes');
  }, []);

  const sendControl = useCallback((action: 'pause' | 'resume' | 'end') => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.warn('[WS] Cannot send control: not connected');
      return;
    }

    const message: WSClientControlMessage = { type: 'control', action };
    wsRef.current.send(JSON.stringify(message));
    console.log('[WS] Sent control:', action);
  }, []);

  // ---------------------------------------------------------------------------
  // Utility Functions
  // ---------------------------------------------------------------------------

  const clearHistory = useCallback(() => {
    setTriageHistory([]);
    setTranscripts([]);
    setAlerts([]);
    setCurrentTriage(null);
  }, []);

  // ---------------------------------------------------------------------------
  // Effects
  // ---------------------------------------------------------------------------

  // Auto-connect when sessionId changes (if enabled)
  useEffect(() => {
    if (autoConnect && sessionId) {
      connect();
    }
  }, [autoConnect, sessionId, connect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Reset state when session changes
  useEffect(() => {
    if (sessionId !== sessionIdRef.current) {
      clearHistory();
    }
  }, [sessionId, clearHistory]);

  // ---------------------------------------------------------------------------
  // Return
  // ---------------------------------------------------------------------------

  return {
    connectionStatus,
    currentTriage,
    triageHistory,
    transcripts,
    alerts,
    lastError,
    sendText,
    sendSimulatedAudio,
    sendAudioChunk,
    sendControl,
    connect,
    disconnect,
    clearHistory,
  };
}

export default useTriageSocket;
