/**
 * CrisisTriage AI - Frontend Type Definitions
 *
 * TypeScript types mirroring backend API schemas.
 * Keep in sync with backend/app/api/schemas.py and backend/app/api/websocket.py
 *
 * WebSocket Protocol Summary:
 * ===========================
 * Client → Server:
 *   - Binary frames: Raw audio (PCM 16-bit, 16kHz, mono)
 *   - Text frames (JSON):
 *     {"type": "text", "data": {"text": "message content"}}
 *     {"type": "control", "action": "pause" | "resume" | "end"}
 *
 * Server → Client:
 *   - {"type": "connected", "session_id": "...", "message": "...", "protocol_version": "1.0"}
 *   - {"type": "transcript", "data": {...}}
 *   - {"type": "triage", "data": {...}}
 *   - {"type": "alert", "level": "...", "message": "...", "timestamp_ms": ...}
 *   - {"type": "status", "message": "..."}
 *   - {"type": "error", "message": "..."}
 */

// =============================================================================
// Enums (match backend/app/core/types.py)
// =============================================================================

export type EmotionalState = 'calm' | 'anxious' | 'distressed' | 'panicked' | 'unknown';

export type RiskLevel = 'low' | 'medium' | 'high' | 'imminent' | 'unknown';

export type RecommendedAction =
  | 'continue_listening'
  | 'ask_followup'
  | 'escalate_to_human'
  | 'immediate_intervention';

// =============================================================================
// Session Types (REST API)
// =============================================================================

export interface SessionCreateRequest {
  metadata?: Record<string, string>;
  sample_rate?: number;
  channels?: number;
}

export interface SessionCreateResponse {
  session_id: string;
  websocket_url: string;
  status: string;
}

export interface SessionStatus {
  session_id: string;
  status: 'active' | 'paused' | 'ended';
  created_at: string;
  duration_seconds: number;
  latest_triage?: TriageSnapshot;
}

// =============================================================================
// Triage Types
// =============================================================================

/** Feature contribution for explainability */
export interface FeatureContribution {
  feature_name: string;
  contribution: number;
  value?: string | number;
  category: 'text' | 'prosody';
}

/** Explanation of triage decision */
export interface FeatureExplanation {
  text_features: FeatureContribution[];
  prosody_features: FeatureContribution[];
  top_contributors?: FeatureContribution[];
  summary: string;
  method?: string;
}

/** Core triage result from backend */
export interface TriageSnapshot {
  emotional_state: EmotionalState;
  risk_level: RiskLevel;
  urgency_score: number; // 0-100
  recommended_action: RecommendedAction;
  explanation: FeatureExplanation;
  confidence: number; // 0-1
  timestamp_ms: number;
  processing_time_ms?: number;
}

/** Request payload for REST triage endpoint */
export interface TriageRequest {
  text: string;
  context?: Record<string, unknown>;
}

export interface TranscriptSegment {
  text: string;
  is_final: boolean;
  start_ms?: number;
  end_ms?: number;
  timestamp_ms?: number;
  confidence?: number;
}

// =============================================================================
// WebSocket Messages: Client → Server
// =============================================================================

/** Text message for triage */
export interface WSClientTextMessage {
  type: 'text';
  data: {
    text: string;
  };
}

/** Control message (pause/resume/end) */
export interface WSClientControlMessage {
  type: 'control';
  action: 'pause' | 'resume' | 'end';
}

/** All client-to-server message types */
export type WSClientMessage = WSClientTextMessage | WSClientControlMessage;

// =============================================================================
// WebSocket Messages: Server → Client
// =============================================================================

export interface WSConnectedMessage {
  type: 'connected';
  session_id: string;
  message: string;
  protocol_version?: string;
}

export interface WSTriageMessage {
  type: 'triage';
  data: TriageSnapshot;
}

export interface WSTranscriptMessage {
  type: 'transcript';
  data: TranscriptSegment;
}

export interface WSAlertMessage {
  type: 'alert';
  level: 'warning' | 'high' | 'critical';
  message: string;
  timestamp_ms: number;
}

export interface WSStatusMessage {
  type: 'status';
  message: string;
}

export interface WSErrorMessage {
  type: 'error';
  message: string;
}

/** All server-to-client message types */
export type WSServerMessage =
  | WSConnectedMessage
  | WSTriageMessage
  | WSTranscriptMessage
  | WSAlertMessage
  | WSStatusMessage
  | WSErrorMessage;

// =============================================================================
// API Response Types
// =============================================================================

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  components: Record<string, string>;
  version: string;
}

export interface APIError {
  detail: string;
  status_code?: number;
}

// =============================================================================
// Connection State
// =============================================================================

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

// =============================================================================
// UI State Types
// =============================================================================

export interface TriageHistoryEntry {
  id: string;
  timestamp_ms: number;
  urgency_score: number;
  risk_level: RiskLevel;
  emotional_state: EmotionalState;
  recommended_action: RecommendedAction;
  confidence: number;
  explanation_summary: string;
}

export interface SessionState {
  sessionId: string | null;
  connectionStatus: ConnectionStatus;
  isRecording: boolean;
  transcript: TranscriptSegment[];
  currentTriage: TriageSnapshot | null;
  triageHistory: TriageHistoryEntry[];
  alerts: WSAlertMessage[];
  lastError: string | null;
}

// =============================================================================
// Utility Types
// =============================================================================

/** Risk level color mapping */
export const RISK_LEVEL_COLORS: Record<RiskLevel, string> = {
  low: 'bg-green-500',
  medium: 'bg-yellow-500',
  high: 'bg-orange-500',
  imminent: 'bg-red-500',
  unknown: 'bg-gray-500',
};

/** Emotional state display mapping */
export const EMOTIONAL_STATE_LABELS: Record<EmotionalState, string> = {
  calm: 'Calm',
  anxious: 'Anxious',
  distressed: 'Distressed',
  panicked: 'Panicked',
  unknown: 'Unknown',
};

/** Recommended action display mapping */
export const ACTION_LABELS: Record<RecommendedAction, string> = {
  continue_listening: 'Continue Listening',
  ask_followup: 'Ask Follow-up',
  escalate_to_human: 'Escalate to Human',
  immediate_intervention: 'Immediate Intervention',
};


// =============================================================================
// Analytics Types
// =============================================================================

/** Input modality for analytics */
export type InputModality = 'text' | 'audio' | 'mixed';

/** A single triage event in analytics */
export interface TriageEvent {
  timestamp: string; // ISO date string
  session_id: string; // Anonymized/truncated
  risk_level: RiskLevel;
  emotional_state: EmotionalState;
  urgency_score: number;
  confidence: number;
  modality: InputModality;
  processing_time_ms?: number;
  text_snippet?: string; // Only if privacy allows
}

/** Statistics for a single risk level */
export interface RiskLevelStats {
  count: number;
  percentage: number;
  avg_urgency: number;
  avg_confidence: number;
  example_snippets: string[];
}

/** Aggregated analytics across triage events */
export interface TriageAnalytics {
  // Totals
  total_events: number;
  events_last_hour: number;
  events_last_24h: number;
  
  // Risk level distribution
  risk_counts: Record<string, number>;
  risk_percentages: Record<string, number>;
  
  // Emotional state distribution
  emotion_counts: Record<string, number>;
  emotion_percentages: Record<string, number>;
  
  // Modality distribution
  modality_counts: Record<string, number>;
  
  // Averages
  avg_urgency_score: number;
  avg_confidence: number;
  avg_processing_time_ms: number;
  
  // Per-risk stats
  risk_level_stats: Record<string, RiskLevelStats>;
  
  // Sessions
  unique_sessions: number;
}

/** Response when analytics is disabled */
export interface AnalyticsDisabledResponse {
  message: string;
  enabled: false;
}

/** Modality display labels */
export const MODALITY_LABELS: Record<InputModality, string> = {
  text: 'Text',
  audio: 'Audio',
  mixed: 'Mixed',
};

/** Modality colors */
export const MODALITY_COLORS: Record<InputModality, string> = {
  text: 'bg-blue-500',
  audio: 'bg-purple-500',
  mixed: 'bg-indigo-500',
};
