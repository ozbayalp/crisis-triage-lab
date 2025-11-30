/**
 * CrisisTriage AI - Frontend Configuration
 *
 * Centralized configuration reading from environment variables.
 * All backend URLs and feature flags are managed here.
 *
 * Environment Variables:
 *   NEXT_PUBLIC_BACKEND_HTTP_URL - REST API base URL (e.g., http://localhost:8000)
 *   NEXT_PUBLIC_BACKEND_WS_URL   - WebSocket base URL (e.g., ws://localhost:8000)
 */

// =============================================================================
// Environment Variable Access
// =============================================================================

/**
 * Get backend HTTP URL from environment.
 * Falls back to localhost:8000 for development.
 */
export function getBackendHttpUrl(): string {
  return process.env.NEXT_PUBLIC_BACKEND_HTTP_URL || 'http://localhost:8000';
}

/**
 * Get backend WebSocket URL from environment.
 * Falls back to localhost:8000 for development.
 */
export function getBackendWsUrl(): string {
  return process.env.NEXT_PUBLIC_BACKEND_WS_URL || 'ws://localhost:8000';
}

// =============================================================================
// Derived URLs
// =============================================================================

/**
 * Get full REST API URL for a path.
 * @param path - API path (e.g., '/api/sessions')
 */
export function getApiUrl(path: string): string {
  const base = getBackendHttpUrl();
  // Ensure no double slashes
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${base}${cleanPath}`;
}

/**
 * Get WebSocket URL for a session.
 * @param sessionId - UUID of the session
 */
export function getSessionWsUrl(sessionId: string): string {
  const base = getBackendWsUrl();
  return `${base}/ws/session/${sessionId}`;
}

// =============================================================================
// API Endpoints
// =============================================================================

export const API_ENDPOINTS = {
  /** Health check endpoint */
  health: '/api/health',

  /** Create a new session */
  createSession: '/api/sessions',

  /** Get session status (append session ID) */
  sessionStatus: (sessionId: string) => `/api/sessions/${sessionId}`,

  /** Submit text for triage (append session ID) */
  triage: (sessionId: string) => `/api/sessions/${sessionId}/triage`,

  /** Get latest triage result (append session ID) */
  latestTriage: (sessionId: string) => `/api/sessions/${sessionId}/triage/latest`,

  /** Get triage history (append session ID) */
  triageHistory: (sessionId: string) => `/api/sessions/${sessionId}/triage/history`,
} as const;

// =============================================================================
// Configuration Object
// =============================================================================

export const config = {
  /** Backend HTTP base URL */
  httpUrl: getBackendHttpUrl(),

  /** Backend WebSocket base URL */
  wsUrl: getBackendWsUrl(),

  /** Feature flags */
  features: {
    /** Enable explainability display in UI */
    showExplanations: true,

    /** Enable simulated audio button (for testing without mic) */
    enableSimulatedAudio: true,

    /** Maximum triage history entries to keep in memory */
    maxHistoryEntries: 100,

    /** WebSocket reconnection settings */
    wsReconnect: {
      enabled: true,
      maxAttempts: 5,
      delayMs: 1000,
    },
  },
} as const;

export default config;
