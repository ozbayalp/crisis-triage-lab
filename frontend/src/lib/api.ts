/**
 * CrisisTriage AI - API Client
 * 
 * HTTP client for REST API endpoints.
 * All requests are made to the backend server configured in config.ts.
 * 
 * IMPORTANT SAFETY NOTICE:
 *   This is a RESEARCH AND SIMULATION tool only.
 *   NOT a medical device. NOT suitable for real crisis intervention.
 */

import { getBackendHttpUrl } from './config';
import type {
  TriageAnalytics,
  TriageEvent,
  AnalyticsDisabledResponse,
} from './types';

// =============================================================================
// Error Types
// =============================================================================

export class APIError extends Error {
  constructor(
    message: string,
    public status: number,
    public detail?: string,
  ) {
    super(message);
    this.name = 'APIError';
  }
}

// =============================================================================
// HTTP Helpers
// =============================================================================

async function fetchJSON<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const baseUrl = getBackendHttpUrl();
  const url = `${baseUrl}${endpoint}`;
  
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  });
  
  if (!response.ok) {
    let detail: string | undefined;
    try {
      const errorBody = await response.json();
      detail = errorBody.detail || errorBody.message;
    } catch {
      // Ignore JSON parse errors
    }
    throw new APIError(
      `API request failed: ${response.status}`,
      response.status,
      detail,
    );
  }
  
  // Handle 204 No Content
  if (response.status === 204) {
    return undefined as T;
  }
  
  return response.json();
}

// =============================================================================
// Analytics API
// =============================================================================

/**
 * Fetch aggregated analytics summary.
 * 
 * Returns distribution of risk levels, emotions, and other aggregate stats.
 * If analytics is disabled, returns an AnalyticsDisabledResponse.
 */
export async function fetchAnalyticsSummary(): Promise<TriageAnalytics | AnalyticsDisabledResponse> {
  return fetchJSON<TriageAnalytics | AnalyticsDisabledResponse>('/api/analytics/summary');
}

/**
 * Check if a response is the disabled response.
 */
export function isAnalyticsDisabled(
  response: TriageAnalytics | AnalyticsDisabledResponse
): response is AnalyticsDisabledResponse {
  return 'enabled' in response && response.enabled === false;
}

/**
 * Fetch recent triage events.
 * 
 * Returns the most recent events in reverse chronological order.
 */
export async function fetchRecentEvents(limit: number = 100): Promise<TriageEvent[]> {
  return fetchJSON<TriageEvent[]>(`/api/analytics/recent?limit=${limit}`);
}

/**
 * Clear all analytics data.
 * 
 * Use with caution - permanently deletes all stored events.
 */
export async function clearAnalytics(): Promise<void> {
  await fetchJSON<void>('/api/analytics/clear', { method: 'DELETE' });
}

// =============================================================================
// Health API
// =============================================================================

export interface HealthResponse {
  status: string;
  components: Record<string, string>;
  version: string;
}

/**
 * Check backend health status.
 */
export async function fetchHealth(): Promise<HealthResponse> {
  return fetchJSON<HealthResponse>('/api/health');
}
