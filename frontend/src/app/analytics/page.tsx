'use client';

import { useState, useEffect, useCallback } from 'react';
import {
  fetchAnalyticsSummary,
  fetchRecentEvents,
  clearAnalytics,
  isAnalyticsDisabled,
} from '@/lib/api';
import type {
  TriageAnalytics,
  TriageEvent,
  RiskLevel,
  EmotionalState,
  InputModality,
} from '@/lib/types';
import {
  RISK_LEVEL_COLORS,
  EMOTIONAL_STATE_LABELS,
  MODALITY_LABELS,
  MODALITY_COLORS,
} from '@/lib/types';

/**
 * CrisisTriage AI - Analytics Dashboard
 *
 * Displays aggregated analytics across triage sessions.
 * All data is from simulated/research sessions only.
 *
 * IMPORTANT SAFETY NOTICE:
 *   This is a RESEARCH AND SIMULATION tool only.
 *   NOT a medical device. NOT suitable for real crisis intervention.
 */

// =============================================================================
// Helper Components
// =============================================================================

/** Summary stat card */
function StatCard({
  label,
  value,
  subtext,
  color,
}: {
  label: string;
  value: string | number;
  subtext?: string;
  color?: string;
}) {
  return (
    <div 
      className="rounded-lg p-4"
      style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-primary)' }}
    >
      <div 
        className="text-xs uppercase tracking-wide mb-1"
        style={{ color: 'var(--text-tertiary)', letterSpacing: '0.04em' }}
      >{label}</div>
      <div className={`text-2xl font-bold ${color || ''}`}>{value}</div>
      {subtext && <div className="text-xs mt-1" style={{ color: 'var(--text-tertiary)' }}>{subtext}</div>}
    </div>
  );
}

/** Horizontal bar chart for distributions */
function DistributionBar({
  label,
  count,
  percentage,
  color,
}: {
  label: string;
  count: number;
  percentage: number;
  color: string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <span className="capitalize" style={{ color: 'var(--text-secondary)' }}>{label}</span>
        <span style={{ color: 'var(--text-tertiary)' }}>{count} ({percentage.toFixed(1)}%)</span>
      </div>
      <div className="h-2 rounded-full overflow-hidden" style={{ background: 'var(--bg-tertiary)' }}>
        <div
          className={`h-full ${color} transition-all duration-300`}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>
    </div>
  );
}

/** Risk level badge */
function RiskBadge({ level }: { level: RiskLevel }) {
  const bgColor = RISK_LEVEL_COLORS[level] || 'bg-gray-500';
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${bgColor} text-white capitalize`}>
      {level}
    </span>
  );
}

/** Modality badge */
function ModalityBadge({ modality }: { modality: InputModality }) {
  const bgColor = MODALITY_COLORS[modality] || 'bg-gray-500';
  const label = MODALITY_LABELS[modality] || modality;
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${bgColor} text-white`}>
      {label}
    </span>
  );
}

// =============================================================================
// Test Scenarios
// =============================================================================

const TRIAGE_SCENARIOS = [
  { id: 'low_1', label: 'Low-risk: gratitude', text: 'Thank you for listening. I feel a bit better now.' },
  { id: 'low_2', label: 'Low-risk: coping', text: 'I talked to a friend and it really helped me see things differently.' },
  { id: 'med_1', label: 'Moderate: overwhelmed', text: "I'm feeling overwhelmed but not thinking of hurting myself." },
  { id: 'med_2', label: 'Moderate: anxious', text: "I can't stop worrying about everything. It's been weeks since I slept well." },
  { id: 'high_1', label: 'High-risk: ideation', text: "I don't see a way out anymore. I've been thinking about ending it all." },
  { id: 'high_2', label: 'High-risk: hopeless', text: "Everyone would be better off without me. I'm just a burden." },
];

// =============================================================================
// Main Component
// =============================================================================

export default function AnalyticsPage() {
  // State
  const [analytics, setAnalytics] = useState<TriageAnalytics | null>(null);
  const [recentEvents, setRecentEvents] = useState<TriageEvent[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isDisabled, setIsDisabled] = useState(false);
  const [isClearing, setIsClearing] = useState(false);

  // Fetch data
  const loadData = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const [summaryResult, eventsResult] = await Promise.all([
        fetchAnalyticsSummary(),
        fetchRecentEvents(50),
      ]);

      if (isAnalyticsDisabled(summaryResult)) {
        setIsDisabled(true);
        setAnalytics(null);
      } else {
        setIsDisabled(false);
        setAnalytics(summaryResult);
      }
      setRecentEvents(eventsResult);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Failed to load analytics');
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Load on mount and refresh periodically
  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [loadData]);

  // Handle clear
  const handleClear = async () => {
    if (!confirm('Are you sure you want to clear all analytics data? This cannot be undone.')) {
      return;
    }
    setIsClearing(true);
    try {
      await clearAnalytics();
      await loadData();
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      }
    } finally {
      setIsClearing(false);
    }
  };

  // Disabled state
  if (isDisabled) {
    return (
      <div className="max-w-7xl mx-auto">
        <div className="text-center py-16">
          <h1 className="text-2xl font-bold mb-4" style={{ color: 'var(--text-secondary)' }}>Analytics Disabled</h1>
          <p style={{ color: 'var(--text-tertiary)' }}>
            Analytics is disabled in this deployment.
            Set <code style={{ background: 'var(--bg-tertiary)', padding: '4px 8px', borderRadius: '4px' }}>ENABLE_ANALYTICS=true</code> to enable.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header with disclaimer */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 style={{ fontSize: '32px', fontWeight: 600, letterSpacing: '-0.01em' }}>Triage Analytics</h1>
            <p style={{ color: 'var(--text-secondary)', marginTop: '4px' }}>
              Aggregated insights across sessions{' '}
              <span style={{ color: 'var(--warning)', fontSize: '13px' }}>(simulated data only)</span>
            </p>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={loadData}
              disabled={isLoading}
              className="px-4 py-2 text-sm font-medium rounded-lg transition-colors disabled:opacity-50"
              style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)', border: '1px solid var(--border-primary)' }}
            >
              {isLoading ? 'Refreshing...' : 'Refresh'}
            </button>
            <button
              onClick={handleClear}
              disabled={isClearing || isLoading}
              className="px-4 py-2 text-sm font-medium rounded-lg transition-colors disabled:opacity-50"
              style={{ background: 'var(--bg-tertiary)', color: 'var(--danger)', border: '1px solid var(--danger)' }}
            >
              {isClearing ? 'Clearing...' : 'Clear Data'}
            </button>
          </div>
        </div>

        {/* Research disclaimer */}
        <div 
          className="p-3 rounded-lg text-sm"
          style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--warning)', color: 'var(--warning)' }}
        >
          <strong>Research Only:</strong> Analytics are computed on simulated data for research purposes.
          No real user or clinical data is used or stored.
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div 
          className="mb-6 p-4 rounded-lg"
          style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--danger)', color: 'var(--danger)' }}
        >
          Error: {error}
        </div>
      )}

      {/* Loading state */}
      {isLoading && !analytics && (
        <div className="text-center py-16">
          <div style={{ color: 'var(--text-tertiary)' }}>Loading analytics...</div>
        </div>
      )}

      {/* Main content */}
      {analytics && (
        <div className="space-y-6">
          {/* Summary cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard
              label="Total Events"
              value={analytics.total_events}
              subtext={`${analytics.unique_sessions} sessions`}
            />
            <StatCard
              label="Last Hour"
              value={analytics.events_last_hour}
              color="text-blue-400"
            />
            <StatCard
              label="Avg Urgency"
              value={analytics.avg_urgency_score.toFixed(1)}
              subtext="0-100 scale"
              color={analytics.avg_urgency_score > 50 ? 'text-orange-400' : 'text-green-400'}
            />
            <StatCard
              label="Avg Confidence"
              value={`${(analytics.avg_confidence * 100).toFixed(0)}%`}
              subtext={`${analytics.avg_processing_time_ms.toFixed(0)}ms avg`}
            />
          </div>

          {/* Distributions */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Risk Distribution */}
            <div 
              className="rounded-lg p-6"
              style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-primary)' }}
            >
              <h2 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>Risk Distribution</h2>
              <div className="space-y-3">
                {(['low', 'medium', 'high', 'imminent'] as RiskLevel[]).map((level) => {
                  const count = analytics.risk_counts[level] || 0;
                  const pct = analytics.risk_percentages[level] || 0;
                  return (
                    <DistributionBar
                      key={level}
                      label={level}
                      count={count}
                      percentage={pct}
                      color={RISK_LEVEL_COLORS[level]}
                    />
                  );
                })}
              </div>
            </div>

            {/* Emotion Distribution */}
            <div 
              className="rounded-lg p-6"
              style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-primary)' }}
            >
              <h2 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>Emotion Distribution</h2>
              <div className="space-y-3">
                {(['calm', 'anxious', 'distressed', 'panicked'] as EmotionalState[]).map((emotion) => {
                  const count = analytics.emotion_counts[emotion] || 0;
                  const pct = analytics.emotion_percentages[emotion] || 0;
                  const colors: Record<EmotionalState, string> = {
                    calm: 'bg-green-500',
                    anxious: 'bg-yellow-500',
                    distressed: 'bg-orange-500',
                    panicked: 'bg-red-500',
                    unknown: 'bg-gray-500',
                  };
                  return (
                    <DistributionBar
                      key={emotion}
                      label={EMOTIONAL_STATE_LABELS[emotion]}
                      count={count}
                      percentage={pct}
                      color={colors[emotion]}
                    />
                  );
                })}
              </div>
            </div>

            {/* Modality Distribution */}
            <div 
              className="rounded-lg p-6"
              style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-primary)' }}
            >
              <h2 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>Input Modality</h2>
              <div className="space-y-3">
                {(['text', 'audio', 'mixed'] as InputModality[]).map((modality) => {
                  const count = analytics.modality_counts[modality] || 0;
                  const total = analytics.total_events || 1;
                  const pct = (count / total) * 100;
                  return (
                    <DistributionBar
                      key={modality}
                      label={MODALITY_LABELS[modality]}
                      count={count}
                      percentage={pct}
                      color={MODALITY_COLORS[modality]}
                    />
                  );
                })}
              </div>
            </div>
          </div>

          {/* Recent Events Table */}
          <div 
            className="rounded-lg p-6"
            style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-primary)' }}
          >
            <div className="flex items-center justify-between mb-4">
              <h2 style={{ fontSize: '16px', fontWeight: 600 }}>Recent Events</h2>
              <span className="text-sm" style={{ color: 'var(--text-tertiary)' }}>{recentEvents.length} events</span>
            </div>
            
            {recentEvents.length === 0 ? (
              <div className="text-center py-8" style={{ color: 'var(--text-tertiary)' }}>
                No events recorded yet. Use the Live Triage page to generate data.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr 
                      className="text-left"
                      style={{ color: 'var(--text-tertiary)', borderBottom: '1px solid var(--border-primary)' }}
                    >
                      <th className="pb-2 pr-4">Time</th>
                      <th className="pb-2 pr-4">Session</th>
                      <th className="pb-2 pr-4">Risk</th>
                      <th className="pb-2 pr-4">Emotion</th>
                      <th className="pb-2 pr-4">Urgency</th>
                      <th className="pb-2 pr-4">Modality</th>
                      <th className="pb-2">Snippet</th>
                    </tr>
                  </thead>
                  <tbody style={{ borderColor: 'var(--border-primary)' }}>
                    {recentEvents.slice(0, 20).map((event, idx) => (
                      <tr 
                        key={idx} 
                        style={{ 
                          color: 'var(--text-secondary)',
                          borderBottom: '1px solid var(--border-primary)'
                        }}
                      >
                        <td className="py-2 pr-4 whitespace-nowrap" style={{ color: 'var(--text-tertiary)' }}>
                          {new Date(event.timestamp).toLocaleTimeString()}
                        </td>
                        <td className="py-2 pr-4 font-mono text-xs" style={{ color: 'var(--text-tertiary)' }}>
                          {event.session_id}
                        </td>
                        <td className="py-2 pr-4">
                          <RiskBadge level={event.risk_level} />
                        </td>
                        <td className="py-2 pr-4 capitalize">
                          {EMOTIONAL_STATE_LABELS[event.emotional_state]}
                        </td>
                        <td className="py-2 pr-4 font-mono">
                          {event.urgency_score}
                        </td>
                        <td className="py-2 pr-4">
                          <ModalityBadge modality={event.modality} />
                        </td>
                        <td className="py-2 max-w-xs truncate" style={{ color: 'var(--text-tertiary)' }}>
                          {event.text_snippet || '(hidden)'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Scenario Presets */}
          <div 
            className="rounded-lg p-6"
            style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-primary)' }}
          >
            <h2 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px', color: 'var(--text-primary)' }}>Test Scenarios</h2>
            <p className="text-sm mb-4" style={{ color: 'var(--text-tertiary)' }}>
              Quick presets for testing. Go to <a href="/" className="hover:underline" style={{ color: 'var(--info)' }}>Live Triage</a> and paste these messages.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {TRIAGE_SCENARIOS.map((scenario) => (
                <div
                  key={scenario.id}
                  className="p-3 rounded-lg transition-colors cursor-pointer"
                  style={{ 
                    background: 'var(--bg-tertiary)', 
                    border: '1px solid var(--border-primary)'
                  }}
                  onClick={() => {
                    navigator.clipboard.writeText(scenario.text);
                    alert('Copied to clipboard! Paste in Live Triage.');
                  }}
                >
                  <div className="text-xs mb-1" style={{ color: 'var(--text-tertiary)' }}>{scenario.label}</div>
                  <div className="text-sm line-clamp-2" style={{ color: 'var(--text-secondary)' }}>{scenario.text}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
