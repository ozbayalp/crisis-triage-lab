'use client';

import { useState, useEffect, useCallback } from 'react';
import { getBackendHttpUrl } from '@/lib/config';

/**
 * CrisisTriage AI - Phone Calls Dashboard
 * 
 * Displays active and recent phone call sessions.
 * Phone integration requires Twilio configuration.
 */

const API_BASE = getBackendHttpUrl();

interface CallSession {
  call_id_masked: string;
  session_id: string;
  status: string;
  from_number: string;
  to_number: string;
  direction: string;
  initiated_at: string;
  started_at: string | null;
  ended_at: string | null;
  duration_seconds: number | null;
  triage_events: number;
  highest_risk: string | null;
}

interface TelephonyStatus {
  enabled: boolean;
  provider: string;
  max_concurrent_calls: number;
  audio_sample_rate: number;
}

export default function CallsPage() {
  const [activeCalls, setActiveCalls] = useState<CallSession[]>([]);
  const [recentCalls, setRecentCalls] = useState<CallSession[]>([]);
  const [telephonyStatus, setTelephonyStatus] = useState<TelephonyStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connectEnabled, setConnectEnabled] = useState(false);

  // Fetch telephony status
  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/system/telephony_status`);
      if (response.ok) {
        const data = await response.json();
        setTelephonyStatus(data);
        setError(null);
      } else if (response.status === 503) {
        setError('Telephony integration is disabled on the backend.');
        setTelephonyStatus(null);
      }
    } catch (err) {
      setError('Failed to connect to backend. Is it running?');
      setTelephonyStatus(null);
    }
  }, []);

  // Fetch active calls
  const fetchActiveCalls = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/telephony/calls/active`);
      if (response.ok) {
        const data = await response.json();
        setActiveCalls(data.calls || []);
      }
    } catch (err) {
      console.error('Failed to fetch active calls:', err);
    }
  }, []);

  // Fetch recent calls
  const fetchRecentCalls = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/telephony/calls/recent?limit=20`);
      if (response.ok) {
        const data = await response.json();
        setRecentCalls(data.calls || []);
      }
    } catch (err) {
      console.error('Failed to fetch recent calls:', err);
    }
  }, []);

  // Toggle connect call panel
  const toggleConnect = () => {
    setConnectEnabled(!connectEnabled);
  };

  // Initial load and polling
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await fetchStatus();
      await fetchActiveCalls();
      await fetchRecentCalls();
      setLoading(false);
    };

    loadData();

    // Poll for updates every 5 seconds
    const interval = setInterval(() => {
      fetchActiveCalls();
      fetchRecentCalls();
    }, 5000);

    return () => clearInterval(interval);
  }, [fetchStatus, fetchActiveCalls, fetchRecentCalls]);

  // Risk level colors
  const getRiskColor = (risk: string | null) => {
    switch (risk?.toLowerCase()) {
      case 'imminent': return 'text-red-500 bg-red-500/20';
      case 'high': return 'text-orange-500 bg-orange-500/20';
      case 'medium': return 'text-yellow-500 bg-yellow-500/20';
      case 'low': return 'text-green-500 bg-green-500/20';
      default: return 'text-gray-400 bg-gray-500/20';
    }
  };

  // Status colors
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'in_progress': return 'text-green-400';
      case 'ringing': return 'text-yellow-400';
      case 'completed': return 'text-blue-400';
      case 'failed': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  return (
    <div style={{ background: 'var(--bg-primary)', color: 'var(--text-primary)', minHeight: '100vh' }}>
      <div className="py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-12">
          <div>
            <h1 style={{ fontSize: '32px', fontWeight: 600, letterSpacing: '-0.01em' }}>Phone Calls</h1>
            <p style={{ color: 'var(--text-secondary)', marginTop: '4px', fontSize: '14px' }}>
              Monitor and manage phone call triage sessions
            </p>
          </div>
          <a 
            href="/"
            className="px-4 py-2 rounded-lg text-sm font-medium transition-colors"
            style={{ 
              background: 'var(--accent)', 
              color: 'var(--bg-primary)',
            }}
          >
            Back to Dashboard
          </a>
        </div>

        {/* Error State */}
        {error && (
          <div 
            className="rounded-lg p-4 mb-8"
            style={{ background: 'var(--danger)', opacity: 0.1, border: '1px solid var(--danger)' }}
          >
            <p style={{ color: 'var(--danger)' }}>{error}</p>
            <p style={{ fontSize: '13px', color: 'var(--text-tertiary)', marginTop: '8px' }}>
              To enable telephony, set <code style={{ background: 'var(--bg-tertiary)', padding: '2px 4px', borderRadius: '4px' }}>ENABLE_TELEPHONY_INTEGRATION=true</code> in your backend .env file.
            </p>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="text-center py-16">
            <div 
              className="animate-spin w-8 h-8 border-2 border-t-transparent rounded-full mx-auto"
              style={{ borderColor: 'var(--accent)', borderTopColor: 'transparent' }}
            ></div>
            <p style={{ color: 'var(--text-tertiary)', marginTop: '16px' }}>Loading telephony status...</p>
          </div>
        )}

        {/* Telephony Status */}
        {telephonyStatus && !loading && (
          <>
            {/* Status Bar */}
            <div 
              className="rounded-lg p-5 mb-8 flex items-center justify-between"
              style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-primary)' }}
            >
              <div className="flex items-center gap-8">
                <div>
                  <span style={{ color: 'var(--text-tertiary)', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Status</span>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="w-2 h-2 rounded-full" style={{ background: 'var(--success)' }}></span>
                    <span className="font-medium" style={{ color: 'var(--success)' }}>Enabled</span>
                  </div>
                </div>
                <div>
                  <span style={{ color: 'var(--text-tertiary)', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Provider</span>
                  <p className="font-medium mt-1">{telephonyStatus.provider === 'generic' ? 'Unknown' : telephonyStatus.provider}</p>
                </div>
                <div>
                  <span style={{ color: 'var(--text-tertiary)', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Active Calls</span>
                  <p className="font-medium text-xl mt-1">{activeCalls.length}</p>
                </div>
                <div>
                  <span style={{ color: 'var(--text-tertiary)', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Max Concurrent</span>
                  <p className="font-medium mt-1">{telephonyStatus.max_concurrent_calls}</p>
                </div>
              </div>
              <button
                onClick={toggleConnect}
                className="px-4 py-2 rounded-lg font-medium text-sm transition-colors"
                style={{ 
                  background: connectEnabled ? 'var(--danger)' : 'var(--success)', 
                  color: '#FFFFFF' 
                }}
              >
                {connectEnabled ? 'Disconnect' : 'Connect Call'}
              </button>
            </div>

            {/* Twilio Notice - Shows when Connect is toggled */}
            {connectEnabled && (
              <div 
                className="rounded-lg p-6 mb-8"
                style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--warning)' }}
              >
                <h3 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--warning)', marginBottom: '8px' }}>
                  Phone Call Integration Unavailable
                </h3>
                <p style={{ color: 'var(--text-secondary)', marginBottom: '12px', fontSize: '14px' }}>
                  Real phone call integration requires a <strong>Twilio account</strong> with 
                  an active subscription. Twilio is a paid cloud communications platform that 
                  provides programmable voice, SMS, and video APIs.
                </p>
                <div 
                  className="rounded p-4 mb-3"
                  style={{ background: 'var(--bg-secondary)' }}
                >
                  <p style={{ fontSize: '13px', color: 'var(--text-tertiary)', marginBottom: '8px' }}>To enable phone call functionality:</p>
                  <ol style={{ fontSize: '13px', color: 'var(--text-secondary)' }} className="list-decimal list-inside space-y-1">
                    <li>Sign up for a Twilio account at <span style={{ color: 'var(--info)' }}>twilio.com</span></li>
                    <li>Purchase a phone number (~$1/month)</li>
                    <li>Add your Twilio credentials to the backend <code style={{ background: 'var(--bg-tertiary)', padding: '2px 6px', borderRadius: '4px' }}>.env</code> file</li>
                    <li>Configure the webhook URL to point to this server</li>
                  </ol>
                </div>
                <p style={{ fontSize: '13px', color: 'var(--text-tertiary)' }}>
                  The backend telephony infrastructure is fully implemented and ready to receive 
                  calls once Twilio is configured. This includes real-time audio streaming, 
                  transcription, and triage analysis.
                </p>
              </div>
            )}

            {/* Active Calls */}
            <div className="mb-12">
              <h2 style={{ fontSize: '20px', fontWeight: 600, marginBottom: '16px' }} className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full" style={{ background: 'var(--success)' }}></span>
                Active Calls ({activeCalls.length})
              </h2>
              {activeCalls.length === 0 ? (
                <div 
                  className="rounded-lg p-12 text-center"
                  style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-primary)' }}
                >
                  <p style={{ color: 'var(--text-secondary)' }}>No active calls</p>
                  <p style={{ fontSize: '13px', color: 'var(--text-tertiary)', marginTop: '8px' }}>
                    Click "Connect Call" above to learn about phone integration
                  </p>
                </div>
              ) : (
                <div className="grid gap-4">
                  {activeCalls.map((call) => (
                    <div 
                      key={call.call_id_masked}
                      className="bg-gray-800 rounded-lg p-4 border border-gray-700"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                          <div className="w-12 h-12 bg-green-500/20 rounded-full flex items-center justify-center">
                            <span className="text-2xl">📞</span>
                          </div>
                          <div>
                            <p className="font-medium">Call {call.call_id_masked}</p>
                            <p className="text-sm text-gray-400">
                              From: {call.from_number} → To: {call.to_number}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-6">
                          <div className="text-center">
                            <p className="text-sm text-gray-400">Status</p>
                            <p className={`font-medium ${getStatusColor(call.status)}`}>
                              {call.status.replace('_', ' ')}
                            </p>
                          </div>
                          <div className="text-center">
                            <p className="text-sm text-gray-400">Triage Events</p>
                            <p className="font-medium">{call.triage_events}</p>
                          </div>
                          <div className="text-center">
                            <p className="text-sm text-gray-400">Risk Level</p>
                            <span className={`px-2 py-1 rounded text-sm font-medium ${getRiskColor(call.highest_risk)}`}>
                              {call.highest_risk || 'N/A'}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Recent Calls */}
            <div>
              <h2 style={{ fontSize: '20px', fontWeight: 600, marginBottom: '16px' }} className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full" style={{ background: 'var(--danger)' }}></span>
                Recent Calls
              </h2>
              {recentCalls.length === 0 ? (
                <div 
                  className="rounded-lg p-12 text-center"
                  style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-primary)' }}
                >
                  <p style={{ color: 'var(--text-secondary)' }}>No recent calls</p>
                </div>
              ) : (
                <div 
                  className="rounded-lg overflow-hidden"
                  style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-primary)' }}
                >
                  <table className="w-full">
                    <thead style={{ background: 'var(--bg-tertiary)' }}>
                      <tr>
                        <th className="text-left p-3" style={{ fontSize: '12px', fontWeight: 500, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Call ID</th>
                        <th className="text-left p-3" style={{ fontSize: '12px', fontWeight: 500, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>From</th>
                        <th className="text-left p-3" style={{ fontSize: '12px', fontWeight: 500, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Status</th>
                        <th className="text-left p-3" style={{ fontSize: '12px', fontWeight: 500, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Duration</th>
                        <th className="text-left p-3" style={{ fontSize: '12px', fontWeight: 500, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Events</th>
                        <th className="text-left p-3" style={{ fontSize: '12px', fontWeight: 500, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Risk</th>
                        <th className="text-left p-3" style={{ fontSize: '12px', fontWeight: 500, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Time</th>
                      </tr>
                    </thead>
                    <tbody style={{ borderTop: '1px solid var(--border-primary)' }}>
                      {recentCalls.map((call, idx) => (
                        <tr 
                          key={call.call_id_masked} 
                          className="transition-colors"
                          style={{ 
                            borderBottom: idx < recentCalls.length - 1 ? '1px solid var(--border-primary)' : 'none',
                            background: 'var(--bg-secondary)'
                          }}
                        >
                          <td className="p-3 font-mono" style={{ fontSize: '13px' }}>{call.call_id_masked}</td>
                          <td className="p-3" style={{ fontSize: '13px' }}>{call.from_number}</td>
                          <td className={`p-3 ${getStatusColor(call.status)}`} style={{ fontSize: '13px' }}>
                            {call.status}
                          </td>
                          <td className="p-3" style={{ fontSize: '13px' }}>
                            {call.duration_seconds ? `${call.duration_seconds}s` : '-'}
                          </td>
                          <td className="p-3" style={{ fontSize: '13px' }}>{call.triage_events}</td>
                          <td className="p-3">
                            <span className={`px-2 py-0.5 rounded text-xs font-medium ${getRiskColor(call.highest_risk)}`}>
                              {call.highest_risk || 'N/A'}
                            </span>
                          </td>
                          <td className="p-3" style={{ fontSize: '13px', color: 'var(--text-tertiary)' }}>
                            {new Date(call.initiated_at).toLocaleTimeString()}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </>
        )}

        {/* Safety Notice */}
        <div 
          className="mt-12 p-4 rounded-lg"
          style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--border-primary)' }}
        >
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            <strong>Research System Only:</strong> This telephony integration is for research 
            and simulation purposes only. Phone numbers are masked and no audio is persisted.
          </p>
        </div>
      </div>
    </div>
  );
}
