'use client';

import { useState, useEffect, useCallback } from 'react';

/**
 * CrisisTriage AI - Phone Calls Dashboard
 * 
 * Displays active and recent phone call sessions.
 * Allows simulating incoming calls for testing.
 */

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_HTTP_URL || 'http://localhost:8000';

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
  const [simulating, setSimulating] = useState(false);

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

  // Simulate incoming call
  const simulateCall = async () => {
    setSimulating(true);
    try {
      const response = await fetch(`${API_BASE}/api/telephony/simulate/incoming`, {
        method: 'POST',
      });
      if (response.ok) {
        const data = await response.json();
        console.log('Simulated call:', data);
        // Refresh calls
        await fetchActiveCalls();
      } else {
        const errData = await response.json();
        alert(`Failed to simulate call: ${errData.detail || response.statusText}`);
      }
    } catch (err) {
      console.error('Failed to simulate call:', err);
      alert('Failed to simulate call. Check console for details.');
    } finally {
      setSimulating(false);
    }
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
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold">📞 Phone Calls</h1>
            <p className="text-gray-400 mt-1">
              Monitor and manage phone call triage sessions
            </p>
          </div>
          <a 
            href="/"
            className="px-4 py-2 rounded bg-gray-700 hover:bg-gray-600 transition"
          >
            ← Back to Dashboard
          </a>
        </div>

        {/* Error State */}
        {error && (
          <div className="bg-red-900/50 border border-red-700 rounded-lg p-4 mb-6">
            <p className="text-red-300">{error}</p>
            <p className="text-sm text-gray-400 mt-2">
              To enable telephony, set <code className="bg-gray-800 px-1 rounded">ENABLE_TELEPHONY_INTEGRATION=true</code> in your backend .env file.
            </p>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="text-center py-12">
            <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto"></div>
            <p className="text-gray-400 mt-4">Loading telephony status...</p>
          </div>
        )}

        {/* Telephony Status */}
        {telephonyStatus && !loading && (
          <>
            {/* Status Bar */}
            <div className="bg-gray-800 rounded-lg p-4 mb-6 flex items-center justify-between">
              <div className="flex items-center gap-6">
                <div>
                  <span className="text-gray-400 text-sm">Status</span>
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                    <span className="font-medium text-green-400">Enabled</span>
                  </div>
                </div>
                <div>
                  <span className="text-gray-400 text-sm">Provider</span>
                  <p className="font-medium">{telephonyStatus.provider}</p>
                </div>
                <div>
                  <span className="text-gray-400 text-sm">Active Calls</span>
                  <p className="font-medium text-xl">{activeCalls.length}</p>
                </div>
                <div>
                  <span className="text-gray-400 text-sm">Max Concurrent</span>
                  <p className="font-medium">{telephonyStatus.max_concurrent_calls}</p>
                </div>
              </div>
              <button
                onClick={simulateCall}
                disabled={simulating}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed transition"
              >
                {simulating ? 'Simulating...' : '📱 Simulate Incoming Call'}
              </button>
            </div>

            {/* Active Calls */}
            <div className="mb-8">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                Active Calls ({activeCalls.length})
              </h2>
              {activeCalls.length === 0 ? (
                <div className="bg-gray-800/50 rounded-lg p-8 text-center">
                  <p className="text-gray-400">No active calls</p>
                  <p className="text-sm text-gray-500 mt-2">
                    Click "Simulate Incoming Call" to test the telephony integration
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
              <h2 className="text-xl font-semibold mb-4">Recent Calls</h2>
              {recentCalls.length === 0 ? (
                <div className="bg-gray-800/50 rounded-lg p-8 text-center">
                  <p className="text-gray-400">No recent calls</p>
                </div>
              ) : (
                <div className="bg-gray-800 rounded-lg overflow-hidden">
                  <table className="w-full">
                    <thead className="bg-gray-700">
                      <tr>
                        <th className="text-left p-3 font-medium">Call ID</th>
                        <th className="text-left p-3 font-medium">From</th>
                        <th className="text-left p-3 font-medium">Status</th>
                        <th className="text-left p-3 font-medium">Duration</th>
                        <th className="text-left p-3 font-medium">Events</th>
                        <th className="text-left p-3 font-medium">Risk</th>
                        <th className="text-left p-3 font-medium">Time</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-700">
                      {recentCalls.map((call) => (
                        <tr key={call.call_id_masked} className="hover:bg-gray-700/50">
                          <td className="p-3 font-mono text-sm">{call.call_id_masked}</td>
                          <td className="p-3 text-sm">{call.from_number}</td>
                          <td className={`p-3 text-sm ${getStatusColor(call.status)}`}>
                            {call.status}
                          </td>
                          <td className="p-3 text-sm">
                            {call.duration_seconds ? `${call.duration_seconds}s` : '-'}
                          </td>
                          <td className="p-3 text-sm">{call.triage_events}</td>
                          <td className="p-3">
                            <span className={`px-2 py-0.5 rounded text-xs font-medium ${getRiskColor(call.highest_risk)}`}>
                              {call.highest_risk || 'N/A'}
                            </span>
                          </td>
                          <td className="p-3 text-sm text-gray-400">
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
        <div className="mt-8 p-4 bg-amber-900/30 border border-amber-700 rounded-lg">
          <p className="text-amber-300 text-sm">
            ⚠️ <strong>Research System Only:</strong> This telephony integration is for research 
            and simulation purposes only. Phone numbers are masked and no audio is persisted.
          </p>
        </div>
      </div>
    </div>
  );
}
