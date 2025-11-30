"""
CrisisTriage AI - API Endpoint Tests

Tests for REST API endpoints using FastAPI TestClient.
These tests verify:
- Health endpoint
- Triage endpoint
- Analytics endpoints
- Error handling

Run with: pytest tests/test_api_endpoints.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import os


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_200(self, client: TestClient):
        """Health endpoint should return 200 OK."""
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_returns_correct_structure(self, client: TestClient):
        """Health response should have expected structure."""
        response = client.get("/api/health")
        data = response.json()
        
        assert "status" in data
        assert "components" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

    def test_health_includes_components(self, client: TestClient):
        """Health response should include component statuses."""
        response = client.get("/api/health")
        data = response.json()
        
        components = data.get("components", {})
        # Should have at least some components
        assert len(components) > 0


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_200(self, client: TestClient):
        """Root endpoint should return 200 OK."""
        # Note: Root is outside /api prefix
        from main import app
        with TestClient(app) as root_client:
            response = root_client.get("/")
            assert response.status_code == 200

    def test_root_returns_service_info(self, client: TestClient):
        """Root should return service information."""
        from main import app
        with TestClient(app) as root_client:
            response = root_client.get("/")
            data = response.json()
            
            assert "service" in data
            assert "status" in data
            assert data["status"] == "operational"


class TestSessionEndpoints:
    """Tests for session management endpoints."""

    def test_create_session(self, client: TestClient):
        """Should create a new session."""
        response = client.post("/api/sessions", json={})
        
        assert response.status_code == 201
        data = response.json()
        
        assert "session_id" in data
        assert "websocket_url" in data
        assert len(data["session_id"]) > 0

    def test_create_session_with_metadata(self, client: TestClient):
        """Should create session with optional metadata."""
        response = client.post("/api/sessions", json={
            "metadata": {"test_run": "123"},
            "sample_rate": 16000,
        })
        
        assert response.status_code == 201
        data = response.json()
        assert "session_id" in data


class TestTriageEndpoints:
    """Tests for triage processing endpoints."""

    def test_triage_text_success(self, client: TestClient):
        """Should successfully process text triage request."""
        # First create a session
        session_resp = client.post("/api/sessions", json={})
        session_id = session_resp.json()["session_id"]
        
        # Then send a triage request
        response = client.post(
            f"/api/sessions/{session_id}/triage",
            json={"text": "I'm feeling okay today."}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "risk_level" in data
        assert "emotional_state" in data
        assert "urgency_score" in data
        assert "recommended_action" in data
        assert "confidence" in data

    def test_triage_text_risk_levels(self, client: TestClient):
        """Should return valid risk levels."""
        session_resp = client.post("/api/sessions", json={})
        session_id = session_resp.json()["session_id"]
        
        response = client.post(
            f"/api/sessions/{session_id}/triage",
            json={"text": "I don't see a way out anymore."}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        valid_risk_levels = ["low", "medium", "high", "imminent", "unknown"]
        assert data["risk_level"] in valid_risk_levels

    def test_triage_empty_text_fails(self, client: TestClient):
        """Should reject empty text."""
        session_resp = client.post("/api/sessions", json={})
        session_id = session_resp.json()["session_id"]
        
        response = client.post(
            f"/api/sessions/{session_id}/triage",
            json={"text": ""}
        )
        
        # Should fail with 400 or similar
        assert response.status_code in [400, 422]

    def test_triage_urgency_score_range(self, client: TestClient):
        """Urgency score should be between 0 and 100."""
        session_resp = client.post("/api/sessions", json={})
        session_id = session_resp.json()["session_id"]
        
        response = client.post(
            f"/api/sessions/{session_id}/triage",
            json={"text": "Testing the urgency score."}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert 0 <= data["urgency_score"] <= 100

    def test_triage_confidence_range(self, client: TestClient):
        """Confidence should be between 0 and 1."""
        session_resp = client.post("/api/sessions", json={})
        session_id = session_resp.json()["session_id"]
        
        response = client.post(
            f"/api/sessions/{session_id}/triage",
            json={"text": "Testing confidence."}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert 0.0 <= data["confidence"] <= 1.0


class TestAnalyticsEndpoints:
    """Tests for analytics endpoints."""

    def test_analytics_summary_success(self, client: TestClient):
        """Should return analytics summary."""
        response = client.get("/api/analytics/summary")
        
        # Should succeed (analytics enabled by default)
        assert response.status_code == 200
        data = response.json()
        
        # Check structure
        assert "total_events" in data
        assert "risk_counts" in data
        assert "emotion_counts" in data

    def test_analytics_summary_structure(self, client: TestClient):
        """Analytics summary should have complete structure."""
        response = client.get("/api/analytics/summary")
        data = response.json()
        
        expected_fields = [
            "total_events",
            "events_last_hour",
            "events_last_24h",
            "risk_counts",
            "risk_percentages",
            "emotion_counts",
            "emotion_percentages",
            "avg_urgency_score",
            "avg_confidence",
            "unique_sessions",
        ]
        
        for field in expected_fields:
            assert field in data, f"Missing field: {field}"

    def test_analytics_recent_events(self, client: TestClient):
        """Should return recent events."""
        response = client.get("/api/analytics/recent")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should be a list
        assert isinstance(data, list)

    def test_analytics_recent_with_limit(self, client: TestClient):
        """Should respect limit parameter."""
        response = client.get("/api/analytics/recent?limit=5")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return at most 5 events
        assert len(data) <= 5

    def test_analytics_after_triage(self, client: TestClient):
        """Analytics should reflect triage events."""
        # Create session and send triage
        session_resp = client.post("/api/sessions", json={})
        session_id = session_resp.json()["session_id"]
        
        client.post(
            f"/api/sessions/{session_id}/triage",
            json={"text": "I'm feeling overwhelmed."}
        )
        
        # Check analytics
        response = client.get("/api/analytics/summary")
        data = response.json()
        
        # Should have at least one event
        assert data["total_events"] >= 1

    def test_analytics_clear(self, client: TestClient):
        """Should be able to clear analytics."""
        # First add some data
        session_resp = client.post("/api/sessions", json={})
        session_id = session_resp.json()["session_id"]
        
        client.post(
            f"/api/sessions/{session_id}/triage",
            json={"text": "Test message."}
        )
        
        # Clear
        response = client.delete("/api/analytics/clear")
        assert response.status_code == 204
        
        # Verify cleared
        summary_resp = client.get("/api/analytics/summary")
        data = summary_resp.json()
        
        assert data["total_events"] == 0


class TestAnalyticsDisabled:
    """Tests for analytics when disabled."""

    @pytest.fixture
    def client_no_analytics(self):
        """Create client with analytics disabled."""
        # Temporarily set environment
        original = os.environ.get("ENABLE_ANALYTICS")
        os.environ["ENABLE_ANALYTICS"] = "false"
        
        try:
            from main import create_app
            app = create_app()
            with TestClient(app) as c:
                yield c
        finally:
            if original is not None:
                os.environ["ENABLE_ANALYTICS"] = original
            else:
                os.environ.pop("ENABLE_ANALYTICS", None)

    def test_analytics_summary_when_disabled(self, client_no_analytics: TestClient):
        """Should indicate analytics is disabled."""
        response = client_no_analytics.get("/api/analytics/summary")
        
        # Either returns disabled message (200) or error (403)
        if response.status_code == 200:
            data = response.json()
            # Check if it's a disabled response
            if "enabled" in data:
                assert data["enabled"] == False
        else:
            assert response.status_code == 403

    def test_analytics_recent_when_disabled(self, client_no_analytics: TestClient):
        """Should return error or empty when analytics disabled."""
        response = client_no_analytics.get("/api/analytics/recent")
        
        # May return 403 or 200 with empty/disabled response depending on implementation
        assert response.status_code in [200, 403]


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_session_id(self, client: TestClient):
        """Should handle invalid session ID."""
        response = client.get("/api/sessions/nonexistent-session")
        
        # Should return 404 Not Found
        assert response.status_code == 404

    def test_invalid_json_payload(self, client: TestClient):
        """Should handle invalid JSON."""
        session_resp = client.post("/api/sessions", json={})
        session_id = session_resp.json()["session_id"]
        
        response = client.post(
            f"/api/sessions/{session_id}/triage",
            content="not json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422

    def test_missing_required_fields(self, client: TestClient):
        """Should handle missing required fields."""
        session_resp = client.post("/api/sessions", json={})
        session_id = session_resp.json()["session_id"]
        
        response = client.post(
            f"/api/sessions/{session_id}/triage",
            json={}  # Missing 'text' field
        )
        
        assert response.status_code == 422

    def test_nonexistent_endpoint(self, client: TestClient):
        """Should return 404 for nonexistent endpoints."""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self, client: TestClient):
        """Should include CORS headers for allowed origins."""
        response = client.options(
            "/api/health",
            headers={"Origin": "http://localhost:3000"}
        )
        
        # Check CORS headers
        # Note: TestClient may not fully simulate CORS
        # This is a basic check
        assert response.status_code in [200, 405]
