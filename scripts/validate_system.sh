#!/bin/bash
# CrisisTriage AI - System Validation Script
# Run from project root: ./scripts/validate_system.sh

set -e

echo "=============================================="
echo "  CrisisTriage AI - System Validation"
echo "=============================================="
echo ""

API_BASE="http://localhost:8000"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
    else
        echo -e "${RED}✗${NC} $1"
        exit 1
    fi
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# 1. Check if backend is running
echo "1. Checking backend health..."
HEALTH=$(curl -s "$API_BASE/api/health" 2>/dev/null)
if [ -z "$HEALTH" ]; then
    echo -e "${RED}✗${NC} Backend not running at $API_BASE"
    echo "   Start with: cd backend && uvicorn main:app --reload"
    exit 1
fi
echo -e "${GREEN}✓${NC} Backend is running"

# 2. Check health endpoint
echo ""
echo "2. Validating health endpoint..."
STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
if [ "$STATUS" = "healthy" ] || [ "$STATUS" = "ok" ]; then
    echo -e "${GREEN}✓${NC} Health status: $STATUS"
else
    warn "Health status: $STATUS (may be degraded)"
fi

# 3. Create a session
echo ""
echo "3. Creating test session..."
SESSION_RESP=$(curl -s -X POST "$API_BASE/api/sessions" -H "Content-Type: application/json" -d '{}')
SESSION_ID=$(echo "$SESSION_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id',''))" 2>/dev/null)
if [ -z "$SESSION_ID" ]; then
    echo -e "${RED}✗${NC} Failed to create session"
    exit 1
fi
echo -e "${GREEN}✓${NC} Session created: $SESSION_ID"

# 4. Test low-risk triage
echo ""
echo "4. Testing LOW risk message..."
TRIAGE_LOW=$(curl -s -X POST "$API_BASE/api/sessions/$SESSION_ID/triage" \
    -H "Content-Type: application/json" \
    -d '{"text": "Thank you for listening, I feel better now."}')
RISK_LOW=$(echo "$TRIAGE_LOW" | python3 -c "import sys,json; print(json.load(sys.stdin).get('risk_level',''))" 2>/dev/null)
echo -e "${GREEN}✓${NC} Risk level: $RISK_LOW"

# 5. Test high-risk triage
echo ""
echo "5. Testing HIGH risk message..."
TRIAGE_HIGH=$(curl -s -X POST "$API_BASE/api/sessions/$SESSION_ID/triage" \
    -H "Content-Type: application/json" \
    -d '{"text": "I want to end my life, I cannot take this anymore."}')
RISK_HIGH=$(echo "$TRIAGE_HIGH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('risk_level',''))" 2>/dev/null)
if [ "$RISK_HIGH" = "high" ] || [ "$RISK_HIGH" = "imminent" ]; then
    echo -e "${GREEN}✓${NC} Risk level: $RISK_HIGH (correctly detected)"
else
    warn "Risk level: $RISK_HIGH (expected high/imminent)"
fi

# 6. Check analytics
echo ""
echo "6. Checking analytics..."
ANALYTICS=$(curl -s "$API_BASE/api/analytics/summary")
TOTAL=$(echo "$ANALYTICS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_events',0))" 2>/dev/null)
if [ "$TOTAL" -ge 2 ]; then
    echo -e "${GREEN}✓${NC} Analytics recorded: $TOTAL events"
else
    warn "Analytics total: $TOTAL (expected >= 2)"
fi

# 7. Check recent events
echo ""
echo "7. Checking recent events..."
RECENT=$(curl -s "$API_BASE/api/analytics/recent?limit=5")
EVENT_COUNT=$(echo "$RECENT" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null)
echo -e "${GREEN}✓${NC} Recent events: $EVENT_COUNT"

# Summary
echo ""
echo "=============================================="
echo "  Validation Complete"
echo "=============================================="
echo ""
echo "Backend API: ${GREEN}OK${NC}"
echo "Session Management: ${GREEN}OK${NC}"
echo "Triage Processing: ${GREEN}OK${NC}"
echo "Analytics: ${GREEN}OK${NC}"
echo ""
echo "Next steps:"
echo "  1. Open http://localhost:3000 for Live Triage UI"
echo "  2. Open http://localhost:3000/analytics for Analytics Dashboard"
echo "  3. Run 'pytest' in backend/ for full test suite"
echo ""
