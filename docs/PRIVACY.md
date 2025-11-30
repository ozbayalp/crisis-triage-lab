# CrisisTriage AI — Privacy & Data Handling

> Privacy principles, data handling policies, and security considerations.

## Table of Contents

1. [Privacy Principles](#privacy-principles)
2. [Data Classification](#data-classification)
3. [Data Lifecycle](#data-lifecycle)
4. [Consent Model](#consent-model)
5. [Technical Safeguards](#technical-safeguards)
6. [Configuration Reference](#configuration-reference)
7. [Compliance Considerations](#compliance-considerations)
8. [Incident Response](#incident-response)

---

## Privacy Principles

CrisisTriage AI is designed with privacy as a foundational requirement, not an afterthought.

### 1. Local-First Processing

All inference runs locally or on self-hosted infrastructure. No raw audio, transcripts, or identifiable data is transmitted to external services.

### 2. Ephemeral by Default

Raw data (audio, transcripts) exists only in memory during processing and is discarded immediately after. Persistence is opt-in and requires explicit configuration.

### 3. Minimal Data Collection

We extract only the features necessary for triage. Raw representations are transformed into derived features that cannot be reversed to reconstruct the original data.

### 4. Purpose Limitation

Data is used solely for real-time triage assessment. No secondary use (marketing, profiling, training external models) without explicit consent and ethical review.

### 5. Transparency

Every triage decision includes an explanation of contributing features. Operators can understand *why* the system made a recommendation.

---

## Data Classification

| Data Type | Sensitivity Level | Contains PII? | Default Handling |
|-----------|-------------------|---------------|------------------|
| Raw audio | **CRITICAL** | Yes (voice biometrics) | Memory only, immediate deletion |
| Transcripts | **HIGH** | Potentially | Memory only, opt-in persistence |
| Prosody features | **MEDIUM** | Indirect | Ephemeral, opt-in persistence |
| Text embeddings | **MEDIUM** | Indirect | Ephemeral, opt-in persistence |
| Triage outputs | **LOW** | No (aggregated) | Session-scoped, configurable TTL |
| Session metadata | **LOW** | No (if properly designed) | Configurable retention |

### What We Consider PII

- Voice biometrics (raw audio waveforms)
- Named entities in transcripts (names, locations, etc.)
- Content that could identify an individual in context

### What We Do NOT Consider PII

- Anonymized prosody statistics (pitch mean, speech rate)
- Semantic embeddings (non-reversible representations)
- Aggregated triage scores without session linkage

---

## Data Lifecycle

### During a Session

```
┌────────────────────────────────────────────────────────────────┐
│                      ACTIVE SESSION                            │
│                                                                │
│  Audio Chunk ──▶ [Memory Buffer] ──▶ Process ──▶ Delete       │
│                        │                                       │
│                        ▼                                       │
│                 [Derived Features] ──▶ Triage ──▶ Output      │
│                        │                                       │
│                        ▼                                       │
│            [Optional: Feature Cache in Redis]                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### After Session Ends

```
┌────────────────────────────────────────────────────────────────┐
│                      SESSION CLEANUP                           │
│                                                                │
│  1. Raw audio buffers      → DELETED (always)                 │
│  2. Transcript segments    → DELETED (unless PERSIST=true)    │
│  3. Prosody features       → DELETED (unless PERSIST=true)    │
│  4. Triage outputs         → DELETED after TTL (configurable) │
│  5. Session metadata       → DELETED after TTL (configurable) │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Retention Periods

| Data Type | Default TTL | Configurable? | Maximum Recommended |
|-----------|-------------|---------------|---------------------|
| Raw audio | 0 (immediate) | No | N/A (never persist) |
| Transcripts | 0 (immediate) | Yes | 24 hours |
| Features | 0 (immediate) | Yes | 7 days |
| Triage outputs | Session end | Yes | 30 days |
| Aggregate metrics | Indefinite | Yes | Indefinite (anonymized) |

---

## Consent Model

### For Research/Simulation Use

When using CrisisTriage AI for research with simulated data:

- ✅ Synthetic or consented audio data only
- ✅ Clear documentation that data is simulated
- ✅ No real crisis caller data without IRB approval

### For Production Use (Future)

If deploying with real conversations:

1. **Informed Consent**: Callers must be informed that AI analysis is in use
2. **Opt-Out**: Mechanism for callers to decline AI-assisted triage
3. **Purpose Disclosure**: Clear explanation of how data is used
4. **Data Rights**: Access, correction, and deletion rights
5. **Ethical Review**: IRB or ethics board approval required

### Consent Implementation

```
┌─────────────────────────────────────────────────────────────────┐
│                     CONSENT FLOW (Future)                       │
│                                                                 │
│  Caller ──▶ "This call uses AI-assisted triage. Press 1 to    │
│             continue or 2 to speak with a human directly."     │
│                            │                                    │
│              ┌─────────────┴─────────────┐                     │
│              ▼                           ▼                      │
│        [AI Triage ON]              [AI Triage OFF]             │
│        Normal flow                  Bypass pipeline            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Audio Processing Privacy

### Local-Only Speech Recognition

CrisisTriage AI uses **OpenAI Whisper running locally** for speech-to-text. This ensures:

- ✅ **No external API calls**: Audio data never leaves your infrastructure
- ✅ **No cloud transmission**: All processing happens on your local/controlled servers
- ✅ **No third-party storage**: Audio is not sent to OpenAI or any external service
- ✅ **Immediate processing**: Audio is transcribed in real-time and discarded

### Audio Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AUDIO PRIVACY ARCHITECTURE                        │
│                                                                      │
│  Browser Microphone                                                  │
│        │                                                             │
│        ▼ (WebSocket, TLS encrypted)                                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                 LOCAL BACKEND SERVER                         │    │
│  │                                                              │    │
│  │   Audio Buffer (memory only)                                 │    │
│  │        │                                                     │    │
│  │        ├──▶ Whisper (LOCAL) ──▶ Transcript ──▶ Delete       │    │
│  │        │                                                     │    │
│  │        └──▶ Librosa (LOCAL) ──▶ Prosody Features ──▶ Delete │    │
│  │                                                              │    │
│  │   ❌ NO external API calls                                   │    │
│  │   ❌ NO audio stored to disk (by default)                   │    │
│  │   ❌ NO audio sent to cloud services                        │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Audio Processing Configuration

```bash
# Backend service selection (all run locally)
TRANSCRIPTION_BACKEND=whisper    # Local Whisper model
PROSODY_BACKEND=librosa          # Local feature extraction
TRIAGE_MODEL_BACKEND=neural      # Local neural network

# Whisper model selection (smaller = faster, larger = more accurate)
WHISPER_MODEL_NAME=base          # tiny, base, small, medium, large
WHISPER_LANGUAGE=en              # Force English

# Privacy flags
STORE_AUDIO=false                # Never persist raw audio
STORE_RAW_TRANSCRIPTS=false      # Don't log transcript content
ANONYMIZE_LOGS=true              # Minimize identifying info in logs
```

### What Happens to Your Voice

1. **Capture**: Browser records audio as PCM 16-bit, 16kHz
2. **Transmit**: Sent over encrypted WebSocket to local backend
3. **Buffer**: Held in memory for ~1 second chunks
4. **Process**: Whisper transcribes, Librosa extracts features
5. **Delete**: Audio bytes are immediately discarded
6. **Result**: Only text transcript and aggregate features remain

---

## Technical Safeguards

### Network Isolation

```yaml
# docker-compose.yml security settings
services:
  backend:
    networks:
      - internal    # No external network access
    # No ports exposed to host except API
```

### Memory Protection

- Audio buffers use secure memory allocation where available
- Explicit zeroing of memory after use
- No swap for sensitive data (mlockall where supported)

### Encryption

| Data State | Encryption |
|------------|------------|
| In transit (API) | TLS 1.3 required |
| In transit (WS) | WSS required in production |
| At rest (if persisted) | AES-256 encryption |
| In memory | Platform memory protection |

### Access Control

- API authentication required (future: JWT/OAuth2)
- Role-based access (operator, supervisor, admin)
- Audit logging for all data access

---

## Configuration Reference

### Environment Variables

```bash
# ============================================
# Privacy Configuration
# ============================================

# Raw audio persistence (STRONGLY RECOMMENDED: false)
PERSIST_RAW_AUDIO=false

# Transcript persistence
PERSIST_TRANSCRIPTS=false

# Feature persistence
PERSIST_FEATURES=false

# Session data TTL (0 = delete immediately after session)
SESSION_TTL_SECONDS=0

# Enable audit logging
AUDIT_LOG_ENABLED=true
AUDIT_LOG_PATH=/var/log/crisis-triage/audit.log

# PII detection and redaction
PII_REDACTION_ENABLED=true
PII_REDACTION_LEVEL=aggressive  # conservative | moderate | aggressive
```

### Privacy Levels

| Level | Raw Audio | Transcripts | Features | Outputs | Use Case |
|-------|-----------|-------------|----------|---------|----------|
| **Maximum** | ❌ | ❌ | ❌ | Session only | Real-time only |
| **Standard** | ❌ | ❌ | ✅ (7d) | ✅ (30d) | Research analysis |
| **Research** | ❌ | ✅ (24h) | ✅ (30d) | ✅ (90d) | Model improvement |

---

## Compliance Considerations

### HIPAA (US Healthcare)

If used in a healthcare context:

- ✅ Business Associate Agreement (BAA) required
- ✅ Minimum necessary standard applies
- ✅ Audit controls required
- ⚠️ Consult healthcare compliance officer

### GDPR (EU)

If processing data from EU subjects:

- ✅ Legal basis for processing (consent or legitimate interest)
- ✅ Data minimization principle
- ✅ Right to erasure (deletion on request)
- ✅ Data Protection Impact Assessment (DPIA) recommended

### CCPA (California)

If processing California resident data:

- ✅ Notice at collection
- ✅ Right to know / access
- ✅ Right to delete
- ✅ Right to opt-out of sale (N/A - we don't sell data)

---

## Incident Response

### Data Breach Protocol

1. **Detect**: Monitor for unauthorized access attempts
2. **Contain**: Isolate affected systems immediately
3. **Assess**: Determine scope and data affected
4. **Notify**: Inform affected parties per legal requirements
5. **Remediate**: Fix vulnerability, enhance controls
6. **Document**: Full incident report for compliance

### Contact

For privacy concerns or data requests:

- **Internal**: [privacy@your-org.com]
- **DPO**: [dpo@your-org.com] (if applicable)

---

## Analytics & Metrics

### What Analytics Collects

When `ENABLE_ANALYTICS=true`, the system stores aggregate triage events for research evaluation:

| Data Type | Stored | Notes |
|-----------|--------|-------|
| Timestamp | ✅ | When the triage occurred |
| Session ID | ✅ (truncated) | First 8 chars only, anonymized |
| Risk Level | ✅ | low/medium/high/imminent |
| Emotional State | ✅ | calm/anxious/distressed/panicked |
| Urgency Score | ✅ | 0-100 numeric score |
| Confidence | ✅ | Model confidence 0-1 |
| Modality | ✅ | text/audio/mixed |
| Processing Time | ✅ | Latency in milliseconds |
| Text Snippet | ⚠️ | Only if `STORE_ANALYTICS_TEXT_SNIPPETS=true` |

### Data Storage

- **In-Memory**: Default implementation stores events in a bounded buffer
- **Maximum Events**: Configurable via `ANALYTICS_MAX_EVENTS` (default: 10,000)
- **Persistence**: Data is **ephemeral** - lost on server restart
- **No Redis**: Currently in-memory only; Redis backend available for future

### Privacy Controls

```bash
# Enable/disable analytics entirely
ENABLE_ANALYTICS=true

# Control text snippet storage (sensitive!)
STORE_ANALYTICS_TEXT_SNIPPETS=false

# Maximum events in memory
ANALYTICS_MAX_EVENTS=10000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analytics/summary` | GET | Aggregated statistics |
| `/api/analytics/recent?limit=100` | GET | Recent events list |
| `/api/analytics/clear` | DELETE | Clear all stored events |

All endpoints respect `ENABLE_ANALYTICS` flag and return 403 if disabled.

### Text Snippet Behavior

When `STORE_ANALYTICS_TEXT_SNIPPETS=false` (default):
- Raw text is never stored in analytics events
- Example snippets in risk breakdown are empty
- API returns `null` for `text_snippet` field

When `STORE_ANALYTICS_TEXT_SNIPPETS=true`:
- First 100 characters of transcript are stored
- Truncated snippets shown in analytics dashboard
- Use only in controlled research environments

---

## Acknowledgments

This privacy framework is informed by:

- NIST Privacy Framework
- IEEE Ethically Aligned Design
- Mental Health Technology Best Practices (MHA)

---

*Last updated: [Date]*
*Version: 1.0*
