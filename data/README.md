# CrisisTriage AI — Data Directory

> Local data storage for development and testing.

## ⚠️ Privacy Notice

**This directory is gitignored.** No data files should ever be committed to version control.

Raw audio and transcripts are considered sensitive data. Handle with care.

---

## Directory Structure

```
data/
├── README.md           # This file (committed)
├── raw/                # Raw audio files (gitignored)
├── processed/          # Extracted features (gitignored)
└── synthetic/          # Simulated test data (gitignored, except .gitkeep)
```

---

## Subdirectories

### `raw/`

Raw audio files for development/testing.

- **Format**: WAV (16kHz, mono, 16-bit PCM)
- **Naming**: `{session_id}_{timestamp}.wav`
- **Retention**: Delete after use

```
raw/
├── session_001_20241129.wav
└── session_002_20241129.wav
```

### `processed/`

Extracted features and intermediate representations.

- **Format**: JSON, NPY, or Parquet
- **Contents**: Prosody features, text embeddings, triage outputs

```
processed/
├── features/
│   ├── prosody_{session_id}.json
│   └── text_{session_id}.json
└── outputs/
    └── triage_{session_id}.json
```

### `synthetic/`

Simulated data for testing without real recordings.

- **Purpose**: Unit tests, integration tests, demos
- **Generation**: Use `scripts/generate_synthetic.py` (TBD)

```
synthetic/
├── .gitkeep
├── calm_sample.wav
├── distressed_sample.wav
└── test_transcript.json
```

---

## Data Sources

### For Development

1. **Synthetic audio**: Generated using TTS (e.g., Coqui TTS, Bark)
2. **Public datasets**: (with appropriate licenses)
   - LibriSpeech (clean speech)
   - RAVDESS (emotional speech)
   - IEMOCAP (emotional dialogue, requires agreement)

### For Research

If using real data for research:

1. Obtain IRB approval
2. Ensure informed consent
3. Anonymize before use
4. Document data provenance

---

## Data Handling Rules

| Action | Allowed? | Notes |
|--------|----------|-------|
| Commit raw audio to git | ❌ | Never |
| Store on shared drives | ⚠️ | Only with encryption |
| Process locally | ✅ | Ephemeral by default |
| Upload to cloud | ❌ | Not without explicit approval |

---

## Generating Test Data

```bash
# Generate synthetic audio samples (TBD)
python scripts/generate_synthetic.py --output data/synthetic/

# Create test transcript
python scripts/create_test_transcript.py --output data/synthetic/test_transcript.json
```

---

## Cleanup

Remove all data:

```bash
rm -rf data/raw/* data/processed/*
# Keep synthetic .gitkeep
```

---

*See also: [docs/PRIVACY.md](../docs/PRIVACY.md)*
