# Neural Triage Model v1

Pre-trained DistilBERT-based triage classifier for crisis assessment.

## Model Details

- **Base Model**: distilbert-base-uncased
- **Task**: Multi-output classification (risk level, emotional state, urgency score)
- **Training Data**: Synthetic crisis conversation data
- **Format**: SafeTensors

## Files

The model is split into multiple zip parts due to GitHub's 100MB file limit:

- `best_model.zip` - Main archive (56MB)
- `best_model.z01` - Part 1 (90MB)
- `best_model.z02` - Part 2 (90MB)

## Extraction

To extract the model:

```bash
# macOS/Linux
zip -s 0 best_model.zip --out combined.zip
unzip combined.zip

# Windows (PowerShell) - requires 7-Zip installed
# Option 1: Use 7-Zip GUI - right-click best_model.zip -> Extract Here
# Option 2: Command line with 7-Zip:
& "C:\Program Files\7-Zip\7z.exe" x best_model.zip
```

After extraction, you should have a `best_model/` directory containing the model files.

## Usage

After extraction, the `best_model/` directory contains:

- `model.safetensors` - Model weights
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer
- `vocab.txt` - Vocabulary
- `artifact.json` - Training metadata

## Important Notice

**FOR RESEARCH/SIMULATION ONLY** - This model has not been validated on clinical populations and must not be used for real crisis intervention.
