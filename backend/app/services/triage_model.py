"""
CrisisTriage AI - Triage Model Service

Provides risk/emotional state assessment from text and prosody features.

Architecture:
    - Protocol defines the interface for triage models
    - DummyTriageModel: Heuristic placeholder for development/testing
    - NeuralTriageModel: PyTorch production implementation (TODO)

Model Design Considerations:
    1. Multi-task learning: Joint prediction of emotion, risk, urgency, action
    2. Multimodal fusion: Combining text embeddings with prosody features
    3. Uncertainty quantification: Confidence scores and calibration
    4. Interpretability: Feature attribution for every prediction
    5. Temporal modeling: Consider conversation history, not just current turn

Safety Notes:
    - This model provides DECISION SUPPORT, not clinical diagnosis
    - All high-risk predictions should trigger human review
    - Model should fail-safe to "escalate" when uncertain
    - Regular monitoring for bias and drift is essential

Privacy Considerations:
    - Model should not memorize training examples
    - Avoid features that could re-identify individuals
    - Log predictions, not inputs (or sanitized inputs)
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import abstractmethod
from typing import Protocol, Optional, runtime_checkable
import time

from app.core.types import (
    ProsodyFeatures,
    TriageResult,
    TriageExplanation,
    FeatureContribution,
    EmotionalState,
    RiskLevel,
    RecommendedAction,
    SessionId,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Protocol (Interface)
# =============================================================================

@runtime_checkable
class TriageModel(Protocol):
    """
    Protocol for triage assessment models.
    
    Models receive text (from transcription) and optional prosody features,
    returning a complete triage assessment with explanations.
    """

    @abstractmethod
    def predict(
        self,
        session_id: SessionId,
        text: str,
        prosody: Optional[ProsodyFeatures] = None,
        context: Optional[dict] = None,
    ) -> TriageResult:
        """
        Generate triage assessment from text and prosody features.
        
        Args:
            session_id: Session identifier for tracking
            text: Transcribed text to analyze
            prosody: Optional prosody features from audio
            context: Optional context dict (e.g., conversation history)
            
        Returns:
            TriageResult with all assessment outputs and explanation
        """
        ...

    @abstractmethod
    async def predict_async(
        self,
        session_id: SessionId,
        text: str,
        prosody: Optional[ProsodyFeatures] = None,
        context: Optional[dict] = None,
    ) -> TriageResult:
        """Async version for non-blocking inference."""
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return model version identifier."""
        ...

    @abstractmethod
    def warmup(self) -> None:
        """
        Warm up the model (e.g., run dummy inference).
        
        Call at startup to ensure first real inference isn't slow.
        """
        ...


# =============================================================================
# Exceptions
# =============================================================================

class TriageModelError(Exception):
    """Raised when model inference fails."""
    pass


# =============================================================================
# Dummy Implementation (Development/Testing)
# =============================================================================

class DummyTriageModel:
    """
    Heuristic-based triage model for development and testing.
    
    Uses simple keyword matching and threshold rules to generate
    plausible triage outputs. This is intentionally simplistic and
    NOT suitable for any real clinical application.
    
    The heuristics are designed to:
    1. Demonstrate the full output structure
    2. Respond sensibly to obviously concerning content
    3. Provide reproducible outputs for testing
    
    WARNING: This model has NO clinical validity. It exists purely
    to test the pipeline architecture.
    """

    # Keyword lists for heuristic analysis
    # These are intentionally obvious for testing purposes
    HIGH_RISK_KEYWORDS = {
        "suicide", "kill myself", "end my life", "want to die",
        "no reason to live", "better off dead", "can't go on",
        "hurt myself", "self-harm", "overdose",
    }
    
    MEDIUM_RISK_KEYWORDS = {
        "hopeless", "worthless", "burden", "trapped",
        "can't take it", "give up", "no way out", "desperate",
        "scared", "panic", "crisis", "emergency",
    }
    
    DISTRESS_KEYWORDS = {
        "anxious", "worried", "stressed", "overwhelmed",
        "can't sleep", "exhausted", "struggling", "difficult",
        "hard time", "upset", "crying", "lonely",
    }
    
    POSITIVE_KEYWORDS = {
        "better", "improving", "hopeful", "thank you",
        "helpful", "grateful", "okay", "alright", "good",
    }

    def __init__(self, simulated_latency_ms: float = 30.0):
        """
        Initialize dummy triage model.
        
        Args:
            simulated_latency_ms: Artificial delay to simulate inference time
        """
        self._simulated_latency_ms = simulated_latency_ms
        self._call_count = 0
        self._warmed_up = False

    @property
    def model_id(self) -> str:
        return "dummy-triage-v0.0.1"

    def warmup(self) -> None:
        """Simulate model warmup."""
        if not self._warmed_up:
            logger.info("DummyTriageModel: warming up...")
            # In real impl, run dummy inference to load weights
            self._warmed_up = True
            logger.info("DummyTriageModel: warmup complete")

    def predict(
        self,
        session_id: SessionId,
        text: str,
        prosody: Optional[ProsodyFeatures] = None,
        context: Optional[dict] = None,
    ) -> TriageResult:
        """
        Generate triage assessment using keyword heuristics.
        
        The assessment logic:
        1. Check for high-risk keywords → IMMINENT/HIGH risk
        2. Check for medium-risk keywords → MEDIUM risk
        3. Check for distress keywords → MEDIUM/LOW risk
        4. Check for positive keywords → may reduce urgency
        5. Incorporate prosody signals (if available)
        """
        start_time = time.time()
        self._call_count += 1
        
        text_lower = text.lower()
        
        # --- Keyword Analysis ---
        high_risk_matches = self._find_matches(text_lower, self.HIGH_RISK_KEYWORDS)
        medium_risk_matches = self._find_matches(text_lower, self.MEDIUM_RISK_KEYWORDS)
        distress_matches = self._find_matches(text_lower, self.DISTRESS_KEYWORDS)
        positive_matches = self._find_matches(text_lower, self.POSITIVE_KEYWORDS)
        
        # --- Determine Risk Level ---
        if high_risk_matches:
            risk_level = RiskLevel.HIGH
            if any(kw in text_lower for kw in ["now", "tonight", "today", "plan"]):
                risk_level = RiskLevel.IMMINENT
        elif medium_risk_matches:
            risk_level = RiskLevel.MEDIUM
        elif distress_matches:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.LOW
        
        # --- Determine Emotional State ---
        if high_risk_matches or (medium_risk_matches and not positive_matches):
            emotional_state = EmotionalState.PANICKED if len(high_risk_matches) > 1 else EmotionalState.DISTRESSED
        elif distress_matches:
            emotional_state = EmotionalState.ANXIOUS
        elif positive_matches:
            emotional_state = EmotionalState.CALM
        else:
            emotional_state = EmotionalState.UNKNOWN
        
        # --- Incorporate Prosody (if available) ---
        prosody_adjustment = 0
        prosody_contributions = []
        
        if prosody is not None:
            # High pitch variability may indicate distress
            if prosody.pitch_std is not None and prosody.pitch_std > 40:
                prosody_adjustment += 10
                prosody_contributions.append(
                    FeatureContribution(
                        feature_name="pitch_variability",
                        contribution=0.15,
                        value=prosody.pitch_std,
                        category="prosody",
                    )
                )
            
            # Fast speech rate may indicate anxiety
            if prosody.speech_rate is not None and prosody.speech_rate > 5.5:
                prosody_adjustment += 5
                prosody_contributions.append(
                    FeatureContribution(
                        feature_name="speech_rate",
                        contribution=0.10,
                        value=prosody.speech_rate,
                        category="prosody",
                    )
                )
            
            # High pause ratio may indicate hesitation/distress
            if prosody.pause_ratio is not None and prosody.pause_ratio > 0.35:
                prosody_adjustment += 5
                prosody_contributions.append(
                    FeatureContribution(
                        feature_name="pause_ratio",
                        contribution=0.08,
                        value=prosody.pause_ratio,
                        category="prosody",
                    )
                )
        
        # --- Calculate Urgency Score ---
        base_urgency = {
            RiskLevel.IMMINENT: 90,
            RiskLevel.HIGH: 70,
            RiskLevel.MEDIUM: 45,
            RiskLevel.LOW: 20,
            RiskLevel.UNKNOWN: 10,
        }[risk_level]
        
        # Adjust based on keyword density
        keyword_adjustment = min(20, len(high_risk_matches) * 10 + len(medium_risk_matches) * 3)
        
        # Reduce for positive signals
        positive_reduction = min(15, len(positive_matches) * 5)
        
        urgency_score = max(0, min(100, 
            base_urgency + keyword_adjustment + prosody_adjustment - positive_reduction
        ))
        
        # --- Determine Recommended Action ---
        if risk_level == RiskLevel.IMMINENT:
            recommended_action = RecommendedAction.IMMEDIATE_INTERVENTION
        elif risk_level == RiskLevel.HIGH:
            recommended_action = RecommendedAction.ESCALATE_TO_HUMAN
        elif risk_level == RiskLevel.MEDIUM or urgency_score > 40:
            recommended_action = RecommendedAction.ASK_FOLLOWUP
        else:
            recommended_action = RecommendedAction.CONTINUE_LISTENING
        
        # --- Build Explanation ---
        text_contributions = []
        
        for match in high_risk_matches[:3]:  # Top 3
            text_contributions.append(
                FeatureContribution(
                    feature_name="high_risk_keyword",
                    contribution=0.5,
                    value=match,
                    category="text",
                )
            )
        
        for match in medium_risk_matches[:2]:
            text_contributions.append(
                FeatureContribution(
                    feature_name="medium_risk_keyword",
                    contribution=0.2,
                    value=match,
                    category="text",
                )
            )
        
        for match in distress_matches[:2]:
            text_contributions.append(
                FeatureContribution(
                    feature_name="distress_keyword",
                    contribution=0.1,
                    value=match,
                    category="text",
                )
            )
        
        explanation = TriageExplanation(
            top_contributors=text_contributions + prosody_contributions,
            summary=self._generate_summary(risk_level, text_contributions, prosody_contributions),
            method="keyword_heuristic",
        )
        
        # --- Calculate Confidence ---
        # Higher confidence when we have clear signals
        signal_strength = len(high_risk_matches) * 0.3 + len(medium_risk_matches) * 0.2
        if prosody is not None:
            signal_strength += 0.1
        confidence = min(0.95, 0.5 + signal_strength)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        result = TriageResult(
            session_id=session_id,
            emotional_state=emotional_state,
            risk_level=risk_level,
            urgency_score=urgency_score,
            recommended_action=recommended_action,
            confidence=confidence,
            explanation=explanation,
            timestamp_ms=int(time.time() * 1000),
            transcript_segment=text[:200] if len(text) > 200 else text,  # Truncate for storage
            prosody_features=prosody,
            processing_time_ms=processing_time_ms,
            model_version=self.model_id,
        )
        
        logger.debug(
            "DummyTriage: session=%s risk=%s urgency=%d (%.1fms)",
            session_id[:8],
            risk_level.value,
            urgency_score,
            processing_time_ms,
        )
        
        return result

    async def predict_async(
        self,
        session_id: SessionId,
        text: str,
        prosody: Optional[ProsodyFeatures] = None,
        context: Optional[dict] = None,
    ) -> TriageResult:
        """Async wrapper around sync prediction."""
        import asyncio
        
        # Simulate inference latency
        await asyncio.sleep(self._simulated_latency_ms / 1000.0)
        
        return self.predict(session_id, text, prosody, context)

    def _find_matches(self, text: str, keywords: set[str]) -> list[str]:
        """Find all keyword matches in text."""
        matches = []
        for keyword in keywords:
            if keyword in text:
                matches.append(keyword)
        return matches

    def _generate_summary(
        self,
        risk_level: RiskLevel,
        text_contribs: list[FeatureContribution],
        prosody_contribs: list[FeatureContribution],
    ) -> str:
        """Generate human-readable explanation summary."""
        parts = []
        
        if risk_level in (RiskLevel.HIGH, RiskLevel.IMMINENT):
            if text_contribs:
                keywords = [c.value for c in text_contribs[:2]]
                parts.append(f"Detected high-risk language: {', '.join(keywords)}")
        elif text_contribs:
            keywords = [c.value for c in text_contribs[:2]]
            parts.append(f"Noted concerning language: {', '.join(keywords)}")
        
        if prosody_contribs:
            prosody_signals = [c.feature_name.replace("_", " ") for c in prosody_contribs]
            parts.append(f"Voice patterns suggest elevated {', '.join(prosody_signals)}")
        
        if not parts:
            parts.append("No significant risk indicators detected in this segment")
        
        return ". ".join(parts) + "."


# =============================================================================
# Neural Model Implementation
# =============================================================================

class NeuralTriageModel:
    """
    Neural triage model using a fine-tuned transformer classifier.
    
    Loads a trained model artifact from the ml/ package and provides
    inference through the TriageModel protocol.
    
    IMPORTANT SAFETY NOTICE:
        This model is for RESEARCH AND SIMULATION ONLY.
        It is NOT a medical device and NOT suitable for real-world crisis intervention.
        Model predictions should NEVER be used as the sole basis for crisis response.
    
    Architecture:
        - Uses HuggingFace transformer (e.g., DistilBERT) fine-tuned on triage data
        - Predicts risk level from text (prosody integration planned for future)
        - Maps model outputs to domain types (RiskLevel, EmotionalState, etc.)
    
    Usage:
        model = NeuralTriageModel(
            model_dir="./ml/outputs/baseline/best_model",
            settings=settings,
        )
        result = await model.predict_async(session_id, "I'm feeling overwhelmed")
    """
    
    def __init__(
        self,
        model_dir: str,
        settings: Optional["Settings"] = None,
        device: str = "auto",
    ):
        """
        Initialize the neural triage model.
        
        Args:
            model_dir: Path to trained model artifact (contains model weights + artifact.json)
            settings: Application settings (for enable_explainability, etc.)
            device: Device to use ("auto", "cpu", "cuda", "mps")
        """
        import json
        import os
        
        self._model_dir = model_dir
        self._settings = settings
        self._device = self._resolve_device(device)
        self._warmed_up = False
        
        # Load model artifact metadata
        artifact_path = os.path.join(model_dir, "artifact.json")
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(
                f"Model artifact not found at {artifact_path}. "
                "Ensure the model was saved with ml.src.models.TriageClassifier.save_pretrained()"
            )
        
        with open(artifact_path, "r") as f:
            artifact_data = json.load(f)
        
        self._label2id: dict[str, int] = artifact_data["label2id"]
        self._id2label: dict[int, str] = {int(k): v for k, v in artifact_data["id2label"].items()}
        self._model_name = artifact_data.get("model_name", "unknown")
        self._max_seq_length = artifact_data.get("max_seq_length", 256)
        
        logger.info(
            "Loading NeuralTriageModel from %s (base: %s, device: %s)",
            model_dir, self._model_name, self._device
        )
        
        # Load model and tokenizer
        self._load_model()
        
        logger.info("NeuralTriageModel initialized successfully")
        logger.warning(
            "SAFETY NOTICE: This model is for RESEARCH/SIMULATION ONLY. "
            "NOT suitable for real crisis intervention."
        )

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"
        return device

    def _load_model(self) -> None:
        """Load the transformer model and tokenizer."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_dir)
            self._model = AutoModelForSequenceClassification.from_pretrained(self._model_dir)
            self._model.to(self._device)
            self._model.eval()
            
            self._torch = torch
            
        except ImportError as e:
            raise ImportError(
                "NeuralTriageModel requires transformers and torch. "
                "Install with: pip install transformers torch"
            ) from e

    @property
    def model_id(self) -> str:
        return f"neural-triage-{self._model_name}"

    def warmup(self) -> None:
        """Warm up the model with a dummy inference."""
        if self._warmed_up:
            return
        
        logger.info("Warming up NeuralTriageModel...")
        
        # Run a dummy inference
        dummy_text = "This is a warmup inference."
        _ = self._run_inference(dummy_text)
        
        self._warmed_up = True
        logger.info("NeuralTriageModel warmup complete")

    def _run_inference(self, text: str) -> tuple[str, float, dict[str, float]]:
        """
        Run model inference on text.
        
        Returns:
            Tuple of (predicted_label, confidence, all_probabilities)
        """
        import torch.nn.functional as F
        
        # Tokenize
        inputs = self._tokenizer(
            text,
            max_length=self._max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        # Inference
        with self._torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)[0]
        
        # Get prediction
        predicted_id = int(self._torch.argmax(probs).item())
        predicted_label = self._id2label[predicted_id]
        confidence = float(probs[predicted_id].item())
        
        # All probabilities
        all_probs = {
            self._id2label[i]: float(probs[i].item())
            for i in range(len(probs))
        }
        
        return predicted_label, confidence, all_probs

    def _risk_to_emotional_state(self, risk_label: str, confidence: float) -> EmotionalState:
        """
        Infer emotional state from risk level.
        
        This is a simplified mapping. In a full implementation,
        emotional state would be predicted separately.
        """
        # Simple heuristic mapping
        if risk_label == "imminent":
            return EmotionalState.PANICKED
        elif risk_label == "high":
            return EmotionalState.DISTRESSED
        elif risk_label == "medium":
            return EmotionalState.ANXIOUS
        elif risk_label == "low":
            if confidence > 0.8:
                return EmotionalState.CALM
            return EmotionalState.ANXIOUS
        return EmotionalState.UNKNOWN

    def _risk_to_action(self, risk_label: str) -> RecommendedAction:
        """Map risk level to recommended action."""
        mapping = {
            "imminent": RecommendedAction.IMMEDIATE_INTERVENTION,
            "high": RecommendedAction.ESCALATE_TO_HUMAN,
            "medium": RecommendedAction.ASK_FOLLOWUP,
            "low": RecommendedAction.CONTINUE_LISTENING,
        }
        return mapping.get(risk_label, RecommendedAction.ASK_FOLLOWUP)

    def _compute_urgency_score(self, risk_label: str, confidence: float, all_probs: dict[str, float]) -> int:
        """
        Compute urgency score (0-100) from model outputs.
        
        Uses a weighted combination of risk level and probability distribution.
        """
        # Base score by risk level
        base_scores = {
            "low": 15,
            "medium": 40,
            "high": 70,
            "imminent": 90,
        }
        base = base_scores.get(risk_label, 30)
        
        # Adjust by probability of higher-risk classes
        high_prob = all_probs.get("high", 0)
        imminent_prob = all_probs.get("imminent", 0)
        
        # Add urgency for probability mass in severe categories
        adjustment = int((high_prob * 15) + (imminent_prob * 25))
        
        # Scale by confidence
        confidence_factor = 0.7 + (confidence * 0.3)  # Range: 0.7 to 1.0
        
        score = int(base * confidence_factor + adjustment)
        return max(0, min(100, score))

    def _build_explanation(
        self,
        text: str,
        risk_label: str,
        all_probs: dict[str, float],
    ) -> TriageExplanation:
        """
        Build explanation for the prediction.
        
        Currently provides a simple probability-based explanation.
        More sophisticated XAI (attention weights, integrated gradients)
        can be added in future iterations.
        """
        # Build probability-based explanation
        top_contributors = []
        
        # Add probability contributions
        for label, prob in sorted(all_probs.items(), key=lambda x: -x[1])[:3]:
            top_contributors.append(
                FeatureContribution(
                    feature_name=f"prob_{label}",
                    contribution=prob,
                    value=f"{prob:.2%}",
                    category="text",
                )
            )
        
        # Generate summary
        summary_parts = [
            f"Neural model prediction: {risk_label} risk ({all_probs.get(risk_label, 0):.1%} confidence).",
            "Analysis based on text content patterns.",
        ]
        
        if all_probs.get("imminent", 0) > 0.1 or all_probs.get("high", 0) > 0.2:
            summary_parts.append("Elevated probability of severe risk indicators detected.")
        
        # Add disclaimer
        summary_parts.append(
            "[RESEARCH/SIMULATION ONLY - Model output requires human review]"
        )
        
        return TriageExplanation(
            top_contributors=top_contributors,
            summary=" ".join(summary_parts),
            method="neural_text_classifier",
        )

    def predict(
        self,
        session_id: SessionId,
        text: str,
        prosody: Optional[ProsodyFeatures] = None,
        context: Optional[dict] = None,
    ) -> TriageResult:
        """
        Generate triage assessment from text.
        
        Note: Prosody features are not yet used by this text-only model.
        """
        start_time = time.time()
        
        # Run neural inference
        risk_label, confidence, all_probs = self._run_inference(text)
        
        # Map to domain types
        risk_level = RiskLevel(risk_label)
        emotional_state = self._risk_to_emotional_state(risk_label, confidence)
        recommended_action = self._risk_to_action(risk_label)
        urgency_score = self._compute_urgency_score(risk_label, confidence, all_probs)
        
        # Build explanation
        explanation = self._build_explanation(text, risk_label, all_probs)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        result = TriageResult(
            session_id=session_id,
            emotional_state=emotional_state,
            risk_level=risk_level,
            urgency_score=urgency_score,
            recommended_action=recommended_action,
            confidence=confidence,
            explanation=explanation,
            timestamp_ms=int(time.time() * 1000),
            transcript_segment=text[:200] if len(text) > 200 else text,
            prosody_features=prosody,  # Stored but not used yet
            processing_time_ms=processing_time_ms,
            model_version=self.model_id,
        )
        
        logger.debug(
            "NeuralTriage: session=%s risk=%s urgency=%d confidence=%.2f (%.1fms)",
            session_id[:8] if session_id else "none",
            risk_level.value,
            urgency_score,
            confidence,
            processing_time_ms,
        )
        
        return result

    async def predict_async(
        self,
        session_id: SessionId,
        text: str,
        prosody: Optional[ProsodyFeatures] = None,
        context: Optional[dict] = None,
    ) -> TriageResult:
        """Async wrapper around sync prediction."""
        import asyncio
        
        # Run in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.predict(session_id, text, prosody, context)
        )


# Type hint for settings (avoid circular import)
try:
    from app.config import Settings
except ImportError:
    Settings = None
