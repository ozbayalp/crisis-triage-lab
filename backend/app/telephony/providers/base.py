"""
CrisisTriage AI - Telephony Provider Base

Abstract base class for telephony provider implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional


class TelephonyProvider(ABC):
    """
    Abstract base class for telephony providers.
    
    Implementations handle provider-specific:
    - Webhook validation
    - Audio format conversion
    - Response formatting
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        ...
    
    @property
    @abstractmethod
    def audio_encoding(self) -> str:
        """Default audio encoding format."""
        ...
    
    @property
    @abstractmethod
    def audio_sample_rate(self) -> int:
        """Default audio sample rate."""
        ...
    
    @abstractmethod
    def validate_webhook(self, request_data: dict, signature: Optional[str] = None) -> bool:
        """
        Validate webhook request authenticity.
        
        Args:
            request_data: Parsed request body
            signature: Request signature header (if applicable)
        
        Returns:
            True if request is valid
        """
        ...
    
    @abstractmethod
    def format_connect_response(self, stream_url: str) -> str:
        """
        Format response to connect media stream.
        
        Args:
            stream_url: WebSocket URL for media streaming
        
        Returns:
            Provider-specific response (e.g., TwiML for Twilio)
        """
        ...
    
    @abstractmethod
    def parse_call_id(self, request_data: dict) -> str:
        """
        Extract call ID from request data.
        
        Args:
            request_data: Parsed request body
        
        Returns:
            Unique call identifier
        """
        ...
    
    @abstractmethod
    def parse_phone_numbers(self, request_data: dict) -> tuple[str, str]:
        """
        Extract phone numbers from request data.
        
        Args:
            request_data: Parsed request body
        
        Returns:
            Tuple of (from_number, to_number)
        """
        ...
