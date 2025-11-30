"""
CrisisTriage AI - Telephony Providers

Provider-specific implementations for telephony integration.

Supported Providers:
- generic: Basic provider for testing
- twilio: Twilio Voice integration
- simulator: Development call simulator
"""

from .base import TelephonyProvider

__all__ = ["TelephonyProvider"]
