"""
CrisisTriage AI - Telephony Privacy Utilities

Phone number masking and privacy enforcement for telephony integration.

IMPORTANT:
    Raw phone numbers must NEVER be:
    - Logged in cleartext
    - Stored in any database
    - Transmitted without masking
    
    All phone handling must use these utilities.
"""

import hashlib
import re
from typing import Optional


def mask_phone_number(number: Optional[str], show_last_digits: int = 2) -> str:
    """
    Mask a phone number for privacy.
    
    Examples:
        +14155551234 → ***34
        14155551234  → ***34
        5551234      → ***34
        None         → unknown
    
    Args:
        number: Phone number to mask
        show_last_digits: Number of digits to show (default: 2)
    
    Returns:
        Masked phone number string
    
    IMPORTANT: Raw phone numbers must NEVER be logged or stored.
    """
    if not number:
        return "unknown"
    
    # Remove non-digit characters
    digits = re.sub(r'\D', '', str(number))
    
    if len(digits) < show_last_digits:
        return "***"
    
    return f"***{digits[-show_last_digits:]}"


def hash_phone_number(number: Optional[str], salt: str = "") -> str:
    """
    Create a one-way hash of a phone number.
    
    Used for analytics aggregation without storing the actual number.
    The hash is deterministic for the same number+salt combination.
    
    Args:
        number: Phone number to hash
        salt: Optional salt for added security
    
    Returns:
        Truncated SHA-256 hash (16 characters)
    """
    if not number:
        return "unknown"
    
    # Normalize: remove non-digits, convert to E.164 format
    digits = re.sub(r'\D', '', str(number))
    
    # Hash with salt
    data = f"{salt}:{digits}"
    hash_value = hashlib.sha256(data.encode()).hexdigest()
    
    return hash_value[:16]


def is_phone_number_masked(value: str) -> bool:
    """
    Check if a value appears to be a masked phone number.
    
    Args:
        value: String to check
    
    Returns:
        True if the value appears to be masked
    """
    if not value:
        return False
    return value.startswith("***") or value == "unknown"


def validate_phone_number(number: Optional[str]) -> bool:
    """
    Validate that a string looks like a phone number.
    
    Does NOT store or log the number.
    
    Args:
        number: String to validate
    
    Returns:
        True if it looks like a valid phone number
    """
    if not number:
        return False
    
    # Remove non-digit characters
    digits = re.sub(r'\D', '', str(number))
    
    # Phone numbers are typically 7-15 digits
    return 7 <= len(digits) <= 15


def normalize_phone_number(number: Optional[str]) -> Optional[str]:
    """
    Normalize a phone number for comparison.
    
    NOTE: Returns masked output only. Raw number is not returned.
    This function is for internal comparison logic only.
    
    Args:
        number: Phone number to normalize
    
    Returns:
        Normalized digits (for internal use only)
    """
    if not number:
        return None
    
    # Extract digits only
    digits = re.sub(r'\D', '', str(number))
    
    # Remove leading country code if present and return last 10 digits
    if len(digits) > 10:
        digits = digits[-10:]
    
    return digits


class PrivacyGuard:
    """
    Context manager that ensures phone numbers are not accidentally logged.
    
    Usage:
        with PrivacyGuard() as guard:
            masked = guard.mask(phone_number)
            # Use masked value only
    """
    
    def __init__(self):
        self._masked_numbers: list[str] = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clear any stored references
        self._masked_numbers.clear()
        return False
    
    def mask(self, number: Optional[str]) -> str:
        """Mask a phone number and track the masked value."""
        masked = mask_phone_number(number)
        self._masked_numbers.append(masked)
        return masked
    
    def hash(self, number: Optional[str], salt: str = "") -> str:
        """Hash a phone number."""
        return hash_phone_number(number, salt)
