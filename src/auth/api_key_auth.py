"""
API Key authentication for the Legal Copilot system.
Implements authentication as specified in Diagram 1.
"""

import os
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from datetime import datetime, timedelta
import hashlib
import secrets

from src.utils import get_logger

logger = get_logger(__name__)

# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Valid API keys from environment
VALID_API_KEYS = {
    os.getenv("API_KEY_1", "sk-legal-copilot-demo-key-12345"): {
        "name": "Demo User",
        "tier": "free",
        "rate_limit": 10  # per minute
    },
    os.getenv("API_KEY_2", "sk-legal-copilot-admin-key-67890"): {
        "name": "Admin User",
        "tier": "admin",
        "rate_limit": 1000
    }
}

# Rate limiting storage (in-memory for demo, use Redis in production)
rate_limit_store = {}


def validate_api_key(api_key: str = Security(api_key_header)) -> dict:
    """
    Validate API key and return user info.

    Args:
        api_key: API key from request header

    Returns:
        User information dictionary

    Raises:
        HTTPException: If API key is invalid
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required. Provide X-API-Key header."
        )

    if api_key not in VALID_API_KEYS:
        logger.warning(f"Invalid API key attempt: {api_key[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    user_info = VALID_API_KEYS[api_key]
    logger.info(f"Authenticated user: {user_info['name']}")

    return {
        "api_key": api_key,
        **user_info
    }


def check_rate_limit(user_info: dict) -> bool:
    """
    Check if user has exceeded rate limit.

    Args:
        user_info: User information from validate_api_key

    Returns:
        True if within rate limit

    Raises:
        HTTPException: If rate limit exceeded
    """
    api_key = user_info["api_key"]
    rate_limit = user_info["rate_limit"]

    now = datetime.now()
    minute_key = f"{api_key}:{now.strftime('%Y-%m-%d-%H-%M')}"

    # Get current count for this minute
    current_count = rate_limit_store.get(minute_key, 0)

    if current_count >= rate_limit:
        logger.warning(f"Rate limit exceeded for user: {user_info['name']}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {rate_limit} requests per minute."
        )

    # Increment counter
    rate_limit_store[minute_key] = current_count + 1

    # Cleanup old entries (keep last 5 minutes)
    cleanup_old_rate_limits()

    return True


def cleanup_old_rate_limits():
    """Remove rate limit entries older than 5 minutes."""
    now = datetime.now()
    cutoff = now - timedelta(minutes=5)

    keys_to_delete = []
    for key in rate_limit_store.keys():
        try:
            # Extract timestamp from key
            parts = key.split(":")
            if len(parts) == 2:
                timestamp_str = parts[1]
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d-%H-%M')
                if timestamp < cutoff:
                    keys_to_delete.append(key)
        except:
            pass

    for key in keys_to_delete:
        del rate_limit_store[key]


def generate_api_key() -> str:
    """Generate a new API key."""
    return f"sk-legal-copilot-{secrets.token_urlsafe(32)}"


def hash_api_key(api_key: str) -> str:
    """Hash API key for storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()
