"""
QuaRCAA Exponential Backoff Retry Handler
Handles 429 Rate Limits, 500 Server Errors, and API Timeouts safely.
"""
import time
import functools
from typing import Callable, Any

def retry_with_exponential_backoff(
    max_retries: int = 5,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0
) -> Callable:
    """Decorator to retry API calls with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"⚠️ [QuaRCAA Retry Handler] Attempt {attempt}/{max_retries} failed with error: {e}")
                    if attempt == max_retries:
                        raise e
                    print(f"   Waiting {delay:.1f}s before retrying...")
                    time.sleep(delay)
                    delay *= backoff_factor
        return wrapper
    return decorator
