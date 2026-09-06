import os
import time
import threading
from typing import Dict, List, Tuple


class InMemoryRateLimiter:
    """
    Thread-safe, sliding-window rate limiter tracking requests per client IP.
    Default configuration allows 10 verification requests per minute per IP.
    """
    def __init__(self, requests_per_minute: int = 10, window_seconds: int = 60):
        self.requests_per_minute = int(os.getenv("RATE_LIMIT_PER_MINUTE", str(requests_per_minute)))
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, client_ip: str) -> Tuple[bool, int]:
        """
        Determines whether a client request is allowed within the current window.
        Returns:
            (allowed: bool, retry_after_seconds: int)
        """
        now = time.time()
        window_start = now - self.window_seconds

        with self._lock:
            # Clean expired timestamps for this IP
            timestamps = self._requests.get(client_ip, [])
            valid_timestamps = [t for t in timestamps if t > window_start]

            if len(valid_timestamps) >= self.requests_per_minute:
                # Rate limit exceeded
                oldest = valid_timestamps[0]
                retry_after = max(1, int(oldest + self.window_seconds - now))
                self._requests[client_ip] = valid_timestamps
                return False, retry_after

            # Record this request
            valid_timestamps.append(now)
            self._requests[client_ip] = valid_timestamps

            # Periodic cleanup of inactive IPs (if dictionary grows large)
            if len(self._requests) > 1000:
                self._cleanup_stale_ips(window_start)

            return True, 0

    def _cleanup_stale_ips(self, window_start: float):
        """Removes client IPs that have no active timestamps in the window."""
        stale_keys = [
            ip for ip, ts_list in self._requests.items()
            if not ts_list or ts_list[-1] <= window_start
        ]
        for k in stale_keys:
            del self._requests[k]

    def reset(self):
        """Clears all tracked requests (useful for test suites)."""
        with self._lock:
            self._requests.clear()


# Default singleton instance for application use
rate_limiter = InMemoryRateLimiter()
