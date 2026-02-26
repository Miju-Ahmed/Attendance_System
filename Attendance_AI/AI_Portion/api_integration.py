"""
API Integration Module for Attendance System
=============================================
Sends attendance events from the Python AI script to the C# ASP.NET Core backend.
This module is imported by attendance_efficientnetdet.py.

Usage:
    from api_integration import AttendanceAPIClient
    client = AttendanceAPIClient("http://localhost:5000")
    client.send_attendance(employee_id=1, name="MIJU", confidence=0.89, camera_id="CAM-01")
"""

import logging
import threading
from datetime import datetime
from typing import Optional

try:
    import requests
except ImportError:
    requests = None
    logging.warning("requests library not installed. Run: pip install requests")

logger = logging.getLogger(__name__)


class AttendanceAPIClient:
    """Non-blocking client that sends attendance events to the C# backend via REST API."""

    def __init__(self, base_url: str = "http://localhost:5000", timeout: int = 5):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.enabled = requests is not None
        if not self.enabled:
            logger.warning("API client disabled — 'requests' package not available.")

    def _post_async(self, endpoint: str, payload: dict) -> None:
        """Fire-and-forget POST request in a background thread (won't block video pipeline)."""
        if not self.enabled:
            return

        def _do_post():
            url = f"{self.base_url}{endpoint}"
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    logger.info(f"[API] ✓ POST {endpoint} — {payload.get('employeeName', 'unknown')}")
                else:
                    logger.warning(f"[API] ✗ POST {endpoint} returned {resp.status_code}: {resp.text[:200]}")
            except requests.exceptions.ConnectionError:
                logger.warning(f"[API] Backend not reachable at {self.base_url}")
            except Exception as exc:
                logger.warning(f"[API] Error: {exc}")

        thread = threading.Thread(target=_do_post, daemon=True)
        thread.start()

    def send_attendance(
        self,
        employee_id: Optional[int],
        name: str,
        confidence: float,
        camera_id: str = "CAM-01",
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Send a recognized employee attendance event."""
        payload = {
            "employeeId": employee_id,
            "employeeName": name,
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
            "confidence": confidence,
            "cameraId": camera_id,
            "isUnknown": False,
        }
        self._post_async("/api/attendance/mark", payload)

    def send_unknown_detection(
        self,
        confidence: float,
        camera_id: str = "CAM-01",
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Send an unknown face detection event."""
        payload = {
            "employeeId": None,
            "employeeName": "Unknown",
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
            "confidence": confidence,
            "cameraId": camera_id,
            "isUnknown": True,
        }
        self._post_async("/api/attendance/mark", payload)
