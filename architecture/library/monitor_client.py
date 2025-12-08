import requests
from typing import Dict, Any, List, Optional


class MonitorAPIError(Exception):
    """Custom exception for API related errors."""
    pass


class _BaseAPI:
    """Base helper class for REST wrappers."""

    def __init__(self, base_url: str, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str) -> Any:
        url = f"{self.base_url}{path}"
        try:
            r = requests.get(url, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            raise MonitorAPIError(f"Request failed: {url}\n{e}") from e




class StatsAPI(_BaseAPI):
    """Wrapper for /api/stats"""

    def get(self) -> Dict[str, Any]:
        return self._get("/api/stats")


class MessagesAPI(_BaseAPI):
    """Wrapper for /api/messages"""

    def get_all(self) -> List[Dict[str, Any]]:
        return self._get("/api/messages")

    def filter_by_service(self, service_id: str) -> List[Dict[str, Any]]:
        messages = self.get_all()
        return [m for m in messages if m.get("service_id") == service_id]


class ServicesAPI(_BaseAPI):
    """Wrapper for /api/services/input/monitor"""

    def get_all(self) -> Dict[str, Any]:
        return self._get("/api/services/input/monitor")

    def get(self, service_id: str) -> Dict[str, Any]:
        return self._get(f"/api/services/input/monitor/{service_id}")


class HistoryAPI(_BaseAPI):
    """Wrapper for /api/history/<service_id>"""

    def get(self, service_id: str) -> List[Dict[str, Any]]:
        return self._get(f"/api/history/{service_id}")



class MonitorClient:
    """
    High-level client that wraps all REST endpoints of the OutputLayerMonitor.
    
    Example:
        client = MonitorClient()
        stats = client.stats.get()
        messages = client.messages.get_all()
    """

    def __init__(self, base_url: str = "http://152.53.32.66:5000", timeout: float = 5.0):
        self.base_url = base_url
        self.timeout = timeout

        # Sub-APIs
        self.stats = StatsAPI(base_url, timeout)
        self.messages = MessagesAPI(base_url, timeout)
        self.services = ServicesAPI(base_url, timeout)
        self.history = HistoryAPI(base_url, timeout)

    # Optional shorthand methods
    def get_stats(self):
        return self.stats.get()

    def get_messages(self):
        return self.messages.get_all()

    def get_services(self):
        return self.services.get_all()

    def get_service(self, service_id: str):
        return self.services.get(service_id)

    def get_history(self, service_id: str):
        return self.history.get(service_id)
