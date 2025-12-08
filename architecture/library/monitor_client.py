from dataclasses import dataclass
import requests
from typing import Dict, Any, List, Optional

#Entities
from dataclasses import dataclass
from typing import Optional

from .output_layer import OutputLayerMetadata


@dataclass
class ServiceStatus:
    service_id: str
    last_seen: Optional[float]
    online: bool

    def __repr__(self):
        return f"<ServiceStatus id={self.service_id} online={self.online} last_seen={self.last_seen}>"
    
@dataclass
class Stats:
    messages: int
    runtime_sec: float
    total_mb: float    

#Client

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

    def get(self) -> Stats:
        raw = self._get("/api/stats")
        return Stats(
            messages=raw["messages"],
            runtime_sec=raw["runtime_sec"],
            total_mb=raw["total_mb"],
        )


class MessagesAPI(_BaseAPI):
    """Wrapper for /api/messages"""

    def get_all(self) -> List[OutputLayerMetadata]:
        raw = self._get("/api/messages")

        result = []
        for item in raw:

            result.append(
                OutputLayerMetadata(
                    source_id=item["source_id"],
                    service_id=item["service_id"],
                    time_stamp=item["time_stamp"],
                    completed_at=item["completed_at"],
                    result=item["result"],
                )
            )
        return result

    def filter_by_service(self, service_id: str) -> List[OutputLayerMetadata]:
        messages = self.get_all()
        return [m for m in messages if m.get("service_id") == service_id]



class ServicesAPI(_BaseAPI):
    """Wrapper for /api/services/input/monitor"""

    def get_all(self) -> List[ServiceStatus]:
        raw = self._get("/api/services/input/monitor")

        return [
            ServiceStatus(
                service_id=service_id,
                last_seen=data.get("last_seen"),
                online=data.get("online", False)
            )
            for service_id, data in raw.items()
        ]

    def get(self, service_id: str) -> Optional[ServiceStatus]:
        raw = self._get(f"/api/services/input/monitor/{service_id}")

        if service_id not in raw:
            return None

        entry = raw[service_id]

        return ServiceStatus(
            service_id=service_id,
            last_seen=entry.get("last_seen"),
            online=entry.get("online", False)
        )

    

class HistoryAPI(_BaseAPI):
    """Wrapper for /api/history/<service_id>"""

    def get(self, service_id: str) -> List[OutputLayerMetadata]:
        raw = self._get(f"/api/history/{service_id}")

        return [
            OutputLayerMetadata(
                source_id=item["source_id"],
                service_id=item["service_id"],
                time_stamp=item["time_stamp"],
                completed_at=item["completed_at"],
                result=item["result"],
            )
            for item in raw
        ]



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

        self.stats = StatsAPI(base_url, timeout)
        self.messages = MessagesAPI(base_url, timeout)
        self.services = ServicesAPI(base_url, timeout)
        self.history = HistoryAPI(base_url, timeout)

    # Optional shorthand methods
    def get_stats(self) -> Stats:
        return self.stats.get()

    def get_messages(self) -> List[OutputLayerMetadata]:
        return self.messages.get_all()

    def get_services(self) -> List[ServiceStatus]:
        return self.services.get_all()

    def get_online_status(self, service_id: str) -> Optional[ServiceStatus]:
        return self.services.get(service_id)

    def get_history(self, service_id: str) -> List[OutputLayerMetadata]:
        return self.history.get(service_id)
