from __future__ import annotations
import httpx
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List
from ContextProvider.app.model.context_models import (
    EventsContext,
    LocalEvent,
    LocationResolved,
)

async def fetch_local_events_context(
    location: LocationResolved,
    now: datetime,
) -> EventsContext:
    """
    Retrieves events from the Karlsruhe RSS feed and filters them by today's date.
    Is only sent if the position is actually Karlsruhe. 
    Can be replaced by data from other cities, etc., if available.
    """
    if location.city != "Karlsruhe":
        return EventsContext(localToday=[])

    url = "https://kalender.karlsruhe.de/db/termine/rss"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            xml_text = response.text

        root = ET.fromstring(xml_text)
        items = root.findall('.//item')
        
        today_events: List[LocalEvent] = []
        target_date = now.date()

        for it in items:
            title = it.find('title').text if it.find('title') is not None else "Kein Titel"
            link = it.find('link').text if it.find('link') is not None else None
            desc = it.find('description').text if it.find('description') is not None else ""
            pub_date_raw = it.find('pubDate').text if it.find('pubDate') is not None else ""

            try:
                dt = datetime.strptime(pub_date_raw[:25], "%a, %d %b %Y %H:%M:%S")
                
                if dt.date() == target_date:
                    today_events.append(LocalEvent(
                        title=title,
                        description=desc,
                        url=link,
                        startTime=dt.isoformat(),
                        category="Veranstaltung"
                    ))
            except Exception:
                continue

        today_events.sort(key=lambda x: x.startTime if x.startTime else "")

        return EventsContext(localToday=today_events)

    except Exception as e:
        print(f"[EventsService] Fehler beim Abrufen der Karlsruher Events: {e}")
        return EventsContext(localToday=[])
