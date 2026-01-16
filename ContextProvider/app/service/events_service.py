from __future__ import annotations
from datetime import datetime
from ContextProvider.app.model.context_models import (
    EventsContext,
    LocationResolved,
)

async def fetch_local_events_context(
    location: LocationResolved,
    now: datetime,
) -> EventsContext:
    """
    Placeholder implementation for local events.

    Currently returns an empty EventsContext. Later you can plug in:
      - a city event API
      - a university/campus events feed
      - or a custom backend for Smart Interaction

    The structure is already used by the EnvironmentContext, so the
    embodied agent can rely on the shape of the data.
    """
    # TODO: Implement real local event fetching if desired.
    return EventsContext(localToday=[])
