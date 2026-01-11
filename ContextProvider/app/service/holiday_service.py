from __future__ import annotations

from typing import List

import httpx

from ..model.context_models import Holiday

NAGER_BASE_URL = "https://date.nager.at"


async def fetch_holidays(country_code: str, year: int) -> List[Holiday]:
    """
    Fetch public holidays for a given country/year from the Nager.Date API.
    Docs: https://date.nager.at/swagger/index.html
    """
    url = f"{NAGER_BASE_URL}/api/v3/PublicHolidays/{year}/{country_code}"

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
        except httpx.HTTPError:
            # In case of errors we return an empty list and let the caller decide how to handle it.
            return []

        data = response.json()
        holidays: List[Holiday] = []

        for item in data:
            date = item.get("date")
            local_name = item.get("localName")
            if not date or not local_name:
                continue

            # Nager.Date: "counties" enthält optionale Liste von Regionen (z.B. ["DE-BW", "DE-HE"])
            regions = item.get("counties")

            holidays.append(
                Holiday(
                    date=date,
                    localName=local_name,
                    countryCode=country_code,
                    regions=regions,
                )
            )

        return holidays
