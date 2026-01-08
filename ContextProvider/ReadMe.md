# context_provider

`context_provider` is a Python-based **environment context service** built with **FastAPI**.  
It provides real-time **environmental and situational context data** to other systems via:

- classical **HTTP request/response APIs**
- and **automatic server-side push** to an upstream message or routing service.

The service is designed to be used as a **context source for conversational AI systems or smart interaction agents**.

---

## Features

The service provides:

- **Location context**
  - City, country, region
  - Latitude & longitude (from request input or default)
- **Date & time context**
  - ISO timestamp
  - Timezone
  - Weekday
  - Part of day (morning, afternoon, evening, night)
- **Public holidays**
  - Loaded from the Nager.Date API
- **Current weather**
  - Loaded from the Open-Meteo API
- **Locale**
  - Derived from the `Accept-Language` HTTP header
- **Delta updates**
  - Efficient change detection using a stable content hash
- **Automatic background push**
  - Periodic push of context updates to an upstream message service

---

## Architecture Overview

The server runs **two modes of communication in parallel**:

1. **Request/Response API**
   - Clients can actively request the current context via HTTP.

2. **Server Push (Background Task)**
   - The server periodically generates a fresh context snapshot.
   - If the context has changed, it pushes the new context as JSON
     to a configured upstream service via HTTP POST.

Both modes operate **independently and concurrently**.

---

## Requirements

- Python **3.10+**
- Virtual environment recommended

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

# Running the Server
```bash
uvicorn app.main:app --reload
```
After startup, the API will be available at: `http://127.0.0.1:8000`
Interactive API documentation: `http://127.0.0.1:8000/docs`

---

# Environment Variables (Push Configuration)
The background push mechanism is fully configurable via environment variables:

```
PUSH_ENABLED=true
PUSH_WEBHOOK_URL=http://localhost:9000/context-events
PUSH_INTERVAL_SECONDS=600
```
PUSH_ENABLED =	Enables or disables server-side push
PUSH_WEBHOOK_URL =	Target URL of the upstream message service
PUSH_INTERVAL_SECONDS =	Push interval in seconds

---

# API Endpoints

## Health Check
`GET /health`
Response: `{ "ok": true }`

## Context Snapshot
`GET /context`
Optional query parameters:
- lat =	Latitude
- lon =	Longitude
- city =	City name
- countryCode =	ISO country code
- region =	Optional region code

Example:
```
curl "http://127.0.0.1:8000/context?lat=49.0069&lon=8.4037&city=Karlsruhe&countryCode=DE"
```

## Context Delta
`GET /context/delta?sinceHash=<hash>`
- If the hash is unchanged → empty delta {} is returned.
- If the hash differs → full context is returned.
Example:
```
curl "http://127.0.0.1:8000/context/delta?sinceHash=abcd1234"
```

---

# Server Push Behavior
When enabled:
- The server periodically generates a new context snapshot.
- A stable hash is calculated over the context.
- If the hash differs from the last pushed value:
- The following payload is POSTed to the upstream service:
```
{
  "source": "context_provider",
  "payload": { ...ContextEnvelope... }
}
```
This allows upstream systems to react to context changes without polling.

---
