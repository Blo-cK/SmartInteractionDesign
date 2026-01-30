from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.prompts import ChatPromptTemplate

from data_models.data_models import AgentState, NextActionDecision, NextActionDecisionType
from conversational_agents.agent_logic.base_decision_agent import BaseDecisionAgent
from large_language_models.llm_factory import llm_factory


MONITOR_BASE_URL = "http://152.53.32.66:5000"
SERVICE_ID_STATIC = "context_static"
SERVICE_ID_DYNAMIC = "context_dynamic"
SERVICE_ID_REGION_COUNTING = "region_counting"
SERVICE_ID_HEATMAP = "heatmap"
SERVICE_ID_LOUDNESS = "environment_loudness"
SERVICE_ID_BRIGHTNESS = "video_brightness"


class ConversationOnlyDecisionAgent(BaseDecisionAgent):
    """
    Decision agent that focuses on conversational behavior and optionally uses
    environment data from the context provider service.
    """

    def __init__(self) -> None:
        super().__init__()
        self.llm = llm_factory.get_llm()
        self._http_client = httpx.Client(timeout=3.0)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def next_action(self, agent_state: AgentState) -> NextActionDecision:
        """
        Decide what to do next.

        By default, the agent generates a normal answer. If environment data
        from the context provider is available, this data is summarized and
        passed to the LLM as additional "sensor information".
        """
        # Default: no special sensor handling
        default_decision = NextActionDecision(
            type=NextActionDecisionType.GENERATE_ANSWER,
            action=None,
            payload=None,
        )

        env_data = self._fetch_latest_environment_context()

        if not env_data:
            # No environment data available -> fall back to normal behavior
            return default_decision

        environment_information = self._build_environment_description(env_data)

        # Example placeholder for additional sensor/person data
        person_information = (
            "No dedicated person-specific sensor information is available."
        )

        sensor_description = self.aggregate_sensor_information_with_llm(
            agent_state=agent_state,
            person_information=person_information,
            environment_information=environment_information,
        )

        print("ENVIRONMENT INFO:", environment_information)
        print("LLM-aggregated sensor information:", sensor_description)

        return NextActionDecision(
            type=NextActionDecisionType.PROMPT_ADAPTION,
            action="sensor",
            payload={"sensor_information": sensor_description},
        )

    # -------------------------------------------------------------------------
    # Communication with OutputLayer monitor
    # -------------------------------------------------------------------------

    def _fetch_latest_from_service(self, service_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch the latest output message for a given service_id from the
        OutputLayer monitor API.

        Expects the monitor to provide:
            GET /api/history/<service_id>
        returning a list of messages; the newest entry is used.
        """
        url = f"{MONITOR_BASE_URL}/api/history/{service_id}"

        try:
            response = self._http_client.get(url)
            response.raise_for_status()
            messages = response.json()
        except Exception:
            return None

        if not isinstance(messages, list) or not messages:
            return None

        latest = messages[-1]
        result = latest.get("result")
        if isinstance(result, dict):
            return result

        return None

    def _fetch_latest_environment_context(self) -> Optional[Dict[str, Any]]:
        """
        Fetch static and dynamic environment context as well as region counting and heatmap data. Merge all information
        into a single dictionary.

        Static payload (context_static) typically contains:
          - location
          - dayMeta
          - holidays
          - holidaySummary
          - placeContext
          - locale

        Dynamic payload (context_dynamic) typically contains:
          - dateTime
          - weather_current
          - weather_forecast
          - weather_tomorrow
          - daylight
          - comfort

        Region Counting contains:
          - people_in_region

        Heatmap contains:
          - total_people
          - trend
        
        Environment loudness contains:
        - environment loudness

        Video brightness contains:
        - video brightness

        """
        static_payload = self._fetch_latest_from_service(SERVICE_ID_STATIC)
        dynamic_payload = self._fetch_latest_from_service(SERVICE_ID_DYNAMIC)
        region_counting = self._fetch_latest_from_service(SERVICE_ID_REGION_COUNTING)
        heatmap = self._fetch_latest_from_service(SERVICE_ID_HEATMAP)
        loudness_payload = self._fetch_latest_from_service(SERVICE_ID_LOUDNESS)
        brightness_payload = self._fetch_latest_from_service(SERVICE_ID_BRIGHTNESS)

        if not static_payload and not dynamic_payload and not region_counting and not heatmap and not loudness_payload and notbrightness_payload:
            return None

        merged: Dict[str, Any] = {}
        if static_payload:
            merged.update(static_payload)
        if dynamic_payload:
            merged.update(dynamic_payload)
        if region_counting:
            merged.update(region_counting)
        if heatmap:
            merged.update(heatmap)
        if loudness_payload:
            merged.update({"loudness_info": loudness_payload})
        if brightness_payload:
            merged.update({"brightness_info": brightness_payload})

        return merged

    # -------------------------------------------------------------------------
    # Environment text construction
    # -------------------------------------------------------------------------

    def _build_environment_description(self, env: Dict[str, Any]) -> str:
        """
        Turn the merged environment context into a compact, human-readable text
        that can be passed to the LLM as additional context.
        """
        location = env.get("location") or {}
        date_time = env.get("dateTime") or {}
        day_meta = env.get("dayMeta") or {}
        holidays = env.get("holidays") or []
        holiday_summary = env.get("holidaySummary") or {}
        weather_current = env.get("weather_current") or {}
        weather_forecast = env.get("weather_forecast") or []
        weather_tomorrow = env.get("weather_tomorrow") or {}
        daylight = env.get("daylight") or {}
        comfort = env.get("comfort") or {}
        place_ctx = env.get("placeContext") or {}
        locale = env.get("locale") or {}
        people_in_region = env.get("people_in_region") or 0
        total_people = env.get("total_people") or 0
        trend = env.get("trend") or ""
        loudness_info = env.get("loudness_info") or {}
        scaled_value = loudness_info.get("loudness_scaled")
        brightness_info = env.get("brightness_info") or {}
        b_scale = brightness_info.get("brightness_scaled")
        events_data = env.get("events") or {}
        local_today = events_data.get("localToday") or []

        parts: List[str] = []

        # --- Time / date ---
        iso_value = date_time.get("iso")
        timezone_name = date_time.get("timezone") or "local time zone"
        weekday = date_time.get("weekday")
        part_of_day = date_time.get("partOfDay")

        pretty_date_str: Optional[str] = None
        pretty_time_str: Optional[str] = None

        if iso_value:
            try:
                dt_obj = datetime.fromisoformat(iso_value)
                pretty_date_str = dt_obj.strftime("%A, %B %d, %Y")
                pretty_time_str = dt_obj.strftime("%H:%M")
            except ValueError:
                pretty_date_str = None
                pretty_time_str = None

        if pretty_date_str and pretty_time_str:
            parts.append(
                f"Today is {pretty_date_str}, around {pretty_time_str} in the local time zone {timezone_name}."
            )
        elif pretty_date_str:
            parts.append(
                f"Today is {pretty_date_str} in the local time zone {timezone_name}."
            )

        # --- Location ---
        city = location.get("city")
        country = location.get("countryCode")
        region = location.get("region")
        lat = location.get("lat")
        lon = location.get("lon")

        if city and country:
            if region:
                parts.append(
                    f"You and the user are located in {city}, {region}, {country}"
                    + (
                        f" (approx. latitude {lat}, longitude {lon})."
                        if lat is not None and lon is not None
                        else "."
                    )
                )
            else:
                parts.append(
                    f"You and the user are located in {city}, {country}"
                    + (
                        f" (approx. latitude {lat}, longitude {lon})."
                        if lat is not None and lon is not None
                        else "."
                    )
                )
        elif country:
            parts.append(f"You and the user are located in {country}.")

        parts.append(
            "All environment information below is derived from the environment service and refers to this shared location and its local time."
        )

        # --- Locale ---
        lang = locale.get("language")
        loc_str = locale.get("locale")
        if lang and loc_str:
            parts.append(
                f"The preferred language and locale are '{lang}' and '{loc_str}'."
            )

        # --- Day meta / weekday ---
        if weekday and part_of_day:
            parts.append(f"It is {weekday} and currently {part_of_day} at this location.")

        if day_meta.get("isWeekend"):
            parts.append(
                "It is a weekend, so asking how their weekend is going is very natural."
            )
        elif day_meta.get("isBridgeDay"):
            parts.append(
                "Today is a bridge day close to a public holiday, which can also be used as a topic."
            )

        # --- Holidays list ---
        if isinstance(holidays, list) and holidays:
            lines: List[str] = ["Public holidays for this region in the current year are:"]
            for h in holidays:
                h_date = h.get("date")
                h_name = h.get("localName")
                if h_date and h_name:
                    lines.append(f"- {h_date}: {h_name}")
            parts.append("\n".join(lines))

        # --- Holiday summary (last / next / nearest) ---
        if isinstance(holiday_summary, dict):
            last_h = holiday_summary.get("lastHoliday") or {}
            next_h = holiday_summary.get("nextHoliday") or {}
            nearest_h = holiday_summary.get("nearestHoliday") or {}

            last_name = last_h.get("localName")
            last_date = last_h.get("date")
            last_days_ago = last_h.get("daysAgo")

            if last_name and last_date is not None:
                if isinstance(last_days_ago, int):
                    parts.append(
                        f"The most recent public holiday here was {last_name} on {last_date}, which was about {last_days_ago} days ago and might still be fresh in their memory."
                    )
                else:
                    parts.append(
                        f"The most recent public holiday here was {last_name} on {last_date}."
                    )

            next_name = next_h.get("localName")
            next_date = next_h.get("date")
            next_days = next_h.get("daysUntil")

            if next_name and next_date is not None:
                if isinstance(next_days, int):
                    parts.append(
                        f"The next upcoming public holiday here is {next_name} on {next_date}, which is in about {next_days} days and can naturally be used as a topic when asking about their plans."
                    )
                else:
                    parts.append(
                        f"The next upcoming public holiday here is {next_name} on {next_date}."
                    )

            nearest_name = nearest_h.get("localName")
            nearest_date = nearest_h.get("date")
            nearest_diff = nearest_h.get("daysDifference")
            nearest_is_past = nearest_h.get("isPast")

            if nearest_name and nearest_date is not None and isinstance(
                nearest_diff, int
            ):
                if nearest_is_past:
                    parts.append(
                        f"The closest holiday in time relative to today is {nearest_name} on {nearest_date}, which lies about {nearest_diff} days in the past."
                    )
                else:
                    parts.append(
                        f"The closest holiday in time relative to today is {nearest_name} on {nearest_date}, which lies about {nearest_diff} days in the future."
                    )

        # --- Weather (current) ---
        temp = weather_current.get("temperatureC")
        precip = weather_current.get("precipitationMm")
        wind = weather_current.get("windKph")

        if temp is not None:
            parts.append(
                f"The current weather at this shared location is around {temp:.1f} °C."
            )
        if wind is not None:
            parts.append(f"Wind speeds are roughly {wind:.1f} km/h.")
        if precip is not None and precip > 0:
            parts.append(f"There is some precipitation (about {precip:.1f} mm).")

        # --- Weather (forecast next hours) ---
        if isinstance(weather_forecast, list) and weather_forecast:
            temps = [
                fp.get("temperatureC")
                for fp in weather_forecast
                if fp.get("temperatureC") is not None
            ]
            precs = [
                fp.get("precipitationMm")
                for fp in weather_forecast
                if fp.get("precipitationMm") is not None
            ]

            if temps:
                min_t = min(temps)
                max_t = max(temps)
                parts.append(
                    f"In the next hours the temperature is expected to range roughly between {min_t:.1f} °C and {max_t:.1f} °C."
                )

            if precs:
                max_prec = max(precs)
                if max_prec < 0.1:
                    parts.append(
                        "No significant precipitation is expected in the next hours."
                    )
                else:
                    parts.append("Some precipitation is possible in the next hours.")

                # Basic warnings for precipitation and freezing conditions
                if temp is not None and temp <= 0 and max_prec >= 1.0:
                    parts.append(
                        "With sub-zero temperatures and precipitation, there is a noticeable risk of snow and ice on roads and sidewalks."
                    )
                    if max_prec >= 5.0:
                        parts.append(
                            "If the user needs to travel, they should be careful because heavy snowfall is possible."
                        )
                elif temp is not None and temp <= 0 and max_prec > 0.0:
                    parts.append(
                        "Because temperatures are below zero and some precipitation is expected, icy conditions are possible."
                    )
                elif max_prec >= 5.0:
                    parts.append(
                        "There may be periods of heavier rain in the next hours."
                    )

            examples: List[str] = []
            for fp in weather_forecast[:4]:
                t_str = fp.get("time")
                t_temp = fp.get("temperatureC")
                t_prec = fp.get("precipitationMm")
                if t_str and t_temp is not None:
                    if t_prec is not None and t_prec > 0.1:
                        examples.append(
                            f"around {t_str}: {t_temp:.1f} °C with some precipitation"
                        )
                    else:
                        examples.append(
                            f"around {t_str}: {t_temp:.1f} °C, almost no precipitation"
                        )
            if examples:
                parts.append(
                    "Example hourly forecast for the next hours (local time): "
                    + "; ".join(examples)
                    + "."
                )

        # --- Weather (tomorrow) ---
        if isinstance(weather_tomorrow, dict) and weather_tomorrow:
            t_min = weather_tomorrow.get("temperatureMinC")
            t_max = weather_tomorrow.get("temperatureMaxC")
            t_prec = weather_tomorrow.get("precipitationMm")

            if t_min is not None and t_max is not None:
                parts.append(
                    f"For tomorrow, the weather at this location is expected to be between {t_min:.1f} °C and {t_max:.1f} °C."
                )
            if t_prec is not None and t_prec > 0:
                parts.append("There is a chance of precipitation tomorrow as well.")

        # --- Daylight ---
        if daylight:
            is_daylight = daylight.get("isDaylight")
            minutes_to_sunrise = daylight.get("minutesUntilSunrise")
            minutes_to_sunset = daylight.get("minutesUntilSunset")

            sunrise_str = daylight.get("sunrise")
            sunset_str = daylight.get("sunset")

            if sunrise_str and sunset_str:
                parts.append(
                    f"Sunrise today is around {sunrise_str}, and sunset is around {sunset_str} (local time)."
                )

            if is_daylight is True:
                parts.append("It is currently daylight at this location.")
                if minutes_to_sunset is not None:
                    parts.append(
                        f"The sun will set in about {minutes_to_sunset} minutes."
                    )
            elif is_daylight is False:
                parts.append("It is currently dark outside at this location.")
                if minutes_to_sunrise is not None:
                    parts.append(
                        f"The sun will rise in about {minutes_to_sunrise} minutes."
                    )

        # --- Comfort / outdoor hints ---
        if comfort:
            cold_risk = comfort.get("coldRisk")
            heat_risk = comfort.get("heatRisk")
            ice_risk = comfort.get("iceRisk")
            outdoor_recommended = comfort.get("outdoorRecommended")

            if cold_risk in ("medium", "high"):
                parts.append("Conditions are rather cold for being outside.")
            if heat_risk in ("medium", "high"):
                parts.append("Temperatures may feel quite warm outside.")
            if ice_risk in ("medium", "high"):
                parts.append(
                    "There is a noticeable risk of icy conditions, so extra care is advisable when moving outside."
                )
            if outdoor_recommended is False:
                parts.append(
                    "Overall conditions are not ideal for longer outdoor activities, so indoor-focused topics or suggestions may be more appropriate."
                )

        # --- Place context ---
        if place_ctx:
            place_type = place_ctx.get("placeType")
            raw_cat = place_ctx.get("rawCategory")
            raw_type = place_ctx.get("rawType")
            nearby_cats = place_ctx.get("nearbyCategories") or []

            if place_type and place_type != "unknown":
                parts.append(
                    f"The underlying place information suggests category '{place_type}'."
                )
            elif raw_cat or raw_type:
                parts.append("There is additional place information available nearby.")

            if nearby_cats:
                uniq = sorted(set(str(c) for c in nearby_cats))
                parts.append(
                    "Nearby points of interest include categories such as "
                    + ", ".join(uniq)
                    + "."
                )

        # --- Environment loudness ---
        if scaled_value is not None:
            if scaled_value <= 3:
                status = "very quiet and peaceful. Actively say that you find it pleasantly quiet here"
            elif scaled_value <= 7:
                status = "moderately noisy with some background activity"
            else:
                status = "very loud and busy. Actively say that you find it quite loud here"
    
            parts.append(f"The current ambient noise in the room is a {scaled_value} out of 10 and therefore {status}.")

        # --- Video brightness ---
        if b_scale is not None:
            if b_scale <= 3:
                light_desc = "dimly lit or dark. Actively say that you can hardly see anything because it is so dark."
            elif b_scale <= 7:
                light_desc = "well-lit with comfortable brightness"
            else:
                light_desc = "very bright, possibly intense light"
            
            parts.append(f"The room's brightness level is {b_scale} out of 10, so it is currently {light_desc}.")

        # --- Region counting ---
        if people_in_region:
            match people_in_region:
                case 0:
                    parts.append("There is currently no person in the region, please try to actively get a person to come closer.")
                case 1:
                    parts.append("There is only one person in the region, please address only one person while speaking.")
                case _:
                    parts.append("There are multiple people in the region, please address multiple people while speaking.")
            parts.append("If you are asked about the amount of people taking part in the conversation answer with " + str(people_in_region) + " people.")

        # --- Heatmap ---
        if total_people:
            if total_people == 0:
                parts.append("There are currently no people in the frame. If you are talking to a user, please tell them to come closer.")
            elif total_people < 5:
                parts.append("There are currently not many people in the frame.")
            elif total_people > 15:
                parts.append("There are currently many people in the frame.")
            parts.append("In total you are able to see " + str(total_people) + " people inside the frame.")

        if trend != "":
            if trend == "increasing":
                parts.append("The amount of people is currently increasing in the frame.")
            else:
                parts.append("The amount of people is currently decreasing in the frame.")

        if not parts:
            return (
                "No reliable environment context could be derived from the available "
                "sensor data."
            )

        # --- Local Events ---
        if isinstance(local_today, list) and len(local_today) > 0:
            event_lines = ["The following local events are happening in the city today, please remmember all of them:"]

            """
            Use the last 10 events of the day (useful for afternoon events). 
            Consider adjusting how many events are read out, as this can quickly generate a lot of text.
            """

            for event in local_today[-10:]:
                title = event.get("title", "Event")
                start = event.get("startTime", "")
                desc = event.get("description", "")
                
                time_info = ""
                if start and "T" in start:
                    time_info = f" at {start.split('T')[1][:5]}"
                
                # The entire description of an event (or only the first e.g. 100 characters with desc[:100])
                event_lines.append(f"- {title}{time_info}: {desc}")
            parts.append("\n".join(event_lines))
        else:
            parts.append("There are no specific local events listed in the calendar for today.")

        return " ".join(parts)

    # -------------------------------------------------------------------------
    # LLM aggregation
    # -------------------------------------------------------------------------

    """
    Consider shortening the events prompt. So far, the texts have been very long because all the information 
    about the events is to be included for the sake of completeness. If no detailed information is to be provided, 
    it may be sufficient to include only the title, location, and date of the events, for example.
    """

    def aggregate_sensor_information_with_llm(
                self,
                agent_state: AgentState,
                person_information: str,
                environment_information: str,
        ) -> str:
            """
            Aggregate person and environment information into a compact description
            that can be passed as context into the main conversation logic.
            The goal is to keep the text short but still preserve concrete details
            like dates, times and temperatures.
            """
            sensor_data_aggregation_prompt = """
    The following is the recent conversation between you and the user:
    {chat_history}

    Here is structured information about the environment and situation:
    {sensor_data}

    Write a short description (about 6-8 sentences) that can be used as additional
    context for the conversation. Explicitly include:
    - today's date and the approximate local time (e.g. "around 02:30"),
    - where you and the user are located (city, region, country),
    - whether it is a weekday or weekend and roughly which part of the day,
    - which public holiday was most recent and which one is next, including their dates,
    - the current weather (temperature, wind, precipitation),
    - how the weather will develop in the next few hours and tomorrow. 
    - how many people should be addressed while talking.
    - how many people are taking part in the conversation.
    - how many people are in the frame.
    - whether the amount of people in the current location is increasing or decreasing.
    - the current ambient loudness level and if it's currently quiet or noisy.  
    - the current room brightness and if it is dark or bright.                  
    - IMPORTANT: List ALL local events mentioned in the sensor data with their titles and times. Do not shorten this list, as the user might ask for all of them.

    Keep the wording compact but do not omit concrete numerical details such as
    dates, temperatures, times or day distances if they appear in the data.
    Only use information that is present in the sensor data; do not invent facts.
    Please remember to switch from singular you to plural you if multiple people are taking part in the conversation.
    If there are multiple people taking part in the conversation, assume everyone is talking to you.
    """

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You receive structured information about the user and environment. "
                        "Summarise it briefly so it can be used as context for a conversation. "
                        "Preserve important concrete details like dates, times, temperatures and "
                        "distances to holidays, while keeping the text readable and concise."
                        "Please remember to switch from singular you to plural you if multiple people are taking part in the conversation.",
                    ),
                    ("human", sensor_data_aggregation_prompt),
                ]
            )

            chain = prompt | self.llm

            response = chain.invoke(
                {
                    "chat_history": agent_state.chat_history,
                    "sensor_data": "\n".join(
                        part for part in [person_information, environment_information] if part
                    ),
                }
            )

            return response.content