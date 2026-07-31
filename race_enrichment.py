from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


@dataclass(frozen=True)
class CourseMetadata:
    name: str
    country: str
    latitude: float
    longitude: float
    region: str


COURSE_METADATA: dict[str, CourseMetadata] = {
    "aintree": CourseMetadata("Aintree", "GB", 53.4769, -2.9401, "England"),
    "ascot": CourseMetadata("Ascot", "GB", 51.4140, -0.6770, "England"),
    "cheltenham": CourseMetadata("Cheltenham", "GB", 51.9197, -2.0685, "England"),
    "doncaster": CourseMetadata("Doncaster", "GB", 53.5229, -1.1060, "England"),
    "epsom": CourseMetadata("Epsom", "GB", 51.3091, -0.2540, "England"),
    "goodwood": CourseMetadata("Goodwood", "GB", 50.8942, -0.7393, "England"),
    "kempton": CourseMetadata("Kempton", "GB", 51.4209, -0.4066, "England"),
    "lingfield": CourseMetadata("Lingfield", "GB", 51.1693, -0.0038, "England"),
    "newbury": CourseMetadata("Newbury", "GB", 51.3986, -1.3075, "England"),
    "newmarket": CourseMetadata("Newmarket", "GB", 52.2445, 0.4061, "England"),
    "york": CourseMetadata("York", "GB", 53.9399, -1.0973, "England"),
    "curragh": CourseMetadata("Curragh", "IE", 53.1566, -6.8518, "Ireland"),
    "leopardstown": CourseMetadata("Leopardstown", "IE", 53.2660, -6.1904, "Ireland"),
}

ENRICHMENT_COLUMNS = [
    "country",
    "course_latitude",
    "course_longitude",
    "distance_bucket",
    "going_category",
    "race_type",
]
MODEL_ENRICHMENT_COLUMNS = ["country", "distance_bucket", "going_category", "race_type"]
OPEN_METEO_DAILY_VARIABLES = [
    "weather_code",
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "soil_moisture_0_to_7cm_mean",
]


def normalize_track_name(value: Any) -> str:
    return str(value or "").strip().lower().replace(" racecourse", "")


def course_metadata_for(track: Any) -> CourseMetadata | None:
    return COURSE_METADATA.get(normalize_track_name(track))


def distance_to_yards(distance: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([distance]), errors="coerce").iloc[0]
    if not pd.isna(numeric):
        return float(numeric)

    text = str(distance or "").lower().replace(" ", "")
    if not text:
        return None

    miles = furlongs = yards = 0.0
    try:
        if "m" in text:
            head, text = text.split("m", 1)
            miles = float(head or 0)
        if "f" in text:
            head, text = text.split("f", 1)
            furlongs = float(head or 0)
        if "y" in text:
            yards = float(text.split("y", 1)[0] or 0)
    except ValueError:
        return None

    parsed = miles * 1760 + furlongs * 220 + yards
    return parsed or None


def distance_bucket(distance: Any) -> str | None:
    yards = distance_to_yards(distance)
    if yards is None:
        return None
    if yards < 1540:
        return "sprint"
    if yards < 2200:
        return "mile"
    if yards < 3080:
        return "middle"
    return "staying"


def going_category(*values: Any) -> str | None:
    text = " ".join(str(value or "").lower() for value in values)
    if not text.strip():
        return None
    if any(token in text for token in ["heavy", "muddy"]):
        return "heavy"
    if any(token in text for token in ["soft", "yielding"]):
        return "soft"
    if "firm" in text:
        return "firm"
    if "good" in text:
        return "good"
    if any(token in text for token in ["polytrack", "tapeta", "all-weather", "aw", "dirt"]):
        return "all_weather"
    if "turf" in text:
        return "turf"
    return "unknown"


def infer_race_type(*values: Any) -> str:
    text = " ".join(str(value or "").lower() for value in values)
    if any(token in text for token in ["hurdle", "chase", "nhf", "national hunt", "jumps"]):
        return "jumps"
    if any(token in text for token in ["polytrack", "tapeta", "all-weather", "aw", "dirt"]):
        return "flat_aw"
    if "turf" in text or "flat" in text:
        return "flat_turf"
    return "unknown"


def enrich_race_frame(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()
    if "track" not in work_df.columns:
        work_df["track"] = None
    if "distance" not in work_df.columns:
        work_df["distance"] = None
    if "surface" not in work_df.columns:
        work_df["surface"] = None
    if "weather" not in work_df.columns:
        work_df["weather"] = None

    metadata = work_df["track"].map(course_metadata_for)
    work_df["country"] = metadata.map(lambda item: item.country if item else None)
    work_df["course_latitude"] = metadata.map(lambda item: item.latitude if item else None)
    work_df["course_longitude"] = metadata.map(lambda item: item.longitude if item else None)
    work_df["distance_bucket"] = work_df["distance"].map(distance_bucket)
    work_df["going_category"] = [
        going_category(surface, weather)
        for surface, weather in zip(work_df["surface"], work_df["weather"], strict=False)
    ]
    work_df["race_type"] = [
        infer_race_type(surface, weather)
        for surface, weather in zip(work_df["surface"], work_df["weather"], strict=False)
    ]
    return work_df


def open_meteo_daily_weather_url(latitude: float, longitude: float, target_date: str | date) -> str:
    race_date = target_date.isoformat() if isinstance(target_date, date) else str(target_date)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": race_date,
        "end_date": race_date,
        "daily": ",".join(OPEN_METEO_DAILY_VARIABLES),
        "timezone": "UTC",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    return f"https://archive-api.open-meteo.com/v1/archive?{urlencode(params)}"


def fetch_open_meteo_daily_weather(latitude: float, longitude: float, target_date: str | date, timeout: float = 10.0) -> dict[str, Any]:
    url = open_meteo_daily_weather_url(latitude, longitude, target_date)
    with urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    daily = payload.get("daily") or {}
    summary: dict[str, Any] = {"source": "open-meteo", "url": url}
    for field in OPEN_METEO_DAILY_VARIABLES:
        values = daily.get(field) or []
        summary[field] = values[0] if values else None
    return summary


def course_metadata_payload(track: Any) -> dict[str, Any] | None:
    metadata = course_metadata_for(track)
    return asdict(metadata) if metadata else None
