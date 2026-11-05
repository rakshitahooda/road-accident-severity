"""Shared contract: training and inference use exactly the same raw features."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CATEGORICAL = ["Day_of_week", "Area_accident_occured", "Types_of_Junction", "Road_surface_type", "Road_surface_conditions", "Light_conditions", "Weather_conditions", "Type_of_vehicle"]
NUMERIC = ["Hour", "Number_of_vehicles_involved"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Accident_severity"
SEED = 42
SOURCE = "https://raw.githubusercontent.com/avikumart/Road-Traffic-Severity-Classification-Project/main/RTA%20Dataset.csv"
SHA256 = "975716a9e7014e5f49eaca3cf7ba282f3fa7a36db3a30fca79d7cb42e06df053"


def features_from_raw(raw):
    frame = raw.copy()
    frame["Hour"] = pd.to_datetime(frame["Time"], format="%H:%M:%S", errors="coerce").dt.hour
    return frame[FEATURES]


def validate_record(record, schema):
    """Reject silent schema drift; missing values must be explicit nulls."""
    if set(record) != set(FEATURES):
        raise ValueError("Expected exactly these fields: " + ", ".join(FEATURES))
    result = {}
    for key in NUMERIC:
        value = record[key]
        if value is None:
            result[key] = float("nan")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not float(value).is_integer():
            raise ValueError(f"{key} must be an integer or null")
        low, high = (0, 23) if key == "Hour" else (1, 20)
        if not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
        result[key] = value
    for key in CATEGORICAL:
        value = record[key]
        if value is not None and (not isinstance(value, str) or value not in schema["categories"][key]):
            raise ValueError(f"{key}: select a supported category or null")
        result[key] = value if value is not None else float("nan")
    return result
