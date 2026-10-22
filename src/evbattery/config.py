"""Project-wide settings and explicit possible-leakage exclusion policy."""
from __future__ import annotations

TARGET = "battery_failure"
IDENTIFIERS = ("vehicle_id", "battery_serial")
# These are composite / precomputed scores. Their information availability
# before the target outcome is not documented in the uploaded dataset.
SUSPECT_SCORES = (
    "thermal_runaway_risk",
    "battery_stress_index",
    "aging_score",
    "thermal_health_score",
    "charging_quality_score",
    "driving_stress_score",
    "predicted_remaining_life_cycles",
    "maintenance_score",
)
RANDOM_STATE = 42
