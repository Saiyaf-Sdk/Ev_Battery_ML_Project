"""Feature provenance catalogue for all 70 CSV columns (Member A audit).

Each column gets a group, a plain-language meaning and a modelling decision.
"Provenance" = where a value comes from and *when* it would be known. A feature
that is only known after (or computed from) the failure outcome is leakage.

Decisions
---------
keep            -> used as a model input
drop_id         -> identifier, no predictive meaning, would let a model memorise rows
drop_target     -> the label itself
exclude_suspect -> precomputed composite/risk score with undocumented origin;
                   excluded from the baseline, studied only in the ablation
drop_redundant  -> exact deterministic copy of another kept column
"""
from __future__ import annotations

from .config import IDENTIFIERS, REDUNDANT_EXACT, SUSPECT_SCORES, TARGET

# Columns that are exact formulas of other columns (verified in leakage_audit.py):
#   capacity_loss_percent = 100 - battery_health_percent
#   remaining_capacity    = battery_capacity_kwh * battery_health_percent / 100
#   manufacturing_year    ~ 2026 - vehicle_age_years
REDUNDANT_FORMULAS = {
    "capacity_loss_percent": "= 100 - battery_health_percent (exact, every row)",
    "remaining_capacity": "= battery_capacity_kwh x battery_health_percent / 100 (exact to rounding)",
    "manufacturing_year": "= 2026 - vehicle_age_years (to within ~1 year)",
}
assert set(REDUNDANT_FORMULAS) == set(REDUNDANT_EXACT), "keep config.py and feature_catalog.py in sync"

# group, meaning
CATALOG: dict[str, tuple[str, str]] = {
    "vehicle_id": ("identifier", "Unique vehicle code"),
    "battery_serial": ("identifier", "Unique battery serial number"),
    "battery_failure": ("target", "Label: 1 = failed battery, 0 = healthy"),
    # vehicle / static specification
    "vehicle_brand": ("vehicle", "Manufacturer brand (20 brands)"),
    "vehicle_model": ("vehicle", "Model name (42 models)"),
    "vehicle_type": ("vehicle", "Body type, e.g. SUV, Van"),
    "manufacturing_year": ("vehicle", "Year the vehicle was built"),
    "battery_manufacturer": ("vehicle", "Cell supplier"),
    "battery_chemistry": ("vehicle", "Cell chemistry: NMC, LFP, LMO, NCA, LTO"),
    "battery_capacity_kwh": ("vehicle", "Nominal pack capacity (kWh)"),
    "drive_type": ("vehicle", "AWD / RWD / FWD"),
    "fleet_or_private": ("vehicle", "Fleet or private ownership"),
    "odometer_km": ("usage", "Total distance driven (km)"),
    "vehicle_age_years": ("vehicle", "Age of the vehicle (years)"),
    "cycle_count": ("usage", "Lifetime full charge cycles"),
    # battery state (BMS measurements)
    "battery_health_percent": ("battery_state", "Battery health (% of original capacity)"),
    "state_of_charge": ("battery_state", "Current charge level (%)"),
    "depth_of_discharge": ("battery_state", "Typical depth of discharge (%)"),
    "state_of_health": ("battery_state", "BMS state-of-health estimate (%)"),
    "cell_voltage_avg": ("battery_state", "Mean cell voltage (V)"),
    "cell_voltage_std": ("battery_state", "Spread of cell voltages (V)"),
    "pack_voltage": ("battery_state", "Total pack voltage (V)"),
    "cell_temperature_avg": ("battery_state", "Mean cell temperature (C)"),
    "cell_temperature_max": ("battery_state", "Hottest cell temperature (C)"),
    "internal_resistance": ("battery_state", "Internal resistance (ohm, scaled)"),
    "charge_efficiency": ("battery_state", "Charging efficiency (%)"),
    "discharge_efficiency": ("battery_state", "Discharging efficiency (%)"),
    "remaining_capacity": ("battery_state", "Usable capacity left (kWh)"),
    "capacity_loss_percent": ("battery_state", "Capacity lost (%)"),
    # charging behaviour
    "charging_cycles_last_month": ("charging", "Charge sessions in the last month"),
    "fast_charge_ratio": ("charging", "Share of DC fast charging (0-1)"),
    "slow_charge_ratio": ("charging", "Share of slow AC charging (0-1)"),
    "average_charge_power_kw": ("charging", "Mean charging power (kW)"),
    "average_charging_time": ("charging", "Mean session length (min)"),
    "overnight_charging_ratio": ("charging", "Share of overnight sessions (0-1)"),
    "home_charging_ratio": ("charging", "Share of home charging (0-1)"),
    "charging_interruptions": ("charging", "Interrupted charging sessions"),
    "overcharge_events": ("charging", "Overcharge events recorded"),
    # driving behaviour
    "average_speed": ("driving", "Mean speed (km/h)"),
    "average_trip_distance": ("driving", "Mean trip length (km)"),
    "aggressive_acceleration_score": ("driving", "Hard acceleration score (0-100)"),
    "hard_braking_score": ("driving", "Hard braking score (0-100)"),
    "regenerative_braking_usage": ("driving", "Regenerative braking use (%)"),
    "highway_driving_ratio": ("driving", "Share of highway driving (0-1)"),
    "city_driving_ratio": ("driving", "Share of city driving (0-1)"),
    "daily_distance": ("driving", "Mean daily distance (km)"),
    # environment
    "average_ambient_temperature": ("environment", "Mean outside temperature (C)"),
    "maximum_temperature": ("environment", "Max outside temperature (C)"),
    "minimum_temperature": ("environment", "Min outside temperature (C)"),
    "humidity": ("environment", "Relative humidity (%)"),
    "altitude": ("environment", "Altitude (m)"),
    "terrain_type": ("environment", "Flat / Hilly / Mountainous / Coastal / Desert"),
    "dust_exposure": ("environment", "Dust exposure score (0-100)"),
    # maintenance & fault history
    "last_service_days": ("maintenance", "Days since last service"),
    "cooling_system_health": ("maintenance", "Cooling system health (%)"),
    "firmware_updates": ("maintenance", "Firmware updates installed"),
    "previous_faults": ("maintenance", "Earlier recorded faults"),
    "voltage_imbalance": ("battery_state", "Voltage imbalance between cells (mV)"),
    "temperature_variance": ("battery_state", "Variance of cell temperatures (clipped at 32.94)"),
    "sensor_fault_count": ("maintenance", "Sensor faults recorded"),
    "BMS_warning_count": ("maintenance", "Battery-management-system warnings"),
    "abnormal_voltage_events": ("battery_state", "Abnormal voltage events"),
    # composite / precomputed scores (undocumented formulas)
    "maintenance_score": ("composite_score", "Precomputed maintenance score"),
    "thermal_runaway_risk": ("composite_score", "Precomputed thermal-runaway risk"),
    "battery_stress_index": ("composite_score", "Precomputed stress index"),
    "aging_score": ("composite_score", "Precomputed ageing score"),
    "thermal_health_score": ("composite_score", "Precomputed thermal health score"),
    "charging_quality_score": ("composite_score", "Precomputed charging quality score"),
    "driving_stress_score": ("composite_score", "Precomputed driving stress score"),
    "predicted_remaining_life_cycles": ("composite_score", "Model-predicted remaining life (cycles)"),
}


def decision_for(column: str) -> tuple[str, str]:
    """Return (decision, reason) for one raw CSV column."""
    if column == TARGET:
        return "drop_target", "This is the label y; it can never be an input."
    if column in IDENTIFIERS:
        return "drop_id", "Unique per row; carries no physics and lets a model memorise rows."
    if column in SUSPECT_SCORES:
        return ("exclude_suspect",
                "Precomputed score with unknown formula/timing; may be built from the outcome. "
                "Excluded from baseline, tested only in the ablation study.")
    if column in REDUNDANT_FORMULAS:
        return "drop_redundant", "Exact copy: " + REDUNDANT_FORMULAS[column]
    return "keep", "Raw telemetry/specification plausibly available before the outcome."
