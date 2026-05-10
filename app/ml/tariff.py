"""
app/ml/tariff.py
────────────────
Time-of-use (TOU) tariff engine and load-shifting optimization.

Tariff structure
----------------
The default tariff mirrors a typical residential TOU schedule:

  Peak hours   (07:00–23:00 local / UTC)  →  TARIFF_PEAK_RATE     Rs/kWh
  Off-peak     (23:00–07:00)              →  TARIFF_OFF_PEAK_RATE  Rs/kWh

All rates and hour boundaries are configurable via app.config.Settings.

Optimization logic
------------------
Given a 24-hour power forecast:

1. Compute the cost for each forecasted hour at its applicable tariff rate.
2. Identify the CHEAPEST hours (best windows for deferrable loads).
3. Estimate potential savings if peak-hour consumption were shifted to
   the cheapest off-peak window.
4. Generate plain-language tips for the user.

Usage
-----
    from app.ml.tariff import compute_tariff_optimization

    opt = compute_tariff_optimization(forecast_result, now)
    # opt.total_cost_rs, opt.potential_savings_rs, opt.tips …
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from app.ml.forecasting import ForecastResult, HourlyForecastPoint

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# NEPRA Residential Tariff Slabs  (NEPRA Schedule of Tariffs)
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: (max_monthly_units, Rs_per_kWh)
# The slab rate applies to ALL units consumed that month (flat, not progressive).

# Protected / Lifeline consumers (sanctioned load < 5 kW, low-income)
_PROTECTED_SLABS: list[tuple[float, float]] = [
    (50,         3.95),   # Up to 50 units  – Lifeline
    (100,        7.74),   # Up to 100 units – Lifeline
    (100,       10.54),   # 001–100 units   – Protected
    (200,       13.01),   # 101–200 units   – Protected
]

# Un-Protected consumers (sanctioned load < 5 kW, standard)
_UNPROTECTED_SLABS: list[tuple[float, float]] = [
    (100,       22.44),   # 1–100 units
    (200,       28.91),   # 101–200 units
    (300,       33.10),   # 201–300 units
    (400,       36.46),   # 301–400 units
    (500,       38.95),   # 401–500 units
    (600,       40.22),   # 501–600 units
    (700,       41.85),   # 601–700 units
    (float("inf"), 47.20),  # Above 700 units
]

# TOU rates for sanctioned load >= 5 kW
NEPRA_TOU_PEAK_RATE: float    = 46.85   # Rs/kWh  (5 PM – 5 AM PKT)
NEPRA_TOU_OFFPEAK_RATE: float = 34.53   # Rs/kWh  (5 AM – 5 PM PKT)


def get_nepra_slab_rate(
    monthly_kwh: float,
    protected: bool = False,
) -> tuple[float, str]:
    """
    Return (rate_rs_per_kwh, slab_label) for a given monthly consumption.

    Parameters
    ----------
    monthly_kwh :
        Total kWh consumed in the current billing month.
    protected :
        True for lifeline / protected consumers (very low income).

    Notes
    -----
    Pakistan's NEPRA residential tariff is a FLAT slab system: the entire
    monthly consumption is billed at the applicable slab rate (not progressive).
    """
    slabs = _PROTECTED_SLABS if protected else _UNPROTECTED_SLABS

    for max_units, rate in slabs:
        if monthly_kwh <= max_units:
            if max_units == float("inf"):
                label = "Above 700 units"
            else:
                label = f"Up to {int(max_units)} units"
            return rate, label

    # Fallback – should never reach here
    last_rate = slabs[-1][1]
    return last_rate, "Above 700 units"


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TariffSlot:
    """Cost-annotated forecast point."""
    hour_offset: int
    hour_bucket: datetime
    hour_of_day: int               # 0-23 (UTC)
    label: str                     # "peak" | "off_peak"
    rate_per_kwh: float            # Rs / kWh
    predicted_power_w: float       # Watts
    predicted_kwh: float           # kWh consumed in this 1-hour slot
    cost_rs: float                 # Rs for this slot
    confidence: str                # "high" | "medium" | "low"


@dataclass
class TariffOptimization:
    """
    Full tariff cost analysis + load-shifting recommendations
    for the next 24 hours.
    """
    device_id: str
    generated_at: datetime

    # Per-hour cost breakdown (24 items)
    slots: list[TariffSlot] = field(default_factory=list)

    # 24-hour totals
    total_predicted_kwh: float = 0.0
    total_cost_rs: float = 0.0
    peak_cost_rs: float = 0.0
    offpeak_cost_rs: float = 0.0

    # Cheapest / most-expensive windows
    cheapest_hours: list[int] = field(default_factory=list)       # hour-of-day values
    most_expensive_hours: list[int] = field(default_factory=list) # hour-of-day values

    # Savings potential
    shiftable_kwh: float = 0.0          # kWh that could realistically be shifted
    potential_savings_rs: float = 0.0   # Rs saved if shiftable_kwh moved to off-peak

    # Plain-language recommendations
    tips: list[str] = field(default_factory=list)
    alert_level: str = "info"           # "info" | "warning" | "saving_opportunity"


# ─────────────────────────────────────────────────────────────────────────────
# Tariff helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_tariff_label_and_rate(
    hour_of_day: int,
    peak_start: int,
    peak_end: int,
    peak_rate: float,
    offpeak_rate: float,
) -> tuple[str, float]:
    """
    Return ("peak"|"off_peak", rate_rs_per_kwh) for a given hour-of-day.

    Peak window is [peak_start, peak_end). E.g. peak_start=7, peak_end=23
    means hours 7,8,…,22 are peak and 23,0,1,…,6 are off-peak.
    """
    if peak_start < peak_end:
        is_peak = peak_start <= hour_of_day < peak_end
    else:
        # Wraps midnight, e.g. peak 22:00–06:00
        is_peak = hour_of_day >= peak_start or hour_of_day < peak_end

    if is_peak:
        return "peak", peak_rate
    return "off_peak", offpeak_rate


# ─────────────────────────────────────────────────────────────────────────────
# Optimization tips builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_tips(
    slots: list[TariffSlot],
    cheapest_hours: list[int],
    most_expensive_hours: list[int],
    potential_savings_rs: float,
    shiftable_kwh: float,
    currency: str,
) -> tuple[list[str], str]:
    """Return (tips list, alert_level)."""
    tips: list[str] = []
    alert_level = "info"

    # Cheapest hours notice
    if cheapest_hours:
        hours_str = ", ".join(f"{h:02d}:00" for h in cheapest_hours[:4])
        tips.append(
            f"Cheapest hours in the next 24 h: {hours_str}. "
            "Schedule deferrable loads (washing machine, EV charging, water heater) then."
        )

    # Most expensive hours notice
    if most_expensive_hours:
        hours_str = ", ".join(f"{h:02d}:00" for h in most_expensive_hours[:3])
        tips.append(
            f"Highest-cost hours: {hours_str}. Avoid running heavy appliances during these windows."
        )

    # Savings opportunity
    if potential_savings_rs >= 1.0:
        tips.append(
            f"Load-shifting opportunity: moving {shiftable_kwh:.2f} kWh of peak consumption "
            f"to off-peak hours could save approximately {currency} {potential_savings_rs:.2f}."
        )
        alert_level = "saving_opportunity"

    # Off-peak overnight tip
    offpeak_night = [s for s in slots if s.label == "off_peak" and s.predicted_kwh > 0.05]
    if offpeak_night:
        avg_offpeak = sum(s.rate_per_kwh for s in offpeak_night) / len(offpeak_night)
        peak_slots = [s for s in slots if s.label == "peak"]
        avg_peak = (
            sum(s.rate_per_kwh for s in peak_slots) / len(peak_slots) if peak_slots else avg_offpeak
        )
        discount_pct = round((avg_peak - avg_offpeak) / avg_peak * 100) if avg_peak > 0 else 0
        if discount_pct > 0:
            tips.append(
                f"Off-peak electricity is {discount_pct}% cheaper than peak rates. "
                "Running large appliances overnight maximises your savings."
            )

    # High consumption warning
    peak_slots_high = [s for s in slots if s.label == "peak" and s.predicted_power_w > 1500]
    if peak_slots_high:
        tips.append(
            f"{len(peak_slots_high)} hour(s) of high consumption (>1500 W) predicted during "
            "peak tariff — consider spreading this load to reduce your bill."
        )
        alert_level = "warning" if alert_level != "saving_opportunity" else alert_level

    if not tips:
        tips.append(
            "Your predicted consumption pattern looks cost-efficient. "
            "Continue monitoring to maintain low energy costs."
        )

    return tips, alert_level


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_tariff_optimization(
    forecast: ForecastResult,
    peak_start_hour: int = 7,
    peak_end_hour: int = 23,
    peak_rate: float = 25.0,
    offpeak_rate: float = 15.0,
    currency: str = "Rs",
) -> TariffOptimization:
    """
    Annotate a ForecastResult with tariff costs and load-shifting advice.

    Parameters
    ----------
    forecast :
        Output of generate_forecast().
    peak_start_hour :
        First peak-tariff hour (inclusive, 0-23 UTC).  Default = 7 (07:00).
    peak_end_hour :
        First off-peak hour after peak window (exclusive, 0-23 UTC). Default = 23 (23:00).
    peak_rate :
        Rs per kWh during peak hours.
    offpeak_rate :
        Rs per kWh during off-peak hours.
    currency :
        Currency symbol for tips text.

    Returns
    -------
    TariffOptimization populated with per-slot costs and recommendations.
    """
    slots: list[TariffSlot] = []

    for point in forecast.points:
        hour_of_day = point.hour_bucket.hour
        label, rate = get_tariff_label_and_rate(
            hour_of_day, peak_start_hour, peak_end_hour, peak_rate, offpeak_rate
        )
        cost = round(point.predicted_kwh * rate, 4)
        slots.append(
            TariffSlot(
                hour_offset=point.hour_offset,
                hour_bucket=point.hour_bucket,
                hour_of_day=hour_of_day,
                label=label,
                rate_per_kwh=rate,
                predicted_power_w=point.predicted_power_w,
                predicted_kwh=point.predicted_kwh,
                cost_rs=cost,
                confidence=point.confidence,
            )
        )

    # ── Totals ────────────────────────────────────────────────────────────────
    total_kwh = round(sum(s.predicted_kwh for s in slots), 4)
    total_cost = round(sum(s.cost_rs for s in slots), 4)
    peak_cost = round(sum(s.cost_rs for s in slots if s.label == "peak"), 4)
    offpeak_cost = round(total_cost - peak_cost, 4)

    # ── Cheapest / most-expensive windows ────────────────────────────────────
    sorted_by_cost = sorted(slots, key=lambda s: s.cost_rs)
    cheapest_hours = [s.hour_of_day for s in sorted_by_cost[:6] if s.cost_rs >= 0]
    most_expensive = [s.hour_of_day for s in sorted(slots, key=lambda s: -s.cost_rs)[:4]]

    # ── Savings estimation ────────────────────────────────────────────────────
    # Shiftable = peak-hour kWh above a realistic "always-on" baseline.
    # We assume 20 % of peak consumption is non-shiftable (lighting, idle devices).
    peak_slots = [s for s in slots if s.label == "peak"]
    total_peak_kwh = sum(s.predicted_kwh for s in peak_slots)
    shiftable_kwh = round(total_peak_kwh * 0.80, 4)   # 80 % of peak is shiftable

    # Cost if that shiftable kWh were moved to the cheapest off-peak rate
    savings_rs = round(shiftable_kwh * (peak_rate - offpeak_rate), 4)
    savings_rs = max(0.0, savings_rs)

    # ── Tips ──────────────────────────────────────────────────────────────────
    tips, alert_level = _build_tips(
        slots, cheapest_hours, most_expensive, savings_rs, shiftable_kwh, currency
    )

    return TariffOptimization(
        device_id=forecast.device_id,
        generated_at=forecast.generated_at,
        slots=slots,
        total_predicted_kwh=total_kwh,
        total_cost_rs=total_cost,
        peak_cost_rs=peak_cost,
        offpeak_cost_rs=offpeak_cost,
        cheapest_hours=cheapest_hours,
        most_expensive_hours=most_expensive,
        shiftable_kwh=shiftable_kwh,
        potential_savings_rs=savings_rs,
        tips=tips,
        alert_level=alert_level,
    )
