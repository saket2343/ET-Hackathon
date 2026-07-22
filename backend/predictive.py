"""AXON predictive intelligence (MVP).

Anomaly detection (rolling-baseline deviation) + RUL estimate (trend
extrapolation to the ISO 10816 danger limit) on the pump's vibration stream.
Stand-in for the autoencoder/survival-model stack in the design doc.
"""
from __future__ import annotations

from datetime import datetime

ALERT_LIMIT = 4.5   # mm/s, ISO 10816 zone B/C
DANGER_LIMIT = 7.1  # mm/s, ISO 10816 zone C/D

# The asset this sensor stream belongs to. Named here so callers (e.g. the
# Asset360 endpoint) can tell which asset has live condition data instead
# of hard-coding the tag in several places.
ANCHOR_ASSET = "P-101"


def analyze(sensors: list[dict]) -> dict:
    vib = [r["vibration_mm_s"] for r in sensors]
    n = len(vib)
    baseline = sorted(vib[: n // 2])[len(vib[: n // 2]) // 2]  # median of first half
    latest = vib[-1]

    # anomaly: deviation of the recent mean from baseline
    recent = vib[-24:]
    recent_mean = sum(recent) / len(recent)
    anomaly_score = max(0.0, min(1.0, (recent_mean - baseline) / (DANGER_LIMIT - baseline)))
    is_anomalous = recent_mean > ALERT_LIMIT or (recent_mean - baseline) > 0.8

    # RUL: least-squares linear trend over the last 72 h, extrapolated to danger limit
    window = vib[-72:]
    m = len(window)
    xs = list(range(m))
    x_mean = sum(xs) / m
    y_mean = sum(window) / m
    denom = sum((x - x_mean) ** 2 for x in xs) or 1.0
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, window)) / denom  # mm/s per hour

    rul_hours = None
    if slope > 1e-4 and latest < DANGER_LIMIT:
        rul_hours = (DANGER_LIMIT - latest) / slope
    elif latest >= DANGER_LIMIT:
        rul_hours = 0.0

    temps = [r["bearing_temp_c"] for r in sensors[-24:]]
    return {
        "asset": "P-101",
        "signal": "VT-101 vibration (mm/s RMS)",
        "latest_vibration": round(latest, 2),
        "baseline_vibration": round(baseline, 2),
        "recent_mean": round(recent_mean, 2),
        "alert_limit": ALERT_LIMIT,
        "danger_limit": DANGER_LIMIT,
        "in_alert_zone": recent_mean >= ALERT_LIMIT,
        "anomaly": is_anomalous,
        "anomaly_score": round(anomaly_score, 2),
        "trend_mm_s_per_day": round(slope * 24, 2),
        "rul_hours": round(rul_hours, 1) if rul_hours is not None else None,
        "rul_days": round(rul_hours / 24, 1) if rul_hours is not None else None,
        "bearing_temp_recent_c": round(sum(temps) / len(temps), 1),
        "temp_elevated": (sum(temps) / len(temps)) > 85,
        "as_of": sensors[-1]["timestamp"],
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }
