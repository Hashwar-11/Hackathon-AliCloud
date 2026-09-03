"""
Verification Agent — cuts false positives by cross-checking flagged accounts
against non-telemetry operational and grid context BEFORE field-inspection tickets
are dispatched. Deliberately rule-based and transparent for utility auditability.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class CustomerContext:
    consumer_id: str
    feeder_id: str = "FEEDER-01"
    transformer_id: str = "TX-01"
    tariff_category: str = "residential"       # residential, commercial, industrial, agricultural
    sanctioned_load_kw: float = 5.0
    feeder_loss_pct: float = 12.0              # Distribution line loss percentage on this feeder
    recent_audit_result: str = "none"          # "cleared", "confirmed_theft", "meter_fault", "none"
    months_since_last_audit: int = 999
    billing_dispute_open: bool = False
    known_grid_topology_issue: bool = False    # e.g., CT/PT miscalibration, phase imbalance
    has_rooftop_solar: bool = False
    solar_net_metering_active: bool = False
    is_seasonal_occupancy: bool = False        # e.g., agricultural tube-well, vacation home
    tamper_event_count: int = 0                # Smart meter hardware tamper logs (magnetic/lid/tilt)
    meter_seal_broken: bool = False
    reverse_current_alert: bool = False
    historical_mean_kwh: float = 15.0          # Baseline consumption
    recent_30d_mean_kwh: float = 5.0           # Recent consumption
    tariff_rate_per_kwh: float = 35.0          # Local currency (PKR/kWh, $/kWh, etc.)


def verify_flag(context: CustomerContext, raw_theft_probability: float) -> Dict[str, Any]:
    """
    Applies utility domain rules to cross-check telemetry-based theft probability.
    Never silently overrides the score: returns every reason, adjustment, and financial estimate.
    """
    reasons: List[str] = []
    adjustments: List[Dict[str, Any]] = []
    adjusted_probability = float(raw_theft_probability)
    action_recommendation = "MONITOR"

    # Rule 1: Prior Audit Outcome & Recency Decay
    if context.recent_audit_result == "cleared":
        if context.months_since_last_audit <= 6:
            # Linear decay from 0.25 (immediate) up to 1.0 (at 6 months)
            decay = min(1.0, 0.25 + 0.75 * (context.months_since_last_audit / 6.0))
            adjusted_probability *= decay
            reason = f"Audited & cleared {context.months_since_last_audit}mo ago — suppressing false positive (decay factor: {decay:.2f})."
            reasons.append(reason)
            adjustments.append({"rule": "audit_cleared_decay", "factor": decay})
    elif context.recent_audit_result == "confirmed_theft":
        adjusted_probability = max(adjusted_probability, 0.95)
        reason = "Prior confirmed theft on this account — high recidivism risk (probability floored at 0.95)."
        reasons.append(reason)
        adjustments.append({"rule": "prior_theft_recidivism", "override": 0.95})
    elif context.recent_audit_result == "meter_fault":
        adjusted_probability *= 0.40
        reason = "Recent audit logged defective meter hardware — suppress theft raid, schedule meter replacement."
        reasons.append(reason)
        adjustments.append({"rule": "prior_meter_fault", "factor": 0.40})

    # Rule 2: Feeder Line Loss Context
    if context.feeder_loss_pct >= 25.0 and raw_theft_probability >= 0.35:
        adjusted_probability = min(1.0, adjusted_probability * 1.25)
        reason = f"Account sits on high-loss feeder ({context.feeder_loss_pct:.1f}% technical/commercial loss) — high correlation with theft."
        reasons.append(reason)
        adjustments.append({"rule": "high_loss_feeder_boost", "factor": 1.25})
    elif context.feeder_loss_pct <= 6.0 and not context.meter_seal_broken and context.tamper_event_count == 0:
        adjusted_probability *= 0.65
        reason = f"Account is on low-loss feeder ({context.feeder_loss_pct:.1f}% loss) — drop likely due to vacancy/efficiency rather than systemic bypass."
        reasons.append(reason)
        adjustments.append({"rule": "low_loss_feeder_suppression", "factor": 0.65})

    # Rule 3: Grid Hardware / CT-PT Topology Issues
    if context.known_grid_topology_issue:
        adjusted_probability *= 0.45
        reason = "Known CT/PT miscalibration or phase unbalance on feeder — telemetry anomaly flagged as probable metering equipment fault."
        reasons.append(reason)
        adjustments.append({"rule": "grid_topology_hardware_issue", "factor": 0.45})

    # Rule 4: Rooftop Solar & Net Metering
    if context.has_rooftop_solar or context.solar_net_metering_active:
        if not context.meter_seal_broken and context.tamper_event_count == 0:
            adjusted_probability *= 0.40
            reason = "Active rooftop solar / net-metering verified — daylight load drop is consistent with on-site PV generation."
            reasons.append(reason)
            adjustments.append({"rule": "solar_net_metering_suppression", "factor": 0.40})

    # Rule 5: Seasonal & Agricultural Vacancy Patterns
    if context.is_seasonal_occupancy:
        adjusted_probability *= 0.50
        reason = "Registered seasonal / agricultural profile — zero-consumption period matches seasonal crop/vacancy pattern."
        reasons.append(reason)
        adjustments.append({"rule": "seasonal_occupancy_suppression", "factor": 0.50})

    # Rule 6: Hardware Tamper Alerts & Broken Seal (Overrides all suppressions)
    if context.meter_seal_broken or context.reverse_current_alert or context.tamper_event_count >= 2:
        adjusted_probability = max(0.98, adjusted_probability)
        reason = "CRITICAL: Physical tamper event detected (broken seal / reverse polarity / magnetic sensor) — override to confirmed suspect."
        reasons.append(reason)
        adjustments.append({"rule": "critical_hardware_tamper", "override": 0.98})
    elif context.tamper_event_count == 1:
        adjusted_probability = min(1.0, max(0.80, adjusted_probability * 1.35))
        reason = "Single hardware tamper event logged by smart meter — escalating suspicion."
        reasons.append(reason)
        adjustments.append({"rule": "tamper_event_escalation", "factor": 1.35})

    # Rule 7: Billing Dispute
    if context.billing_dispute_open:
        reasons.append("Open billing dispute on account — flag stands for inspection; notify field team of billing review.")

    # Bound final probability to [0.0, 1.0]
    final_prob = max(0.0, min(1.0, adjusted_probability))

    # Revenue & Financial Loss Estimation
    drop_kwh_per_day = max(0.0, context.historical_mean_kwh - context.recent_30d_mean_kwh)
    monthly_stolen_kwh = drop_kwh_per_day * 30.0
    estimated_monthly_loss = monthly_stolen_kwh * context.tariff_rate_per_kwh

    # Determine Action Recommendation
    if final_prob >= 0.85 or context.meter_seal_broken:
        action_recommendation = "DISPATCH_IMMEDIATE_RAID"
    elif final_prob >= 0.65:
        action_recommendation = "SCHEDULE_FIELD_INSPECTION"
    elif context.known_grid_topology_issue or context.recent_audit_result == "meter_fault":
        action_recommendation = "MAINTENANCE_METER_REPLACE"
    elif context.has_rooftop_solar:
        action_recommendation = "VERIFY_SOLAR_INVERTER"
    else:
        action_recommendation = "MONITOR"

    return {
        "consumer_id": context.consumer_id,
        "feeder_id": context.feeder_id,
        "raw_theft_probability": round(float(raw_theft_probability), 4),
        "adjusted_theft_probability": round(final_prob, 4),
        "reasons": reasons,
        "adjustments": adjustments,
        "action_recommendation": action_recommendation,
        "financial_impact": {
            "drop_kwh_per_day": round(drop_kwh_per_day, 2),
            "estimated_monthly_stolen_kwh": round(monthly_stolen_kwh, 2),
            "estimated_monthly_loss_currency": round(estimated_monthly_loss, 2),
            "tariff_rate": context.tariff_rate_per_kwh,
        },
    }
if __name__ == "__main__":
    # Case 1: High-loss feeder, no red flags — theft-consistent pattern
    ctx1 = CustomerContext(
        consumer_id="V1001",
        feeder_id="FEEDER-09",
        feeder_loss_pct=30.0,
        historical_mean_kwh=50.0,
        recent_30d_mean_kwh=10.0,
    )
    result1 = verify_flag(ctx1, raw_theft_probability=0.60)
    print("--- Case 1: High-loss feeder ---")
    print(result1)
    print()

    # Case 2: Rooftop solar customer, legit consumption drop
    ctx2 = CustomerContext(
        consumer_id="V2002",
        has_rooftop_solar=True,
        solar_net_metering_active=True,
        historical_mean_kwh=20.0,
        recent_30d_mean_kwh=6.0,
    )
    result2 = verify_flag(ctx2, raw_theft_probability=0.70)
    print("--- Case 2: Solar suppression ---")
    print(result2)