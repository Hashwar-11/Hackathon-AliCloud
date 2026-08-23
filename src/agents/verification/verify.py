"""
Verification Agent — cuts false positives by cross-checking a flagged
account against non-telemetry context BEFORE a field-inspection ticket
is created. Deliberately rule-based, not a black box: whoever reads a
dispatch ticket needs to see why it fired.

This is a placeholder ruleset. You need real fields from your utility's
billing/audit system to make this meaningful — it cannot run on SGCC's
data alone, since SGCC has no billing-history or audit-history columns.
Wire this to your actual utility's CRM/billing database before trusting it.
"""
from dataclasses import dataclass


@dataclass
class CustomerContext:
    consumer_id: str
    recent_audit_result: str = "none"       # "cleared", "confirmed_theft", "none"
    billing_dispute_open: bool = False
    known_grid_topology_issue: bool = False  # e.g. CT/PT miscalibration on that feeder
    months_since_last_audit: int = 999


def verify_flag(context: CustomerContext, raw_theft_probability: float) -> dict:
    """Returns an adjusted decision, never silently overriding the model score without
    stating why in the returned dict."""
    reasons = []
    adjusted_probability = raw_theft_probability

    if context.recent_audit_result == "cleared" and context.months_since_last_audit < 3:
        adjusted_probability *= 0.3
        reasons.append("Recently audited and cleared — suppressing false positive.")

    if context.known_grid_topology_issue:
        adjusted_probability *= 0.5
        reasons.append("Known CT/PT calibration issue on this feeder — signal may be a hardware fault, not theft.")

    if context.billing_dispute_open:
        reasons.append("Open billing dispute — flag stands, but note for the field team.")

    if context.recent_audit_result == "confirmed_theft":
        adjusted_probability = max(adjusted_probability, 0.95)
        reasons.append("Prior confirmed theft on this account — reinforcing flag.")

    return {
        "consumer_id": context.consumer_id,
        "raw_theft_probability": raw_theft_probability,
        "adjusted_theft_probability": min(adjusted_probability, 1.0),
        "reasons": reasons,
    }
