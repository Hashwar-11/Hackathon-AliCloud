"""
Coordinator Agent — routes accounts to the appropriate detection model,
calibrates raw anomaly errors into legitimate probabilities, executes the
Verification Agent, assigns operational risk tiers, and builds prioritized
field inspection queues.
"""
from dataclasses import dataclass, field, asdict
import math
from typing import Optional, List, Dict, Any

from src.agents.verification.verify import verify_flag, CustomerContext


@dataclass
class ScoringResult:
    consumer_id: str
    feeder_id: str
    consumer_type: str                  # "residential", "industrial", "commercial"
    raw_model_score: float              # Raw sigmoid prob or calibrated anomaly score
    verified_theft_probability: float   # Probability after Verification Agent rules
    risk_tier: str                      # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    is_theft_suspect: bool              # True if probability exceeds threshold
    action_recommendation: str          # "DISPATCH_IMMEDIATE_RAID", "SCHEDULE_FIELD_INSPECTION", etc.
    reasons: List[str]                  # Human-readable justification strings
    adjustments: List[Dict[str, Any]]   # Rule adjustments trace
    financial_impact: Dict[str, Any]    # Estimated stolen kWh and currency loss

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calibrate_industrial_anomaly_score(reconstruction_error: float, tau: float, steepness: float = 4.0) -> float:
    """
    Calibrates reconstruction error to a sigmoid probability curve centered at tau.
    P(theft) = 1 / (1 + exp(-steepness * (err - tau) / tau))
    
    Ensures that when error == tau, P = 0.50 (normal/anomaly boundary),
    rather than a naive linear ratio that misclassifies normal samples.
    """
    if tau <= 0 or math.isnan(reconstruction_error):
        return 0.0
    
    # Normalized error offset
    delta = (reconstruction_error - tau) / max(tau, 1e-6)
    # Clip to avoid math overflow
    clipped_exp_arg = max(-20.0, min(20.0, -steepness * delta))
    prob = 1.0 / (1.0 + math.exp(clipped_exp_arg))
    return round(float(prob), 4)


def assign_risk_tier(probability: float, context: CustomerContext) -> str:
    """Categorizes suspected accounts into actionable utility response tiers."""
    if context.meter_seal_broken or context.reverse_current_alert or probability >= 0.85:
        return "CRITICAL"
    elif probability >= 0.65:
        return "HIGH"
    elif probability >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"


def coordinate(
    consumer_id: str,
    consumer_type: str,
    residential_probability: Optional[float] = None,
    industrial_reconstruction_error: Optional[float] = None,
    industrial_tau: Optional[float] = None,
    context: Optional[CustomerContext] = None,
    threshold: float = 0.65,
) -> ScoringResult:
    """
    Single-account coordinator routing and arbitration pipeline.
    """
    if context is None:
        context = CustomerContext(consumer_id=consumer_id)

    # Step 1: Model Routing & Score Extraction
    if consumer_type in ("residential", "commercial"):
        if residential_probability is None:
            raise ValueError(f"residential_probability required for {consumer_type}-typed customer")
        raw_score = float(residential_probability)
    elif consumer_type == "industrial":
        if industrial_reconstruction_error is None or industrial_tau is None:
            raise ValueError("industrial_reconstruction_error and industrial_tau required for industrial customer")
        raw_score = calibrate_industrial_anomaly_score(industrial_reconstruction_error, industrial_tau)
    else:
        raise ValueError(f"Unknown consumer_type: {consumer_type}")

    # Step 2: Verification Agent Cross-Checking
    verification = verify_flag(context, raw_score)
    final_prob = verification["adjusted_theft_probability"]

    # Step 3: Risk Tier Assignment
    risk_tier = assign_risk_tier(final_prob, context)
    is_suspect = final_prob >= threshold

    return ScoringResult(
        consumer_id=consumer_id,
        feeder_id=context.feeder_id,
        consumer_type=consumer_type,
        raw_model_score=round(raw_score, 4),
        verified_theft_probability=round(final_prob, 4),
        risk_tier=risk_tier,
        is_theft_suspect=is_suspect,
        action_recommendation=verification["action_recommendation"],
        reasons=verification["reasons"],
        adjustments=verification["adjustments"],
        financial_impact=verification["financial_impact"],
    )


def coordinate_batch(
    scored_items: List[Dict[str, Any]],
    threshold: float = 0.65,
) -> Dict[str, Any]:
    """
    Processes and ranks a batch of scored accounts, aggregating feeder-level risk KPIs.
    
    Returns:
        prioritized_queue: List of ScoringResults sorted by verified_theft_probability descending
        feeder_summary: Feeder-level metrics (theft count, estimated total loss, risk index)
    """
    results: List[ScoringResult] = []
    feeder_stats: Dict[str, Dict[str, Any]] = {}

    for item in scored_items:
        res = coordinate(
            consumer_id=item["consumer_id"],
            consumer_type=item.get("consumer_type", "residential"),
            residential_probability=item.get("residential_probability"),
            industrial_reconstruction_error=item.get("industrial_reconstruction_error"),
            industrial_tau=item.get("industrial_tau"),
            context=item.get("context"),
            threshold=threshold,
        )
        results.append(res)

        fid = res.feeder_id
        if fid not in feeder_stats:
            feeder_stats[fid] = {
                "total_meters": 0,
                "suspect_count": 0,
                "critical_count": 0,
                "high_count": 0,
                "total_loss_currency": 0.0,
                "total_stolen_kwh": 0.0,
            }
        feeder_stats[fid]["total_meters"] += 1
        if res.is_theft_suspect:
            feeder_stats[fid]["suspect_count"] += 1
        if res.risk_tier == "CRITICAL":
            feeder_stats[fid]["critical_count"] += 1
        elif res.risk_tier == "HIGH":
            feeder_stats[fid]["high_count"] += 1
        
        feeder_stats[fid]["total_loss_currency"] += res.financial_impact.get("estimated_monthly_loss_currency", 0.0)
        feeder_stats[fid]["total_stolen_kwh"] += res.financial_impact.get("estimated_monthly_stolen_kwh", 0.0)

    # Sort results by priority: CRITICAL first, then highest theft probability
    results.sort(key=lambda r: (r.risk_tier == "CRITICAL", r.verified_theft_probability), reverse=True)

    # Finalize feeder risk rankings
    for fid, stat in feeder_stats.items():
        suspect_rate = stat["suspect_count"] / max(1, stat["total_meters"])
        stat["theft_rate_pct"] = round(suspect_rate * 100.0, 2)
        stat["total_loss_currency"] = round(stat["total_loss_currency"], 2)
        stat["total_stolen_kwh"] = round(stat["total_stolen_kwh"], 2)

    return {
        "prioritized_queue": [r.to_dict() for r in results],
        "feeder_summary": feeder_stats,
        "total_analyzed": len(results),
        "total_suspects": sum(1 for r in results if r.is_theft_suspect),
        "total_estimated_monthly_loss": round(sum(r.financial_impact.get("estimated_monthly_loss_currency", 0.0) for r in results), 2),
    }
if __name__ == "__main__":
    # Example 1: Normal residential account, likely a false positive (cleared audit)
    ctx1 = CustomerContext(
        consumer_id="C1001",
        feeder_id="FEEDER-07",
        recent_audit_result="cleared",
        months_since_last_audit=2,
    )
    result1 = coordinate(
        consumer_id="C1001",
        consumer_type="residential",
        residential_probability=0.78,
        context=ctx1,
    )
    print("--- Example 1: Cleared audit account ---")
    print(result1.to_dict())
    print()

    # Example 2: Suspicious account with a broken meter seal
    ctx2 = CustomerContext(
        consumer_id="C2002",
        feeder_id="FEEDER-03",
        meter_seal_broken=True,
        tamper_event_count=2,
        historical_mean_kwh=40.0,
        recent_30d_mean_kwh=5.0,
    )
    result2 = coordinate(
        consumer_id="C2002",
        consumer_type="residential",
        residential_probability=0.55,
        context=ctx2,
    )
    print("--- Example 2: Broken seal / tamper flags ---")
    print(result2.to_dict())