"""
Coordinator Agent — the single entry point that routes a customer to the
correct specialist, runs verification, and returns the final decision.
This is intentionally plain Python/control-flow, NOT an LLM. If you later
add an LLM-based justification-writer on top of this, run it asynchronously
AFTER this function returns — do not put it in the synchronous scoring path.
"""
from dataclasses import dataclass
from typing import Optional

from src.agents.verification.verify import verify_flag, CustomerContext


@dataclass
class ScoringResult:
    consumer_id: str
    consumer_type: str          # "residential" or "industrial" -- proxy label, see preprocess.py
    theft_probability: float
    is_theft_suspect: bool
    reasons: list


def coordinate(
    consumer_id: str,
    consumer_type: str,
    residential_probability: Optional[float],
    industrial_reconstruction_error: Optional[float],
    industrial_tau: Optional[float],
    context: CustomerContext,
    threshold: float = 0.65,
) -> ScoringResult:

    if consumer_type == "residential":
        if residential_probability is None:
            raise ValueError("residential_probability required for a residential-typed customer")
        raw_probability = residential_probability
    elif consumer_type == "industrial":
        if industrial_reconstruction_error is None or industrial_tau is None:
            raise ValueError("industrial_reconstruction_error and industrial_tau required for industrial customer")
        # convert reconstruction error into a comparable 0-1 score relative to tau
        raw_probability = min(industrial_reconstruction_error / industrial_tau, 1.0) if industrial_tau > 0 else 0.0
    else:
        raise ValueError(f"Unknown consumer_type: {consumer_type}")

    verification = verify_flag(context, raw_probability)
    final_probability = verification["adjusted_theft_probability"]

    return ScoringResult(
        consumer_id=consumer_id,
        consumer_type=consumer_type,
        theft_probability=final_probability,
        is_theft_suspect=final_probability > threshold,
        reasons=verification["reasons"],
    )
