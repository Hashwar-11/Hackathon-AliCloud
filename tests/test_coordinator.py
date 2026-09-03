"""
Unit tests for Coordinator Agent (src/agents/coordinator/coordinator.py).
Tests calibration, risk tier assignment, single-account routing, and batch feeder queue orchestration.
"""
import pytest
from src.agents.verification.verify import CustomerContext
from src.agents.coordinator.coordinator import (
    calibrate_industrial_anomaly_score,
    assign_risk_tier,
    coordinate,
    coordinate_batch,
    ScoringResult,
)


def test_calibrate_industrial_anomaly_score():
    tau = 0.05
    # Exact threshold should yield 0.50
    assert calibrate_industrial_anomaly_score(0.05, tau) == 0.50
    # High error should yield high probability
    assert calibrate_industrial_anomaly_score(0.15, tau) > 0.90
    # Low error should yield low probability
    assert calibrate_industrial_anomaly_score(0.01, tau) < 0.20
    # Invalid inputs
    assert calibrate_industrial_anomaly_score(0.05, tau=0.0) == 0.0


def test_assign_risk_tier():
    ctx_normal = CustomerContext(consumer_id="U1")
    ctx_tamper = CustomerContext(consumer_id="U2", meter_seal_broken=True)

    assert assign_risk_tier(0.90, ctx_normal) == "CRITICAL"
    assert assign_risk_tier(0.70, ctx_normal) == "HIGH"
    assert assign_risk_tier(0.45, ctx_normal) == "MEDIUM"
    assert assign_risk_tier(0.20, ctx_normal) == "LOW"
    # Broken seal forces CRITICAL even at low probability
    assert assign_risk_tier(0.10, ctx_tamper) == "CRITICAL"


def test_coordinate_residential():
    ctx = CustomerContext(consumer_id="RES_101", feeder_id="FDR-01")
    res = coordinate(
        consumer_id="RES_101",
        consumer_type="residential",
        residential_probability=0.75,
        context=ctx,
        threshold=0.65,
    )
    assert isinstance(res, ScoringResult)
    assert res.consumer_id == "RES_101"
    assert res.raw_model_score == 0.75
    assert res.is_theft_suspect is True
    assert res.risk_tier == "HIGH"


def test_coordinate_industrial():
    ctx = CustomerContext(consumer_id="IND_202", feeder_id="FDR-02")
    res = coordinate(
        consumer_id="IND_202",
        consumer_type="industrial",
        industrial_reconstruction_error=0.10,
        industrial_tau=0.05,
        context=ctx,
        threshold=0.65,
    )
    assert isinstance(res, ScoringResult)
    assert res.consumer_id == "IND_202"
    assert res.raw_model_score > 0.50
    assert res.consumer_type == "industrial"


def test_coordinate_batch_ranking():
    ctx1 = CustomerContext(consumer_id="A1", feeder_id="FDR-A")
    ctx2 = CustomerContext(consumer_id="A2", feeder_id="FDR-A", meter_seal_broken=True)
    ctx3 = CustomerContext(consumer_id="A3", feeder_id="FDR-B")

    scored_items = [
        {"consumer_id": "A1", "consumer_type": "residential", "residential_probability": 0.70, "context": ctx1},
        {"consumer_id": "A2", "consumer_type": "residential", "residential_probability": 0.50, "context": ctx2},
        {"consumer_id": "A3", "consumer_type": "residential", "residential_probability": 0.20, "context": ctx3},
    ]

    batch = coordinate_batch(scored_items, threshold=0.65)
    queue = batch["prioritized_queue"]

    assert len(queue) == 3
    # A2 has broken seal -> CRITICAL risk tier -> must be ranked first
    assert queue[0]["consumer_id"] == "A2"
    assert queue[0]["risk_tier"] == "CRITICAL"
    assert batch["total_analyzed"] == 3
    assert "feeder_summary" in batch
    assert "FDR-A" in batch["feeder_summary"]
