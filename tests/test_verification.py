"""
Unit tests for Verification Agent (src/agents/verification/verify.py).
Tests all 7 utility domain rules, probability adjustments, action recommendations,
and financial loss impact assessment.
"""
import pytest
from src.agents.verification.verify import verify_flag, CustomerContext


def test_audit_cleared_recency_decay():
    # Immediate post-audit (0 months): decay is 0.25
    ctx_0mo = CustomerContext(
        consumer_id="C001",
        recent_audit_result="cleared",
        months_since_last_audit=0,
    )
    res_0mo = verify_flag(ctx_0mo, raw_theft_probability=0.80)
    assert res_0mo["adjusted_theft_probability"] == pytest.approx(0.80 * 0.25, abs=1e-3)
    assert any("Audited & cleared" in r for r in res_0mo["reasons"])

    # 3 months post-audit: decay is 0.25 + 0.75 * (3/6) = 0.625
    ctx_3mo = CustomerContext(
        consumer_id="C001",
        recent_audit_result="cleared",
        months_since_last_audit=3,
    )
    res_3mo = verify_flag(ctx_3mo, raw_theft_probability=0.80)
    assert res_3mo["adjusted_theft_probability"] == pytest.approx(0.80 * 0.625, abs=1e-3)


def test_prior_confirmed_theft_recidivism():
    ctx = CustomerContext(
        consumer_id="C002",
        recent_audit_result="confirmed_theft",
    )
    res = verify_flag(ctx, raw_theft_probability=0.30)
    assert res["adjusted_theft_probability"] >= 0.95
    assert res["action_recommendation"] == "DISPATCH_IMMEDIATE_RAID"
    assert any("Prior confirmed theft" in r for r in res["reasons"])


def test_prior_meter_fault_suppression():
    ctx = CustomerContext(
        consumer_id="C003",
        recent_audit_result="meter_fault",
    )
    res = verify_flag(ctx, raw_theft_probability=0.70)
    assert res["adjusted_theft_probability"] == pytest.approx(0.70 * 0.40, abs=1e-3)
    assert any("defective meter hardware" in r for r in res["reasons"])


def test_high_loss_feeder_boost():
    ctx = CustomerContext(
        consumer_id="C004",
        feeder_loss_pct=30.0,
    )
    res = verify_flag(ctx, raw_theft_probability=0.60)
    assert res["adjusted_theft_probability"] == pytest.approx(min(1.0, 0.60 * 1.25), abs=1e-3)
    assert any("high-loss feeder" in r for r in res["reasons"])


def test_low_loss_feeder_suppression():
    ctx = CustomerContext(
        consumer_id="C005",
        feeder_loss_pct=4.5,
    )
    res = verify_flag(ctx, raw_theft_probability=0.70)
    assert res["adjusted_theft_probability"] == pytest.approx(0.70 * 0.65, abs=1e-3)
    assert any("low-loss feeder" in r for r in res["reasons"])


def test_grid_topology_issue_suppression():
    ctx = CustomerContext(
        consumer_id="C006",
        known_grid_topology_issue=True,
    )
    res = verify_flag(ctx, raw_theft_probability=0.70)
    assert res["adjusted_theft_probability"] == pytest.approx(0.70 * 0.45, abs=1e-3)
    assert any("CT/PT miscalibration" in r for r in res["reasons"])


def test_rooftop_solar_suppression():
    ctx = CustomerContext(
        consumer_id="C007",
        has_rooftop_solar=True,
        solar_net_metering_active=True,
    )
    res = verify_flag(ctx, raw_theft_probability=0.75)
    assert res["adjusted_theft_probability"] == pytest.approx(0.75 * 0.40, abs=1e-3)
    assert any("rooftop solar" in r for r in res["reasons"])


def test_seasonal_occupancy_suppression():
    ctx = CustomerContext(
        consumer_id="C008",
        is_seasonal_occupancy=True,
    )
    res = verify_flag(ctx, raw_theft_probability=0.80)
    assert res["adjusted_theft_probability"] == pytest.approx(0.80 * 0.50, abs=1e-3)
    assert any("seasonal / agricultural" in r for r in res["reasons"])


def test_hardware_tamper_override():
    ctx = CustomerContext(
        consumer_id="C009",
        meter_seal_broken=True,
        tamper_event_count=2,
    )
    res = verify_flag(ctx, raw_theft_probability=0.20)
    assert res["adjusted_theft_probability"] >= 0.98
    assert res["action_recommendation"] == "DISPATCH_IMMEDIATE_RAID"
    assert any("CRITICAL: Physical tamper event" in r for r in res["reasons"])


def test_financial_impact_estimation():
    ctx = CustomerContext(
        consumer_id="C010",
        historical_mean_kwh=30.0,
        recent_30d_mean_kwh=10.0,
        tariff_rate_per_kwh=40.0,
    )
    res = verify_flag(ctx, raw_theft_probability=0.85)
    fin = res["financial_impact"]
    assert fin["drop_kwh_per_day"] == 20.0
    assert fin["estimated_monthly_stolen_kwh"] == 600.0
    assert fin["estimated_monthly_loss_currency"] == 24000.0
