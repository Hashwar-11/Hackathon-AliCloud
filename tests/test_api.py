"""
Integration tests for FastAPI Enterprise Service (src/api/app.py).
Tests health checks, model info, single prediction, batch prediction, inspection queue, and feedback logging.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert "models_loaded" in data


def test_models_info(client):
    response = client.get("/api/v1/models/info")
    assert response.status_code == 200
    data = response.json()
    assert "architecture" in data
    assert "industrial_tau" in data
    assert "active_threshold" in data


def test_predict_single_residential(client):
    payload = {
        "consumer_id": "TEST_RES_001",
        "client_type": "residential",
        "daily_kwh": [10.0, 10.5, 9.8, 1.2, 0.5, 0.2, 0.1] * 10,
        "context": {
            "feeder_id": "FDR-TEST-01",
            "feeder_loss_pct": 28.0,
            "recent_audit_result": "none",
            "historical_mean_kwh": 10.0,
            "recent_30d_mean_kwh": 0.5,
        },
    }
    response = client.post("/api/v1/predict/single", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["consumer_id"] == "TEST_RES_001"
    assert "theft_probability" in data
    assert "risk_tier" in data
    assert "is_theft_suspect" in data
    assert "action_recommendation" in data
    assert "financial_impact" in data


def test_predict_single_with_solar_suppression(client):
    payload = {
        "consumer_id": "TEST_SOLAR_002",
        "client_type": "residential",
        "daily_kwh": [12.0, 12.0, 4.0, 3.5, 3.0] * 10,
        "context": {
            "feeder_id": "FDR-TEST-02",
            "has_rooftop_solar": True,
            "solar_net_metering_active": True,
            "feeder_loss_pct": 6.0,
        },
    }
    response = client.post("/api/v1/predict/single", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Verification rule should adjust and add reason
    assert any("rooftop solar" in r.lower() for r in data["reasons"])


def test_predict_batch(client):
    payload = {
        "items": [
            {
                "consumer_id": "BATCH_01",
                "client_type": "residential",
                "daily_kwh": [15.0] * 50,
            },
            {
                "consumer_id": "BATCH_02",
                "client_type": "residential",
                "daily_kwh": [15.0] * 25 + [0.5] * 25,
            },
        ],
        "threshold": 0.65,
    }
    response = client.post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prioritized_queue" in data
    assert "feeder_summary" in data
    assert data["total_analyzed"] == 2


def test_inspection_action_logging(client):
    ticket_payload = {
        "ticket_id": "TCK-TEST-9999",
        "inspector_id": "INSP-ALI-07",
        "action_outcome": "CONFIRMED_THEFT",
        "actual_theft_found": True,
        "penalty_imposed_currency": 50000.0,
        "notes": "Direct bypass tap confirmed on main line.",
        "consumer_id": "BATCH_02",
    }
    response = client.post("/api/v1/inspections/TCK-TEST-9999/action", json=ticket_payload)
    assert response.status_code == 200
    data = response.json()
    assert "ticket" in data
    assert data["ticket"]["status"] == "COMPLETED"


def test_feeders_summary(client):
    response = client.get("/api/v1/feeders/summary")
    assert response.status_code == 200
    data = response.json()
    assert "feeders" in data
    assert len(data["feeders"]) > 0


def test_list_consumers_and_details(client):
    res = client.get("/api/v1/consumers")
    assert res.status_code == 200
    data = res.json()
    assert "consumers" in data
    assert data["count"] > 0
    first_id = data["consumers"][0]["consumer_id"]

    res_detail = client.get(f"/api/v1/consumers/{first_id}")
    assert res_detail.status_code == 200
    detail_data = res_detail.json()
    assert detail_data["consumer_id"] == first_id
    assert "context" in detail_data


def test_system_statistics(client):
    res = client.get("/api/v1/statistics")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "total_monitored_consumers" in data
    assert "total_tickets" in data


def test_assistant_summarize(client):
    payload = {
        "consumer_id": "PK001",
        "consumer_type": "residential",
        "theft_probability": 0.92,
        "risk_tier": "CRITICAL",
        "action_recommendation": "DISPATCH_IMMEDIATE_RAID",
        "reasons": ["Sustained severe consumption drop detected", "Broken meter seal override"],
        "financial_impact": {
            "drop_kwh_per_day": 43.6,
            "estimated_monthly_stolen_kwh": 1308.0,
            "estimated_monthly_loss_currency": 36624.0,
        },
    }
    res = client.post("/api/v1/assistant/summarize", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "summary" in data
    assert "CRITICAL" in data["summary"]
    assert "source" in data

