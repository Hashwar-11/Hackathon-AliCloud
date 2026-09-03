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
