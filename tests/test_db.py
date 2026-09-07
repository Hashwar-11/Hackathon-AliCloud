"""
Persistence tests for the ETD operational database (src/db.py, src/models_db.py)
and the DB-backed inspection endpoints in src/api/app.py.

The key guarantee these prove: a field-inspection record written in one session is
visible in a *fresh* session -- i.e. it is durable and survives beyond a single
request/restart, which the previous in-memory dicts (INSPECTION_TICKETS /
INSPECTION_FEEDBACK) did NOT.
"""
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect as sa_inspect

from src.api.app import app
from src.db import SessionLocal, engine, init_db
from src.models_db import Inspection, InspectionFeedback


@pytest.fixture(scope="module")
def client():
    # Entering the context runs the app lifespan (init_db + model load attempt).
    with TestClient(app) as c:
        yield c


def _unique_ticket(prefix: str) -> str:
    """Collision-free ticket id so tests are isolated across runs."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def test_init_db_creates_tables():
    init_db()
    names = set(sa_inspect(engine).get_table_names())
    assert "inspections" in names
    assert "inspection_feedback" in names


def test_record_persists_across_sessions():
    """Write in one session, read back in a NEW session -> proves durability."""
    init_db()
    ticket = _unique_ticket("TCK-PERSIST")

    s1 = SessionLocal()
    try:
        s1.add(Inspection(
            ticket_id=ticket, consumer_id="C-PERSIST", feeder_id="FDR-1",
            theft_probability=0.91, risk_tier="CRITICAL", status="PENDING_DISPATCH",
        ))
        s1.commit()
    finally:
        s1.close()

    # A completely separate session, exactly as a new request/restart would use.
    s2 = SessionLocal()
    try:
        found = s2.query(Inspection).filter(Inspection.ticket_id == ticket).first()
        assert found is not None, "record did not persist to a fresh session"
        assert found.risk_tier == "CRITICAL"
        assert found.status == "PENDING_DISPATCH"
        assert found.to_dict()["ticket_id"] == ticket
        s2.delete(found)  # cleanup
        s2.commit()
    finally:
        s2.close()


def test_feedback_log_is_append_only():
    init_db()
    ticket = _unique_ticket("TCK-FB")
    s = SessionLocal()
    try:
        s.add(InspectionFeedback(ticket_id=ticket, consumer_id="C-FB",
                                 outcome="CONFIRMED_THEFT", actual_theft=True,
                                 penalty=12345.0, notes="n1"))
        s.add(InspectionFeedback(ticket_id=ticket, consumer_id="C-FB",
                                 outcome="FALSE_POSITIVE", actual_theft=False,
                                 penalty=0.0, notes="n2"))
        s.commit()
        rows = s.query(InspectionFeedback).filter(InspectionFeedback.ticket_id == ticket).all()
        assert len(rows) == 2, "both feedback entries must coexist (append-only log)"
        for r in rows:  # cleanup
            s.delete(r)
        s.commit()
    finally:
        s.close()


def test_api_inspection_action_persists_to_db(client):
    """POST an outcome via the API, then confirm it actually landed in the DB."""
    ticket = _unique_ticket("TCK-API")
    payload = {
        "ticket_id": ticket,
        "inspector_id": "INSP-TEST-01",
        "action_outcome": "CONFIRMED_THEFT",
        "actual_theft_found": True,
        "penalty_imposed_currency": 75000.0,
        "notes": "Bypass confirmed via API test.",
        "consumer_id": "C-API",
    }
    resp = client.post(f"/api/v1/inspections/{ticket}/action", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticket"]["status"] == "COMPLETED"
    assert body["ticket"]["action_outcome"] == "CONFIRMED_THEFT"

    # Independently verify persistence (not just the response echo).
    s = SessionLocal()
    try:
        rec = s.query(Inspection).filter(Inspection.ticket_id == ticket).first()
        assert rec is not None
        assert rec.status == "COMPLETED"
        assert rec.penalty_imposed == 75000.0
        fb = s.query(InspectionFeedback).filter(InspectionFeedback.ticket_id == ticket).all()
        assert len(fb) >= 1
        s.delete(rec)  # cleanup
        for f in fb:
            s.delete(f)
        s.commit()
    finally:
        s.close()


def test_api_queue_reflects_db_state(client):
    """A logged ticket shows up in the DB-backed dispatch queue endpoint."""
    ticket = _unique_ticket("TCK-QUEUE")
    client.post(f"/api/v1/inspections/{ticket}/action", json={
        "ticket_id": ticket, "inspector_id": "INSP-Q", "action_outcome": "DEFECTIVE_METER",
        "actual_theft_found": False, "penalty_imposed_currency": 0.0, "notes": "",
        "consumer_id": "C-QUEUE",
    })
    resp = client.get("/api/v1/inspections/queue")
    assert resp.status_code == 200
    ids = [t["ticket_id"] for t in resp.json()["tickets"]]
    assert ticket in ids

    s = SessionLocal()  # cleanup
    try:
        rec = s.query(Inspection).filter(Inspection.ticket_id == ticket).first()
        if rec:
            s.delete(rec)
        for f in s.query(InspectionFeedback).filter(InspectionFeedback.ticket_id == ticket).all():
            s.delete(f)
        s.commit()
    finally:
        s.close()
