"""
ORM models for the ETD operational database.

These replace the previous in-memory stores (``INSPECTION_TICKETS`` /
``INSPECTION_FEEDBACK``) that were wiped on every API restart. Two tables:

  - ``inspections``:          current state of each field-inspection ticket (upserted)
  - ``inspection_feedback``:  append-only audit log of every outcome ever recorded

Column types mirror the JSON the API already returned, and ``Inspection.to_dict``
reproduces that exact shape, so switching to a durable backend does not change any
API response -- only where the data lives.
"""
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.db import Base, utcnow


class Inspection(Base):
    """A field-inspection ticket.

    Created automatically when an account is flagged as a theft suspect
    (``status='PENDING_DISPATCH'``) and updated when a field officer logs the
    physical inspection outcome (``status='COMPLETED'``).
    """
    __tablename__ = "inspections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    ticket_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    consumer_id: Mapped[str] = mapped_column(String(64), index=True, default="UNKNOWN")
    feeder_id: Mapped[str] = mapped_column(String(64), default="FEEDER-MAIN")
    theft_probability: Mapped[float] = mapped_column(Float, default=0.0)
    risk_tier: Mapped[str] = mapped_column(String(16), default="LOW")
    action_recommendation: Mapped[str] = mapped_column(String(64), default="MONITOR")
    estimated_loss_currency: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(24), default="PENDING_DISPATCH", index=True)

    # Populated when a field officer records the outcome:
    action_outcome: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    actual_theft_found: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    penalty_imposed: Mapped[float] = mapped_column(Float, default=0.0)
    inspector_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[object] = mapped_column(DateTime, default=utcnow)
    resolved_at: Mapped[Optional[object]] = mapped_column(DateTime, nullable=True)

    def to_dict(self) -> dict:
        """Serialise to the exact JSON shape the API returned before the database."""
        def _ts(dt):
            return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else None

        return {
            "ticket_id": self.ticket_id,
            "consumer_id": self.consumer_id,
            "feeder_id": self.feeder_id,
            "theft_probability": self.theft_probability,
            "risk_tier": self.risk_tier,
            "action_recommendation": self.action_recommendation,
            "estimated_loss_currency": self.estimated_loss_currency,
            "status": self.status,
            "action_outcome": self.action_outcome,
            "actual_theft_found": self.actual_theft_found,
            "penalty_imposed": self.penalty_imposed,
            "inspector_id": self.inspector_id,
            "notes": self.notes,
            "created_at": _ts(self.created_at),
            "resolved_at": _ts(self.resolved_at),
        }


class InspectionFeedback(Base):
    """Append-only log of every field outcome ever submitted.

    This is the ground-truth history used to recalibrate the Verification Agent and
    to report real precision/recall of the dispatch queue over time.
    """
    __tablename__ = "inspection_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    ticket_id: Mapped[str] = mapped_column(String(64), index=True)
    consumer_id: Mapped[str] = mapped_column(String(64), default="UNKNOWN")
    outcome: Mapped[str] = mapped_column(String(32), default="")
    actual_theft: Mapped[bool] = mapped_column(Boolean, default=False)
    penalty: Mapped[float] = mapped_column(Float, default=0.0)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    recorded_at: Mapped[object] = mapped_column(DateTime, default=utcnow)
