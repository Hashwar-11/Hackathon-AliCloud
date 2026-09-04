"""
Database engine & session management for the ETD API.

The field-inspection workflow (tickets + audit feedback) is *transactional* data
that must survive API restarts, so it lives in a real database rather than the
in-memory dicts it replaces.

Backend selection is driven entirely by the ``DATABASE_URL`` environment variable:

  - Local dev / tests (zero config):  ``sqlite:///./etd.db``            (default)
  - Docker Compose / AliCloud RDS:     ``postgresql://user:pass@host:5432/db``

Every other module (models, endpoints, migrations) is backend-agnostic, so moving
from SQLite to PostgreSQL is a single env-var change -- no code edits.
"""
import os
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# SQLite needs check_same_thread=False because FastAPI serves requests from a
# threadpool. PostgreSQL (and other server backends) manage this themselves.
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./etd.db")
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)


class Base(DeclarativeBase):
    """Declarative base class for all ORM models (SQLAlchemy 2.0 style)."""
    pass


def utcnow() -> datetime:
    """Naive UTC 'now'.

    Used as the default for timestamp columns. Avoids ``datetime.utcnow()``, which
    is deprecated on Python 3.12+. Naive (tzinfo stripped) keeps values uniformly
    sortable across SQLite and PostgreSQL.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def get_db():
    """FastAPI dependency that yields a session and always closes it afterwards.

    Usage in an endpoint::

        @app.get("/...")
        def handler(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create any missing tables.

    Convenient for local/dev boot and for the test suite. In production prefer
    Alembic migrations (``alembic upgrade head``) so schema changes are versioned
    and reversible rather than implicitly auto-created.
    """
    # Deferred import registers the tables on Base.metadata and avoids a circular
    # import (models_db imports Base from this module).
    from src import models_db  # noqa: F401
    Base.metadata.create_all(bind=engine)
