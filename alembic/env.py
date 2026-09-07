"""Alembic migration environment for the ETD database.

Wired to the application's own SQLAlchemy metadata (``src.db.Base``) and the
``DATABASE_URL`` environment variable, so migrations always target the same schema
and backend the API uses -- SQLite locally, PostgreSQL in docker-compose / AliCloud
RDS -- with no duplicated connection settings.
"""
from logging.config import fileConfig

from sqlalchemy import create_engine, pool

from alembic import context

# Import the declarative Base AND the model modules so every table is registered
# on Base.metadata before autogenerate compares it against the database.
from src.db import Base, DATABASE_URL
import src.models_db  # noqa: F401  (registers Inspection / InspectionFeedback)

# Alembic Config object (values from alembic.ini).
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Autogenerate target: the application's ORM metadata.
target_metadata = Base.metadata

# SQLite cannot ALTER most columns in place; batch mode rewrites the table so the
# same migrations stay portable across SQLite (dev) and PostgreSQL (prod).
_RENDER_AS_BATCH = DATABASE_URL.startswith("sqlite")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL to stdout; no DBAPI needed)."""
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_as_batch=_RENDER_AS_BATCH,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (create an Engine and run against it)."""
    connectable = create_engine(DATABASE_URL, poolclass=pool.NullPool, future=True)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=_RENDER_AS_BATCH,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
