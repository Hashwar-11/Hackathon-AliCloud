"""initial inspection tables

Revision ID: 532792a9ce13
Revises: 
Create Date: 2026-09-04 00:14:01.471283

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '532792a9ce13'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _existing_tables() -> set:
    """Names of tables already present in the target database."""
    return set(sa.inspect(op.get_bind()).get_table_names())


def upgrade() -> None:
    """Create the inspection tables.

    Idempotent ON PURPOSE. The API also calls ``Base.metadata.create_all()`` at
    startup (``init_db``) as a zero-config convenience, so the tables may already
    exist. Without these guards, ``alembic upgrade head`` dies with
    "table already exists" on any database the app has already booted against.
    Skipping creation when the table is present lets create_all and Alembic coexist
    on the same database: Alembic simply stamps the revision.
    """
    existing = _existing_tables()

    if 'inspection_feedback' not in existing:
        op.create_table('inspection_feedback',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('ticket_id', sa.String(length=64), nullable=False),
            sa.Column('consumer_id', sa.String(length=64), nullable=False),
            sa.Column('outcome', sa.String(length=32), nullable=False),
            sa.Column('actual_theft', sa.Boolean(), nullable=False),
            sa.Column('penalty', sa.Float(), nullable=False),
            sa.Column('notes', sa.Text(), nullable=True),
            sa.Column('recorded_at', sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint('id'),
        )
        with op.batch_alter_table('inspection_feedback', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_inspection_feedback_id'), ['id'], unique=False)
            batch_op.create_index(batch_op.f('ix_inspection_feedback_ticket_id'), ['ticket_id'], unique=False)

    if 'inspections' not in existing:
        op.create_table('inspections',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('ticket_id', sa.String(length=64), nullable=False),
            sa.Column('consumer_id', sa.String(length=64), nullable=False),
            sa.Column('feeder_id', sa.String(length=64), nullable=False),
            sa.Column('theft_probability', sa.Float(), nullable=False),
            sa.Column('risk_tier', sa.String(length=16), nullable=False),
            sa.Column('action_recommendation', sa.String(length=64), nullable=False),
            sa.Column('estimated_loss_currency', sa.Float(), nullable=False),
            sa.Column('status', sa.String(length=24), nullable=False),
            sa.Column('action_outcome', sa.String(length=32), nullable=True),
            sa.Column('actual_theft_found', sa.Boolean(), nullable=True),
            sa.Column('penalty_imposed', sa.Float(), nullable=False),
            sa.Column('inspector_id', sa.String(length=128), nullable=True),
            sa.Column('notes', sa.Text(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            sa.Column('resolved_at', sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint('id'),
        )
        with op.batch_alter_table('inspections', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_inspections_consumer_id'), ['consumer_id'], unique=False)
            batch_op.create_index(batch_op.f('ix_inspections_id'), ['id'], unique=False)
            batch_op.create_index(batch_op.f('ix_inspections_status'), ['status'], unique=False)
            batch_op.create_index(batch_op.f('ix_inspections_ticket_id'), ['ticket_id'], unique=True)


def downgrade() -> None:
    """Drop the inspection tables (guarded so it is safe on a partial schema)."""
    existing = _existing_tables()

    if 'inspections' in existing:
        with op.batch_alter_table('inspections', schema=None) as batch_op:
            batch_op.drop_index(batch_op.f('ix_inspections_ticket_id'))
            batch_op.drop_index(batch_op.f('ix_inspections_status'))
            batch_op.drop_index(batch_op.f('ix_inspections_id'))
            batch_op.drop_index(batch_op.f('ix_inspections_consumer_id'))
        op.drop_table('inspections')

    if 'inspection_feedback' in existing:
        with op.batch_alter_table('inspection_feedback', schema=None) as batch_op:
            batch_op.drop_index(batch_op.f('ix_inspection_feedback_ticket_id'))
            batch_op.drop_index(batch_op.f('ix_inspection_feedback_id'))
        op.drop_table('inspection_feedback')
