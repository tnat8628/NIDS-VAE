"""Add database-backed VAE inference results.

Revision ID: 20260702_0002
Revises: 20260702_0001
Create Date: 2026-07-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260702_0002"
down_revision: str | None = "20260702_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "inference_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("upload_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("total_flows", sa.Integer(), nullable=False),
        sa.Column("anomaly_count", sa.Integer(), nullable=False),
        sa.Column("normal_count", sa.Integer(), nullable=False),
        sa.Column("anomaly_rate", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["upload_id"],
            ["csv_uploads.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_inference_runs_upload_id"),
        "inference_runs",
        ["upload_id"],
        unique=False,
    )

    op.create_table(
        "flow_predictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("inference_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("csv_row_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("row_index", sa.Integer(), nullable=False),
        sa.Column("reconstruction_error", sa.Float(), nullable=False),
        sa.Column("prediction", sa.SmallInteger(), nullable=False),
        sa.Column("prediction_label", sa.String(length=16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "prediction IN (0, 1)",
            name="ck_flow_predictions_prediction",
        ),
        sa.ForeignKeyConstraint(
            ["csv_row_id"],
            ["csv_rows.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["inference_run_id"],
            ["inference_runs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "inference_run_id",
            "row_index",
            name="uq_flow_predictions_run_row_index",
        ),
    )
    op.create_index(
        op.f("ix_flow_predictions_csv_row_id"),
        "flow_predictions",
        ["csv_row_id"],
        unique=False,
    )
    op.create_index(
        "ix_flow_predictions_run_row_index",
        "flow_predictions",
        ["inference_run_id", "row_index"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_flow_predictions_run_row_index",
        table_name="flow_predictions",
    )
    op.drop_index(
        op.f("ix_flow_predictions_csv_row_id"),
        table_name="flow_predictions",
    )
    op.drop_table("flow_predictions")
    op.drop_index(
        op.f("ix_inference_runs_upload_id"),
        table_name="inference_runs",
    )
    op.drop_table("inference_runs")