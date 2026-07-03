"""Add indexes used by upload management and global flow explorer.

Revision ID: 20260702_0003
Revises: 20260702_0002
Create Date: 2026-07-02
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260702_0003"
down_revision: str | None = "20260702_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_csv_uploads_created_at",
        "csv_uploads",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "ix_flow_predictions_inference_run_id",
        "flow_predictions",
        ["inference_run_id"],
        unique=False,
    )
    op.create_index(
        "ix_flow_predictions_prediction",
        "flow_predictions",
        ["prediction"],
        unique=False,
    )
    op.create_index(
        "ix_flow_predictions_row_index",
        "flow_predictions",
        ["row_index"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_flow_predictions_row_index", table_name="flow_predictions")
    op.drop_index("ix_flow_predictions_prediction", table_name="flow_predictions")
    op.drop_index(
        "ix_flow_predictions_inference_run_id",
        table_name="flow_predictions",
    )
    op.drop_index("ix_csv_uploads_created_at", table_name="csv_uploads")