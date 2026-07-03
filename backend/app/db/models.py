"""Database models for persisted CSV uploads."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.database import Base


class CsvUpload(Base):
    """Metadata for one successfully uploaded CSV file."""

    __tablename__ = "csv_uploads"
    __table_args__ = (Index("ix_csv_uploads_created_at", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    col_count: Mapped[int] = mapped_column(Integer, nullable=False)
    columns: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    file_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    rows: Mapped[list[CsvRow]] = relationship(
        back_populates="upload",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    inference_runs: Mapped[list[InferenceRun]] = relationship(
        back_populates="upload",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class CsvRow(Base):
    """One row from an uploaded CSV, stored as a flexible JSONB payload."""

    __tablename__ = "csv_rows"
    __table_args__ = (
        UniqueConstraint("upload_id", "row_index", name="uq_csv_rows_upload_row_index"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    upload_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("csv_uploads.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    payload: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    upload: Mapped[CsvUpload] = relationship(back_populates="rows")
    predictions: Mapped[list[FlowPrediction]] = relationship(
        back_populates="csv_row",
        passive_deletes=True,
    )


class InferenceRun(Base):
    """One complete VAE inference execution for a persisted upload."""

    __tablename__ = "inference_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    upload_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("csv_uploads.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    total_flows: Mapped[int] = mapped_column(Integer, nullable=False)
    anomaly_count: Mapped[int] = mapped_column(Integer, nullable=False)
    normal_count: Mapped[int] = mapped_column(Integer, nullable=False)
    anomaly_rate: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    upload: Mapped[CsvUpload] = relationship(back_populates="inference_runs")
    predictions: Mapped[list[FlowPrediction]] = relationship(
        back_populates="inference_run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class FlowPrediction(Base):
    """Persisted per-row result produced by an inference run."""

    __tablename__ = "flow_predictions"
    __table_args__ = (
        CheckConstraint("prediction IN (0, 1)", name="ck_flow_predictions_prediction"),
        UniqueConstraint(
            "inference_run_id",
            "row_index",
            name="uq_flow_predictions_run_row_index",
        ),
        Index(
            "ix_flow_predictions_run_row_index",
            "inference_run_id",
            "row_index",
        ),
        Index("ix_flow_predictions_prediction", "prediction"),
        Index("ix_flow_predictions_row_index", "row_index"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    inference_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("inference_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    csv_row_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("csv_rows.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    reconstruction_error: Mapped[float] = mapped_column(Float, nullable=False)
    prediction: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    prediction_label: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    inference_run: Mapped[InferenceRun] = relationship(back_populates="predictions")
    csv_row: Mapped[CsvRow] = relationship(back_populates="predictions")