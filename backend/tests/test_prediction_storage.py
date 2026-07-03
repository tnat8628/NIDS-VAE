"""Unit tests for atomic persistence of inference runs."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.services.prediction_storage import (
    PredictionStorageError,
    save_prediction_run,
)


def _raw_result(row_count: int = 3) -> dict[str, object]:
    items = [
        {
            "row_index": index,
            "reconstruction_error": 0.1 + index,
            "prediction": 1 if index == row_count - 1 else 0,
            "prediction_label": "anomaly" if index == row_count - 1 else "normal",
        }
        for index in range(row_count)
    ]
    return {
        "total_flows": row_count,
        "anomaly_count": 1,
        "normal_count": row_count - 1,
        "anomaly_rate": 1 / row_count,
        "threshold": 1.5,
        "results": items,
    }


def test_save_prediction_run_inserts_run_and_every_row_atomically() -> None:
    session = MagicMock(spec=Session)
    run_id = uuid.uuid4()
    session.add.side_effect = lambda run: setattr(run, "id", run_id)
    row_ids = {index: uuid.uuid4() for index in range(3)}

    result = save_prediction_run(
        session,
        upload_id=uuid.uuid4(),
        row_ids=row_ids,
        raw_result=_raw_result(),
    )

    assert result == run_id
    session.begin.assert_called_once_with()
    inserted = session.execute.call_args.args[1]
    assert [item["row_index"] for item in inserted] == [0, 1, 2]
    assert [item["csv_row_id"] for item in inserted] == list(row_ids.values())


def test_save_prediction_run_rejects_result_count_mismatch() -> None:
    session = MagicMock(spec=Session)

    with pytest.raises(PredictionStorageError, match="không khớp"):
        save_prediction_run(
            session,
            upload_id=uuid.uuid4(),
            row_ids={0: uuid.uuid4()},
            raw_result=_raw_result(),
        )

    session.begin.assert_not_called()


def test_save_prediction_run_rolls_back_when_row_insert_fails() -> None:
    session = MagicMock(spec=Session)
    session.add.side_effect = lambda run: setattr(run, "id", uuid.uuid4())
    session.execute.side_effect = SQLAlchemyError("insert failed")

    with pytest.raises(PredictionStorageError):
        save_prediction_run(
            session,
            upload_id=uuid.uuid4(),
            row_ids={index: uuid.uuid4() for index in range(3)},
            raw_result=_raw_result(),
        )

    transaction = session.begin.return_value
    assert transaction.__exit__.call_args.args[0] is SQLAlchemyError