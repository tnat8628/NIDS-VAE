"""Unit tests for transactional CSV persistence helpers."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.services.upload_storage import (
    UploadStorageError,
    dataframe_rows,
    save_csv_upload,
)


def test_dataframe_rows_preserves_duplicates_and_normalizes_non_finite_values() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "value": 7.0,
                "missing": np.nan,
                "positive_inf": np.inf,
                "when": pd.NaT,
                "text_inf": "Infinity",
            },
            {
                "value": 7.0,
                "missing": np.nan,
                "positive_inf": -np.inf,
                "when": pd.NaT,
                "text_inf": "-Infinity",
            },
        ]
    )

    rows = list(dataframe_rows(dataframe))

    assert [row_index for row_index, _ in rows] == [0, 1]
    assert rows[0][1] == {
        "value": 7.0,
        "missing": None,
        "positive_inf": None,
        "when": None,
        "text_inf": None,
    }
    assert rows[1][1] == rows[0][1]


def test_save_csv_upload_inserts_metadata_and_every_row_in_one_transaction() -> None:
    session = MagicMock(spec=Session)
    upload_id = uuid.uuid4()

    def assign_upload_id(upload) -> None:
        upload.id = upload_id

    session.add.side_effect = assign_upload_id
    dataframe = pd.DataFrame([{"source": "a", "score": 1}, {"source": "a", "score": 1}])

    result = save_csv_upload(
        session,
        original_filename="flows.csv",
        contents=b"source,score\na,1\na,1\n",
        dataframe=dataframe,
    )

    assert result == upload_id
    session.begin.assert_called_once_with()
    session.add.assert_called_once()
    stored_upload = session.add.call_args.args[0]
    assert stored_upload.original_filename == "flows.csv"
    assert stored_upload.row_count == 2
    assert stored_upload.col_count == 2
    assert stored_upload.columns == ["source", "score"]
    assert len(stored_upload.file_sha256) == 64

    inserted_rows = session.execute.call_args.args[1]
    assert [row["row_index"] for row in inserted_rows] == [0, 1]
    assert [row["payload"] for row in inserted_rows] == [
        {"source": "a", "score": 1},
        {"source": "a", "score": 1},
    ]


def test_save_csv_upload_surfaces_failure_after_transaction_context_rolls_back() -> None:
    session = MagicMock(spec=Session)
    session.add.side_effect = lambda upload: setattr(upload, "id", uuid.uuid4())
    session.execute.side_effect = SQLAlchemyError("insert failed")

    with pytest.raises(UploadStorageError):
        save_csv_upload(
            session,
            original_filename="flows.csv",
            contents=b"value\n1\n",
            dataframe=pd.DataFrame([{"value": 1}]),
        )

    transaction = session.begin.return_value
    assert transaction.__exit__.call_args.args[0] is SQLAlchemyError