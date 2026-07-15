"""Unit tests for atomic persistence of inference runs."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.services import prediction_storage
from backend.app.services.prediction_storage import (
    PredictionStorageError,
    _prediction_item,
    compare_prediction,
    extract_ground_truth_label,
    get_paginated_results,
    normalize_ground_truth_label,
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


@pytest.mark.parametrize(
    ("raw_label", "expected"),
    [
        ("BENIGN", 0),
        ("benign", 0),
        ("normal", 0),
        ("DDoS", 1),
        ("PortScan", 1),
        ("1", 1),
        (None, None),
    ],
)
def test_normalize_ground_truth_label(raw_label: object | None, expected: int | None) -> None:
    assert normalize_ground_truth_label(raw_label) == expected


def test_extract_ground_truth_label_supports_known_payload_keys() -> None:
    assert extract_ground_truth_label({" Flow Duration": 10, " Label ": "BENIGN"}) == (
        "BENIGN",
        0,
    )
    assert extract_ground_truth_label({"value": 1}) == (None, None)


@pytest.mark.parametrize(
    ("prediction", "actual_binary", "expected"),
    [
        (1, 1, "TP"),
        (1, 0, "FP"),
        (0, 0, "TN"),
        (0, 1, "FN"),
        (1, None, "N/A"),
    ],
)
def test_compare_prediction_returns_confusion_matrix_cell(
    prediction: int,
    actual_binary: int | None,
    expected: str,
) -> None:
    assert compare_prediction(prediction, actual_binary) == expected


@pytest.mark.parametrize(
    ("payload", "expected_label", "expected_binary"),
    [
        ({"Label": "BENIGN"}, "BENIGN", 0),
        ({" Label ": "DDoS"}, "DDoS", 1),
        ({"binary_label": 1}, 1, 1),
        ({"Flow Duration": 42}, None, None),
    ],
)
def test_prediction_item_includes_ground_truth_from_payload(
    payload: dict[str, object],
    expected_label: object | None,
    expected_binary: int | None,
) -> None:
    flow = SimpleNamespace(
        row_index=7,
        reconstruction_error=2.5,
        prediction=1,
        prediction_label="anomaly",
    )

    item = _prediction_item(flow, threshold=1.0, payload=payload)

    assert item["actual_label"] == expected_label
    assert item["actual_binary"] == expected_binary


def test_get_paginated_results_includes_ground_truth_for_items_and_top_anomalies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = SimpleNamespace(
        id=uuid.uuid4(),
        total_flows=2,
        anomaly_count=1,
        normal_count=1,
        anomaly_rate=0.5,
        threshold=1.0,
    )
    normal_flow = SimpleNamespace(
        row_index=0,
        reconstruction_error=0.4,
        prediction=0,
        prediction_label="normal",
    )
    anomaly_flow = SimpleNamespace(
        row_index=1,
        reconstruction_error=2.4,
        prediction=1,
        prediction_label="anomaly",
    )
    items_result = MagicMock()
    items_result.all.return_value = [
        (normal_flow, {"Label": "BENIGN"}),
        (anomaly_flow, {" Label ": "DDoS"}),
    ]
    top_result = MagicMock()
    top_result.all.return_value = [(anomaly_flow, {" Label ": "DDoS"})]
    session = MagicMock(spec=Session)
    session.execute.side_effect = [items_result, top_result]

    monkeypatch.setattr(prediction_storage, "_find_run", lambda *_, **__: run)
    monkeypatch.setattr(prediction_storage, "_histogram", lambda *_: [{"bin": "0.0", "normal": 1, "anomaly": 1}])

    result = get_paginated_results(
        session,
        upload_id=uuid.uuid4(),
        page=1,
        page_size=25,
        inference_run_id=run.id,
    )

    assert result["summary"]["total_flows"] == 2
    assert result["pagination"]["total_items"] == 2
    assert result["items"][0]["actual_label"] == "BENIGN"
    assert result["items"][0]["actual_binary"] == 0
    assert result["items"][1]["actual_label"] == "DDoS"
    assert result["items"][1]["actual_binary"] == 1
    assert result["aggregates"]["top_anomalies"][0]["actual_label"] == "DDoS"
    assert result["aggregates"]["top_anomalies"][0]["actual_binary"] == 1