"""
validate_endpoints.py
-----------------------
Script kiểm tra nhanh các FastAPI endpoints sau khi triển khai.
Chạy sau khi server đã khởi động tại http://localhost:8000.
"""
import json
import math
import sys
import urllib.request

BASE_URL = "http://127.0.0.1:8001"


def check_health():
    """Kiểm tra GET /health trả về status=ok và tất cả artifacts loaded."""
    print("=== GET /health ===")
    with urllib.request.urlopen(f"{BASE_URL}/health") as r:
        health = json.loads(r.read())
    print(json.dumps(health, indent=2))
    assert health["status"] == "ok", f"Health not ok: {health}"
    assert health["model_loaded"] is True
    assert health["threshold_loaded"] is True
    print("PASS: /health\n")


def check_predict():
    """Kiểm tra POST /predict với fixed_batch.csv trả về 128 flows không NaN."""
    print("=== POST /predict ===")

    with open("artifacts/sample_batch/fixed_batch.csv", "rb") as f:
        csv_data = f.read()

    boundary = "----ValidationBoundary"
    body_parts = [
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="file"; filename="fixed_batch.csv"\r\n',
        b"Content-Type: text/csv\r\n\r\n",
        csv_data,
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    body = b"".join(body_parts)

    req = urllib.request.Request(
        f"{BASE_URL}/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    with urllib.request.urlopen(req) as r:
        result = json.loads(r.read())

    summary = result["summary"]
    print("status:", result["status"])
    print("total_flows:", summary["total_flows"])
    print("anomaly_count:", summary["anomaly_count"])
    print("normal_count:", summary["normal_count"])
    print("anomaly_rate:", summary["anomaly_rate"])
    print("threshold:", summary["threshold"])
    print("len results:", len(result["results"]))
    print("sample result[0]:", result["results"][0])

    assert summary["total_flows"] == 128, f'Expected 128, got {summary["total_flows"]}'
    assert len(result["results"]) == 128
    bad = [r for r in result["results"] if not math.isfinite(r["reconstruction_error"])]
    assert len(bad) == 0, f"NaN/Inf in errors: {bad[:3]}"
    assert result["status"] == "ok"
    print("PASS: /predict total_flows=128, no NaN/Inf\n")
    return result


def check_results():
    """Kiểm tra GET /results trả về kết quả cache từ /predict."""
    print("=== GET /results ===")
    with urllib.request.urlopen(f"{BASE_URL}/results") as r:
        res = json.loads(r.read())
    assert res["summary"]["total_flows"] == 128
    print("total_flows cached:", res["summary"]["total_flows"])
    print("PASS: /results cached correctly\n")


if __name__ == "__main__":
    sys.path.insert(0, ".")
    try:
        check_health()
        check_predict()
        check_results()
        print("=" * 50)
        print("ALL VALIDATION PASSED")
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
