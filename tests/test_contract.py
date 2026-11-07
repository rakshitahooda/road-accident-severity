import json
import math
import pytest
from fastapi.testclient import TestClient
from roadlab.api import app
from roadlab.core import ROOT, FEATURES, validate_record

client = TestClient(app)


@pytest.fixture
def example():
    return json.loads((ROOT / "artifacts/examples.json").read_text())[0]


def test_health_and_ui():
    assert client.get("/health").json() == {"status": "ready"}
    assert "RoadLab" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_live_prediction(example):
    response = client.post("/api/predict", json={"records": [example]})
    assert response.status_code == 200
    prediction = response.json()["predictions"][0]
    assert prediction["label"] in prediction["scores"]
    assert sum(prediction["scores"].values()) == pytest.approx(1)
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in prediction["scores"].values())


def test_batch_same_as_single(example):
    single = client.post("/api/predict", json={"records": [example]}).json()["predictions"][0]
    batch = client.post("/api/predict", json={"records": [example, example]}).json()["predictions"]
    assert [item["label"] for item in batch] == [single["label"], single["label"]]
    for item in batch:
        assert item["scores"] == pytest.approx(single["scores"])


@pytest.mark.parametrize("change", [{"Hour": -1}, {"Hour": 24}, {"Hour": True}, {"Hour": 3.2}, {"Number_of_vehicles_involved": 0}, {"Weather_conditions": "invented"}, {"Accident_severity": "Fatal injury"}])
def test_invalid_input_rejected(example, change):
    assert client.post("/api/predict", json={"records": [{**example, **change}]}).status_code == 422


def test_missing_feature_rejected(example):
    example.pop("Hour")
    assert client.post("/api/predict", json={"records": [example]}).status_code == 422


def test_explicit_nulls():
    response = client.post("/api/predict", json={"records": [dict.fromkeys(FEATURES)]})
    assert response.status_code == 200


def test_batch_limits(example):
    assert client.post("/api/predict", json={"records": []}).status_code == 422
    assert client.post("/api/predict", json={"records": [example] * 101}).status_code == 422
    assert client.post("/api/predict", json={"records": [example], "extra": 1}).status_code == 422


def test_audit_and_leakage_contract():
    report = json.loads((ROOT / "artifacts/metrics.json").read_text())
    assert report["split"]["profile_overlap"] == 0
    assert report["dataset"]["raw_rows"] == 12316
    assert not set(FEATURES) & {"Accident_severity", "Casualty_severity", "Number_of_casualties", "Cause_of_accident"}
    split = json.loads((ROOT / "artifacts/split_indices.json").read_text())
    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["test"]).isdisjoint(split["train"] + split["validation"])


def test_validation_direct(example):
    schema = json.loads((ROOT / "artifacts/schema.json").read_text())
    assert set(validate_record(example, schema)) == set(FEATURES)


def test_missing_artifact_is_service_unavailable(tmp_path, monkeypatch):
    import roadlab.api as api
    api.resources.cache_clear()
    with monkeypatch.context() as context:
        context.setattr(api, "ROOT", tmp_path)
        assert client.get("/health").status_code == 503
        assert client.get("/api/metadata").status_code == 503
    api.resources.cache_clear()


def test_changed_dataset_fails_closed(tmp_path):
    from roadlab.train import run
    data = tmp_path / "changed.csv"
    data.write_text("Accident_severity\nFatal injury\n")
    with pytest.raises(ValueError, match="checksum"):
        run(data, tmp_path / "output")


def test_openapi_documents_batch_limits():
    contract = client.get("/openapi.json").json()
    records = contract["components"]["schemas"]["PredictionRequest"]["properties"]["records"]
    assert records["minItems"] == 1
    assert records["maxItems"] == 100

def test_profile_cluster_bootstrap_is_reproducible():
    import numpy as np
    from roadlab.train import profile_cluster_bootstrap_macro_f1

    truth = np.array(["fatal", "fatal", "serious", "serious", "slight", "slight"])
    predicted = np.array(["fatal", "slight", "serious", "fatal", "slight", "slight"])
    groups = np.array(["profile-a", "profile-a", "profile-b", "profile-b", "profile-c", "profile-c"])
    result = profile_cluster_bootstrap_macro_f1(truth, predicted, groups, ["fatal", "serious", "slight"], n_resamples=100)

    assert result == profile_cluster_bootstrap_macro_f1(
        truth, predicted, groups, ["fatal", "serious", "slight"], n_resamples=100)
    assert result["replicates"] == 100
    assert result["test_profile_groups"] == 3
    assert result["replicates_missing_class"] >= 0
    assert len(result["interval"]) == 2
    assert 0 <= result["interval"][0] <= result["interval"][1] <= 1


def test_profile_cluster_bootstrap_rejects_misaligned_rows():
    import numpy as np
    from roadlab.train import profile_cluster_bootstrap_macro_f1

    with pytest.raises(ValueError, match="same nonzero length"):
        profile_cluster_bootstrap_macro_f1(np.array(["a"]), np.array(["a", "b"]), np.array(["g"]), ["a", "b"])
