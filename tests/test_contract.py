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


def test_metadata_reports_both_bootstrap_intervals():
    metrics = client.get("/api/metadata").json()["metrics"]
    profile = metrics["profile_cluster_bootstrap"]

    assert len(metrics["selected_macro_f1_bootstrap_95_ci"]) == 2
    assert profile["interval"] == metrics["selected_macro_f1_profile_cluster_bootstrap_95_ci"]
    assert profile["replicates"] == metrics["bootstrap_replicates"] == 300
    assert profile["test_profile_groups"] == 2258
    audit = metrics["validation_fatal_error_analysis"]
    assert audit["split"] == "validation"
    assert audit["fatal_rows"] == 33
    assert audit["correctly_classified"] + audit["missed"] == audit["fatal_rows"]
    assert sum(item["count"] for item in audit["missed_predictions"]) == audit["missed"]
    assert "not an independent performance estimate" in audit["interpretation"]

def test_development_grouped_cv_metadata_excludes_test_holdout():
    metrics = json.loads((ROOT / "artifacts/metrics.json").read_text())
    cv = metrics["development_grouped_cross_validation"]
    split = metrics["split"]

    assert cv["method"] == "StratifiedGroupKFold"
    assert cv["n_splits"] == 5
    assert cv["development_rows"] == split["rows"]["train"] + split["rows"]["validation"]
    assert "final test holdout is excluded" in cv["scope"].lower()
    assert "no cutoff is selected or recommended" in cv["fatal_score_threshold_method"].lower()
    assert len(cv["folds"]) == 5
    assert all(fold["group_overlap"] == 0 and fold["validation_fatal_rows"] > 0 for fold in cv["folds"])
    development_fatal_rows = (
        split["class_counts"]["train"]["Fatal injury"]
        + split["class_counts"]["validation"]["Fatal injury"]
    )
    assert set(cv["models"]) == {"majority", "balanced_logistic", "balanced_tree", "balanced_forest"}
    for model in cv["models"].values():
        assert model["oof_fatal_rows"] == development_fatal_rows
        assert model["oof_fatal_correct"] <= model["oof_fatal_rows"]
        assert 0 <= model["oof_fatal_recall"] <= 1
        assert 0 <= model["oof_macro_f1"] <= 1
        assert len(model["fold_fatal_recall"]) == 5
        calibration = model["fatal_score_calibration"]
        assert calibration["oof_rows"] == cv["development_rows"]
        assert calibration["fatal_rows"] == development_fatal_rows
        assert 0 <= calibration["brier_score"] <= 1
        assert sum(item["count"] for item in calibration["bins"]) == cv["development_rows"]
        assert "no calibration transform is fitted" in calibration["interpretation"]
        curve = model["fatal_score_threshold_curve"]
        assert len(curve) == 21
        assert [point["threshold"] for point in curve] == pytest.approx([i / 20 for i in range(21)])
        assert all(point["fatal_rows"] == development_fatal_rows for point in curve)
        assert all(0 <= point["fatal_recall"] <= 1 for point in curve)
        assert all(0 <= point["nonfatal_false_alarm_rate"] <= 1 for point in curve)
        assert all(point["fatal_detected"] <= point["fatal_rows"] for point in curve)
        assert all(point["nonfatal_flagged"] <= point["nonfatal_rows"] for point in curve)


def test_fatal_recall_wilson_interval_reflects_small_test_count():
    from roadlab.train import wilson_interval

    lower, upper = wilson_interval(11, 31)
    assert 0 < lower < 11 / 31 < upper < 1
    assert wilson_interval(0, 31)[0] == 0
    assert wilson_interval(31, 31)[1] == 1
    with pytest.raises(ValueError, match="counts"):
        wilson_interval(32, 31)


def test_metrics_include_fatal_recall_interval_for_fixed_test_set():
    metrics = json.loads((ROOT / "artifacts/metrics.json").read_text())
    counts = metrics["selected_fatal_recall_test_counts"]
    interval = metrics["selected_fatal_recall_wilson_95_ci"]
    fatal_rows = metrics["split"]["class_counts"]["test"]["Fatal injury"]
    recall = metrics["models"][metrics["selected"]]["test"]["per_class"]["Fatal injury"]["recall"]

    assert counts == {"detected": 11, "fatal_rows": fatal_rows}
    assert counts["fatal_rows"] == 31
    assert interval[0] < recall < interval[1]
    assert interval == pytest.approx([0.2111665909, 0.5305204594])
    assert 0 <= interval[0] < interval[1] <= 1



def test_fatal_score_calibration_reports_raw_oof_reliability_only():
    from roadlab.train import fatal_score_calibration

    result = fatal_score_calibration(
        ["Fatal injury", "Slight Injury", "Fatal injury", "Serious Injury"],
        [.1, .2, .7, .9],
        n_bins=2,
    )

    assert result["brier_score"] == pytest.approx(.4375)
    assert result["oof_rows"] == 4
    assert result["fatal_rows"] == 2
    assert len(result["bins"]) == 2
    assert result["bins"][0] == {"lower": 0.0, "upper": 0.5, "count": 2, "mean_score": pytest.approx(.15), "fatal_rate": .5}
    assert result["bins"][1] == {"lower": 0.5, "upper": 1.0, "count": 2, "mean_score": pytest.approx(.8), "fatal_rate": .5}
    assert "no calibration transform is fitted" in result["interpretation"]


@pytest.mark.parametrize(("truth", "scores", "n_bins", "match"), [
    (["Fatal injury"], [], 10, "aligned"),
    (["Fatal injury"], [float("nan")], 10, "finite"),
    (["Fatal injury"], [1.1], 10, "finite"),
    (["Fatal injury"], [.5], 1, "at least two"),
    (["Fatal injury"], [.5], True, "integer"),
])
def test_fatal_score_calibration_rejects_invalid_inputs(truth, scores, n_bins, match):
    from roadlab.train import fatal_score_calibration

    with pytest.raises(ValueError, match=match):
        fatal_score_calibration(truth, scores, n_bins=n_bins)

def test_grouped_cv_splits_keep_feature_profiles_together():
    import numpy as np
    from roadlab.train import grouped_cv_splits

    groups = np.repeat(np.arange(30), 2)
    y = np.repeat(np.repeat(["Fatal injury", "Serious Injury", "Slight Injury"], 10), 2)
    validation_counts = np.zeros(len(y), dtype=int)
    classes = set(y)

    folds = grouped_cv_splits(y, groups, n_splits=5)
    assert len(folds) == 5
    for train_rows, validation_rows in folds:
        assert set(groups[train_rows]).isdisjoint(groups[validation_rows])
        assert set(y[train_rows]) == classes
        assert set(y[validation_rows]) == classes
        validation_counts[validation_rows] += 1
    assert np.all(validation_counts == 1)


def test_fatal_score_threshold_curve_exposes_recall_false_alarm_tradeoff():
    from roadlab.train import fatal_score_threshold_curve

    curve = fatal_score_threshold_curve(
        ["Fatal injury", "Fatal injury", "Fatal injury", "Serious Injury", "Slight Injury", "Slight Injury"],
        [.9, .4, .1, .8, .3, .05],
        thresholds=[.5, .2],
    )

    assert curve[0] == {
        "threshold": .5,
        "fatal_detected": 1,
        "fatal_rows": 3,
        "fatal_recall": pytest.approx(1 / 3),
        "nonfatal_flagged": 1,
        "nonfatal_rows": 3,
        "nonfatal_false_alarm_rate": pytest.approx(1 / 3),
        "all_rows_flagged": 2,
    }
    assert curve[1]["fatal_detected"] == 2
    assert curve[1]["fatal_recall"] == pytest.approx(2 / 3)
    assert curve[1]["nonfatal_flagged"] == 2
    assert curve[1]["nonfatal_false_alarm_rate"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(("truth", "scores", "thresholds", "match"), [
    (["Fatal injury"], [], None, "aligned"),
    (["Fatal injury"], [float("nan")], None, "finite"),
    (["Fatal injury"], [1.1], None, "finite"),
    (["Fatal injury"], [.5], [-.1], "thresholds"),
    (["Fatal injury"], [.5], [.5, .5], "thresholds"),
])
def test_fatal_score_threshold_curve_rejects_invalid_inputs(truth, scores, thresholds, match):
    from roadlab.train import fatal_score_threshold_curve

    with pytest.raises(ValueError, match=match):
        fatal_score_threshold_curve(truth, scores, thresholds=thresholds)


def test_validation_fatal_error_analysis_uses_aggregate_counts():
    from roadlab.train import fatal_error_analysis

    result = fatal_error_analysis(
        ["Fatal injury", "Fatal injury", "Fatal injury", "Serious Injury", "Slight Injury"],
        ["Fatal injury", "Serious Injury", "Slight Injury", "Fatal injury", "Slight Injury"],
        ["Fatal injury", "Serious Injury", "Slight Injury"],
    )

    assert result["split"] == "validation"
    assert result["fatal_rows"] == 3
    assert result["correctly_classified"] == 1
    assert result["missed"] == 2
    assert result["recall"] == pytest.approx(1 / 3)
    assert result["missed_predictions"] == [
        {"label": "Serious Injury", "count": 1},
        {"label": "Slight Injury", "count": 1},
    ]
    assert "not an independent performance estimate" in result["interpretation"]


def test_validation_fatal_error_analysis_handles_no_fatal_rows():
    from roadlab.train import fatal_error_analysis

    result = fatal_error_analysis(["Slight Injury"], ["Slight Injury"], ["Fatal injury", "Slight Injury"])

    assert result["fatal_rows"] == 0
    assert result["recall"] is None
    assert result["missed_predictions"] == []


def test_validation_fatal_error_analysis_rejects_misaligned_rows():
    from roadlab.train import fatal_error_analysis

    with pytest.raises(ValueError, match="aligned"):
        fatal_error_analysis(["Fatal injury"], [], ["Fatal injury", "Slight Injury"])


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
