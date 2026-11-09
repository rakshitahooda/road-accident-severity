"""Reproducible grouped holdout experiment; never optimize against the test set."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from urllib.request import urlretrieve
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from roadlab.core import ROOT, FEATURES, NUMERIC, CATEGORICAL, TARGET, SEED, SOURCE, SHA256, features_from_raw


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def evaluate(y, prediction, probabilities, labels):
    return {"accuracy": accuracy_score(y, prediction), "balanced_accuracy": balanced_accuracy_score(y, prediction),
            "macro_f1": f1_score(y, prediction, average="macro", zero_division=0),
            "log_loss": log_loss(y, probabilities, labels=labels),
            "per_class": classification_report(y, prediction, labels=labels, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, prediction, labels=labels).tolist()}


def fatal_error_analysis(y, prediction, labels, fatal_label="Fatal injury"):
    """Summarize fatal-labelled validation rows without exporting row-level examples."""
    truth = np.asarray(y)
    predicted = np.asarray(prediction)
    labels = list(labels)
    if truth.ndim != 1 or predicted.ndim != 1 or len(truth) != len(predicted):
        raise ValueError("y and prediction must be aligned one-dimensional arrays")
    if fatal_label not in labels:
        raise ValueError("fatal_label must be present in labels")

    fatal_rows = truth == fatal_label
    total = int(fatal_rows.sum())
    correctly_classified = int(np.sum(fatal_rows & (predicted == fatal_label)))
    missed_predictions = [
        {"label": label, "count": int(np.sum(fatal_rows & (predicted == label)))}
        for label in labels if label != fatal_label and np.any(fatal_rows & (predicted == label))
    ]
    missed_predictions.sort(key=lambda item: (-item["count"], item["label"]))
    missed = total - correctly_classified
    return {
        "split": "validation",
        "fatal_label": fatal_label,
        "fatal_rows": total,
        "correctly_classified": correctly_classified,
        "missed": missed,
        "recall": correctly_classified / total if total else None,
        "missed_predictions": missed_predictions,
        "interpretation": (
            "The validation split was used to select the model; this error review is exploratory, "
            "not an independent performance estimate. Only aggregate counts are reported."
        ),
    }


def profile_cluster_bootstrap_macro_f1(y, prediction, groups, labels, n_resamples=300, seed=SEED):
    """Resample complete identical-feature-profile groups for a descriptive sensitivity interval."""
    truth = np.asarray(y)
    predicted = np.asarray(prediction)
    group_values = np.asarray(groups)
    if len(truth) == 0 or len(truth) != len(predicted) or len(truth) != len(group_values):
        raise ValueError("y, prediction and groups must have the same nonzero length")
    if not labels or n_resamples < 1:
        raise ValueError("labels and at least one bootstrap resample are required")
    group_codes, unique_groups = pd.factorize(group_values, sort=False)
    if (group_codes < 0).any():
        raise ValueError("groups must not contain missing values")
    rows_by_group = [np.flatnonzero(group_codes == code) for code in range(len(unique_groups))]
    rng = np.random.default_rng(seed)
    scores = np.empty(n_resamples)
    missing_class_resamples = 0
    expected_labels = set(labels)
    for replicate in range(n_resamples):
        sampled_groups = rng.integers(0, len(rows_by_group), size=len(rows_by_group))
        sampled_rows = np.concatenate([rows_by_group[index] for index in sampled_groups])
        missing_class_resamples += int(not expected_labels.issubset(set(truth[sampled_rows])))
        scores[replicate] = f1_score(truth[sampled_rows], predicted[sampled_rows], labels=labels,
                                     average="macro", zero_division=0)
    return {"interval": np.quantile(scores, [.025, .975]).tolist(),
            "replicates": int(n_resamples), "replicates_missing_class": int(missing_class_resamples),
            "test_profile_groups": int(len(unique_groups))}


def run(path, output):
    output.mkdir(parents=True, exist_ok=True)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != SHA256:
        raise ValueError("Dataset checksum differs from the reviewed source snapshot")
    raw = pd.read_csv(path)
    data = raw.drop_duplicates().reset_index(drop=True)
    X = features_from_raw(data)
    y = data[TARGET]
    if y.isna().any() or y.nunique() != 3:
        raise ValueError("Expected three nonmissing severity labels")
    # Keep identical selected-feature profiles together, even if they have conflicting labels.
    # This is stricter than a random row split, but not a real temporal/geographic split.
    groups = pd.util.hash_pandas_object(X, index=False)
    development, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(X, y, groups))
    train_rel, validation_rel = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED + 1).split(X.iloc[development], y.iloc[development], groups.iloc[development]))
    train, validation = development[train_rel], development[validation_rel]
    for indices in [train, validation, test]:
        if y.iloc[indices].nunique() != 3:
            raise ValueError("Every split must retain all three classes")
    for a, b in [(train, validation), (train, test), (validation, test)]:
        assert set(groups.iloc[a]).isdisjoint(set(groups.iloc[b]))
    def preprocess():
        return ColumnTransformer([
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL)])
    candidates = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "balanced_logistic": LogisticRegression(class_weight="balanced", max_iter=1500, random_state=SEED),
        "balanced_tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=15, class_weight="balanced", random_state=SEED),
        "balanced_forest": RandomForestClassifier(n_estimators=160, min_samples_leaf=8, max_depth=14, class_weight="balanced_subsample", n_jobs=2, random_state=SEED)}
    models, validation_scores = {}, {}
    for name, estimator in candidates.items():
        model = Pipeline([("preprocess", preprocess()), ("classifier", estimator)])
        model.fit(X.iloc[train], y.iloc[train])
        models[name] = model
        validation_scores[name] = f1_score(y.iloc[validation], model.predict(X.iloc[validation]), average="macro", zero_division=0)
    selected_name = max((n for n in models if n != "majority"), key=validation_scores.get)
    selected = models[selected_name]
    labels = list(selected.classes_)
    validation_fatal_analysis = fatal_error_analysis(
        y.iloc[validation].to_numpy(), selected.predict(X.iloc[validation]), labels)
    # A single final holdout evaluation after all models and selection rules are fixed.
    reports = {name: {"validation_macro_f1": validation_scores[name], "test": evaluate(y.iloc[test], m.predict(X.iloc[test]), m.predict_proba(X.iloc[test]), labels)} for name, m in models.items()}
    pred, truth = selected.predict(X.iloc[test]), y.iloc[test].to_numpy()
    rng = np.random.default_rng(SEED)
    # Stratified bootstrap preserves the small fatal-class count in each resample.
    by_class = [np.flatnonzero(truth == label) for label in labels]
    bootstrap = []
    for _ in range(300):
        idx = np.concatenate([rng.choice(rows, len(rows), replace=True) for rows in by_class])
        bootstrap.append(f1_score(truth[idx], pred[idx], labels=labels, average="macro", zero_division=0))
    interval = np.quantile(bootstrap, [.025, .975]).tolist()
    profile_bootstrap = profile_cluster_bootstrap_macro_f1(
        truth, pred, groups.iloc[test].to_numpy(), labels, n_resamples=300, seed=SEED)
    schema = {"features": FEATURES, "numeric": {"Hour": {"min": 0, "max": 23}, "Number_of_vehicles_involved": {"min": 1, "max": 20}},
              "categories": {c: sorted(X.iloc[train][c].dropna().unique().tolist()) for c in CATEGORICAL}}
    # Permutation importance is computed on validation, not used for model selection.
    perm = permutation_importance(selected, X.iloc[validation], y.iloc[validation], n_repeats=4, random_state=SEED, scoring="f1_macro", n_jobs=2)
    importance = sorted([{"feature": f, "mean": float(a), "std": float(b)} for f, a, b in zip(FEATURES, perm.importances_mean, perm.importances_std)], key=lambda a: a["mean"], reverse=True)
    counts = {name: y.iloc[idx].value_counts().to_dict() for name, idx in [("train", train), ("validation", validation), ("test", test)]}
    report = {"selected": selected_name, "selection_metric": "validation macro-F1", "seed": SEED, "labels": labels,
              "dataset": {"source": SOURCE, "sha256": actual, "raw_rows": len(raw), "unique_rows": len(data), "removed_duplicates": len(raw)-len(data), "selected_feature_groups": int(groups.nunique())},
              "split": {"method": "GroupShuffleSplit on identical selected-feature profiles", "rows": {"train": len(train), "validation": len(validation), "test": len(test)}, "class_counts": counts, "profile_overlap": 0},
              "models": reports, "validation_fatal_error_analysis": validation_fatal_analysis,
              "selected_macro_f1_bootstrap_95_ci": interval,
              "selected_macro_f1_profile_cluster_bootstrap_95_ci": profile_bootstrap["interval"],
              "bootstrap_replicates": 300, "profile_cluster_bootstrap": profile_bootstrap,
              "validation_permutation_importance": importance,
              "runtime": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
              "limitations": ["Retrospective classification; not pre-crash forecasting or emergency triage.", "One grouped holdout; not temporal, geographic or prospective validation.", "Fatal cases are scarce; all class recalls must be considered.", "Model probabilities are uncalibrated scores, not reliable individual risk estimates.", "Source mirror licensing and original data provenance require review before redistribution or production use."]}
    joblib.dump(selected, output / "model.joblib", compress=3)
    dump(output / "metrics.json", report)
    dump(output / "schema.json", schema)
    # Samples from training only; no held-out test data is exposed as interactive examples.
    sample_frame = X.iloc[train].head(4).astype(object).where(pd.notna(X.iloc[train].head(4)), None)
    dump(output / "examples.json", sample_frame.to_dict(orient="records"))
    pd.DataFrame({"row": test, "actual": truth, "predicted": pred}).to_csv(output / "test_predictions.csv", index=False)
    dump(output / "split_indices.json", {"train": train.tolist(), "validation": validation.tolist(), "test": test.tolist()})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    cm = confusion_matrix(truth, pred, labels=labels)
    axes[0].imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, str(cm[i, j]), ha="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    short = ["Fatal", "Serious", "Slight"]
    axes[0].set(xticks=range(3), yticks=range(3), xticklabels=short, yticklabels=short, xlabel="Predicted", ylabel="Observed", title="Untouched grouped test set")
    names = list(reports)
    axes[1].barh(names, [reports[n]["test"]["macro_f1"] for n in names], color="#15746b")
    axes[1].set(xlim=(0, 1), xlabel="Macro-F1", title="Model comparison — no test tuning")
    fig.savefig(output / "evaluation.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"selected": selected_name, "test_macro_f1": reports[selected_name]["test"]["macro_f1"],
                      "stratified_row_bootstrap_95_ci": interval,
                      "profile_cluster_bootstrap_95_ci": profile_bootstrap["interval"],
                      "profile_cluster_bootstrap": profile_bootstrap,
                      "validation_fatal_error_analysis": validation_fatal_analysis,
                      "split": report["split"]}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "road-accidents.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    if not args.data.exists():
        args.data.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(SOURCE, args.data)
    run(args.data, args.output)
