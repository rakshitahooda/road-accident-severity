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
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
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


def preprocess():
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                                  ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL)])


def candidate_estimators():
    return {
        "majority": DummyClassifier(strategy="most_frequent"),
        "balanced_logistic": LogisticRegression(class_weight="balanced", max_iter=1500, random_state=SEED),
        "balanced_tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=15, class_weight="balanced", random_state=SEED),
        "balanced_forest": RandomForestClassifier(n_estimators=160, min_samples_leaf=8, max_depth=14,
                                                   class_weight="balanced_subsample", n_jobs=2, random_state=SEED),
    }


def make_model(estimator):
    return Pipeline([("preprocess", preprocess()), ("classifier", estimator)])


def grouped_cv_splits(y, groups, n_splits=5):
    """Return stratified folds while keeping each selected-feature profile in one fold."""
    truth = np.asarray(y)
    group_values = np.asarray(groups)
    if truth.ndim != 1 or group_values.ndim != 1 or len(truth) != len(group_values) or not len(truth):
        raise ValueError("y and groups must be aligned nonempty one-dimensional arrays")
    if pd.isna(group_values).any():
        raise ValueError("groups must not contain missing values")
    if n_splits < 2 or pd.Series(group_values).nunique() < n_splits:
        raise ValueError("n_splits must be at least two and no greater than the number of groups")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(splitter.split(np.zeros((len(truth), 1)), truth, group_values))


def fatal_score_threshold_curve(y, fatal_scores, thresholds=None, fatal_label="Fatal injury"):
    """Describe fatal-score cutoffs without selecting a threshold or using test rows."""
    truth = np.asarray(y)
    scores = np.asarray(fatal_scores, dtype=float)
    if truth.ndim != 1 or scores.ndim != 1 or len(truth) == 0 or len(truth) != len(scores):
        raise ValueError("y and fatal_scores must be aligned nonempty one-dimensional arrays")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("fatal_scores must be finite values between zero and one")
    if thresholds is None:
        cutoffs = np.round(np.arange(0, 1.0001, .05), 2)
    else:
        cutoffs = np.asarray(thresholds, dtype=float)
    if (cutoffs.ndim != 1 or not len(cutoffs) or not np.isfinite(cutoffs).all()
            or np.any((cutoffs < 0) | (cutoffs > 1)) or len(np.unique(cutoffs)) != len(cutoffs)):
        raise ValueError("thresholds must be unique finite values between zero and one")

    is_fatal = truth == fatal_label
    fatal_rows = int(is_fatal.sum())
    nonfatal_rows = int((~is_fatal).sum())
    curve = []
    for cutoff in cutoffs:
        flagged = scores >= cutoff
        detected = int(np.sum(is_fatal & flagged))
        false_alarms = int(np.sum((~is_fatal) & flagged))
        curve.append({
            "threshold": float(cutoff),
            "fatal_detected": detected,
            "fatal_rows": fatal_rows,
            "fatal_recall": detected / fatal_rows if fatal_rows else None,
            "nonfatal_flagged": false_alarms,
            "nonfatal_rows": nonfatal_rows,
            "nonfatal_false_alarm_rate": false_alarms / nonfatal_rows if nonfatal_rows else None,
            "all_rows_flagged": int(flagged.sum()),
        })
    return curve


def development_grouped_cross_validation(X, y, groups, n_splits=5):
    """Compare candidates with out-of-fold predictions from development rows only."""
    truth = np.asarray(y)
    group_values = np.asarray(groups)
    if len(X) != len(truth) or len(truth) != len(group_values):
        raise ValueError("X, y and groups must have the same length")
    labels = sorted(pd.unique(truth).tolist())
    fatal_label = "Fatal injury"
    if fatal_label not in labels:
        raise ValueError("development data must contain fatal rows")
    splits = grouped_cv_splits(truth, group_values, n_splits=n_splits)
    names = list(candidate_estimators())
    fold_metrics = {name: [] for name in names}
    out_of_fold = {name: np.empty(len(truth), dtype=object) for name in names}
    out_of_fold_fatal_scores = {name: np.full(len(truth), np.nan, dtype=float) for name in names}
    assignment_counts = np.zeros(len(truth), dtype=np.int8)
    fold_reports = []

    for fold_number, (train_rows, validation_rows) in enumerate(splits, start=1):
        train_groups = set(group_values[train_rows])
        validation_groups = set(group_values[validation_rows])
        if not train_groups.isdisjoint(validation_groups):
            raise AssertionError("a feature-profile group crossed a CV fold")
        if not set(labels).issubset(set(truth[train_rows])):
            raise ValueError(f"CV fold {fold_number} training data is missing a class")
        assignment_counts[validation_rows] += 1
        fold_fatal_rows = truth[validation_rows] == fatal_label
        fold_reports.append({
            "fold": fold_number,
            "train_rows": int(len(train_rows)),
            "validation_rows": int(len(validation_rows)),
            "validation_fatal_rows": int(fold_fatal_rows.sum()),
            "train_profile_groups": int(len(train_groups)),
            "validation_profile_groups": int(len(validation_groups)),
            "group_overlap": 0,
        })
        for name, estimator in candidate_estimators().items():
            model = make_model(estimator)
            model.fit(X.iloc[train_rows], truth[train_rows])
            validation_X = X.iloc[validation_rows]
            prediction = model.predict(validation_X)
            class_order = list(model.classes_)
            if fatal_label not in class_order:
                raise ValueError(f"CV fold {fold_number} model {name} is missing the fatal class")
            fatal_column = class_order.index(fatal_label)
            out_of_fold_fatal_scores[name][validation_rows] = model.predict_proba(validation_X)[:, fatal_column]
            out_of_fold[name][validation_rows] = prediction
            correct_fatal = int(np.sum(fold_fatal_rows & (prediction == fatal_label)))
            fold_metrics[name].append({
                "fatal_correct": correct_fatal,
                "fatal_rows": int(fold_fatal_rows.sum()),
                "fatal_recall": correct_fatal / int(fold_fatal_rows.sum()) if fold_fatal_rows.any() else None,
                "macro_f1": float(f1_score(truth[validation_rows], prediction, labels=labels,
                                            average="macro", zero_division=0)),
            })

    if not np.all(assignment_counts == 1):
        raise AssertionError("each development row must receive exactly one out-of-fold prediction")
    if any(not np.isfinite(scores).all() for scores in out_of_fold_fatal_scores.values()):
        raise AssertionError("each development row must receive one finite out-of-fold fatal score")

    summaries = {}
    total_fatal = int(np.sum(truth == fatal_label))
    for name, fold_values in fold_metrics.items():
        fatal_correct = int(sum(item["fatal_correct"] for item in fold_values))
        recalls = np.asarray([item["fatal_recall"] for item in fold_values], dtype=float)
        macro_scores = np.asarray([item["macro_f1"] for item in fold_values], dtype=float)
        summaries[name] = {
            "oof_fatal_correct": fatal_correct,
            "oof_fatal_rows": total_fatal,
            "oof_fatal_recall": fatal_correct / total_fatal,
            "fold_fatal_recall": recalls.tolist(),
            "mean_fold_fatal_recall": float(np.mean(recalls)),
            "sd_fold_fatal_recall": float(np.std(recalls, ddof=1)),
            "oof_macro_f1": float(f1_score(truth, out_of_fold[name], labels=labels,
                                           average="macro", zero_division=0)),
            "mean_fold_macro_f1": float(np.mean(macro_scores)),
            "sd_fold_macro_f1": float(np.std(macro_scores, ddof=1)),
            "fatal_score_threshold_curve": fatal_score_threshold_curve(
                truth, out_of_fold_fatal_scores[name]),
        }

    return {
        "method": "StratifiedGroupKFold",
        "n_splits": n_splits,
        "seed": SEED,
        "development_rows": int(len(truth)),
        "development_profile_groups": int(pd.Series(group_values).nunique()),
        "grouping": "Hash of the ten selected model-input features only; labels are not part of the group key.",
        "scope": "Development data only; the final test holdout is excluded from all CV folds.",
        "fatal_score_threshold_method": (
            "Descriptive one-vs-rest cutoffs over out-of-fold fatal-class scores; scores are uncalibrated. "
            "No cutoff is selected or recommended, and these cutoffs do not change the live three-class model."
        ),
        "folds": fold_reports,
        "models": summaries,
    }


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
    # Cross-validate only development rows (the original train+validation partition).
    # Test rows remain excluded from every CV fit, score and model comparison here.
    development_cv = development_grouped_cross_validation(
        X.iloc[development].reset_index(drop=True), y.iloc[development].reset_index(drop=True),
        groups.iloc[development].to_numpy())
    candidates = candidate_estimators()
    models, validation_scores = {}, {}
    for name, estimator in candidates.items():
        model = make_model(estimator)
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
              "models": reports, "development_grouped_cross_validation": development_cv,
              "validation_fatal_error_analysis": validation_fatal_analysis,
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
                      "development_grouped_cross_validation": development_cv,
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
