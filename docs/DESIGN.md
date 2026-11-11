# Design decisions

## Separate the experiment from the interface
`core.py` owns feature names and validation. `train.py` constructs a single sklearn Pipeline, writes its measured report, and serializes it. `api.py` loads that artifact once and exposes a bounded batch endpoint. The UI obtains input choices from the actual training schema, not a handwritten approximation. It makes real HTTP inference calls; it does not display hardcoded predictions as live output.

## Narrow features instead of overclaiming advance prediction
The original data contains post-outcome casualty fields. We exclude those and use a conservative scene/vehicle subset. Even so, vehicles involved and some conditions are retrospectively recorded, so the application explicitly classifies a recorded scene rather than forecasting a crash.

## Why grouped splitting
Full-row deduplication is insufficient if two records have the same model inputs but differ in unused columns or labels. Hashing selected-feature profiles and using GroupShuffleSplit prevents those profiles appearing in both training and test. Hashes are local split keys, not personal identifiers. Python/pandas versions are pinned for reproducibility. Missing values are handled consistently before model inference. Grouping is stronger than an ordinary random row split, but is not a substitute for real incident IDs or a time split.

## Why these metrics
Macro-F1 gives each class equal weight; balanced accuracy averages recalls. Both remain visible alongside accuracy and raw confusion counts. The selected model is chosen on validation only. A five-fold StratifiedGroupKFold compares candidate fatal-recall trade-offs on development rows; each fold groups identical selected-feature profiles, and the final test holdout is excluded from CV. In the measured run, out-of-fold fatal recall was 0.496 for balanced logistic regression (63/127), 0.457 for the tree (58/127), and 0.213 for the forest (27/127); macro-F1 was highest for the forest (0.425), versus 0.282 for logistic regression and 0.365 for the tree. This is exploratory because the prior validation split selected the current model. Those profiles are not known incident IDs, so this does not establish independent accidents or measure incident-, time- or location-level uncertainty. The report also keeps a stratified row bootstrap and adds a sensitivity bootstrap that resamples complete groups of identical input profiles. Test results are not used to pick an alternate winner.

## Bound the service
The endpoint accepts at most 100 records, rejects extra fields and impossible demo ranges, and requires explicit nulls for missing fields. No broad CORS policy, uploads, arbitrary code execution or external prediction provider is present. It is an unauthenticated localhost prototype, not hardened production software. Joblib deserialization executes trusted Python artifacts and must never be used for untrusted uploads.

## Reproducibility and CI
A byte-level dataset SHA-256 check fails closed on source changes. CI installs exact direct dependencies, rebuilds artifacts from the public CSV, runs API/contract tests and lint. Builds depend on the public mirror and Python package services remaining available; a downloaded dataset with a different hash is a deliberate failure requiring review. Runtime dependency versions are recorded in the measured report. No production hosting has been deployed.
