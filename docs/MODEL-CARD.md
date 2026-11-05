# RoadLab model card

## Intended use
An educational analysis of retrospective road accident records and demonstration of a reproducible ML service. Intended users are students learning imbalanced classification and software engineering. No safety, emergency-response, insurance, policing or individual-risk decision use is supported.

## Inputs and excluded information
Ten fields: hour, vehicles involved, day, area type, junction, road surface type/condition, light, weather and vehicle type. Some fields are available only after an accident. Excluded fields include casualty severity, casualty counts, casualty demographics, accident cause, collision type and driver demographics. Excluding sensitive fields does not establish fairness; remaining variables can be proxies.

## Training and evaluation
12,316 records; no exact full-row duplicates removed. Identical selected-feature profiles form 11,290 groups. GroupShuffleSplit with fixed seeds creates 7,861 training, 1,962 validation and 2,493 test rows; group overlap is checked as zero. This is not stratified by label, so every split is checked for all classes. Each selected-feature profile stays in a single split, including profiles with conflicting labels. This prevents one easy form of leakage but cannot establish that distinct rows represent independent accidents; unique incident identifiers and dates are unavailable in the CSV.

All learned imputation/encoding/scaling occurs within a training-fitted Pipeline. Three fixed balanced models compete on validation macro-F1; majority is a reference, not a production candidate. The selected model is not refitted on validation, so the artifact is exactly the model evaluated on the held-out test set. There is no tuning after test inspection. See `artifacts/metrics.json` for full metrics, runtime versions and class counts.

## Findings
Balanced random forest: test macro-F1 0.442, accuracy 0.731, fatal recall 0.355 (11 of 31 fatal test cases), serious recall 0.387, slight recall 0.794. Stratified test bootstrap interval for macro-F1 is 0.412–0.474 over 300 resamples. It preserves observed class counts and conditions on this particular trained model/test set; it does not estimate uncertainty over new datasets, retraining or selection.

## Important limitations
- Scarce fatal cases mean unstable rare-class estimates and many missed fatal outcomes.
- A random grouped split cannot assess different locations, time periods or deployment conditions.
- Source labels, recording practices, under-reporting, imbalance and missingness can create bias.
- Probability outputs are uncalibrated model scores. There is no abstention, prospective validation or operational threshold policy.
- Validation permutation importance is an associational sensitivity measure, not causation. Correlated inputs can share/mask importance.
- A majority baseline reaches 84.7% accuracy while recalling none of the serious/fatal cases. Higher accuracy is not sufficient evidence of a better model.
- Schema validation bounds vehicle counts at 20 for this demo; that is an application guardrail, not a physical claim about possible accidents.

## Next research steps
Obtain a licensed, independently documented dataset with incident identifiers, actual dates and geographic holdouts; collect representative validation data; compare calibrated/ordinal models and cost-sensitive objectives; investigate errors and fairness with a clear, ethical use case. None of these are claimed complete.

The stratified row-bootstrap interval is descriptive and conditional on the fixed holdout. It does not model dependence between rows sharing feature profiles; a grouped bootstrap or repeated grouped experiment is an appropriate follow-up.
