# RoadLab model card

## Intended use
An educational analysis of retrospective road accident records and demonstration of a reproducible ML service. Intended users are students learning imbalanced classification and software engineering. No safety, emergency-response, insurance, policing or individual-risk decision use is supported.

## Inputs and excluded information
Ten fields: hour, vehicles involved, day, area type, junction, road surface type/condition, light, weather and vehicle type. Some fields are available only after an accident. Excluded fields include casualty severity, casualty counts, casualty demographics, accident cause, collision type and driver demographics. Excluding sensitive fields does not establish fairness; remaining variables can be proxies.

## Training and evaluation
12,316 records; no exact full-row duplicates removed. Identical selected-feature profiles form 11,290 groups. GroupShuffleSplit with fixed seeds creates 7,861 training, 1,962 validation and 2,493 test rows; group overlap is checked as zero. This is not stratified by label, so every split is checked for all classes. Each selected-feature profile stays in a single split, including profiles with conflicting labels. This prevents one easy form of leakage but cannot establish that distinct rows represent independent accidents; unique incident identifiers and dates are unavailable in the CSV.

All learned imputation/encoding/scaling occurs within a training-fitted Pipeline. Three fixed balanced models compete on validation macro-F1; majority is a reference, not a production candidate. A separate five-fold `StratifiedGroupKFold` compares these candidates on the 9,823 development rows only, grouping by hashes of the ten selected input features; the final test holdout is excluded from every CV fit and score. The earlier validation split already selected the current model, so this CV comparison is exploratory rather than an independent post-selection estimate. Identical input profiles are not known incident IDs. The selected model is not refitted on validation, so the artifact is exactly the model evaluated on the held-out test set. There is no tuning after test inspection. See `artifacts/metrics.json` for full metrics, runtime versions and class counts.

Development-only grouped CV (out-of-fold, 5 folds; 127 fatal-labelled rows): majority 0/127 fatal recall 0.000 and macro-F1 0.305; balanced logistic 63/127 (0.496) and 0.282; balanced tree 58/127 (0.457) and 0.365; balanced forest 27/127 (0.213) and 0.425. The forest leads on macro-F1; logistic has the highest fatal recall. These are exploratory comparisons, not independent post-selection estimates, and identical input profiles are not incident IDs. The accompanying fatal-score cutoff curves summarize out-of-fold development predictions only; scores are uncalibrated, no cutoff is selected or recommended, and the deployed three-class model is unchanged.

## Findings
Balanced random forest: test macro-F1 0.442, accuracy 0.731, fatal recall 0.355 (11 of 31 fatal test cases), serious recall 0.387, slight recall 0.794. The 95% Wilson interval for fatal recall is 0.211–0.531 (11/31); it describes sampling variation for this fixed model and test set only. The 300-resample stratified row-bootstrap 95% interval for macro-F1 is 0.412–0.474. A second 300-resample bootstrap of whole identical-feature-profile groups gives 0.411–0.473; it sampled 2,258 test groups, and none of its replicates omitted a class. Both intervals are descriptive and conditional on this fixed model and holdout. The group bootstrap addresses dependence among rows with identical input profiles only; profiles are not incident identifiers, and neither interval estimates uncertainty over new datasets, retraining or model selection. An exploratory audit on the validation split found 3 of 33 fatal-labelled rows correctly classified and 30 missed (19 predicted slight, 11 serious). Validation was also used for model selection, so this is not an independent performance estimate; only aggregate counts are published.

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

The profile-cluster bootstrap is a sensitivity analysis, not incident-level inference: the source has no incident identifiers, so distinct profiles may still belong to the same event. Obtaining a licensed dataset with incident IDs and dates, followed by incident-level and temporal/geographic validation, remains an appropriate follow-up.
