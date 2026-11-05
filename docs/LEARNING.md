# Explain the project, don't just show it

## Walk through one prediction
1. Trace a form field through the JSON request, `validate_record`, DataFrame, preprocessing and classifier.
2. Explain how `null` differs from an unseen category and why we reject unknown category strings at the API boundary.
3. Find a predicted class whose score is largest. Explain why the score is not a trustworthy individual probability.

## Defend the evaluation
- Why can a majority predictor reach 84.7% accuracy and still fail the useful part of the task?
- Why is the fatal-class recall based on only 31 cases? What changes if one additional fatal case is correctly classified?
- What leakage would occur if imputation or one-hot encoding were fit before splitting?
- Why is keeping identical input profiles together useful, and why does it not prove the rows are independent incidents?
- Why does the pipeline not refit the selected model on validation? What extra evaluation would a refitted artifact need?
- Why cannot the new macro-F1 be called an improvement over a different earlier split?

## Engineering exercises
- Add a test for a missing model artifact returning 503, without loading untrusted joblib files.
- Add calibration using development data only, then assess a proper scoring rule and calibration error on the untouched test set.
- Introduce nested validation or a properly justified repeated-group split while preserving a final locked test set.
- Add an explainable error-analysis page using aggregate slices and minimum sample sizes. Avoid exposing individual casualty records.
- Document one genuine design change you make, run tests, and commit it with an accurate message. Do not create empty commits for graph activity.

This guide is a learning aid, not a script to claim unaided work or a completed institution assessment. Write your own explanation and disclose AI assistance according to your course rules.
