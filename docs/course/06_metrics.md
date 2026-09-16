# Module 06 — Evaluation Metrics

## Accuracy
(TP + TN) / all observations

Useful but can be misleading on imbalanced data.

## Precision
TP / (TP + FP)

Question:
“When the model flags churn, how often is it correct?”

## Recall
TP / (TP + FN)

Question:
“Of all real churners, how many did we catch?”

## F1
Harmonic mean of precision and recall.

## ROC-AUC
Measures ranking quality across thresholds.

## Log Loss
Penalizes incorrect probability estimates, especially confident wrong predictions.

## Business mapping
If retention offers are expensive, precision may matter more.
If missing churners is very costly, recall may matter more.
