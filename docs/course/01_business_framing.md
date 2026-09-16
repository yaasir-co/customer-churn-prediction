# Module 01 — Business Framing

## Goal
Turn a vague request like “predict churn” into a machine-learning problem that creates business value.

## Questions an ML engineer should ask
- What exactly counts as churn?
- How early must we predict it?
- Which customers can the business actually intervene with?
- What action will happen after a customer is flagged?
- What is the cost of a false positive?
- What is the cost of a false negative?
- How will success be measured: model metric, saved customers, retained revenue, or all three?

## Prediction horizon
A production system should define a horizon such as “predict whether a customer will churn in the next 30 days using only information available today.”

## Key lesson
A technically accurate model can still fail if the business has no useful intervention or if the target is defined incorrectly.

## Exercise
Write a one-paragraph product requirement for a telecom retention team using this model.
