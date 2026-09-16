# Module 08 — Explainability

## Global explainability
Which features matter most overall?

This project uses permutation importance:
1. Measure model performance.
2. Shuffle one feature.
3. Measure performance again.
4. Large performance drop => important feature.

## Local explanation
For an individual profile, inspect the customer attributes associated with higher or lower risk.

## Critical limitation
Feature importance is not causal inference.

A model may learn correlation without proving that changing the feature will change the outcome.
