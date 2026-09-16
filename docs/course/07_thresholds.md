# Module 07 — Decision Thresholds

A classifier outputs probability, not a business action.

Default rule:
Predict churn if probability >= 0.50

But 0.50 is not automatically optimal.

## Lower threshold
- More customers flagged
- Usually higher recall
- Usually lower precision
- More intervention cost

## Higher threshold
- Fewer customers flagged
- Usually higher precision
- Usually lower recall

## Production approach
Choose a threshold using:
- Retention offer cost
- Customer lifetime value
- Available retention-team capacity
- Cost of missed churn
