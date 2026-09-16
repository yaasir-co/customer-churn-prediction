# Module 02 — Data, Labels and Leakage

## Data schema
The project combines numeric variables such as tenure and monthly charges with categorical variables such as contract type and payment method.

## Target
`churn = 1` means the customer churned; `churn = 0` means the customer stayed.

## Leakage
Data leakage occurs when training uses information that would not be known at real prediction time.

Examples:
- Cancellation date
- Final account status
- Refund issued after cancellation
- A support ticket created after churn

## Why leakage is dangerous
The model can appear extremely accurate during evaluation but fail in production.

## Exercise
Identify three possible leakage features in a real telecom database.
