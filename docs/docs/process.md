# Electricity Theft Detection Using Pakistani Household Data

## Objective
Develop a machine learning model to classify electricity consumers as:

- **0 = Normal Consumer**
- **1 = Theft Consumer**

---

## Data Collection

- Collected smart meter data from **42 Pakistani residential houses**.
- Data includes:
  - Date_Time
  - Usage_kW
  - AC Consumption
  - Kitchen Consumption
  - UPS Consumption

---

## Data Preprocessing

- Removed invalid and duplicate records.
- Standardized date and time format.
- Verified all required features.
- Assigned a unique House_ID to each house.

---

## Normal Dataset

- Original household consumption data was considered legitimate.
- No modifications were applied.
- All records were assigned:

```text
Label = 0
```

---

## Theft Data Generation

Since no public Pakistani electricity theft dataset was available, theft scenarios were created based on common electricity theft practices in Pakistan.

Generated theft behaviors include:

- Partial Meter Tampering
- Direct Hook Connection (Kunda)
- Night-Time Theft
- Intermittent Theft

All generated theft records were assigned:

```text
Label = 1
```

---

## Dataset Labels

| Label | Meaning |
|---------|---------|
| 0 | Normal |
| 1 | Theft |

---

## Final Dataset

### Features

```text
Date_Time
Usage_kW
AC_kW
Kitchen_kW
UPS_kW
House_ID
Label
```

### Dataset Summary

```text
Normal Houses = 42
Theft Houses  = 42
Total Houses  = 84
```

Class Distribution:

```text
Normal = 50%
Theft  = 50%
```

---

## Machine Learning Usage

The dataset will be used to train and evaluate:

- Random Forest
- Decision Tree
- XGBoost
- LightGBM
- LSTM

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

## Conclusion

A balanced Pakistani electricity consumption dataset was prepared using data from 42 residential houses. Original records were labeled as normal consumers, while realistic theft scenarios were generated and labeled as theft consumers. The final dataset is ready for machine learning based electricity theft detection.