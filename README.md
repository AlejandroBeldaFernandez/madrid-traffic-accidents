# Traffic Accident Injury Prediction — Madrid 2019–2023

Supervised binary classification project to predict whether a traffic accident in Madrid will result in at least one injured person, using open data from the Madrid City (2019–2023).

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Business Value](#business-value)
3. [Dataset](#dataset)
4. [Data Challenges and Transformations](#data-challenges-and-transformations)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Methodology](#methodology)
7. [Preprocessing Pipeline](#preprocessing-pipeline)
8. [Models and Hyperparameter Optimisation](#models-and-hyperparameter-optimisation)
9. [Results and Evaluation](#results-and-evaluation)
10. [Explainability — SHAP Analysis](#explainability--shap-analysis)
11. [Problems Encountered](#problems-encountered)
12. [Conclusions](#conclusions)
13. [Possible Improvements](#possible-improvements)
14. [Project Structure](#project-structure)
15. [Requirements](#requirements)

---

## Problem Statement

Traffic accidents in Madrid generate thousands of incident reports every year. Each report contains information about the time, location, type of accident, vehicles involved, weather conditions, and the outcome for every person present. Emergency response teams must decide quickly how many resources and what type of personnel to dispatch.

The question this project addresses is:

> **Given information available immediately after an accident is reported, can we predict whether at least one person will be injured?**

This is framed as a **binary classification** problem:

| Label | Meaning |
|---|---|
| `injured` | At least one person sustained any injury |
| `no_injury` | No person involved sustained an injury |

---

## Business Value

A reliable real-time injury predictor has direct operational value for emergency services:

- **Faster medical dispatch.** If the model flags an accident as likely to result in injuries, ambulances and medical personnel can be dispatched simultaneously with police, reducing response time.
- **Resource prioritisation.** In high-incident periods, the model can help prioritise which accidents require immediate medical attention and which can be handled by traffic officers alone.
- **Decision support, not replacement.** The model is designed as a support tool. Its predictions are probabilistic and act as a first filter — human operators retain final decision authority.

The features used (accident type, time, district, number of vehicles, vehicle types, weather) are all available from the initial incident report, making real-time inference feasible.

---

## Dataset

- **Source:** Open data portal of the Madrid City Council
- **Period:** 2019–2023 (five years)
- **Granularity:** One row per person involved in an accident
- **Size:** ~170,000 rows before cleaning
- **Key columns:**

| Column | Description |
|---|---|
| `fecha` | Date of the accident |
| `hora` | Time of the accident |
| `distrito` | Madrid district |
| `tipo_accidente` | Accident type (rear-end, pedestrian knockdown, etc.) |
| `estado_meteorológico` | Weather conditions |
| `tipo_vehiculo` | Vehicle type of the person's vehicle |
| `lesividad` | Injury severity of the person |
| `coordenada_x_utm`, `coordenada_y_utm` | GPS coordinates (UTM) |

---

## Data Challenges and Transformations

### 1. Person-level to accident-level aggregation

The original dataset is at **person level** — one row per person per accident. The prediction target must be at **accident level** — one row per accident. This required a full aggregation step:

- **Target construction:** An accident is labelled `injured` if at least one person's `lesividad` is not `"Sin asistencia sanitaria"` (no medical assistance required). This is computed with a lambda aggregation that checks whether any person received medical attention.
- **Vehicle flags:** Binary flags created for each vehicle category (`flag_moto`, `flag_car`, `flag_van_truck`, `flag_bike_scooter`, `flag_bus`, `flag_other`) by checking whether any person in the accident was driving that vehicle type.
- **Counts:** `num_vehicles` (number of distinct vehicles) and `num_persons` (number of people involved) derived per accident.
- **Conditions:** District, accident type, weather, and time slot taken from the first record of each accident (consistent within an accident).

### 2. Severity mapping

The original `lesividad` column has eight categories. These were mapped to a binary label:

| Original | Binary label |
|---|---|
| Sin asistencia sanitaria | `no_injury` |
| All other categories (any medical attention) | `injured` |

### 3. Feature engineering

- **`time_slot`:** Hour extracted from `hora` and binned into: `dawn` (0–6), `morning` (7–11), `afternoon` (12–17), `rush_hour` (18–20), `night` (21–23).
- **`season`:** Month mapped to meteorological season.
- **`year`:** Extracted from `fecha` for temporal analysis.

### 4. Coordinate cleaning

The dataset includes UTM coordinates for each accident, which were included as numeric features in the model. During the cleaning phase, a systematic data entry error was discovered: approximately 11,000 accidents had coordinate values exactly 1,000 times larger than the valid Madrid UTM range. Plotting the raw data made the pattern immediately visible — the outlier points formed a perfect scaled replica of the city's street layout. The fix was dividing both coordinate columns by 1,000 for all affected rows, recovering all records with no data loss.

---

## Exploratory Data Analysis

The EDA was conducted at accident level after aggregation. Key findings:

### Target distribution

After aggregation, the class distribution shifts significantly compared to person-level data:

- **Person level:** ~54% injured, ~46% no_injury
- **Accident level:** ~84% injured, ~16% no_injury

This shift is expected and correct. An accident that involved five people, four of whom had no injury and one of whom did, is still an injured accident. The dataset is **imbalanced**, with no_injury accidents being the minority class.

### District

All districts show similar proportions of injured accidents. There is no single district that stands out dramatically, though some variation exists.

### Time slot

Late night has the highest proportion of injured accidents. This may reflect increased risk due to fatigue and lower traffic volume (fewer accidents overall but proportionally more severe).

### Vehicle type

Motorcycles, scooters, and buses are the vehicle types most associated with injured outcomes. Motorcycle and scooter injuries likely reflect the lack of physical protection for the rider; bus injuries may be related to the mass and size of the vehicle.

### Weather conditions

Weather conditions show minimal variation across injury rates. The differences between categories are small and the model treats weather as a weak signal.

### Accident type

Pedestrian knockdowns and road departures show the highest proportion of injured outcomes. Rear-end and lateral collisions, despite being the most frequent accident types, result in fewer injuries proportionally.

### Number of vehicles

Accidents involving two or more vehicles are more likely to result in injury than single-vehicle accidents.

### Alcohol and drugs

Drug-positive tests show a slightly stronger association with injured outcomes than alcohol. However, both flags have low global importance due to the rarity of positive tests.

### Temporal trend

The proportion of injured accidents peaked in 2020, likely influenced by reduced traffic volume during COVID-19 lockdowns — fewer but proportionally higher-risk journeys. No clear upward or downward trend is observed across the full period.

---

## Methodology

The project follows a structured supervised learning workflow:

1. **Data loading and cleaning** — handle missing values, fix data types, remove duplicates.
2. **Target construction** — aggregate from person level to accident level, derive binary label.
3. **Feature engineering** — time slots, season, vehicle flags, counts.
4. **Exploratory data analysis** — visualise distributions and relationships with the target.
5. **Train/test split** — stratified 80/20 split to preserve class distribution.
6. **Preprocessing pipeline** — encoding and scaling via `ColumnTransformer`.
7. **Model training** — three models trained with hyperparameter optimisation via Optuna.
8. **Evaluation** — ROC AUC, balanced accuracy, per-class F1, confusion matrices.
9. **Explainability** — SHAP values on the best model (CatBoost).

---

## Preprocessing Pipeline

A `ColumnTransformer` applies different transformations to each feature group:

| Feature group | Transformation |
|---|---|
| High-cardinality categorical |  Target Encoding |
| Ordered categorical | Ordinal Encoding |
| Low-cardinality categorical | One-Hot Encoding |
| Numeric  | Standard Scaling |
| Binary flags | Passthrough |
| Temporal | Passthrough |

All transformers are fitted exclusively on the training set. Target Encoding uses cross-validated smoothing to prevent target leakage.

---

## Models and Hyperparameter Optimisation

Three models were trained, all with class imbalance handling:

### Logistic Regression

- **Imbalance handling:** `class_weight='balanced'`
- **Optimised hyperparameters:** `C` (regularisation strength), `penalty` (`l1` / `l2`), solver selected automatically based on penalty
- **Optimisation:** Optuna, 50 trials, stratified 5-fold CV, balanced accuracy as objective

### Random Forest

- **Imbalance handling:** `class_weight` (`balanced` / `balanced_subsample`)
- **Optimised hyperparameters:** `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- **Optimisation:** Optuna, 50 trials, stratified 5-fold CV, balanced accuracy as objective

### CatBoost

- **Imbalance handling:** `auto_class_weights` (`Balanced` / `SqrtBalanced`) as a hyperparameter
- **Optimised hyperparameters:** `depth`, `learning_rate`, `l2_leaf_reg`, `bagging_temperature`, `iterations`, `auto_class_weights`
- **Optimisation:** Optuna with manual `StratifiedKFold` loop (required to pass `verbose=0` to CatBoost fit)
- **Note:** `bagging_temperature` and `subsample` are mutually exclusive in CatBoost (they correspond to Bayesian vs Bernoulli bootstrap modes). Only `bagging_temperature` was used.

Balanced accuracy was chosen as the cross-validation metric because it accounts for class imbalance by averaging recall across both classes, giving equal weight to correct identification of injured and no_injury accidents.

---

## Results and Evaluation

### Test set performance

| Model | ROC AUC | Balanced Accuracy | F1 no_injury | F1 injured | Macro F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.851 | 0.787 | 0.55 | 0.85 | 0.70 |
| Random Forest | 0.862 | 0.790 | 0.58 | 0.88 | 0.73 |
| CatBoost | 0.873 | 0.801 | 0.59 | 0.88 | 0.73 |

### Key observations

- All three models perform at a similar level. The differences in ROC AUC and balanced accuracy are small (< 0.03).
- **CatBoost achieves the best results** across all metrics.
- Precision on the `no_injury` class is low across all models (~0.40–0.45), reflecting the difficulty of identifying the minority class in this imbalanced setting.
- Recall on `injured` is high (> 0.90 for all models), meaning the models are effective at catching genuine injury accidents — the most operationally important outcome.

### Production recommendation

In a production environment, **Logistic Regression would be the preferred model**. It delivers comparable quality with far lower computational cost, is interpretable by non-technical stakeholders, and can be retrained quickly as new accident data arrives. CatBoost's performance advantage is real but marginal, and does not justify the additional complexity in most deployment scenarios.

---

## Explainability — SHAP Analysis

SHAP (SHapley Additive exPlanations) was applied to the CatBoost model to understand what drives predictions.

### Global feature importance

The most influential features, ranked by mean absolute SHAP value:

1. **Accident type** — the strongest predictor. Pedestrian knockdowns and road departures push strongly towards `injured`; rear-end and parking accidents push towards `no_injury`.
2. **Number of vehicles** — multi-vehicle collisions are strongly associated with injury.
3. **District** — provides moderate contextual signal; some districts are consistently riskier.
4. **Time slot** — late night pushes slightly towards `injured`.
5. **Vehicle flags** — motorcycle and scooter presence increases injury probability.

**Least relevant features:** Season and weather conditions show very low SHAP values, confirming that these variables do not meaningfully differentiate injured from no-injury accidents. Alcohol and drug flags are also weak predictors globally due to the rarity of positive tests.

### Error analysis

**False negatives (injured predicted as no_injury):** These are accidents where the injury signal is weak — typically single-vehicle, low-risk accident type, off-peak hours. The model is not wrong to be uncertain — the features genuinely resemble a no-injury accident.

**False positives (no_injury predicted as injured):** These accidents share structural characteristics with the injured majority — multiple vehicles, risky time, high-risk accident type — but happened not to cause injury. The model over-generalises on structural risk factors, which is a reasonable behaviour in a triage context.

Both error types concentrate on the same top features, confirming the model is failing on structurally ambiguous cases rather than random noise.

---

## Problems Encountered

### 1. Low precision on the minority class

The dataset is heavily imbalanced (~84% injured, ~16% no_injury). Despite class weighting, precision for `no_injury` remains low across all models. This means the model generates a significant number of false positives — accidents flagged as injured that resulted in no injury. In an operational context where false negatives (missed injuries) are far more costly than false positives, this is an acceptable trade-off, but it limits the model's use as a strict binary filter.

### 2. Dataset not prepared for the prediction target

The original dataset was collected for administrative and statistical purposes, not for machine learning. It records outcomes at person level after the fact, making it necessary to reconstruct accident-level information through aggregation. Decisions such as how to handle conflicting severity records within the same accident, and how to derive a single coherent label, required careful design. The target variable does not exist in the raw data — it had to be engineered entirely.

### 3. Corrupted coordinate values

The dataset includes UTM GPS coordinates for each accident. During cleaning we discovered that ~11,000 records had coordinate values exactly 1,000 times larger than the valid Madrid UTM range — a systematic data entry error. The problem was not obvious from a simple inspection of the raw numbers, but became clear when plotting: the outlier points formed a perfect scaled replica of Madrid's street layout. The fix was straightforward — dividing both coordinate columns by 1,000 for all affected rows — and no records were lost. The corrected coordinates were included as numeric features in the model.

---

## Conclusions

This project demonstrates that it is possible to predict traffic accident injuries in Madrid using only information available at the moment the incident is reported, before any medical assessment takes place.

The best model (CatBoost) achieves a ROC AUC of 0.873 and a balanced accuracy of 0.801. More importantly, the two strongest predictors are accident type and number of vehicles involved, both of which are captured in the initial report filed by the responding officer. This means the prediction can be made in real time, at the scene, with no additional data collection required.

The practical implication is concrete: emergency coordination services could use a model of this kind to prioritise resource dispatch. Accidents flagged as high injury probability would trigger faster ambulance allocation or alert nearby medical units, reducing response time in the cases where it matters most.

The main current limitation is precision on the no injury class, which generates false positives. In an emergency context this is the preferable type of error  it is safer to dispatch resources unnecessarily than to fail to respond to a genuine injury  but it has operational cost implications that would need to be evaluated against the budget constraints of the deploying organisation.

For a production deployment, Logistic Regression is the recommended model. It matches CatBoost closely enough in performance that its interpretability and low computational cost justify the choice, particularly in environments where decisions need to be auditable and explainable to non-technical stakeholders.

---

## Possible Improvements

- **Resampling with SMOTE or ADASYN.** Applying synthetic minority over-sampling (SMOTE, BorderlineSMOTE, ADASYN) or combined over/under-sampling strategies from the `imbalanced-learn` library could improve recall and precision on the `no_injury` class.
- **Imbalanced-learn native classifiers.** Ensemble methods designed specifically for imbalanced problems — `BalancedRandomForest`, `EasyEnsembleClassifier`, `RUSBoostClassifier` — could outperform the current models, particularly on the minority class.
- **Threshold optimisation.** Rather than using the default 0.5 classification threshold, tuning it on a validation set to minimise false negatives subject to a precision constraint would better align the model with operational requirements.
- **Spatial features.** If coordinate data quality improves, or if addresses can be geocoded, spatial features (road type, zone speed limit, hospital proximity, intersection density) would likely add meaningful predictive signal.
- **External data integration.** Adding road infrastructure data, real-time traffic volume, or historical accident density per location could significantly improve model performance.
- **Temporal modelling.** Exploring whether accident risk patterns have changed meaningfully post-COVID (2022–2023 vs 2019–2020) could justify training separate models per period or adding temporal drift detection.

---


## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
catboost
optuna
shap
```

Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn catboost optuna shap imbalanced-learn
```

---

*Data source: https://www.kaggle.com/datasets/leomed666/traffic-accidents-in-madrid-spain-from-2019-2023*
