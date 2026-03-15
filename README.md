# MultiClassification-App: In-Vehicle Coupon Recommendation

BITS Pilani WILP - Machine Learning (Assignment 2)

## Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models to predict whether a driver will accept a coupon delivered to their smartphone while driving. The dataset captures various contextual, demographic, and behavioral attributes of drivers and the coupons offered to them. This is a **Binary Classification** task where the target variable `Y` indicates whether the coupon was accepted (`1`) or rejected (`0`). Six different ML models — Logistic Regression, Decision Tree, k-Nearest Neighbors (kNN), Naive Bayes, Random Forest (Ensemble), and XGBoost (Ensemble) — are trained, tuned via RandomizedSearchCV, and compared across standard evaluation metrics including Accuracy, AUC, Precision, Recall, F1 Score, and MCC.

## Dataset Description

The project uses the **In-Vehicle Coupon Recommendation Dataset**, sourced from the UCI Machine Learning Repository.

- **Total Instances:** 12,684 records (after deduplication)
- **Features:** 25 attributes covering:
    - **Contextual Features:** Destination, Passenger, Weather, Temperature, Time, Expiration.
    - **Coupon Features:** Coupon type (e.g., Bar, Coffee House, Carry out & Take away).
    - **User Demographics:** Gender, Age, Marital Status, Education, Occupation, Income.
    - **User Behavior:** Frequency of visiting bars, coffee houses, and restaurants (`Bar`, `CoffeeHouse`, `CarryAway`, `RestaurantLessThan20`, `Restaurant20To50`).
    - **Distance Metrics:** Proximity to the coupon location (`toCoupon_GEQ5min`, `toCoupon_GEQ15min`, `toCoupon_GEQ25min`, `direction_same`, `direction_opp`).
- **Target Variable:** `Y` — 1 (Coupon Accepted) / 0 (Coupon Not Accepted).
- **Preprocessing Applied:**
    - Dropped `car` column (>90% null values) and duplicate rows.
    - Ordinal encoding for ordered categorical features (age, income, frequency columns, expiration).
    - Numeric mapping of `time` column.
    - Feature engineering: `total_visits` (sum of frequency columns) and `is_alone` (binary flag from passenger column).
    - StandardScaler for numerical features; OneHotEncoder for remaining categorical features.

## Model Comparison Table

Six different classification models were implemented, tuned with RandomizedSearchCV (30 iterations, 5-fold CV, `f1_weighted` scoring), and evaluated on the **training set**. The table below compares their performance:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 67.89% | 0.7383 | 68.46% | 67.89% | 68.03% | 0.3558 |
| Decision Tree | 84.72% | 0.9395 | 84.87% | 84.72% | 84.76% | 0.6912 |
| kNN | 99.83% | 1.0000 | 99.83% | 99.83% | 99.83% | 0.9966 |
| Naive Bayes | 63.60% | 0.6715 | 64.33% | 63.60% | 63.76% | 0.2716 |
| Random Forest (Ensemble) | 82.98% | 0.9224 | 83.19% | 82.98% | 83.02% | 0.6541 |
| XGBoost (Ensemble) | 97.78% | 0.9978 | 97.79% | 97.78% | 97.78% | 0.9548 |

## Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the second-lowest accuracy (67.89%) with an AUC of 0.7383 and MCC of 0.3558. As a linear model, it struggles to capture the complex, non-linear interactions among the heterogeneous features (contextual, demographic, behavioral). However, it serves as a valuable baseline for comparison due to its simplicity and interpretability. |
| **Decision Tree** | Performed reasonably well with 84.72% accuracy and an AUC of 0.9395. The use of `class_weight='balanced'` and hyperparameter tuning (max_depth, min_samples_split/leaf) helped control overfitting. The MCC of 0.6912 indicates solid discriminative ability, though it lags significantly behind ensemble methods. |
| **kNN** | Achieved near-perfect training metrics (99.83% accuracy, AUC of 1.0, MCC of 0.9966). This extremely high training performance is characteristic of kNN's instance-based learning nature — it memorizes the training data. While these numbers look impressive, kNN is highly likely to overfit and may not generalize as well to unseen data. Distance-weighted voting and hyperparameter tuning helped optimize its nearest-neighbor search. |
| **Naive Bayes** | Recorded the lowest performance across all metrics (63.60% accuracy, AUC of 0.6715, MCC of 0.2716). The strong feature-independence assumption of Gaussian Naive Bayes does not hold well for this dataset, where demographic, behavioral, and contextual features are highly correlated. This makes Naive Bayes the weakest model for this particular classification task. |
| **Random Forest (Ensemble)** | Achieved 82.98% accuracy with an AUC of 0.9224 and MCC of 0.6531. As a bagging-based ensemble, Random Forest reduces variance by averaging predictions from multiple decision trees trained on bootstrapped subsets. The `class_weight='balanced'` parameter ensures equitable treatment of both classes. Performance is comparable to Decision Tree, confirming that the ensemble benefit is primarily in variance reduction rather than bias improvement for this dataset. |
| **XGBoost (Ensemble)** | Achieved 97.79% accuracy, AUC of 0.9978, and MCC of 0.9550, making it one of the strongest models. As a boosting-based ensemble, XGBoost sequentially builds trees that correct the errors of prior trees. The extensive hyperparameter tuning (learning rate, max_depth, subsample, regularization parameters) allows it to learn complex patterns while mitigating overfitting. Its performance is very close to Random Forest on the training set. |

## Streamlit Web Application

An interactive Streamlit application is provided to test these models with new data. link - https://multiclassification-app-biswajit.streamlit.app/

- **Features:**
    - Upload a CSV file for prediction with any of the 6 models.
    - View baseline training performance metrics for the selected model.
    - New data evaluation with confusion matrix when ground-truth labels (`Y` column) are present.
    - Download prediction results as CSV.
- **Run Locally:** `streamlit run app.py`
