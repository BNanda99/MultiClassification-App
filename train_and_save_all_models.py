"""
Train and save all ML classification model pipelines as .pkl files.
Each pipeline includes the same preprocessing (imputation + scaling/encoding)
followed by the specific classifier.

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
import pickle
import json
import os


# --- Ordinal mappings for ordered categorical columns ---
FREQUENCY_ORDER = ['never', 'less1', '1~3', '4~8', 'gt8']
AGE_ORDER = ['below21', '21', '26', '31', '36', '41', '46', '50plus']
INCOME_ORDER = [
    'Less than $12500', '$12500 - $24999', '$25000 - $37499',
    '$37500 - $49999', '$50000 - $62499', '$62500 - $74999',
    '$75000 - $87499', '$87500 - $99999', '$100000 or More'
]
EXPIRATION_ORDER = ['2h', '1d']
TIME_MAP = {'7AM': 7, '10AM': 10, '2PM': 14, '6PM': 18, '10PM': 22}

# Columns that have ordinal (ordered) values
FREQUENCY_COLS = ['Bar', 'CoffeeHouse', 'CarryAway', 'RestaurantLessThan20', 'Restaurant20To50']


def load_and_preprocess_data():
    """Load dataset and perform shared preprocessing with feature engineering."""
    file_path = os.path.join(os.path.dirname(__file__), 'in-vehicle-coupon-recommendation.csv')
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)

    # Drop 'car' column if >90% null
    if 'car' in df.columns and df['car'].isnull().mean() > 0.9:
        print("Dropping 'car' column due to excessive missing values.")
        df = df.drop(columns=['car'])

    # Drop duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Dropped {duplicates} duplicates. New shape: {df.shape}")

    # --- Ordinal encoding (before splitting into X/y) ---
    # Map frequency columns to numeric
    freq_map = {v: i for i, v in enumerate(FREQUENCY_ORDER)}
    for col in FREQUENCY_COLS:
        if col in df.columns:
            df[col] = df[col].map(freq_map)  # NaN stays NaN

    # Map age to numeric
    age_map = {v: i for i, v in enumerate(AGE_ORDER)}
    if 'age' in df.columns:
        df['age'] = df['age'].map(age_map)

    # Map income to numeric
    income_map = {v: i for i, v in enumerate(INCOME_ORDER)}
    if 'income' in df.columns:
        df['income'] = df['income'].map(income_map)

    # Map expiration to numeric
    exp_map = {v: i for i, v in enumerate(EXPIRATION_ORDER)}
    if 'expiration' in df.columns:
        df['expiration'] = df['expiration'].map(exp_map)

    # Map time to numeric hour
    if 'time' in df.columns:
        df['time'] = df['time'].map(TIME_MAP)

    # --- Feature Engineering ---
    # Total visits: sum of all frequency columns (already numeric now)
    freq_available = [c for c in FREQUENCY_COLS if c in df.columns]
    if freq_available:
        df['total_visits'] = df[freq_available].sum(axis=1)
        print(f"Added 'total_visits' feature (sum of {freq_available})")

    # Is alone flag
    if 'passanger' in df.columns:
        df['is_alone'] = (df['passanger'] == 'Alone').astype(int)
        print("Added 'is_alone' feature")

    # Separate target and features
    X = df.drop(columns=['Y'])
    y = df['Y']

    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

    return X, y, numerical_cols, categorical_cols


def build_preprocessor(numerical_cols, categorical_cols):
    """Build the shared ColumnTransformer preprocessor."""
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    return preprocessor


def get_models():
    """Return a dictionary of model name -> classifier instance."""
    return {
        'logistic_regression_pipeline': LogisticRegression(
            max_iter=5000,
            random_state=42,
            class_weight='balanced'
        ),
        'knn_pipeline': KNeighborsClassifier(),
        'decision_tree_pipeline': DecisionTreeClassifier(
            random_state=42,
            class_weight='balanced'
        ),
        'naive_bayes_pipeline': GaussianNB(),
        'random_forest_pipeline': RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        'xgboost_pipeline': XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        ),
    }


def get_param_grids():
    """Return a dictionary of model name -> hyperparameter grid."""
    return {
        'logistic_regression_pipeline': {
            'classifier__C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
            'classifier__solver': ['lbfgs', 'liblinear', 'saga'],
        },
        'knn_pipeline': {
            'classifier__n_neighbors': [3, 5, 7, 9, 11, 15, 20, 25],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__p': [1, 2],
            'classifier__leaf_size': [20, 30, 50],
        },
        'decision_tree_pipeline': {
            'classifier__max_depth': [5, 10, 15, 20, 25, 30, 40, None],
            'classifier__min_samples_split': [2, 3, 5, 10, 15],
            'classifier__min_samples_leaf': [1, 2, 3, 4, 5],
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_features': ['sqrt', 'log2', None],
        },
        'random_forest_pipeline': {
            'classifier__n_estimators': [50, 75, 100],
            'classifier__max_depth': [8, 10, 12, 15],
            'classifier__min_samples_split': [5, 10, 15],
            'classifier__min_samples_leaf': [4, 6, 8],
            'classifier__max_features': ['sqrt', 'log2'],
        },
        'xgboost_pipeline': {
            'classifier__n_estimators': [100, 200, 300, 500, 700],
            'classifier__learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7, 9, 11],
            'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'classifier__min_child_weight': [1, 3, 5, 7],
            'classifier__gamma': [0, 0.1, 0.3, 0.5],
            'classifier__reg_alpha': [0, 0.01, 0.1, 1],
            'classifier__reg_lambda': [0.5, 1, 1.5, 2],
        },
        'naive_bayes_pipeline': {
            'classifier__var_smoothing': np.logspace(-12, -6, 30),
        }
    }


def main():
    X, y, numerical_cols, categorical_cols = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}\n")

    output_dir = os.path.join(os.path.dirname(__file__), 'model')
    os.makedirs(output_dir, exist_ok=True)

    models = get_models()
    param_grids = get_param_grids()

    for name, classifier in models.items():
        print(f"{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")

        preprocessor = build_preprocessor(numerical_cols, categorical_cols)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])

        param_grid = param_grids.get(name, {})
        
        if param_grid:
            print(f"Running randomized search for {name} (30 iter, 5-fold CV)...")
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=30,
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            print(f"Best params: {search.best_params_}")
            print(f"Best CV score: {search.best_score_:.4f}")
        else:
            print(f"No params to tune for {name}, training default model...")
            pipeline.fit(X_train, y_train)
            best_model = pipeline

        # --- Helper for metrics calculation ---
        def calculate_metrics(model, X, y_true):
            y_pred = model.predict(X)
            if hasattr(model, "predict_proba"):
                 y_proba = model.predict_proba(X)[:, 1]
            else:
                 y_proba = np.zeros(len(y_true))
            
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='weighted')
            rec = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            try:
                auc = roc_auc_score(y_true, y_proba)
            except ValueError:
                auc = 0.0
            mcc = matthews_corrcoef(y_true, y_pred)
            
            return {
                'accuracy': round(acc, 4),
                'precision': round(prec, 4),
                'recall': round(rec, 4),
                'f1_score': round(f1, 4),
                'auc_score': round(auc, 4),
                'mcc_score': round(mcc, 4)
            }

        # Calculate metrics for Train set only
        train_metrics = calculate_metrics(best_model, X_train, y_train)
        
        # Merge metrics with prefixes
        combined_metrics = {}
        for k, v in train_metrics.items():
            combined_metrics[f'train_{k}'] = v

        print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Train AUC: {train_metrics['auc_score']:.4f}")

        output_path = os.path.join(output_dir, f'{name}.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(best_model, f)

        metrics_path = os.path.join(output_dir, f'{name}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(combined_metrics, f, indent=2)
        print(f"Saved to {output_path}")
        print(f"Metrics saved to {metrics_path}\n")

    print("All models trained and saved successfully!")


if __name__ == "__main__":
    main()
