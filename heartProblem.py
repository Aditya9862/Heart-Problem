import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Load dataset
df = pd.read_csv("heart.csv")

# 2. Identify the target column
if "target" not in df.columns and "Disease" in df.columns:
    df = df.rename(columns={"Disease": "target"})

if "target" not in df.columns:
    raise ValueError(
        "Target column not found. Expected 'target' or 'Disease'."
    )

# 3. Select features
selected_features = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "thalach",
    "exang",
    "oldpeak"
]

missing_features = [
    col for col in selected_features if col not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing features in heart.csv: {missing_features}"
    )

X = df[selected_features].copy()
y = df["target"].copy()

# Remove rows with missing target labels
valid_rows = y.notna()
X = X.loc[valid_rows].reset_index(drop=True)
y = y.loc[valid_rows].reset_index(drop=True)

if y.nunique() != 2:
    raise ValueError(
        "This script expects a binary target with exactly two classes."
    )

print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("\nDataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nTarget distribution:")
print(y.value_counts())

# 4. Separate numerical and categorical features
categorical_features = ["sex", "cp", "exang"]

numeric_features = [
    col for col in selected_features
    if col not in categorical_features
]

print("\nNumerical features:", numeric_features)
print("Categorical features:", categorical_features)

# 5. Build preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)

# 6. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 7. Define model pipelines
pipelines = {
    "logreg": Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=2000, random_state=42))
    ]),

    "rf": Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(random_state=42))
    ])
}

# 8. Define hyperparameter grids
param_grids = {
    "logreg": {
        "clf__C": [0.01, 0.1, 1, 10]
    },

    "rf": {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 6, 10],
        "clf__min_samples_split": [2, 5]
    }
}

# 9. Cross-validation configuration
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

best_models = {}

# 10. Train, tune, and evaluate each model
for name, pipeline in pipelines.items():
    print(f"\n{'=' * 50}")
    print(f"Training model: {name}")
    print("=" * 50)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[name],
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise"
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # Predictions on the held-out test set
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)

    # Locate the probability column for the positive class (label 1)
    classes = list(best_model.named_steps["clf"].classes_)

    if 1 not in classes:
        raise ValueError(
            "Expected positive target class 1 for ROC-AUC evaluation."
        )

    positive_index = classes.index(1)
    positive_probabilities = probabilities[:, positive_index]

    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, positive_probabilities)

    print("\nBest parameters:", grid.best_params_)
    print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    best_models[name] = {
        "model": best_model,
        "cv_roc_auc": grid.best_score_,
        "test_accuracy": accuracy,
        "test_roc_auc": roc_auc
    }

# 11. Select the best model using cross-validation ROC-AUC
best_name = max(
    best_models,
    key=lambda name: best_models[name]["cv_roc_auc"]
)

best_model = best_models[best_name]["model"]

print(f"\n{'=' * 50}")
print("BEST MODEL:", best_name)
print(
    "Cross-validation ROC-AUC:",
    f"{best_models[best_name]['cv_roc_auc']:.4f}"
)
print(
    "Test accuracy:",
    f"{best_models[best_name]['test_accuracy']:.4f}"
)
print(
    "Test ROC-AUC:",
    f"{best_models[best_name]['test_roc_auc']:.4f}"
)

# 12. Save the complete pipeline
joblib.dump(best_model, "heart_model1.joblib")

print("\nModel saved successfully as heart_model1.joblib")

    accuracy_score, roc_auc_score, confusion_matrix, classification_report
)
import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 1) Load data
df = pd.read_csv("heart.csv")  # place your dataset here

# Quick target check - adapt if your target column has another name
if 'target' not in df.columns and 'Disease' in df.columns:
    df = df.rename(columns={'Disease': 'target'})

# 2) Basic EDA (optional prints)
print("Shape:", df.shape)
print(df.head())

# 3) Prepare features and target
X = df.drop(columns=['target'])
y = df['target']

# 4) Column types
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
# If some numeric columns are actually categorical (like 'cp', 'thal', 'slope', 'ca', 'sex', 'fbs', 'restecg', 'exang'),
# list them manually and treat as categorical:
cat_cols = []
possible_cat = ['cp','thal','slope','ca','sex','fbs','restecg','exang']
for c in possible_cat:
    if c in X.columns:
        cat_cols.append(c)
        if c in num_cols:
            num_cols.remove(c)

print("Numerical cols:", num_cols)
print("Categorical cols:", cat_cols)

# 5) Preprocessing
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    # We'll let Random Forest handle label-encoded ints; if your categories are non-numeric,
    # consider OneHotEncoder(drop='first') instead.
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
], remainder='passthrough')  # passthrough for any columns not listed

# 6) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 7) Define pipelines for models
pipelines = {
    'logreg': Pipeline([('pre', preprocessor), ('clf', LogisticRegression(max_iter=1000))]),
    'rf': Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(random_state=42))])
}

# Optional: XGBoost (uncomment if xgboost installed)
# try:
#     from xgboost import XGBClassifier
#     pipelines['xgb'] = Pipeline([('pre', preprocessor), ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])
# except Exception:
#     print("XGBoost not available — skipping.")

# 8) Simple hyperparameter grid (small)
param_grids = {
    'logreg': {
        'clf__C': [0.01, 0.1, 1, 10]
    },
    'rf': {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 6, 10]
    },
    'xgb': {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [3,6]
    }
}

# 9) Grid search and evaluation
best_models = {}
for name, pipe in pipelines.items():
    print(f"\nTraining {name}...")
    grid = GridSearchCV(pipe, param_grid=param_grids.get(name, {}), cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best params for {name}:", grid.best_params_)
    preds = grid.predict(X_test)
    probs = grid.predict_proba(X_test)[:, 1] if hasattr(grid.best_estimator_, "predict_proba") else None
    acc = accuracy_score(y_test, preds)
    roc = roc_auc_score(y_test, probs) if probs is not None else None
    print(f"{name} Accuracy: {acc:.4f}")
    print(f"{name} ROC AUC: {roc:.4f}" if roc is not None else f"{name} ROC AUC: N/A")
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    print("Classification report:\n", classification_report(y_test, preds))
    best_models[name] = {'model': grid.best_estimator_, 'acc': acc, 'roc': roc}

# 10) Choose best model (by roc auc if available, otherwise accuracy)
best_name = None
best_score = -np.inf
for name, info in best_models.items():
    score = info['roc'] if info['roc'] is not None else info['acc']
    if score > best_score:
        best_score = score
        best_name = name

print(f"\nBest model: {best_name} with score {best_score:.4f}")

# 11) Save the best model
best_model = best_models[best_name]['model']
joblib.dump(best_model, 'heart_model.joblib')
print("Saved best model to heart_model.joblib")
