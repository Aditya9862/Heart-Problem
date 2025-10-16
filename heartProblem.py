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