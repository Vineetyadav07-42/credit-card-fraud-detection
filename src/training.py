import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix
)

DATA_PATH = r"C:\Users\Vineet\OneDrive\Desktop\credit_card_resume\data\creditcard.csv"
data = pd.read_csv(DATA_PATH)

X = data.drop(columns="Class")
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

best_params = dict(
    n_estimators=200,
    min_samples_leaf=2,
    max_samples=0.5,
    max_features=0.3,
    max_depth=20,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)

# STEP 1: train on X_train only and evaluate once on the test set
eval_rf = RandomForestClassifier(**best_params)
eval_rf.fit(X_train, y_train)

probs = eval_rf.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)

print(f"PR-AUC    : {average_precision_score(y_test, probs):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, probs):.4f}")
print(f"Precision : {precision_score(y_test, preds):.4f}")
print(f"Recall    : {recall_score(y_test, preds):.4f}")
print(f"F1        : {f1_score(y_test, preds):.4f}")
print(confusion_matrix(y_test, preds))

# STEP 2: refit the same params on the full data and save this model
final_rf = RandomForestClassifier(**best_params)
final_rf.fit(X, y)

joblib.dump(final_rf, "model.joblib")

print("Saved: rf_model.joblib, feature_columns.joblib")