import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
)

# === Tạo thư mục lưu kết quả ===
result_dir = "evaluation/model_gan_1_tuning"
os.makedirs(result_dir, exist_ok=True)

# Đọc dữ liệu đã ghép
df = pd.read_csv('data/full_dataset_with_gan_optimized.csv')
X = df.drop(columns=['Fault_Type'])
y = df['Fault_Type']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==== Tính class_weight ====
classes, counts = np.unique(y_train, return_counts=True)
class_weights = {k: max(counts)/v for k, v in zip(classes, counts)}
print("Class weights:", class_weights)
sample_weight = np.array([class_weights[yy] for yy in y_train])

# ==== Tuning XGBoost (with sample_weight) ====
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.05],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
grid_xgb = GridSearchCV(xgb, xgb_params, cv=3, n_jobs=-1, scoring='f1_macro', verbose=1)
grid_xgb.fit(X_train, y_train, sample_weight=sample_weight)
best_xgb = grid_xgb.best_estimator_

y_pred_xgb = best_xgb.predict(X_test)
y_proba_xgb = best_xgb.predict_proba(X_test)

print("\n===== XGBoost (tuned + class_weight) =====")
print(classification_report(y_test, y_pred_xgb, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_xgb))
roc_auc_xgb = roc_auc_score(y_test, y_proba_xgb, multi_class='ovr')
print(f"ROC-AUC (OvR): {roc_auc_xgb:.4f}")

# === Lưu kết quả XGBoost ===
with open(f"{result_dir}/XGBoost_report.txt", "w", encoding="utf-8") as f:
    f.write("===== XGBoost (tuned + class_weight) =====\n")
    f.write(classification_report(y_test, y_pred_xgb, digits=4))
    f.write("\nConfusion matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred_xgb)))
    f.write(f"\nROC-AUC (OvR): {roc_auc_xgb:.4f}\n")
    acc = accuracy_score(y_test, y_pred_xgb)
    f1 = f1_score(y_test, y_pred_xgb, average='macro')
    precision = precision_score(y_test, y_pred_xgb, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred_xgb, average='macro', zero_division=0)
    f.write(f"\nAccuracy:   {acc:.4f}\nF1-macro:   {f1:.4f}\nPrecision:  {precision:.4f}\nRecall:     {recall:.4f}\n")

# ==== Tuning LightGBM (with class_weight) ====
lgbm_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7, -1],
    'learning_rate': [0.1, 0.05],
    'subsample': [0.8, 1.0],
    'class_weight': [class_weights],
    'verbose': [-1]
}
lgbm = LGBMClassifier(random_state=42, verbose=-1)
grid_lgbm = GridSearchCV(lgbm, lgbm_params, cv=3, n_jobs=-1, scoring='f1_macro', verbose=1)
grid_lgbm.fit(X_train, y_train)
best_lgbm = grid_lgbm.best_estimator_

y_pred_lgbm = best_lgbm.predict(X_test)
y_proba_lgbm = best_lgbm.predict_proba(X_test)
print("\n===== LightGBM (tuned + class_weight) =====")
print(classification_report(y_test, y_pred_lgbm, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_lgbm))
roc_auc_lgbm = roc_auc_score(y_test, y_proba_lgbm, multi_class='ovr')
print(f"ROC-AUC (OvR): {roc_auc_lgbm:.4f}")

# === Lưu kết quả LightGBM ===
with open(f"{result_dir}/LightGBM_report.txt", "w", encoding="utf-8") as f:
    f.write("===== LightGBM (tuned + class_weight) =====\n")
    f.write(classification_report(y_test, y_pred_lgbm, digits=4))
    f.write("\nConfusion matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred_lgbm)))
    f.write(f"\nROC-AUC (OvR): {roc_auc_lgbm:.4f}\n")
    acc = accuracy_score(y_test, y_pred_lgbm)
    f1 = f1_score(y_test, y_pred_lgbm, average='macro')
    precision = precision_score(y_test, y_pred_lgbm, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred_lgbm, average='macro', zero_division=0)
    f.write(f"\nAccuracy:   {acc:.4f}\nF1-macro:   {f1:.4f}\nPrecision:  {precision:.4f}\nRecall:     {recall:.4f}\n")
