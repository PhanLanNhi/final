import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

# Đọc dữ liệu đã ghép
df = pd.read_csv('data/full_dataset_with_gan_optimized.csv')
save_dir = 'evaluation/model_gan_1'
os.makedirs(save_dir, exist_ok=True)

X = df.drop(columns=['Fault_Type'])
y = df['Fault_Type']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

def save_results(y_test, y_pred, y_proba, model_name, save_dir):
    # Ghi báo cáo ra txt
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

    report_txt = os.path.join(save_dir, f"{model_name}_report.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(f"{model_name} results\n")
        f.write(f"Accuracy:      {acc:.4f}\n")
        f.write(f"F1-macro:      {f1:.4f}\n")
        f.write(f"Precision:     {precision:.4f}\n")
        f.write(f"Recall:        {recall:.4f}\n")
        f.write(f"ROC-AUC (OvR): {roc_auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(confusion_matrix(y_test, y_pred)))
        f.write("\n\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred, digits=4))
    print(f"Saved report to {report_txt}")

    # Lưu confusion matrix hình ảnh
    cm_plot = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.figure(figsize=(6,5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {model_name}", fontsize=13, pad=20)
    plt.xlabel("Predicted", fontsize=11)
    plt.ylabel("True", fontsize=11)
    plt.subplots_adjust(top=0.85)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(cm_plot)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_plot}")

# --- XGBoost ---
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_proba_xgb = xgb.predict_proba(X_test)

print("===== XGBoost =====")
print(classification_report(y_test, y_pred_xgb))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_xgb))
roc_auc_xgb = roc_auc_score(y_test, y_proba_xgb, multi_class='ovr')
print(f"ROC-AUC (OvR): {roc_auc_xgb:.4f}")

save_results(y_test, y_pred_xgb, y_proba_xgb, "XGBoost", save_dir)

# --- LightGBM ---
lgbm = LGBMClassifier(random_state=42,  verbose=-1)
lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_test)
y_proba_lgbm = lgbm.predict_proba(X_test)

print("\n===== LightGBM =====")
print(classification_report(y_test, y_pred_lgbm))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_lgbm))
roc_auc_lgbm = roc_auc_score(y_test, y_proba_lgbm, multi_class='ovr')
print(f"ROC-AUC (OvR): {roc_auc_lgbm:.4f}")

save_results(y_test, y_pred_lgbm, y_proba_lgbm, "LightGBM", save_dir)
