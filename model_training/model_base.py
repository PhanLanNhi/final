import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, f1_score, precision_score, recall_score
)

# ==== Đường dẫn dữ liệu ====
file_full = "data/processed_iot_dataset_full.csv"
file_no_fft = "data/processed_iot_dataset_no_fft_anomaly.csv"
save_dir = "evaluation/model_base"
os.makedirs(save_dir, exist_ok=True)

# ==== Hàm vẽ và lưu permutation importance ====
def plot_and_save_importance(imp_df, model_type, file_tag, save_dir):
    plt.figure(figsize=(8, 6))
    sns.barplot(data=imp_df, x="Importance", y="Feature", palette="viridis")
    plt.title(f"Permutation Importance - {model_type.upper()} ({file_tag})")
    plt.xlabel("Permutation Importance (mean)")
    plt.ylabel("Feature")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{file_tag}_{model_type}_feature_importance.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance plot to {path}")

# ==== Hàm lưu các độ đo ====
def save_results(y_test, y_pred, y_proba, model_type, file_tag, save_dir):
    # Lưu báo cáo
    report_txt = os.path.join(save_dir, f"{file_tag}_{model_type}_report.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
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

    # Lưu confusion matrix plot
    cm_plot = os.path.join(save_dir, f"{file_tag}_{model_type}_confusion_matrix.png")
    plt.figure(figsize=(6,5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {model_type.upper()} - {file_tag}", fontsize=13, pad=20)
    plt.xlabel("Predicted", fontsize=11)
    plt.ylabel("True", fontsize=11)
    plt.tight_layout()
    plt.savefig(cm_plot, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix plot to {cm_plot}")

# ==== Huấn luyện, đánh giá, lưu các độ đo và vẽ permutation importance riêng ====
for file_path in [file_full, file_no_fft]:
    for model_type in ['xgb', 'lgbm']:
        df = pd.read_csv(file_path)
        file_tag = os.path.basename(file_path).replace('.csv','')
        X = df.drop(columns=['Fault_Type'])
        y = df['Fault_Type']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        if model_type == 'xgb':
            model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        elif model_type == 'lgbm':
            model = LGBMClassifier(random_state=42, verbose=-1)
        else:
            raise ValueError('Unknown model type')

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Lưu các độ đo
        save_results(y_test, y_pred, y_proba, model_type, file_tag, save_dir)

        # Vẽ permutation importance riêng
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': result.importances_mean
        }).sort_values(by='Importance', ascending=False)
        plot_and_save_importance(importance_df, model_type, file_tag, save_dir)
