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
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, roc_auc_score,
    accuracy_score, precision_score, recall_score
)

# --- Thư mục lưu kết quả ---
result_dir = "evaluation/model_gan_2"
os.makedirs(result_dir, exist_ok=True)

# --- 1. Load data ---
df_real = pd.read_csv("data/processed_iot_dataset_full.csv")
df_gan = pd.read_csv("data/ctgan_synthetic_optimized.csv")

features = [
    'Temperature', 'Vibration', 'Pressure', 'Voltage', 'Current',
    'FFT_Feature1', 'FFT_Feature2', 'Anomaly_Score'
]

# --- 2. Chuẩn bị từng class thật ---
df_real_0 = df_real[df_real['Fault_Type'] == 0].copy()
df_real_1 = df_real[df_real['Fault_Type'] == 1].copy()
df_real_2 = df_real[df_real['Fault_Type'] == 2].copy()
df_real_3 = df_real[df_real['Fault_Type'] == 3].copy()

# --- 3. GAN lỗi từng class ---
df_gan_1 = df_gan[df_gan['Fault_Type'] == 1].copy()
df_gan_2 = df_gan[df_gan['Fault_Type'] == 2].copy()
df_gan_3 = df_gan[df_gan['Fault_Type'] == 3].copy()

# --- 4. Lọc GAN lỗi chất lượng ---
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
scaler = StandardScaler()
df_real_faults = pd.concat([df_real_1, df_real_2, df_real_3], ignore_index=True)
real_faults_scaled = scaler.fit_transform(df_real_faults[features])

def filter_gan_quality(df_gan_fault):
    if len(df_gan_fault) == 0:
        return df_gan_fault
    gan_scaled = scaler.transform(df_gan_fault[features])
    dists = pairwise_distances(gan_scaled, real_faults_scaled)
    df_gan_fault = df_gan_fault.copy()
    df_gan_fault['distance'] = dists.min(axis=1)
    df_gan_fault['score'] = df_gan_fault['distance'] - df_gan_fault['Anomaly_Score']
    return df_gan_fault.sort_values('score')

df_gan_1_q = filter_gan_quality(df_gan_1)
df_gan_2_q = filter_gan_quality(df_gan_2)
df_gan_3_q = filter_gan_quality(df_gan_3)

# --- 5. Cân bằng số lượng mỗi nhãn ---
real_ratio = 0.6
n_0 = len(df_real_0)
n_1 = int(len(df_real_1) + len(df_gan_1_q))
n_2 = int(len(df_real_2) + len(df_gan_2_q))
n_3 = int(len(df_real_3) + len(df_gan_3_q))
n_sample = min(n_0, n_1, n_2, n_3)
n_real_per_fault = int(n_sample * real_ratio)
n_gan_per_fault = n_sample - n_real_per_fault

def safe_sample(df, n):
    if len(df) >= n:
        return df.sample(n=n, random_state=42)
    else:
        return df.copy()

def safe_head(df, n):
    return df.head(n) if len(df) >= n else df.copy()

datasets = {
    "Base (chỉ thật)": pd.concat([
        safe_sample(df_real_0, n_sample),
        safe_sample(df_real_1, n_sample),
        safe_sample(df_real_2, n_sample),
        safe_sample(df_real_3, n_sample),
    ], ignore_index=True),

    "Base + GAN": pd.concat([
        safe_sample(df_real_0, n_sample),
        safe_sample(df_real_1, n_real_per_fault),
        safe_head(df_gan_1, n_gan_per_fault),
        safe_sample(df_real_2, n_real_per_fault),
        safe_head(df_gan_2, n_gan_per_fault),
        safe_sample(df_real_3, n_real_per_fault),
        safe_head(df_gan_3, n_gan_per_fault),
    ], ignore_index=True),

    "Base + GAN chất lượng": pd.concat([
        safe_sample(df_real_0, n_sample),
        safe_sample(df_real_1, n_real_per_fault),
        safe_head(df_gan_1_q, n_gan_per_fault),
        safe_sample(df_real_2, n_real_per_fault),
        safe_head(df_gan_2_q, n_gan_per_fault),
        safe_sample(df_real_3, n_real_per_fault),
        safe_head(df_gan_3_q, n_gan_per_fault),
    ], ignore_index=True)
}

def save_results(y_test, y_pred, y_proba, case, model_name, save_dir):
    # Độ đo tổng hợp
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    except:
        roc_auc = None

    # Lưu txt
    report_txt = os.path.join(save_dir, f"{case.replace(' ', '_')}_{model_name}_report.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(f"{case} | {model_name}\n")
        f.write(f"Accuracy:      {acc:.4f}\n")
        f.write(f"F1-macro:      {f1:.4f}\n")
        f.write(f"Precision:     {precision:.4f}\n")
        f.write(f"Recall:        {recall:.4f}\n")
        if roc_auc is not None:
            f.write(f"ROC-AUC (OvR): {roc_auc:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(confusion_matrix(y_test, y_pred)))
        f.write("\n\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred, digits=4))
    print(f"Saved report to {report_txt}")

    # Lưu confusion matrix hình ảnh
    cm_plot = os.path.join(save_dir, f"{case.replace(' ', '_')}_{model_name}_confusion_matrix.png")
    plt.figure(figsize=(6,5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {case} - {model_name}", fontsize=13, pad=20)
    plt.xlabel("Predicted", fontsize=11)
    plt.ylabel("True", fontsize=11)
    plt.subplots_adjust(top=0.85)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(cm_plot)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_plot}")

def train_evaluate_xgb_lgbm(df, label):
    X = df[features]
    y = df['Fault_Type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    models = [
        ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
        ("LightGBM", LGBMClassifier(random_state=42))
    ]

    results = []

    for name, model in models:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='macro')
        try:
            auc = roc_auc_score(y_test, proba, multi_class='ovr', average='macro')
        except:
            auc = None
        precision = precision_score(y_test, pred, average='macro', zero_division=0)
        recall = recall_score(y_test, pred, average='macro', zero_division=0)
        print(f"\n== {label} | {name} ==")
        print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred, digits=4))
        results.append({
            "Case": label,
            "Model": name,
            "F1": f1,
            "AUC": auc,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall
        })
        save_results(y_test, pred, proba, label, name, result_dir)
    return results

# --- RUN ---
all_results = []
for name, df_case in datasets.items():
    print(f"\n--- Running: {name} ---")
    all_results.extend(train_evaluate_xgb_lgbm(df_case, name))

# --- Tổng hợp kết quả ---
df_results = pd.DataFrame(all_results)
print(df_results)

# Lưu bảng tổng hợp kết quả ra CSV
csv_path = os.path.join(result_dir, "summary_results.csv")
df_results.to_csv(csv_path, index=False)
print(f"Lưu bảng tổng hợp các chỉ số vào: {csv_path}")

# Lưu hình so sánh
plt.figure(figsize=(12, 5))
sns.barplot(data=df_results, x="Case", y="F1", hue="Model")
plt.title("So sánh F1-score giữa các mô hình và kịch bản (đa lớp)")
plt.tight_layout()
f1_fig_path = os.path.join(result_dir, "compare_f1score.png")
plt.savefig(f1_fig_path)
plt.show()
print(f"Lưu ảnh F1-score tại: {f1_fig_path}")

plt.figure(figsize=(12, 5))
sns.barplot(data=df_results, x="Case", y="AUC", hue="Model")
plt.title("So sánh AUC giữa các mô hình và kịch bản (đa lớp)")
plt.tight_layout()
auc_fig_path = os.path.join(result_dir, "compare_auc.png")
plt.savefig(auc_fig_path)
plt.show()
print(f"Lưu ảnh AUC tại: {auc_fig_path}")
