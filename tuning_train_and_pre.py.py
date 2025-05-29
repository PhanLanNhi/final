
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# ==== Load & Preprocess ====
df_real = pd.read_csv("data/processed_iot_dataset.csv").dropna()
df_gan = pd.read_csv("data/processed_ctgan_generated_faults.csv").dropna()

features = ['Temperature', 'Vibration', 'Pressure', 'Voltage', 'Current',
            'FFT_Feature1', 'FFT_Feature2', 'Anomaly_Score']

df_real_faults = df_real[df_real['Fault_Status'] == 1].copy()
df_real_nonfaults = df_real[df_real['Fault_Status'] == 0].copy()

# ==== Filtering GAN Samples ====
scaler = StandardScaler()
real_scaled = scaler.fit_transform(df_real_faults[features])
gan_scaled = scaler.transform(df_gan[features])
df_gan['distance'] = pairwise_distances(gan_scaled, real_scaled).min(axis=1)
df_gan['score'] = df_gan['distance'] - df_gan['Anomaly_Score']
df_gan_filtered = df_gan.sort_values("score")

# ==== Define Training Strategies ====
np.random.seed(42)
strategies = {
    "Chỉ lỗi GAN (5K+5K)": pd.concat([
        df_gan_filtered.head(5000), 
        df_real_nonfaults.sample(5000)
        ], ignore_index=True),
    "Lỗi GAN + Lỗi thật (2.5K+2.5K+5K)": pd.concat([
        df_gan_filtered.head(2500), 
        df_real_faults.sample(2500), df_real_nonfaults.sample(5000)
        ], ignore_index=True),
    "Chỉ lỗi thật (5K+5K)": pd.concat([
        df_real_faults.sample(5000), 
        df_real_nonfaults.sample(5000)
        ], ignore_index=True),
    "Adaptive (3K+3K+4K)": pd.concat([
        df_gan_filtered.head(3000), 
        df_real_faults.sample(3000), 
        df_real_nonfaults.sample(4000)
        ], ignore_index=True),
    "GAN Confidence (Top 3K) + 2K + 5K": pd.concat([
        df_gan_filtered.head(3000), 
        df_real_faults.sample(2000), 
        df_real_nonfaults.sample(5000)
        ], ignore_index=True)
}

results = []

# ==== Tuning Function ====
def tune_model(X, y, model_type):
    if model_type == "Random Forest":
        model = RandomForestClassifier(random_state=42)
        params = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        }
    else:
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        params = {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.1, 0.05]
        }

    grid = GridSearchCV(model, params, scoring='f1', cv=3, n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_

# ==== Training & Evaluation ====
def train_and_evaluate(df, label):
    X = df[features]
    y = df['Fault_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    for model_type in ["Random Forest", "XGBoost"]:
        best_model = tune_model(X_train, y_train, model_type)
        preds = best_model.predict(X_test)
        proba = best_model.predict_proba(X_test)[:, 1]
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)
        print(f"\n=== Case: {label} - {model_type} ===")
        print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
        print(classification_report(y_test, preds))
        print("ROC AUC:", auc)
        results.append({"Case": label, "Model": model_type, "F1": f1, "AUC": auc})

# ==== Run All Strategies ====
for label, df_case in strategies.items():
    train_and_evaluate(df_case, label)

# ==== Fine-tuning (XGBoost + Random Forest) ====
def run_fine_tuning():
    print("\nFine-tuned Models")
    gan_train = pd.concat([df_gan_filtered.head(5000), df_real_nonfaults.sample(5000)], ignore_index=True)
    real_finetune = pd.concat([df_real_faults.sample(2500), df_real_nonfaults.sample(2500)], ignore_index=True)

    X_gan, y_gan = gan_train[features], gan_train['Fault_Status']
    X_real, y_real = real_finetune[features], real_finetune['Fault_Status']
    X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size=0.3, random_state=42)

    # XGBoost Fine-tuned
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_gan, y_gan)
    xgb.fit(X_real, y_real, xgb_model=xgb)
    pred_xgb = xgb.predict(X_test)
    auc_xgb = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
    f1_xgb = f1_score(y_test, pred_xgb)
    print("XGBoost Fine-tuned:")
    print(confusion_matrix(y_test, pred_xgb))
    print(classification_report(y_test, pred_xgb))
    print("ROC AUC:", auc_xgb)
    results.append({"Case": "Fine-tuned", "Model": "XGBoost", "F1": f1_xgb, "AUC": auc_xgb})

run_fine_tuning()

# ==== Visualization ====
df_results = pd.DataFrame(results)
print("\nKết quả tổng hợp:")
print(df_results)

order = ["Chỉ lỗi GAN (5K+5K)", "Lỗi GAN + Lỗi thật (2.5K+2.5K+5K)", "Chỉ lỗi thật (5K+5K)",
         "Fine-tuned", "Adaptive (3K+3K+4K)", "GAN Confidence (Top 3K) + 2K + 5K"]

plt.figure(figsize=(10, 5))
sns.barplot(data=df_results, x="Case", y="F1", hue="Model", order=order)
plt.title("So sánh F1-score giữa các chiến lược huấn luyện (Tuned)")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(data=df_results, x="Case", y="AUC", hue="Model", order=order)
plt.title("So sánh AUC giữa các chiến lược huấn luyện (Tuned)")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()
