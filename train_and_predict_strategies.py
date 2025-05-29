import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances

# ===== LOAD DATA =====
df_real = pd.read_csv("data/processed_iot_dataset.csv").dropna()
df_gan = pd.read_csv("data/processed_ctgan_generated_faults.csv").dropna()

features = ['Temperature', 'Vibration', 'Pressure', 'Voltage', 'Current',
            'FFT_Feature1', 'FFT_Feature2', 'Anomaly_Score']

df_real_faults = df_real[df_real['Fault_Status'] == 1].copy()
df_real_nonfaults = df_real[df_real['Fault_Status'] == 0].copy()

# ===== LỌC GAN THEO DISTANCE + ANOMALY =====
scaler = StandardScaler()
real_scaled = scaler.fit_transform(df_real_faults[features])
gan_scaled = scaler.transform(df_gan[features])
df_gan['distance'] = pairwise_distances(gan_scaled, real_scaled).min(axis=1)
df_gan_filtered = df_gan.copy()
df_gan_filtered['score'] = df_gan_filtered['distance'] - df_gan_filtered['Anomaly_Score']
df_gan_filtered = df_gan_filtered.sort_values('score')  # score thấp hơn → tốt hơn

# ===== DEFINE STRATEGIES =====
np.random.seed(42)
strategies = {
    # (1) Chỉ lỗi GAN
    "Chỉ lỗi GAN (5K+5K)": pd.concat([
        df_gan_filtered.head(5000),
        df_real_nonfaults.sample(5000)
    ], ignore_index=True),

    # (2) Lỗi GAN + lỗi thật
    "Lỗi GAN + Lỗi thật (2.5K+2.5K+5K)": pd.concat([
        df_gan_filtered.head(2500),
        df_real_faults.sample(2500),
        df_real_nonfaults.sample(5000)
    ], ignore_index=True),

    # (3) Chỉ lỗi thật (5K+5K)
    "Chỉ lỗi thật (5K+5K)": pd.concat([
        df_real_faults.sample(5000),
        df_real_nonfaults.sample(5000)
    ], ignore_index=True),

    # (4) Fine-tuning
    "Fine-tuned": None,  # xử lý riêng bên dưới
    # (5) Adaptive Ratio (3K+3K+4K)
    "Adaptive (3K+3K+4K)": pd.concat([
        df_gan_filtered.head(2500),
        df_real_faults.sample(2500),
        df_real_nonfaults.sample(5000)
    ], ignore_index=True),

    # (6) GAN Confidence (low distance + high anomaly)
    "GAN Confidence (Top 3K) + 2K + 5K": pd.concat([
        df_gan_filtered.head(3000),
        df_real_faults.sample(2000),
        df_real_nonfaults.sample(5000)
    ], ignore_index=True)
}

results = []

def train_and_evaluate(df, label):
    X = df[features]
    y = df['Fault_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    f1_rf = f1_score(y_test, pred_rf)
    auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f"\n=== Case: {label} - Random Forest ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred_rf))
    print(classification_report(y_test, pred_rf))
    print("ROC AUC:", auc_rf)
    results.append({"Case": label, "Model": "Random Forest", "F1": f1_rf, "AUC": auc_rf})

    # XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)
    f1_xgb = f1_score(y_test, pred_xgb)
    auc_xgb = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
    print(f"\n=== Case: {label} - XGBoost ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred_xgb))
    print(classification_report(y_test, pred_xgb))
    print("ROC AUC:", auc_xgb)
    results.append({"Case": label, "Model": "XGBoost", "F1": f1_xgb, "AUC": auc_xgb})

# ===== RUN STRATEGIES =====
for label, df_case in strategies.items():
    if label == "Fine-tuned":
        # Step 1: train on GAN
        df_gan_train = pd.concat([
            df_gan_filtered.head(5000),
            df_real_nonfaults.sample(5000)
        ], ignore_index=True)

        X1 = df_gan_train[features]
        y1 = df_gan_train['Fault_Status']
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X1, y1)

        # Step 2: fine-tune on real faults
        df_real_ft = pd.concat([
            df_real_faults.sample(2500),
            df_real_nonfaults.sample(2500)
        ], ignore_index=True)
        X2 = df_real_ft[features]
        y2 = df_real_ft['Fault_Status']
        model.fit(X2, y2, xgb_model=model)  # fine-tuning

        # Step 3: test
        X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.3, random_state=42)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        f1 = f1_score(y_test, pred)
        auc = roc_auc_score(y_test, proba)

        print(f"\n=== Case: {label} - Fine-tuned XGBoost ===")
        print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred))
        print(f"ROC AUC: {auc:.4f}")

        results.append({"Case": label, "Model": "XGBoost", "F1": f1, "AUC": auc})


    else:
        train_and_evaluate(df_case, label)

# ===== PLOT RESULTS =====
df_results = pd.DataFrame(results)
print(df_results)

plt.figure(figsize=(10, 5))
sns.barplot(data=df_results, x="Case", y="F1", hue="Model")
plt.title("So sánh F1-score giữa các chiến lược huấn luyện")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(data=df_results, x="Case", y="AUC", hue="Model")
plt.title("So sánh AUC giữa các chiến lược huấn luyện")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()
