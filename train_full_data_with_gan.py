import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load dữ liệu
df = pd.read_csv("data/full_dataset_with_gan.csv")
df = df.dropna()

# Cột đặc trưng và target
features = ['Temperature', 'Vibration', 'Pressure', 'Voltage', 'Current',
            'FFT_Feature1', 'FFT_Feature2', 'Anomaly_Score']
X = df[features]
y = df['Fault_Status']

# Chuẩn hóa
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia tập train-test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42)

# Huấn luyện Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
roc_rf = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

print("=== Random Forest ===")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("ROC AUC:", roc_rf)

# Huấn luyện XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
roc_xgb = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])

print("=== XGBoost ===")
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
print("ROC AUC:", roc_xgb)

# Vẽ confusion matrix
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues", ax=axs[0])
axs[0].set_title("Random Forest")

sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt="d", cmap="Oranges", ax=axs[1])
axs[1].set_title("XGBoost")

plt.tight_layout()
plt.show()
