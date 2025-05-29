
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ==== 1. Load & chuẩn hóa dữ liệu ====
df = pd.read_csv("data/full_dataset_with_gan.csv").dropna()

features = ['Temperature', 'Vibration', 'Pressure', 'Voltage', 'Current',
            'FFT_Feature1', 'FFT_Feature2', 'Anomaly_Score']
X = df[features]
y = df['Fault_Status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

# ==== 2. Hàm tuning với GridSearchCV ====
def tune_model(model_type):
    if model_type == "Random Forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        }
    else:  # XGBoost
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1]
        }

    grid = GridSearchCV(model, param_grid, scoring='f1', cv=2, n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f" Best params for {model_type}: {grid.best_params_}")
    return grid.best_estimator_

# ==== 3. Huấn luyện & đánh giá ====
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"\n {name}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {auc:.4f}")
    return y_pred

# Random Forest (Tuned)
best_rf = tune_model("Random Forest")
y_pred_rf = evaluate_model(best_rf, X_test, y_test, "Random Forest (Tuned)")

# XGBoost (Tuned)
best_xgb = tune_model("XGBoost")
y_pred_xgb = evaluate_model(best_xgb, X_test, y_test, "XGBoost (Tuned)")

# ==== 4. Vẽ confusion matrix ====
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues", ax=axs[0])
axs[0].set_title("Random Forest (Tuned)")

sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt="d", cmap="Oranges", ax=axs[1])
axs[1].set_title("XGBoost (Tuned)")

plt.tight_layout()
plt.show()
