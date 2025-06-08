import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os

# Đọc dữ liệu
file_path = os.path.join('data/iot_equipment_monitoring_dataset.csv')
df = pd.read_csv(file_path)

# ====================== 1. TIỀN XỬ LÝ CHUNG ======================
print("Thông tin dữ liệu ban đầu:")
print(df.info())
print(df.head())

# Xử lý thời gian
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Year'] = df['Timestamp'].dt.year
df['Month'] = df['Timestamp'].dt.month
df['Day'] = df['Timestamp'].dt.day
df['Hour'] = df['Timestamp'].dt.hour
df['Minute'] = df['Timestamp'].dt.minute
df.drop(columns=['Timestamp'], inplace=True)

# Đưa cột thời gian lên đầu
time_columns = ['Day', 'Month', 'Year', 'Hour', 'Minute']
df = df[time_columns + [col for col in df.columns if col not in time_columns]]

# Làm sạch Sensor_ID
if 'Sensor_ID' in df.columns:
    df['Sensor_ID'] = df['Sensor_ID'].astype(str).str.extract(r'(\d+)').astype(int)

# Xóa các cột đã chuẩn hóa nếu có
columns_to_drop = [
    'Normalized_Temp', 'Normalized_Vibration',
    'Normalized_Pressure', 'Normalized_Voltage', 'Normalized_Current'
]
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Làm sạch và mã hóa Fault_Type
df['Fault_Type'] = df['Fault_Type'].fillna('normal') \
                                   .astype(str).str.strip().str.lower()

fault_type_mapping = {
    'normal': 0,
    'electrical fault': 1,
    'mechanical failure': 2,
    'overheating': 3
}
df['Fault_Type'] = df['Fault_Type'].map(fault_type_mapping)
df['Fault_Type'] = df['Fault_Type'].fillna(0).astype(int)

# Xoá cột Fault_Status nếu tồn tại
if 'Fault_Status' in df.columns:
    df.drop(columns=['Fault_Status'], inplace=True)

# ====================== 2. CHUẨN HÓA DỮ LIỆU ======================
features = ['Temperature', 'Vibration', 'Pressure', 'Voltage', 'Current']
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# ====================== 3. LOẠI OUTLIERS (IQR) ======================
def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for column in columns:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
    return df_clean

df_cleaned = remove_outliers_iqr(df, features)
print(f"Kích thước sau khi loại outliers: {df_cleaned.shape}")
print(df.head())

# ====================== 4. LƯU FILE ======================

# Bản giữ nguyên FFT & Anomaly
out_full = os.path.join('data/processed_iot_dataset_full.csv')
df_cleaned.to_csv(out_full, index=False)
print(f"Đã lưu: {out_full}")

# Bản đã xoá FFT & Anomaly
fft_cols = ['FFT_Feature1', 'FFT_Feature2', 'Anomaly_Score']
df_no_fft = df_cleaned.drop(columns=[col for col in fft_cols if col in df_cleaned.columns])
out_no_fft = os.path.join('data/processed_iot_dataset_no_fft_anomaly.csv')
df_no_fft.to_csv(out_no_fft, index=False)
print(f"Đã lưu: {out_no_fft}")

# ====================== 5. TRỰC QUAN ======================

# Boxplot xem outliers
plt.figure(figsize=(12, 8))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(data=df, x=col)
    plt.title(f'Box Plot - {col}')
plt.tight_layout()
plt.show()

# Biểu đồ mất cân bằng lớp lỗi
plt.figure(figsize=(10, 6))
sns.countplot(x='Fault_Type', data=df_cleaned)
plt.title('Phân phối các loại lỗi sau xử lý')
plt.xlabel('Fault Type (0: Normal, 1: Electrical, 2: Mechanical, 3: Overheating)')
plt.ylabel('Số lượng')
plt.tight_layout()
plt.show()

# Ma trận tương quan
plt.figure(figsize=(10, 8))
sns.heatmap(df_cleaned.corr(), annot=True, cmap='coolwarm')
plt.title('Ma trận tương quan')
plt.tight_layout()
plt.show()
