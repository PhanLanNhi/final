import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os

# Đọc dữ liệu
file_path = os.path.join('D:/dow/test1/txl/data/iot_equipment_monitoring_dataset.csv')
df = pd.read_csv(file_path)

# Hiển thị thông tin tổng quan về dữ liệu
print("Thông tin dữ liệu:")
print(df.info())
print("\nSố lượng giá trị bị thiếu trong mỗi cột:")
print(df.isnull().sum())

# Chuyển Timestamp về datetime và tách thành các cột con
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Year'] = df['Timestamp'].dt.year
df['Month'] = df['Timestamp'].dt.month
df['Day'] = df['Timestamp'].dt.day
df['Hour'] = df['Timestamp'].dt.hour
df['Minute'] = df['Timestamp'].dt.minute
df.drop(columns=['Timestamp'], inplace=True)

# Đưa cột thời gian lên đầu
time_columns = ['Day', 'Month', 'Year', 'Hour', 'Minute']
other_columns = [col for col in df.columns if col not in time_columns]
df = df[time_columns + other_columns]

# Mã hóa Sensor_ID (bỏ tiền tố 'S')
if 'Sensor_ID' in df.columns:
    df['Sensor_ID'] = df['Sensor_ID'].astype(str).str.extract('(\d+)').astype(int)

# Xoá các cột đã chuẩn hóa sẵn
columns_to_drop = ['Normalized_Temp', 'Normalized_Vibration', 'Normalized_Pressure', 'Normalized_Voltage', 'Normalized_Current']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Mã hóa Fault_Type
# Mã hóa trực tiếp cột Fault_Type
fault_type_mapping = {'Normal': 0, 'Electrical Fault': 1, 'Mechanical Failure': 2, 'Overheating': 3}
df['Fault_Type'] = df['Fault_Type'].fillna('Normal').map(fault_type_mapping)

# df['Fault_Type'] = df['Fault_Type'].fillna('Normal')
# fault_type_mapping = {'Normal': 0, 'Electrical Fault': 1, 'Mechanical Failure': 2, 'Overheating': 3}
# df['Fault_Type_Encoded'] = df['Fault_Type'].map(fault_type_mapping)

# # Kiểm tra và loại bỏ cột object không cần thiết
# for col in df.select_dtypes(include=['object']).columns:
#     print(f"Cột '{col}' có dữ liệu không phải số -> loại bỏ.")
#     df.drop(columns=[col], inplace=True)

# Chuẩn hóa dữ liệu
features = ['Temperature', 'Vibration', 'Pressure', 'Voltage', 'Current']
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Vẽ Boxplot để xem ngoại lai
sensor_columns = ['Temperature', 'Vibration', 'Pressure', 'Voltage', 'Current']

plt.figure(figsize=(12, 8))
for i, col in enumerate(sensor_columns):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(data=df, x=col)
    plt.title(f'Box Plot - {col}')
plt.tight_layout()
plt.show()


# Hàm xử lý outlier bằng IQR
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

# Áp dụng loại bỏ outlier cho các cột cảm biến gốc
sensor_columns = ['Temperature', 'Vibration', 'Pressure', 'Voltage', 'Current']
df_no_outliers = remove_outliers_iqr(df, sensor_columns)
print(f"\nKích thước dữ liệu sau khi loại bỏ outliers: {df_no_outliers.shape}")

# Trực quan phân phối Fault_Status
print("\nPhân phối dữ liệu (Fault_Status):")
print(df['Fault_Status'].value_counts())
imbalance_ratio = df['Fault_Status'].value_counts(normalize=True)
print("\nTỷ lệ mất cân bằng dữ liệu:")
print(imbalance_ratio)

plt.figure(figsize=(10, 6))
sns.countplot(x='Fault_Status', data=df)
plt.title("Sự mất cân bằng dữ liệu trong cột Fault_Status")
plt.xlabel("Fault_Status")
plt.ylabel("Số lượng")
plt.show()

# Trực quan hóa phân phối loại lỗi
plt.figure(figsize=(10, 6))
sns.countplot(x='Fault_Type', data=df)
plt.title('Phân phối các loại lỗi')
plt.xlabel('Fault Type (0: Normal, 1: Electrical, 2: Mechanical, 3: Overheating)')
plt.ylabel('Số lượng')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Ma trận tương quan
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', cbar=True)
plt.title('Ma trận tương quan')
plt.show()

# Lưu dữ liệu đã xử lý
processed_file_path = os.path.join('D:/dow/test1/txl/data/processed_iot_dataset.csv')
df.to_csv(processed_file_path, index=False)
print("Đã lưu file processed_iot_dataset.csv thành công!")
