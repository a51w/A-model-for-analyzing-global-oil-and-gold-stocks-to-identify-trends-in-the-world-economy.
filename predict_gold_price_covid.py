import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

# ====== ตั้งค่า reproducibility ======
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# ====== โหลดข้อมูล ======
df = pd.read_csv('Log_IQR_Gold.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

# ====== สร้างฟีเจอร์เสริม ======
df['DayOfWeek'] = df.index.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
df['EMA_7'] = df['TH_PriceTHB'].ewm(span=7, adjust=False).mean()
df['MA_14'] = df['TH_PriceTHB'].rolling(window=14).mean()
df['Momentum_3'] = df['TH_PriceTHB'] - df['TH_PriceTHB'].shift(3)

# lag features 1–7 วัน
for lag in range(1, 8):
    df[f'lag_{lag}'] = df['TH_PriceTHB'].shift(lag)

df.dropna(inplace=True)

# ====== Features และ Target ======
features = ['GLD_USD/oz', 'USD/THB_Exchange_Rate', 'Federal_Funds_Effective_Rate%',
            'DayOfWeek', 'IsWeekend', 'EMA_7', 'MA_14', 'Momentum_3'] + [f'lag_{i}' for i in range(1, 8)]
target = 'TH_PriceTHB'

# ====== แบ่ง train/test ======
test_size = 150
split_index = len(df) - test_size
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

# ====== Scaling ======
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
scaled_features_train = feature_scaler.fit_transform(train_df[features])
scaled_target_train = target_scaler.fit_transform(train_df[[target]])
scaled_features_test = feature_scaler.transform(test_df[features])
scaled_target_test = target_scaler.transform(test_df[[target]])

train_data = np.hstack((scaled_features_train, scaled_target_train))
test_data = np.hstack((scaled_features_test, scaled_target_test))

# ====== สร้าง Sequence ======
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, :-1])
        y.append(data[i, -1])
    return np.array(X), np.array(y)

seq_length = 10
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

if len(X_test) == 0:
    raise ValueError(" Test set is too small. ลองลด seq_length หรือเพิ่ม test_size")

# ====== สร้างโมเดล LSTM ชั้นเดียว ======
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse')

# ====== ฝึกโมเดล ======
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
model.fit(X_train, y_train, validation_split=0.1, epochs=150, batch_size=16,
          callbacks=[early_stop], verbose=1)

# ====== ทำนาย ======
future_pred = model.predict(X_test)
predicted_prices = target_scaler.inverse_transform(future_pred)
true_prices = target_scaler.inverse_transform(y_test.reshape(-1, 1))

# ====== ทำนายย้อนหลัง (Backcast) ======
backcast_pred = model.predict(X_train)
backcast_prices = target_scaler.inverse_transform(backcast_pred)
backcast_true = target_scaler.inverse_transform(y_train.reshape(-1, 1))

# ====== Metrics ======
rmse = np.sqrt(mean_squared_error(true_prices, predicted_prices))
mae = mean_absolute_error(true_prices, predicted_prices)
r2 = r2_score(true_prices, predicted_prices)

# ====== ผลลัพธ์ ======
pd.DataFrame({
    'Actual_Price': true_prices.flatten(),
    'Predicted_Price': predicted_prices.flatten()
}).to_csv('predicted_vs_actual.csv', index=False)

# ====== แสดงกราฟ ======
plt.figure(figsize=(12, 6))
plt.plot(true_prices, label='Actual Price (Future)')
plt.plot(predicted_prices, label='Predicted Price (Future)')
plt.title('Gold Price Forecast (THB)')
plt.xlabel('Time')
plt.ylabel('Price (THB)')
plt.legend()
plt.grid(True)
plt.show()

# ====== สรุปผล ======
print("\n===== สรุปผลการทำนายราคาทองคำ =====")
print(f"1. RMSE = {rmse:.2f} บาท")
print(f"2. MAE = {mae:.2f} บาท")
print(f"3. R² Score = {r2:.4f}")
if r2 > 0.8:
    print("โมเดลแม่นยำสูงมาก พร้อมใช้งานจริง")
elif r2 > 0.5:
    print("โมเดลใช้ได้ดี มีศักยภาพสูง")
else:
    print("ยังไม่พอ อาจต้องปรับข้อมูลหรือฟีเจอร์เพิ่มเติม")

# ====== แสดงตัวอย่างทำนาย ======
print("\nตัวอย่างราคาจริงเทียบกับที่โมเดลทำนายได้:")
for i in range(min(10, len(true_prices))):
    actual = true_prices[i][0]
    predicted = predicted_prices[i][0]
    print(f"วันที่ {i+1}: จริง = {actual:.2f} บาท | ทำนาย = {predicted:.2f} บาท | คลาดเคลื่อน = {abs(actual - predicted):.2f} บาท")
