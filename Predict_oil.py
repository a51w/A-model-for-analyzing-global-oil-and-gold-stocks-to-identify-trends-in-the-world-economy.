import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# โหลดข้อมูล
df = pd.read_csv("C:\\Users\\May\\Downloads\\Log_IQR_Crude_Oil.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df = df.dropna()

# สร้างฟีเจอร์จาก Date และฟีเจอร์เพิ่มเติม
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Is_Covid'] = df['Date'].between('2020-03-01', '2021-12-31').astype(int)

# เพิ่มฟีเจอร์ lag เพื่อดูความสัมพันธ์ในอดีต
df['PTTEP_Lag1'] = df['PTTEP_Close_Price_THB'].shift(1)
df['PTTEP_Lag5'] = df['PTTEP_Close_Price_THB'].shift(5)
df['Oil_Lag1'] = df['Avg_Crude_Oil_Price_USD/barrel'].shift(1)
df['Oil_Lag5'] = df['Avg_Crude_Oil_Price_USD/barrel'].shift(5)

# ลบแถวที่มีค่า NaN (เกิดจาก shift)
df = df.dropna()

# เพิ่มฟีเจอร์ technical indicators
df['PTTEP_5d_MA'] = df['PTTEP_Close_Price_THB'].rolling(window=5).mean()
df['PTTEP_10d_MA'] = df['PTTEP_Close_Price_THB'].rolling(window=10).mean()
df['Oil_5d_MA'] = df['Avg_Crude_Oil_Price_USD/barrel'].rolling(window=5).mean()
df['Oil_10d_MA'] = df['Avg_Crude_Oil_Price_USD/barrel'].rolling(window=10).mean()

# ลบแถวที่มีค่า NaN (เกิดจาก rolling)
df = df.dropna()

# ตรวจสอบความสัมพันธ์ระหว่างราคาน้ำมันและราคาหุ้น PTTEP
plt.figure(figsize=(10, 6))
plt.scatter(df['Avg_Crude_Oil_Price_USD/barrel'], df['PTTEP_Close_Price_THB'], alpha=0.5)
plt.title('ความสัมพันธ์ระหว่างราคาน้ำมันดิบและราคาหุ้น PTTEP')
plt.xlabel('ราคาน้ำมันดิบ (USD/บาร์เรล)')
plt.ylabel('ราคาหุ้น PTTEP (บาท)')
plt.grid(True)
plt.savefig('oil_pttep_correlation.png')
plt.close()

# คำนวณค่าสหสัมพันธ์
correlation = df['Avg_Crude_Oil_Price_USD/barrel'].corr(df['PTTEP_Close_Price_THB'])
print(f'ค่าสหสัมพันธ์ระหว่างราคาน้ำมันและราคาหุ้น PTTEP: {correlation:.4f}')

# เลือก features + target
features = [
    'Avg_Crude_Oil_Price_USD/barrel',
    'USD/THB_Exchange_Rate',
    'PTT_Close_Price_THB',
    'EPS_THB',
    'ROE_pct',
    'Year', 'Month', 'DayOfWeek', 'Is_Covid',
    'PTTEP_Lag1', 'PTTEP_Lag5',
    'Oil_Lag1', 'Oil_Lag5',
    'PTTEP_5d_MA', 'PTTEP_10d_MA',
    'Oil_5d_MA', 'Oil_10d_MA'
]
target = 'PTTEP_Close_Price_THB'

# Backtesting Function
def backtest(model, X, y, scaler_y):
    predictions = model.predict(X)
    y_pred = scaler_y.inverse_transform(predictions)
    y_actual = scaler_y.inverse_transform(y.reshape(-1, 1))
    mse = mean_squared_error(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_pred)
    print(f'Backtest MSE: {mse:.4f}')
    print(f'Backtest RMSE: {rmse:.4f}')
    print(f'Backtest MAE: {mae:.4f}')
    print(f'Backtest R²: {r2:.4f}')
    return y_pred, mse, mae, rmse, r2


# ตรวจสอบการกระจายตัวของข้อมูลก่อน scaling
print("ข้อมูลสถิติก่อน scaling:")
print(df[features + [target]].describe())

# Scale ข้อมูลทั้งหมด
data = df[features + [target]]
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# แยก scale features และ target
X_scaled = scaler_X.fit_transform(data[features])
y_scaled = scaler_y.fit_transform(data[[target]])

# รวม X_scaled และ y_scaled เพื่อใช้ในการสร้าง sequences
scaled_data = np.hstack((X_scaled, y_scaled))

# สร้าง sequences สำหรับ LSTM
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, :-1])  # เฉพาะ features
        y.append(data[i, -1])  # เฉพาะ target
    return np.array(X), np.array(y)

# ทดลองใช้ window size ที่เหมาะสมกับข้อมูลหุ้น (ประมาณ 1 เดือนการซื้อขาย)
window_size = 20
X, y = create_sequences(scaled_data, window_size)

# แบ่งข้อมูลเป็น train และ test set (ใช้ช่วงเวลาล่าสุด 20% เป็น test เพื่อจำลองการใช้งานจริง)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# เพิ่มแบ่ง validation set จาก train set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

print(f"Shape ของข้อมูล: X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

# สร้างโมเดล LSTM ที่ซับซ้อนขึ้น
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# ใช้ callbacks เพื่อหยุดการเทรนเมื่อไม่มีการพัฒนา และเก็บโมเดลที่ดีที่สุด
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_pttep_model.h5', save_best_only=True, monitor='val_loss')

# เทรนโมเดลพร้อม validation set
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

# แสดงกราฟประวัติการเทรน
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

# ทำนายข้อมูล
y_pred_scaled = model.predict(X_test)

# แปลงกลับเป็นค่าจริง
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_actual = scaler_y.inverse_transform(y_pred_scaled).flatten()

# คำนวณ metrics สำหรับผลลัพธ์
mse = mean_squared_error(y_test_actual, y_pred_actual)
mae = mean_absolute_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_actual, y_pred_actual)

print(f'Mean Squared Error: {mse:.4f}')
print(f'Root Mean Squared Error: {rmse:.4f}')
print(f'Mean Absolute Error: {mae:.4f}')
print(f'R² Score: {r2:.4f}')

# วิเคราะห์ความแม่นยำเทียบกับราคาปัจจุบัน
avg_price = np.mean(y_test_actual)
print(f'ราคาเฉลี่ยของหุ้น PTTEP ในชุดทดสอบ: {avg_price:.2f} บาท')
print(f'RMSE คิดเป็น {(rmse/avg_price)*100:.2f}% ของราคาเฉลี่ย')

# สร้างกราฟเปรียบเทียบค่าจริงกับค่าทำนาย
plt.figure(figsize=(12, 6))
plt.plot(df['Date'].iloc[-len(y_test_actual):], y_test_actual, label='Actual PTTEP Price')
plt.plot(df['Date'].iloc[-len(y_pred_actual):], y_pred_actual, label='Predicted PTTEP Price')
plt.title('PTTEP Stock Price: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Price (THB)')
plt.legend()
plt.grid(True)
plt.savefig('prediction_results.png')
plt.close()

# เรียกใช้ฟังก์ชัน backtest เพื่อทำการทดสอบย้อนหลังอย่างสมบูรณ์
print("\n----- BACKTESTING RESULTS -----")
# ทดสอบบน validation set
print("\nValidation Set Backtesting:")
y_val_pred, val_mse, val_mae, val_rmse, val_r2 = backtest(model, X_val, y_val, scaler_y)

# ทดสอบบน test set
print("\nTest Set Backtesting:")
y_test_pred, test_mse, test_mae, test_rmse, test_r2 = backtest(model, X_test, y_test, scaler_y)

# สร้าง backtesting walk-forward
print("\nWalk-forward Backtesting:")
# แบ่งชุดข้อมูลทดสอบเป็น 3 ช่วงเพื่อจำลองการใช้งานในช่วงเวลาต่างๆ
split_size = len(X_test) // 3
for i in range(3):
    start_idx = i * split_size
    end_idx = (i + 1) * split_size if i < 2 else len(X_test)
    
    X_period = X_test[start_idx:end_idx]
    y_period = y_test[start_idx:end_idx]
    
    # หาช่วงเวลาที่กำลังทดสอบ
    date_period = df['Date'].iloc[-(len(y_test)) + start_idx:-(len(y_test)) + end_idx]
    date_range = f"{date_period.iloc[0].strftime('%Y-%m-%d')} to {date_period.iloc[-1].strftime('%Y-%m-%d')}"
    
    print(f"\nPeriod {i+1} ({date_range}):")
    _, period_mse, period_mae, period_rmse, period_r2 = backtest(model, X_period, y_period, scaler_y)

# ทดสอบการทำนายข้อมูลล่าสุด 1 ชุด
last_sequence = X_test[-1:]
next_pred_scaled = model.predict(last_sequence)
next_pred = scaler_y.inverse_transform(next_pred_scaled)[0, 0]
print(f'ทำนายราคาหุ้น PTTEP สำหรับวันถัดไป: {next_pred:.2f} บาท')

# บันทึกโมเดลและ scaler - แก้ไขการบันทึกโมเดลให้ถูกต้อง
from tensorflow.keras.models import save_model
import joblib

# แก้ไขการบันทึกโมเดลโดยเพิ่มนามสกุล .keras ที่เหมาะสม
save_model(model, 'pttep_lstm_model.keras')
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

print("บันทึกโมเดลและ scaler เรียบร้อยแล้ว")

# สร้างฟังก์ชันเพื่อใช้โมเดลในการทำนายข้อมูลใหม่
def prepare_new_data(new_data_dict, window_data, scaler_X):
    """
    เตรียมข้อมูลใหม่สำหรับการทำนาย
    
    Parameters:
    new_data_dict (dict): ข้อมูลใหม่ในรูปแบบ dict โดยมี key เป็นชื่อ feature
    window_data (numpy.ndarray): ข้อมูล window ล่าสุดที่ใช้
    scaler_X (MinMaxScaler): scaler สำหรับ features
    
    Returns:
    numpy.ndarray: sequence สำหรับการทำนาย
    """
    # แปลงข้อมูลใหม่เป็น array และ scale
    new_data_array = np.array([[new_data_dict[feature] for feature in features]])
    new_data_scaled = scaler_X.transform(new_data_array)
    
    # สร้าง sequence ใหม่โดยเลื่อนข้อมูลเก่าออกและใส่ข้อมูลใหม่เข้าไป
    new_sequence = window_data.copy()
    new_sequence[0, :-1, :] = new_sequence[0, 1:, :]  # เลื่อนข้อมูลเก่า
    new_sequence[0, -1, :] = new_data_scaled[0]  # ใส่ข้อมูลใหม่
    
    return new_sequence

# ฟังก์ชันสำหรับทำนายราคาในอนาคต
def predict_future(model, last_sequence, scaler_y, days=5):
    """ทำนายราคาหุ้นในอนาคต X วัน"""
    curr_sequence = last_sequence.copy()
    predicted_prices = []
    
    for _ in range(days):
        # ทำนายวันถัดไป
        pred = model.predict(curr_sequence, verbose=0)
        predicted_prices.append(pred[0, 0])
        
        # ปรับ sequence สำหรับวันถัดไป (อย่างง่าย)
        # หมายเหตุ: ในการใช้งานจริงต้องอัปเดตฟีเจอร์ทั้งหมดอย่างเหมาะสม
        new_seq = curr_sequence.copy()
        new_seq[0, :-1, :] = new_seq[0, 1:, :]  # เลื่อนข้อมูลเก่า
        curr_sequence = new_seq
    
    # แปลงกลับเป็นราคาจริง
    predicted_prices = scaler_y.inverse_transform(np.array(predicted_prices).reshape(-1, 1)).flatten()
    return predicted_prices

# ทำนายราคาในอนาคต 5 วัน
future_prices = predict_future(model, X_test[-1:], scaler_y, days=5)
print("\nทำนายราคาหุ้น PTTEP ในอีก 5 วันข้างหน้า:")
for i, price in enumerate(future_prices, 1):
    print(f"วันที่ {i}: {price:.2f} บาท")

# การประเมินความเสี่ยงของการลงทุน
price_std = np.std(y_test_actual)
print(f"\nค่าเบี่ยงเบนมาตรฐานของราคาในชุดทดสอบ: {price_std:.2f} บาท")
print(f"ความผันผวนราคา (เทียบกับราคาเฉลี่ย): {(price_std/avg_price)*100:.2f}%")

# เพิ่มการวิเคราะห์การเทรดด้วย backtesting
def simulate_trading(actual_prices, predicted_prices, initial_capital=100000, position_size=0.9):
    """
    จำลองการเทรดโดยใช้ผลการทำนาย
    
    Parameters:
    actual_prices (array): ราคาจริง
    predicted_prices (array): ราคาที่ทำนาย
    initial_capital (float): เงินทุนเริ่มต้น
    position_size (float): สัดส่วนเงินทุนที่ใช้ต่อการเทรด (0-1)
    
    Returns:
    dict: ผลลัพธ์การเทรด
    """
    capital = initial_capital
    shares = 0
    trades = []
    positions = []
    
    # สร้างสัญญาณซื้อขาย (1=ซื้อ, -1=ขาย, 0=ถือ)
    signals = []
    for i in range(1, len(predicted_prices)):
        if predicted_prices[i] > predicted_prices[i-1]:
            signals.append(1)  # Buy signal
        elif predicted_prices[i] < predicted_prices[i-1]:
            signals.append(-1)  # Sell signal
        else:
            signals.append(0)  # Hold signal
    
    # จำลองการเทรด
    for i in range(len(signals)):
        if signals[i] == 1 and shares == 0:  # ซื้อเมื่อไม่มีหุ้นในมือ
            max_shares = int((capital * position_size) / actual_prices[i])
            shares = max_shares
            cost = shares * actual_prices[i]
            capital -= cost
            trades.append({
                'action': 'BUY',
                'price': actual_prices[i],
                'shares': shares,
                'value': cost,
                'capital': capital
            })
        elif signals[i] == -1 and shares > 0:  # ขายเมื่อมีหุ้นในมือ
            value = shares * actual_prices[i]
            capital += value
            trades.append({
                'action': 'SELL',
                'price': actual_prices[i],
                'shares': shares,
                'value': value,
                'capital': capital
            })
            shares = 0
        
        # บันทึกมูลค่าพอร์ตทุกวัน
        total_value = capital + (shares * actual_prices[i])
        positions.append({
            'day': i+1,
            'capital': capital,
            'shares': shares,
            'share_value': shares * actual_prices[i],
            'total_value': total_value
        })
    
    # ปิดตำแหน่งท้ายสุด (ถ้ายังมีหุ้นในมือ)
    if shares > 0:
        value = shares * actual_prices[-1]
        capital += value
        trades.append({
            'action': 'CLOSE',
            'price': actual_prices[-1],
            'shares': shares,
            'value': value,
            'capital': capital
        })
        shares = 0
    
    # คำนวณผลตอบแทน
    final_value = capital
    return_pct = ((final_value - initial_capital) / initial_capital) * 100
    
    # คำนวณผลตอบแทนของ Buy & Hold กลยุทธ์
    max_shares_hold = int(initial_capital / actual_prices[0])
    value_hold = max_shares_hold * actual_prices[-1]
    return_hold_pct = ((value_hold - initial_capital) / initial_capital) * 100
    
    return {
        'initial_capital': initial_capital,
        'final_capital': final_value,
        'return_pct': return_pct,
        'return_hold_pct': return_hold_pct,
        'trades': trades,
        'positions': positions
    }

# ทดสอบกลยุทธ์การเทรดด้วย backtest
print("\n----- TRADING STRATEGY BACKTEST -----")
backtest_result = simulate_trading(y_test_actual, y_pred_actual)

print(f"เงินทุนเริ่มต้น: {backtest_result['initial_capital']:,.2f} บาท")
print(f"เงินทุนสุดท้าย: {backtest_result['final_capital']:,.2f} บาท")
print(f"ผลตอบแทน: {backtest_result['return_pct']:.2f}%")
print(f"ผลตอบแทนแบบ Buy & Hold: {backtest_result['return_hold_pct']:.2f}%")
print(f"จำนวนการเทรดทั้งหมด: {len(backtest_result['trades'])}")

# สร้างกราฟมูลค่าพอร์ตโฟลิโอ
portfolio_values = [pos['total_value'] for pos in backtest_result['positions']]
days = [pos['day'] for pos in backtest_result['positions']]

plt.figure(figsize=(12, 6))
plt.plot(days, portfolio_values)
plt.title('มูลค่าพอร์ตโฟลิโอระหว่างการ Backtest')
plt.xlabel('วัน')
plt.ylabel('มูลค่า (บาท)')
plt.grid(True)
plt.savefig('portfolio_value.png')
plt.close()

print("\nสรุปผลการทำนาย:")
print(f"1. ราคาหุ้น PTTEP วันถัดไป: {next_pred:.2f} บาท")
print(f"2. ความแม่นยำของโมเดล (R²): {r2:.4f}")
print(f"3. ค่าความคลาดเคลื่อนเฉลี่ย (MAE): {mae:.2f} บาท")
print(f"4. ผลตอบแทนจากกลยุทธ์การเทรด: {backtest_result['return_pct']:.2f}%")