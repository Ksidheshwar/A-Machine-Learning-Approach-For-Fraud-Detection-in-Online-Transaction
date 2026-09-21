import pandas as pd
import numpy as np
import sqlite3
import pickle
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Transaction_dataset.xlsx')
DB_PATH = os.path.join(BASE_DIR, 'fraud.db')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'cnn_fraud_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'encoders.pkl')

def load_and_preprocess():
    print("[1/6] Loading dataset...")
    df = pd.read_excel(DATA_PATH)
    # Convert USD → INR (1 USD = 83.5 INR)
    INR_RATE = 83.5
    df['TransactionAmount'] = (df['TransactionAmount'] * INR_RATE).round(2)
    df['AccountBalance'] = (df['AccountBalance'] * INR_RATE).round(2)
    df['Hour'] = pd.to_datetime(df['TransactionDate']).dt.hour
    df['DayOfWeek'] = pd.to_datetime(df['TransactionDate']).dt.dayofweek
    df['Month'] = pd.to_datetime(df['TransactionDate']).dt.month
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    df['IsNightTime'] = df['Hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)
    df['AmountToBalanceRatio'] = df['TransactionAmount'] / (df['AccountBalance'] + 1)

    fraud_score = (
        (df['LoginAttempts'] > 2).astype(int) * 3 +
        (df['TransactionAmount'] > df['TransactionAmount'].quantile(0.97)).astype(int) * 2 +
        (df['IsNightTime'] == 1).astype(int) * 1 +
        (df['TransactionDuration'] < 20).astype(int) * 1 +
        (df['AmountToBalanceRatio'] > 0.5).astype(int) * 2 +
        (df['LoginAttempts'] >= 5).astype(int) * 3
    )
    df['IsFraud'] = (fraud_score >= 4).astype(int)
    print(f"    Fraud cases: {df['IsFraud'].sum()} / {len(df)} ({df['IsFraud'].mean()*100:.2f}%)")

    encoders = {}
    cat_cols = ['TransactionType', 'Location', 'Channel', 'CustomerOccupation']
    for col in cat_cols:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col])
        encoders[col] = le

    features = [
        'TransactionAmount', 'CustomerAge', 'TransactionDuration',
        'LoginAttempts', 'AccountBalance', 'Hour', 'DayOfWeek',
        'Month', 'IsWeekend', 'IsNightTime', 'AmountToBalanceRatio',
        'TransactionType_enc', 'Location_enc', 'Channel_enc', 'CustomerOccupation_enc'
    ]
    X = df[features].values
    y = df['IsFraud'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X_scaled, y, scaler, encoders, features

def save_to_db(df):
    print("[3/6] Saving to SQLite database...")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        TransactionID TEXT UNIQUE, AccountID TEXT,
        TransactionAmount REAL, TransactionDate TEXT,
        TransactionType TEXT, Location TEXT,
        DeviceID TEXT, IPAddress TEXT, MerchantID TEXT,
        Channel TEXT, CustomerAge INTEGER,
        CustomerOccupation TEXT, TransactionDuration INTEGER,
        LoginAttempts INTEGER, AccountBalance REAL,
        IsFraud INTEGER, FraudProbability REAL, CreatedAt TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS model_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Accuracy REAL, Precision REAL, Recall REAL,
        F1Score REAL, AUC REAL, TrainedAt TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        TransactionAmount REAL, CustomerAge INTEGER,
        Channel TEXT, TransactionType TEXT,
        LoginAttempts INTEGER, AccountBalance REAL,
        TransactionDuration INTEGER, Location TEXT,
        CustomerOccupation TEXT, FraudProbability REAL,
        IsFraud INTEGER, PredictedAt TEXT)''')
    conn.commit()
    records = []
    for _, row in df.iterrows():
        records.append((
            str(row['TransactionID']), str(row['AccountID']),
            float(row['TransactionAmount']), str(row['TransactionDate']),
            str(row['TransactionType']), str(row['Location']),
            str(row['DeviceID']), str(row['IP Address']), str(row['MerchantID']),
            str(row['Channel']), int(row['CustomerAge']),
            str(row['CustomerOccupation']), int(row['TransactionDuration']),
            int(row['LoginAttempts']), float(row['AccountBalance']),
            int(row['IsFraud']), 0.0, datetime.now().isoformat()
        ))
    c.executemany('''INSERT OR IGNORE INTO transactions
        (TransactionID,AccountID,TransactionAmount,TransactionDate,
         TransactionType,Location,DeviceID,IPAddress,MerchantID,
         Channel,CustomerAge,CustomerOccupation,TransactionDuration,
         LoginAttempts,AccountBalance,IsFraud,FraudProbability,CreatedAt)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', records)
    conn.commit()
    conn.close()
    print(f"    Saved {len(records)} transactions to DB")

def train():
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    df, X, y, scaler, encoders, features = load_and_preprocess()
    save_to_db(df)

    print("[4/6] Splitting and training Deep Neural Network model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Deep MLP mimicking CNN depth: 3 hidden layers with decreasing size
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=256,
        learning_rate='adaptive',
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        verbose=True
    )
    model.fit(X_train, y_train)

    print("\n[5/6] Evaluating model...")
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_pred_prob)
    accuracy = report['accuracy']
    precision = report.get('1', {}).get('precision', 0)
    recall = report.get('1', {}).get('recall', 0)
    f1 = report.get('1', {}).get('f1-score', 0)
    print(f"\n    ✅ Accuracy: {accuracy:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")

    conn = sqlite3.connect(DB_PATH)
    conn.execute('INSERT INTO model_metrics (Accuracy,Precision,Recall,F1Score,AUC,TrainedAt) VALUES (?,?,?,?,?,?)',
                 (accuracy, precision, recall, f1, auc, datetime.now().isoformat()))
    conn.commit()
    conn.close()

    print("[6/6] Saving model artifacts...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump({'encoders': encoders, 'features': features}, f)

    print("\n✅ Training complete! All artifacts saved.")
    return accuracy, auc

if __name__ == '__main__':
    train()
