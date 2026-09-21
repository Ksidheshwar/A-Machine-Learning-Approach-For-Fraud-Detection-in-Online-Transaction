# FraudGuard — Real-Time Fraud Detection System

## Quick Start (3 steps)

### Step 1 — Install dependencies
```bash
pip install flask pandas numpy scikit-learn openpyxl
```

### Step 2 — Run the app
```bash
python app.py
```

### Step 3 — Open browser
```
http://localhost:5000
```

---

## Real-Time Features

| Feature | How it works |
|---------|-------------|
| **Live transaction feed** | New transactions auto-generated every 4 seconds by background thread |
| **Auto-refresh dashboard** | Stats and feed refresh every 4 seconds automatically |
| **Submit real transaction** | Fill the form on Overview page — get instant fraud score + Transaction ID |
| **Verify any TX ID** | Works for original dataset (TX000001–TX050000) AND live transactions (TX050001+) |
| **Fraud scoring** | Deep Neural Network (256→128→64) scores every transaction in real-time |

---

## Project Structure
```
fraud_detection/
├── app.py                  ← Flask server + real-time simulation thread
├── train_model.py          ← Retrain the model (optional)
├── requirements.txt
├── fraud.db                ← SQLite — 50,000 transactions pre-loaded
├── data/
│   └── Transaction_dataset.xlsx
├── models/
│   ├── cnn_fraud_model.pkl ← Trained model (ready to use)
│   ├── scaler.pkl
│   └── encoders.pkl
└── templates/
    └── dashboard.html      ← Full dashboard UI
```

---

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/` | Dashboard |
| GET | `/api/stats` | Summary stats + model metrics |
| GET | `/api/live_feed` | Latest 12 transactions |
| GET | `/api/transactions` | Paginated table (search + filter) |
| POST | `/api/submit_transaction` | Submit a real transaction for scoring |
| GET | `/api/verify/<TX_ID>` | Verify any Transaction ID |
| GET | `/api/chart/*` | Chart data endpoints |

---

## Submit Transaction API Example

```python
import requests

response = requests.post('http://localhost:5000/api/submit_transaction', json={
    "TransactionAmount": 150000,
    "AccountBalance":    200000,
    "CustomerAge":       35,
    "LoginAttempts":     1,
    "TransactionDuration": 90,
    "hour":              14,
    "TransactionType":   "Debit",
    "Channel":           "Online",
    "CustomerOccupation":"Engineer",
    "Location":          "Mumbai"
})
print(response.json())
# {'transaction_id': 'TX050045', 'fraud_probability': 2.3,
#  'is_fraud': 0, 'risk_level': 'LOW', 'timestamp': '...'}
```

---

## Model Info
- **Algorithm**: Deep Neural Network (MLPClassifier)
- **Architecture**: 256 → 128 → 64 neurons
- **Accuracy**: 99.79% | **AUC-ROC**: 99.96% | **F1**: 98.02%
- **Currency**: All amounts in Indian Rupees (₹), converted at 1 USD = ₹83.5
