from flask import Flask, render_template, jsonify, request
import sqlite3, pickle, numpy as np, os, random, threading, time
from datetime import datetime

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, 'fraud.db')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'cnn_fraud_model.pkl')
SCALER_PATH= os.path.join(BASE_DIR, 'models', 'scaler.pkl')
ENCODER_PATH=os.path.join(BASE_DIR, 'models', 'encoders.pkl')

app = Flask(__name__)
_model = _scaler = _encoders = None

# Real-time counter — reads max from DB on startup
def _init_counter():
    try:
        conn = sqlite3.connect(DB_PATH)
        r = conn.execute("SELECT MAX(CAST(SUBSTR(TransactionID,3) AS INTEGER)) FROM transactions").fetchone()
        conn.close()
        return (r[0] if r[0] else 50000)
    except:
        return 50000

_tx_counter = _init_counter()
_tx_lock    = threading.Lock()

def _next_txid():
    """Thread-safe auto-incrementing Transaction ID generator."""
    global _tx_counter
    with _tx_lock:
        _tx_counter += 1
        return f'TX{_tx_counter:06d}'

LOCATIONS   = ['Mumbai','Delhi','Bangalore','Chennai','Hyderabad','Pune',
               'Kolkata','Ahmedabad','Surat','Jaipur','Lucknow','Nagpur',
               'Indore','Bhopal','Visakhapatnam','Patna','Vadodara','Agra']
CHANNELS    = ['ATM','Online','Branch']
TX_TYPES    = ['Debit','Credit']
OCCUPATIONS = ['Doctor','Engineer','Student','Retired']
MERCHANTS   = [f'M{i:03d}' for i in range(1,61)]
DEVICES     = [f'D{i:06d}' for i in range(1,401)]

def get_model():
    global _model, _scaler, _encoders
    if _model is None:
        with open(MODEL_PATH,'rb') as f:  _model   = pickle.load(f)
        with open(SCALER_PATH,'rb') as f: _scaler  = pickle.load(f)
        with open(ENCODER_PATH,'rb') as f:
            d = pickle.load(f); _encoders = d['encoders']
    return _model, _scaler, _encoders

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def validate_transaction_inputs(data):
    """
    Validate business rules before scoring or saving a transaction.
    Returns (is_valid, error_message).
    Rules:
      - CustomerAge        : 18 ≤ age ≤ 80
      - LoginAttempts      : 1 ≤ logins ≤ 100
      - TransactionDuration: 1 ≤ duration ≤ 100 seconds
    """
    errors = []

    # Age validation
    try:
        age = int(data.get('CustomerAge', 0))
        if age < 18:
            errors.append(f'Age must be at least 18 (got {age}).')
        elif age > 80:
            errors.append(f'Age must be 80 or below (got {age}).')
    except (ValueError, TypeError):
        errors.append('Age must be a valid number.')

    # Login attempts validation
    try:
        logins = int(data.get('LoginAttempts', 0))
        if logins < 1:
            errors.append(f'Login attempts must be at least 1 (got {logins}).')
        elif logins > 100:
            errors.append(f'Login attempts must be 100 or less (got {logins}).')
    except (ValueError, TypeError):
        errors.append('Login attempts must be a valid number.')

    # Duration validation
    try:
        duration = int(data.get('TransactionDuration', 0))
        if duration < 1:
            errors.append(f'Transaction duration must be at least 1 second (got {duration}).')
        elif duration > 100:
            errors.append(f'Transaction duration must be 100 seconds or less (got {duration}).')
    except (ValueError, TypeError):
        errors.append('Transaction duration must be a valid number.')

    if errors:
        return False, ' | '.join(errors)
    return True, None


def score_transaction(data):
    """Run ML model on a transaction dict, return probability + label."""
    model, scaler, encoders = get_model()
    hour    = int(data.get('hour', 12))
    is_night= 1 if hour < 6 or hour > 22 else 0
    amount  = float(data['TransactionAmount'])
    balance = float(data['AccountBalance'])
    ratio   = amount / (balance + 1)
    def safe_enc(enc, val):
        try: return int(enc.transform([val])[0])
        except: return 0
    feat = np.array([[
        amount, int(data['CustomerAge']), int(data['TransactionDuration']),
        int(data['LoginAttempts']), balance, hour,
        int(data.get('day_of_week', datetime.now().weekday())),
        int(data.get('month', datetime.now().month)),
        int(data.get('is_weekend', 1 if datetime.now().weekday()>=5 else 0)),
        is_night, ratio,
        safe_enc(encoders['TransactionType'],    data['TransactionType']),
        safe_enc(encoders['Location'],           data['Location']),
        safe_enc(encoders['Channel'],            data['Channel']),
        safe_enc(encoders['CustomerOccupation'], data['CustomerOccupation']),
    ]])
    prob     = float(model.predict_proba(scaler.transform(feat))[0][1])

    # --- Hard rules applied ON TOP of the ML model ---
    login_attempts = int(data['LoginAttempts'])
    duration       = int(data['TransactionDuration'])
    rule_triggered = None

    # Rule 1: login attempts > 100 → always fraud
    if login_attempts > 100:
        prob = max(prob, 0.95)
        rule_triggered = f'Login attempts ({login_attempts}) exceeded limit of 100'

    # Rule 2: duration > 100 seconds → always fraud
    if duration > 100:
        prob = max(prob, 0.95)
        rule_triggered = (rule_triggered or '') + (
            ('; ' if rule_triggered else '') +
            f'Transaction duration ({duration}s) exceeded 100s limit'
        )

    # Rule 3: amount/balance ratio >= 0.40 → always fraud (existing behaviour)
    if ratio >= 0.40:
        prob = max(prob, 0.95)
        rule_triggered = (rule_triggered or '') + (
            ('; ' if rule_triggered else '') +
            f'Amount/Balance ratio ({ratio:.1%}) >= 40%'
        )

    is_fraud = 1 if prob >= 0.5 else 0
    return {
        'prob': prob,
        'is_fraud': is_fraud,
        'risk': 'HIGH' if prob >= 0.7 else ('MEDIUM' if prob >= 0.4 else 'LOW'),
        'rule_triggered': rule_triggered  # None when only ML decides
    }

def generate_transaction():
    """Generate one realistic random transaction."""
    tid = _next_txid()

    now      = datetime.now()
    amount   = round(random.choices(
                   [random.uniform(500,8000),
                    random.uniform(8000,50000),
                    random.uniform(50000,200000)],
                   weights=[60,30,10])[0], 2)
    balance  = round(random.uniform(10000, 1500000), 2)
    logins   = random.choices(
                   [random.randint(1,5), random.randint(6,50), random.randint(51,100)],
                   weights=[75, 15, 10])[0]
    duration = random.randint(1, 100)
    location = random.choice(LOCATIONS)
    channel  = random.choice(CHANNELS)
    tx_type  = random.choice(TX_TYPES)
    occ      = random.choice(OCCUPATIONS)
    age      = random.randint(18, 80)

    result = score_transaction({
        'TransactionAmount':  amount,
        'CustomerAge':        age,
        'TransactionDuration':duration,
        'LoginAttempts':      logins,
        'AccountBalance':     balance,
        'hour':               now.hour,
        'TransactionType':    tx_type,
        'Location':           location,
        'Channel':            channel,
        'CustomerOccupation': occ,
    })

    conn = get_db()
    conn.execute('''INSERT INTO transactions
        (TransactionID,AccountID,TransactionAmount,TransactionDate,
         TransactionType,Location,DeviceID,IPAddress,MerchantID,
         Channel,CustomerAge,CustomerOccupation,TransactionDuration,
         LoginAttempts,AccountBalance,IsFraud,FraudProbability,CreatedAt)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
        tid,
        f'AC{random.randint(1,500):05d}',
        amount,
        now.strftime('%Y-%m-%d %H:%M:%S'),
        tx_type, location,
        random.choice(DEVICES),
        f'{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}',
        random.choice(MERCHANTS),
        channel, age, occ, duration, logins, balance,
        result['is_fraud'], result['prob'],
        now.isoformat()
    ))
    conn.commit()
    conn.close()
    return tid, result

def live_transaction_loop():
    """Background thread: generate a new transaction every 4 seconds."""
    # Wait for model to load on first request before starting simulation
    time.sleep(12)
    print('[LIVE] Real-time transaction simulation started.')
    while True:
        try:
            tid, result = generate_transaction()
            status = 'FRAUD' if result['is_fraud'] else 'OK'
            print(f'[LIVE] {tid} | ₹ | {result["risk"]} | {status}')
        except Exception as e:
            print(f'[LIVE] Error: {e}')
        time.sleep(4)

# Start background simulation thread
sim_thread = threading.Thread(target=live_transaction_loop, daemon=True)
sim_thread.start()

# ── API ROUTES ──

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/stats')
def stats():
    conn = get_db()
    total       = conn.execute('SELECT COUNT(*) as c FROM transactions').fetchone()['c']
    fraud       = conn.execute('SELECT COUNT(*) as c FROM transactions WHERE IsFraud=1').fetchone()['c']
    avg_amount  = conn.execute('SELECT AVG(TransactionAmount) as a FROM transactions').fetchone()['a']
    fraud_amt   = conn.execute('SELECT AVG(TransactionAmount) as a FROM transactions WHERE IsFraud=1').fetchone()['a']
    metrics     = conn.execute('SELECT * FROM model_metrics ORDER BY id DESC LIMIT 1').fetchone()
    live_count  = conn.execute("SELECT COUNT(*) as c FROM transactions WHERE TransactionID > 'TX050000'").fetchone()['c']
    conn.close()
    return jsonify({
        'total': total, 'fraud': fraud, 'legit': total-fraud,
        'fraud_rate': round(fraud/total*100, 2) if total else 0,
        'avg_amount': round(avg_amount or 0, 2),
        'fraud_avg_amount': round(fraud_amt or 0, 2),
        'live_count': live_count,
        'metrics': dict(metrics) if metrics else {}
    })

@app.route('/api/live_feed')
def live_feed():
    conn = get_db()
    rows = conn.execute('''SELECT TransactionID,AccountID,TransactionAmount,
        TransactionDate,TransactionType,Channel,Location,CustomerAge,
        LoginAttempts,AccountBalance,IsFraud,FraudProbability,CustomerOccupation
        FROM transactions ORDER BY id DESC LIMIT 12''').fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/chart/channel')
def chart_channel():
    conn = get_db()
    rows = conn.execute('''SELECT Channel,
        SUM(CASE WHEN IsFraud=1 THEN 1 ELSE 0 END) as fraud,
        SUM(CASE WHEN IsFraud=0 THEN 1 ELSE 0 END) as legit
        FROM transactions GROUP BY Channel''').fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/chart/occupation')
def chart_occupation():
    conn = get_db()
    rows = conn.execute('''SELECT CustomerOccupation,
        SUM(CASE WHEN IsFraud=1 THEN 1 ELSE 0 END) as fraud,
        COUNT(*) as total FROM transactions GROUP BY CustomerOccupation''').fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/chart/amount_dist')
def chart_amount_dist():
    conn = get_db()
    rows = conn.execute('''SELECT
        CASE WHEN TransactionAmount<8350  THEN '0-8K'
             WHEN TransactionAmount<25050 THEN '8K-25K'
             WHEN TransactionAmount<50100 THEN '25K-50K'
             WHEN TransactionAmount<83500 THEN '50K-83K'
             ELSE '83K+' END as bucket,
        SUM(CASE WHEN IsFraud=1 THEN 1 ELSE 0 END) as fraud,
        SUM(CASE WHEN IsFraud=0 THEN 1 ELSE 0 END) as legit
        FROM transactions GROUP BY bucket ORDER BY bucket''').fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/chart/hourly')
def chart_hourly():
    conn = get_db()
    rows = conn.execute('''SELECT CAST(SUBSTR(TransactionDate,12,2) AS INTEGER) as hour,
        SUM(CASE WHEN IsFraud=1 THEN 1 ELSE 0 END) as fraud,
        COUNT(*) as total FROM transactions GROUP BY hour ORDER BY hour''').fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/chart/login_attempts')
def chart_login():
    conn = get_db()
    rows = conn.execute('''SELECT LoginAttempts,
        SUM(CASE WHEN IsFraud=1 THEN 1 ELSE 0 END) as fraud,
        COUNT(*) as total FROM transactions GROUP BY LoginAttempts ORDER BY LoginAttempts''').fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/transactions')
def transactions():
    page       = int(request.args.get('page', 1))
    per_page   = 15
    offset     = (page-1)*per_page
    search     = request.args.get('search','')
    fraud_filter = request.args.get('fraud','')
    query      = 'SELECT * FROM transactions WHERE 1=1'
    params     = []
    if search:
        query += ' AND (TransactionID LIKE ? OR AccountID LIKE ? OR Location LIKE ?)'
        params += [f'%{search}%']*3
    if fraud_filter in ('0','1'):
        query += ' AND IsFraud=?'
        params.append(int(fraud_filter))
    conn  = get_db()
    total = conn.execute(f'SELECT COUNT(*) as c FROM ({query})', params).fetchone()['c']
    rows  = conn.execute(query+f' ORDER BY id DESC LIMIT {per_page} OFFSET {offset}', params).fetchall()
    conn.close()
    return jsonify({'data':[dict(r) for r in rows],'total':total,'page':page,'per_page':per_page})

@app.route('/api/verify/<transaction_id>')
def verify(transaction_id):
    conn = get_db()
    row  = conn.execute('SELECT * FROM transactions WHERE UPPER(TransactionID)=UPPER(?)',
                        (transaction_id,)).fetchone()
    conn.close()
    if row is None:
        return jsonify({'found': False})
    return jsonify({'found': True, 'transaction': dict(row)})

@app.route('/api/submit_transaction', methods=['POST'])
def submit_transaction():
    """Manually submit a real transaction for instant fraud scoring."""
    data = request.json
    tid  = _next_txid()
    try:
        # ── Validate inputs before scoring ──
        valid, err_msg = validate_transaction_inputs(data)
        if not valid:
            return jsonify({'error': err_msg}), 400

        result = score_transaction(data)
        now    = datetime.now()
        conn   = get_db()
        conn.execute('''INSERT INTO transactions
            (TransactionID,AccountID,TransactionAmount,TransactionDate,
             TransactionType,Location,DeviceID,IPAddress,MerchantID,
             Channel,CustomerAge,CustomerOccupation,TransactionDuration,
             LoginAttempts,AccountBalance,IsFraud,FraudProbability,CreatedAt)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
            tid,
            data.get('AccountID', f'AC{random.randint(1,999):05d}'),
            float(data['TransactionAmount']),
            now.strftime('%Y-%m-%d %H:%M:%S'),
            data['TransactionType'],
            data['Location'],
            data.get('DeviceID', f'D{random.randint(1,400):06d}'),
            data.get('IPAddress','0.0.0.0'),
            data.get('MerchantID','M001'),
            data['Channel'],
            int(data['CustomerAge']),
            data['CustomerOccupation'],
            int(data['TransactionDuration']),
            int(data['LoginAttempts']),
            float(data['AccountBalance']),
            result['is_fraud'], result['prob'],
            now.isoformat()
        ))
        conn.commit()
        conn.close()
        return jsonify({
            'transaction_id':    tid,
            'fraud_probability': round(result['prob']*100, 2),
            'is_fraud':          result['is_fraud'],
            'risk_level':        result['risk'],
            'rule_triggered':    result.get('rule_triggered'),
            'timestamp':         now.strftime('%d %b %Y %H:%M:%S')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/locations')
def locations():
    conn = get_db()
    rows = conn.execute('SELECT DISTINCT Location FROM transactions ORDER BY Location').fetchall()
    conn.close()
    return jsonify([r['Location'] for r in rows])

if __name__ == '__main__':
    app.run(debug=False, port=5000, threaded=True)

# ═══════════════════ RECORDS MANAGEMENT ═══════════════════════

@app.route('/api/records/add', methods=['POST'])
def add_record():
    """Add one or more manual records to the dataset."""
    data = request.json
    records = data if isinstance(data, list) else [data]
    added, errors = [], []

    for rec in records:
        try:
            now  = datetime.now()
            txid = rec.get('TransactionID') or _next_txid()
            # Check duplicate
            conn = get_db()
            exists = conn.execute('SELECT 1 FROM transactions WHERE UPPER(TransactionID)=UPPER(?)', (txid,)).fetchone()
            if exists:
                errors.append(f'{txid}: already exists')
                conn.close()
                continue

            amount  = float(rec['TransactionAmount'])
            balance = float(rec['AccountBalance'])

            # ── Validate inputs before scoring ──
            valid, err_msg = validate_transaction_inputs(rec)
            if not valid:
                errors.append(f'{txid}: {err_msg}')
                conn.close()
                continue

            # Auto-score if IsFraud not explicitly provided
            if 'IsFraud' in rec and rec['IsFraud'] != '' and rec['IsFraud'] is not None:
                is_fraud = int(rec['IsFraud'])
                prob     = float(rec.get('FraudProbability', is_fraud))
                risk     = 'HIGH' if prob >= 0.7 else ('MEDIUM' if prob >= 0.4 else 'LOW')
            else:
                result   = score_transaction({**rec, 'TransactionAmount': amount, 'AccountBalance': balance})
                is_fraud = result['is_fraud']
                prob     = result['prob']
                risk     = result['risk']

            conn.execute('''INSERT INTO transactions
                (TransactionID,AccountID,TransactionAmount,TransactionDate,
                 TransactionType,Location,DeviceID,IPAddress,MerchantID,
                 Channel,CustomerAge,CustomerOccupation,TransactionDuration,
                 LoginAttempts,AccountBalance,IsFraud,FraudProbability,CreatedAt)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
                txid,
                rec.get('AccountID', f'AC{random.randint(1,999):05d}'),
                amount,
                rec.get('TransactionDate', now.isoformat()),
                rec.get('TransactionType','Debit'),
                rec.get('Location','Unknown'),
                rec.get('DeviceID', f'D{random.randint(1,400):06d}'),
                rec.get('IPAddress','0.0.0.0'),
                rec.get('MerchantID', f'M{random.randint(1,60):03d}'),
                rec.get('Channel','Online'),
                int(rec.get('CustomerAge', 30)),
                rec.get('CustomerOccupation','Engineer'),
                int(rec.get('TransactionDuration', 60)),
                int(rec.get('LoginAttempts', 1)),
                balance,
                is_fraud, prob,
                now.isoformat()
            ))
            conn.commit()
            conn.close()
            added.append({'transaction_id': txid, 'is_fraud': is_fraud,
                          'fraud_probability': round(prob*100,2), 'risk_level': risk})
        except Exception as e:
            errors.append(f'{rec.get("TransactionID","?")}: {str(e)}')

    return jsonify({'added': added, 'errors': errors, 'count': len(added)})


@app.route('/api/records/edit/<transaction_id>', methods=['PUT'])
def edit_record(transaction_id):
    """Edit an existing record by Transaction ID."""
    data = request.json
    conn = get_db()
    row  = conn.execute('SELECT * FROM transactions WHERE UPPER(TransactionID)=UPPER(?)', (transaction_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'Transaction not found'}), 404

    # ── Validate inputs if age / logins / duration are being updated ──
    check_fields = {k: data[k] for k in ('CustomerAge','LoginAttempts','TransactionDuration') if k in data}
    if check_fields:
        # Fill missing fields from existing record so validator has full context
        merged = {**dict(row), **check_fields}
        valid, err_msg = validate_transaction_inputs(merged)
        if not valid:
            conn.close()
            return jsonify({'error': err_msg}), 400

    fields  = ['AccountID','TransactionAmount','TransactionDate','TransactionType',
               'Location','DeviceID','IPAddress','MerchantID','Channel',
               'CustomerAge','CustomerOccupation','TransactionDuration',
               'LoginAttempts','AccountBalance','IsFraud','FraudProbability']
    updates = {k: data[k] for k in fields if k in data}

    if updates:
        set_clause = ', '.join([f'{k}=?' for k in updates])
        conn.execute(f'UPDATE transactions SET {set_clause} WHERE UPPER(TransactionID)=UPPER(?)',
                     list(updates.values()) + [transaction_id])
        conn.commit()
    conn.close()
    return jsonify({'success': True, 'updated': list(updates.keys())})


@app.route('/api/records/delete/<transaction_id>', methods=['DELETE'])
def delete_record(transaction_id):
    """Delete a record by Transaction ID."""
    conn = get_db()
    row  = conn.execute('SELECT 1 FROM transactions WHERE UPPER(TransactionID)=UPPER(?)', (transaction_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'Transaction not found'}), 404
    conn.execute('DELETE FROM transactions WHERE UPPER(TransactionID)=UPPER(?)', (transaction_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'deleted': transaction_id.upper()})


@app.route('/api/records/bulk_delete', methods=['POST'])
def bulk_delete():
    """Delete multiple records."""
    ids = request.json.get('ids', [])
    if not ids:
        return jsonify({'error': 'No IDs provided'}), 400
    conn = get_db()
    placeholders = ','.join(['?' for _ in ids])
    conn.execute(f'DELETE FROM transactions WHERE UPPER(TransactionID) IN ({placeholders})',
                 [i.upper() for i in ids])
    conn.commit()
    deleted = conn.execute('SELECT changes()').fetchone()[0]
    conn.close()
    return jsonify({'success': True, 'deleted_count': deleted})


@app.route('/api/records/export')
def export_records():
    """Export all records as JSON for download."""
    conn = get_db()
    rows = conn.execute('SELECT * FROM transactions ORDER BY id').fetchall()
    conn.close()
    from flask import Response
    import json
    data = json.dumps([dict(r) for r in rows], indent=2, default=str)
    return Response(data, mimetype='application/json',
                    headers={'Content-Disposition': 'attachment;filename=transactions_export.json'})


@app.route('/api/records/search_id')
def search_id():
    """Quick lookup a transaction to prefill edit form."""
    tid  = request.args.get('id','')
    conn = get_db()
    row  = conn.execute('SELECT * FROM transactions WHERE UPPER(TransactionID)=UPPER(?)', (tid,)).fetchone()
    conn.close()
    if not row:
        return jsonify({'found': False})
    return jsonify({'found': True, 'record': dict(row)})
