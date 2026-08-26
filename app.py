import sqlite3
import pandas as pd
import numpy as np
import joblib
import webbrowser
from threading import Timer
from flask import Flask, jsonify, render_template, request

# Initialize Flask app
app = Flask(__name__)

DB_NAME = "transactions.db"

# Automatically create database & table if it doesn't exist
def init_db():
    """Creates the database table automatically if transactions.db does not exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customer_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            month_period TEXT,
            billed_amount REAL,
            paid_amount REAL,
            outstanding_balance REAL,
            payment_behavior TEXT
        )
    """)
    conn.commit()
    conn.close()

# Run database creation check
init_db()

# Load ML Model & Scaler safely
try:
    loaded_object = joblib.load("credit_model.pkl")
    if isinstance(loaded_object, dict):
        model = loaded_object.get("model", loaded_object)
        scaler = loaded_object.get("scaler", None)
    else:
        model = loaded_object
        scaler = None
except Exception as e:
    print(f"Warning: Could not load credit_model.pkl: {e}")
    model = None
    scaler = None

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/dashboard-data", methods=["GET"])
def dashboard_data():
    """Returns store KPIs, yearly revenue, payment behaviors, and top customers."""
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM customer_transactions", conn)
        conn.close()

        if df.empty:
            return jsonify({
                "kpis": {"total_billed": 0, "total_paid": 0, "total_outstanding": 0, "recovery_rate": 0, "total_orders": 0},
                "yearly": {"labels": [], "billed": [], "paid": []},
                "behavior": {"labels": [], "counts": []},
                "top_customers": {"labels": [], "billed": [], "paid": []}
            })

        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        for col in ['billed_amount', 'paid_amount', 'outstanding_balance']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        if 'year' not in df.columns:
            df['year'] = df['month_period'].str.extract(r'(20\d{2})')[0].fillna('2025')

        total_billed = float(df['billed_amount'].sum())
        total_paid = float(df['paid_amount'].sum())
        total_outstanding = float(df['outstanding_balance'].sum())
        recovery_rate = round((total_paid / total_billed * 100), 1) if total_billed > 0 else 0.0
        total_orders = len(df)

        yearly_df = df.groupby('year', as_index=False)[['billed_amount', 'paid_amount']].sum().sort_values('year')
        yearly_labels = yearly_df['year'].astype(str).tolist()
        yearly_billed = yearly_df['billed_amount'].round(2).tolist()
        yearly_paid = yearly_df['paid_amount'].round(2).tolist()

        behavior_series = df['payment_behavior'].str.strip().str.title().value_counts()
        behavior_labels = behavior_series.index.tolist()
        behavior_counts = behavior_series.values.tolist()

        cust_df = df.groupby('customer_id', as_index=False)[['billed_amount', 'paid_amount']].sum()
        top_cust = cust_df.sort_values(by='billed_amount', ascending=False).head(5)
        top_labels = top_cust['customer_id'].tolist()
        top_billed = top_cust['billed_amount'].round(2).tolist()
        top_paid = top_cust['paid_amount'].round(2).tolist()

        return jsonify({
            "kpis": {
                "total_billed": round(total_billed, 2),
                "total_paid": round(total_paid, 2),
                "total_outstanding": round(total_outstanding, 2),
                "recovery_rate": recovery_rate,
                "total_orders": total_orders
            },
            "yearly": {
                "labels": yearly_labels,
                "billed": yearly_billed,
                "paid": yearly_paid
            },
            "behavior": {
                "labels": behavior_labels,
                "counts": behavior_counts
            },
            "top_customers": {
                "labels": top_labels,
                "billed": top_billed,
                "paid": top_paid
            }
        })

    except Exception as e:
        print(f"❌ Dashboard Data Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/customer-lookup", methods=["GET"])
def customer_lookup():
    """Allows checking monthly payment status for a specific customer across all years."""
    cust_id = request.args.get("customer_id", "").strip()
    if not cust_id:
        return jsonify({"error": "Please provide a Customer ID"}), 400

    if cust_id.startswith("cust") and " " not in cust_id:
        cust_id = cust_id.replace("cust", "cust ")

    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM customer_transactions WHERE LOWER(customer_id) = LOWER(?)", conn, params=(cust_id,))
    conn.close()

    if df.empty:
        return jsonify({"error": f"No transactions found for customer '{cust_id}'"}), 404

    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    for col in ['billed_amount', 'paid_amount', 'outstanding_balance']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    total_billed = float(df['billed_amount'].sum())
    total_paid = float(df['paid_amount'].sum())
    total_outstanding = float(df['outstanding_balance'].sum())

    history = df[['month_period', 'billed_amount', 'paid_amount', 'outstanding_balance', 'payment_behavior']].to_dict(orient='records')

    return jsonify({
        "customer_id": cust_id,
        "total_billed": round(total_billed, 2),
        "total_paid": round(total_paid, 2),
        "total_outstanding": round(total_outstanding, 2),
        "history": history
    })

@app.route("/api/create-customer-transaction", methods=["POST"])
def create_customer_transaction():
    """Saves a new transaction directly to the database and calculates ML risk prediction."""
    try:
        data = request.json or {}
        cust_id = str(data.get("customer_id", "")).strip()
        month_period = str(data.get("month_period", "")).strip()
        billed = float(data.get("billed_amount", 0))
        paid = float(data.get("paid_amount", 0))
        behavior = str(data.get("payment_behavior", "Fully Paid")).strip()

        if not cust_id or not month_period:
            return jsonify({"error": "Customer ID and Period are required"}), 400

        if cust_id.startswith("cust") and " " not in cust_id:
            cust_id = cust_id.replace("cust", "cust ")

        outstanding = max(0.0, billed - paid)

        # 1. Insert into SQLite
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO customer_transactions (customer_id, month_period, billed_amount, paid_amount, outstanding_balance, payment_behavior)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (cust_id, month_period, billed, paid, outstanding, behavior))
        conn.commit()

        # 2. Fetch history for prediction
        df = pd.read_sql_query("SELECT * FROM customer_transactions WHERE LOWER(customer_id) = LOWER(?)", conn, params=(cust_id,))
        conn.close()

        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        for col in ['billed_amount', 'paid_amount', 'outstanding_balance']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        total_billed = float(df['billed_amount'].sum())
        total_paid = float(df['paid_amount'].sum())
        total_outstanding = float(df['outstanding_balance'].sum())
        avg_billed = float(df['billed_amount'].mean())
        avg_paid = float(df['paid_amount'].mean())

        behavior_col = df['payment_behavior'].str.strip().str.title() if 'payment_behavior' in df.columns else pd.Series()
        unpaid_count = int((behavior_col == 'Unpaid').sum())
        partially_paid_count = int((behavior_col == 'Partially Paid').sum())
        fully_paid_count = int((behavior_col == 'Fully Paid').sum())

        # 3. Predict Credit Status
        if model is not None:
            raw_features = np.array([[
                total_billed, total_paid, total_outstanding,
                avg_billed, avg_paid,
                unpaid_count, partially_paid_count, fully_paid_count
            ]])

            if scaler is not None:
                scaled_features = scaler.transform(raw_features)
                prediction = model.predict(scaled_features)[0]
            else:
                prediction = model.predict(raw_features)[0]

            credit_status = "Approved (Low Risk)" if prediction == 1 else "Rejected (High Risk)"
        else:
            credit_status = "Approved (Default)"

        return jsonify({
            "success": True,
            "message": "Record saved successfully to database!",
            "customer_id": cust_id,
            "total_billed": round(total_billed, 2),
            "total_paid": round(total_paid, 2),
            "total_outstanding": round(total_outstanding, 2),
            "credit_decision": credit_status,
            "history": df[['month_period', 'billed_amount', 'paid_amount', 'outstanding_balance', 'payment_behavior']].to_dict(orient='records')
        })

    except Exception as e:
        print(f"❌ Error creating record: {e}")
        return jsonify({"error": str(e)}), 500

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    Timer(1.25, open_browser).start()
    app.run(debug=True, port=5000)