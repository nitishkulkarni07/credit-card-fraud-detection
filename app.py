from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import pandas as pd
import joblib
import os
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "credit_card_fraud_secret_key"


# -------------------------------
# Load ML Model
# -------------------------------
if not os.path.exists("transaction_model.pkl") or not os.path.exists("encoders.pkl"):
    print("Model files not found. Please run train_model.py first.")
    exit()

try:
    model = joblib.load("transaction_model.pkl")
    encoders = joblib.load("encoders.pkl")
except EOFError:
    print("Model file is corrupted. Delete pkl files and run train_model.py again.")
    exit()


# -------------------------------
# Database Creation
# -------------------------------
def create_database():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            transaction_type TEXT NOT NULL,
            merchant_category TEXT NOT NULL,
            location TEXT NOT NULL,
            card_type TEXT NOT NULL,
            hour INTEGER NOT NULL,
            previous_transactions INTEGER NOT NULL,
            is_foreign_transaction TEXT NOT NULL,
            result TEXT NOT NULL,
            status TEXT NOT NULL,
            normal_probability REAL NOT NULL,
            fraud_probability REAL NOT NULL,
            checked_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


create_database()


# -------------------------------
# Helper: Get transaction history
# -------------------------------
def get_user_transactions(user_id):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT amount, transaction_type, merchant_category, location, card_type,
               hour, previous_transactions, is_foreign_transaction,
               result, status, normal_probability, fraud_probability, checked_at
        FROM transactions
        WHERE user_id = ?
        ORDER BY id DESC
    """, (user_id,))

    transactions = cursor.fetchall()
    conn.close()

    return transactions


def get_analysis_data(user_id):
    transactions = get_user_transactions(user_id)

    total_checked = len(transactions)
    fraud_count = sum(1 for t in transactions if t[9] == "fraud")
    normal_count = sum(1 for t in transactions if t[9] == "safe")

    recent_transactions = transactions[:7]
    recent_labels = []
    recent_fraud_scores = []

    for t in reversed(recent_transactions):
        recent_labels.append(t[12])
        recent_fraud_scores.append(t[11])

    return {
        "total_checked": total_checked,
        "fraud_count": fraud_count,
        "normal_count": normal_count,
        "recent_labels": recent_labels,
        "recent_fraud_scores": recent_fraud_scores
    }


# -------------------------------
# Login
# -------------------------------
@app.route("/")
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()

        conn.close()

        if user and check_password_hash(user[3], password):
            session["user_id"] = user[0]
            session["user_name"] = user[1]
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid email or password")

    return render_template("login.html")


# -------------------------------
# Register
# -------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        hashed_password = generate_password_hash(password)

        try:
            conn = sqlite3.connect("users.db")
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO users (name, email, password)
                VALUES (?, ?, ?)
            """, (name, email, hashed_password))

            conn.commit()
            conn.close()

            return redirect(url_for("login"))

        except sqlite3.IntegrityError:
            return render_template("register.html", error="Email already exists. Please login.")

    return render_template("register.html")


# -------------------------------
# Dashboard
# -------------------------------
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]

    transactions = get_user_transactions(user_id)
    analysis_data = get_analysis_data(user_id)

    return render_template(
        "dashboard.html",
        user_name=session["user_name"],
        transactions=transactions,
        analysis_data=analysis_data
    )


# -------------------------------
# Prediction
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session:
        return redirect(url_for("login"))

    try:
        amount = float(request.form["amount"])
        transaction_type = request.form["transaction_type"]
        merchant_category = request.form["merchant_category"]
        location = request.form["location"]
        card_type = request.form["card_type"]
        hour = int(request.form["hour"])
        previous_transactions = int(request.form["previous_transactions"])
        is_foreign_transaction = request.form["is_foreign_transaction"]

        input_data = pd.DataFrame([{
            "amount": amount,
            "transaction_type": transaction_type,
            "merchant_category": merchant_category,
            "location": location,
            "card_type": card_type,
            "hour": hour,
            "previous_transactions": previous_transactions,
            "is_foreign_transaction": is_foreign_transaction
        }])

        categorical_columns = [
            "transaction_type",
            "merchant_category",
            "location",
            "card_type",
            "is_foreign_transaction"
        ]

        for col in categorical_columns:
            input_data[col] = encoders[col].transform(input_data[col])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        normal_probability = round(probability[0] * 100, 2)
        fraud_probability = round(probability[1] * 100, 2)

        if prediction == 0:
            result = "NORMAL TRANSACTION"
            status = "safe"
            message = "This transaction looks safe. You can proceed."
        else:
            result = "FRAUDULENT TRANSACTION"
            status = "fraud"
            message = "Warning! This transaction seems suspicious. Please verify before proceeding."

        checked_at = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO transactions (
                user_id, amount, transaction_type, merchant_category, location,
                card_type, hour, previous_transactions, is_foreign_transaction,
                result, status, normal_probability, fraud_probability, checked_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session["user_id"], amount, transaction_type, merchant_category, location,
            card_type, hour, previous_transactions, is_foreign_transaction,
            result, status, normal_probability, fraud_probability, checked_at
        ))

        conn.commit()
        conn.close()

        transactions = get_user_transactions(session["user_id"])
        analysis_data = get_analysis_data(session["user_id"])

        return render_template(
            "dashboard.html",
            user_name=session["user_name"],
            result=result,
            status=status,
            message=message,
            normal_probability=normal_probability,
            fraud_probability=fraud_probability,
            transactions=transactions,
            analysis_data=analysis_data,
            active_tab="prediction"
        )

    except Exception as e:
        transactions = get_user_transactions(session["user_id"])
        analysis_data = get_analysis_data(session["user_id"])

        return render_template(
            "dashboard.html",
            user_name=session["user_name"],
            error="Something went wrong. Please check all input values.",
            transactions=transactions,
            analysis_data=analysis_data
        )


# -------------------------------
# Proceed / Cancel
# -------------------------------
@app.route("/proceed_transaction", methods=["POST"])
def proceed_transaction():
    if "user_id" not in session:
        return redirect(url_for("login"))

    transactions = get_user_transactions(session["user_id"])
    analysis_data = get_analysis_data(session["user_id"])

    return render_template(
        "dashboard.html",
        user_name=session["user_name"],
        success_message="Transaction proceeded successfully!",
        transactions=transactions,
        analysis_data=analysis_data
    )


@app.route("/cancel_transaction", methods=["POST"])
def cancel_transaction():
    if "user_id" not in session:
        return redirect(url_for("login"))

    transactions = get_user_transactions(session["user_id"])
    analysis_data = get_analysis_data(session["user_id"])

    return render_template(
        "dashboard.html",
        user_name=session["user_name"],
        cancel_message="Transaction has been cancelled successfully.",
        transactions=transactions,
        analysis_data=analysis_data
    )


# -------------------------------
# Logout
# -------------------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)