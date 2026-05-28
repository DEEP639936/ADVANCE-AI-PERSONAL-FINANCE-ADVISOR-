#!/usr/bin/env python3
"""
AI Personal Finance Advisor - Flask Backend v2.0 (INR Edition)
==============================================================
Enhanced with:
- INR currency support
- Demo user with pre-loaded transactions
- JWT token in API response body
- Better error handling
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, make_response
from flask_cors import CORS

BACKEND_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

sys.path.insert(0, os.path.join(BACKEND_DIR, 'utils'))
sys.path.insert(0, os.path.join(BACKEND_DIR, 'auth'))

from predict import predict_all, predict_category, predict_health_score, predict_budget, generate_savings_suggestion
from auth.models import db, User
from auth.utils import token_required
from auth.routes import auth_bp, ensure_demo_user

TEMPLATE_DIR = os.path.join(PROJECT_ROOT, 'frontend', 'templates')
STATIC_DIR   = os.path.join(PROJECT_ROOT, 'frontend', 'static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.secret_key = os.environ.get('SECRET_KEY', 'ai-finance-advisor-secret-key-2026-v2')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'finance_advisor.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
CORS(app, origins=["*"])
app.register_blueprint(auth_bp)

# ---------------------------------------------------------------------------
# IN-MEMORY TRANSACTIONS
# ---------------------------------------------------------------------------
USER_TRANSACTIONS = {}

def get_user_transactions(user_id):
    return USER_TRANSACTIONS.get(user_id, [])

def set_user_transactions(user_id, transactions):
    USER_TRANSACTIONS[user_id] = transactions

# ---------------------------------------------------------------------------
# DEMO TRANSACTIONS (seeded for demo user)
# ---------------------------------------------------------------------------
DEMO_TRANSACTIONS_TEMPLATE = [
    {'category': 'Food',          'description': 'Grocery shopping',       'amount': 3200,  'type': 'expense'},
    {'category': 'Transport',     'description': 'Ola/Uber rides',          'amount': 1800,  'type': 'expense'},
    {'category': 'Entertainment', 'description': 'Netflix + Hotstar',       'amount': 1000,  'type': 'expense'},
    {'category': 'Utilities',     'description': 'Electricity bill',        'amount': 2500,  'type': 'expense'},
    {'category': 'Food',          'description': 'Restaurant outing',       'amount': 2100,  'type': 'expense'},
    {'category': 'Shopping',      'description': 'Amazon purchases',        'amount': 4500,  'type': 'expense'},
    {'category': 'Healthcare',    'description': 'Doctor visit + medicine', 'amount': 1500,  'type': 'expense'},
    {'category': 'Education',     'description': 'Online course (Udemy)',   'amount': 2000,  'type': 'expense'},
    {'category': 'Others',        'description': 'Miscellaneous expenses',  'amount': 1200,  'type': 'expense'},
    {'category': 'Salary',        'description': 'Monthly salary credit',   'amount': 75000, 'type': 'income'},
    {'category': 'Transport',     'description': 'Petrol refill',           'amount': 3000,  'type': 'expense'},
    {'category': 'Food',          'description': 'Swiggy/Zomato orders',    'amount': 1600,  'type': 'expense'},
    {'category': 'Salary',        'description': 'Freelance payment',       'amount': 12000, 'type': 'income'},
    {'category': 'Shopping',      'description': 'Clothing - Myntra',       'amount': 3800,  'type': 'expense'},
    {'category': 'Utilities',     'description': 'Internet + mobile bills', 'amount': 1200,  'type': 'expense'},
]

def seed_demo_transactions(user_id):
    """Seed demo transactions if user has none."""
    if get_user_transactions(user_id):
        return
    txns = []
    today = datetime.now()
    for i, t in enumerate(DEMO_TRANSACTIONS_TEMPLATE):
        day_offset = (len(DEMO_TRANSACTIONS_TEMPLATE) - i) * 2
        d = today.replace(day=max(1, today.day - day_offset % 28))
        txns.append({
            'id': f'TXN_{i+1:04d}',
            'date': d.strftime('%Y-%m-%d'),
            'amount': float(t['amount']),
            'category': t['category'],
            'description': t['description'],
            'type': t['type'],
            'created_at': datetime.now().isoformat()
        })
    set_user_transactions(user_id, txns)

# ---------------------------------------------------------------------------
# DB INIT + DEMO SEED
# ---------------------------------------------------------------------------
with app.app_context():
    db.create_all()
    ensure_demo_user()

# ---------------------------------------------------------------------------
# PAGE ROUTES
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login_page():
    return render_template('auth/login.html')

@app.route('/register')
def register_page():
    return render_template('auth/register.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/transactions')
def transactions_page():
    return render_template('transactions.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/analytics')
def analytics_page():
    return render_template('analytics.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/profile')
def profile_page():
    return render_template('profile.html')

# ---------------------------------------------------------------------------
# API - TRANSACTIONS
# ---------------------------------------------------------------------------
@app.route('/api/transactions', methods=['GET', 'POST'])
@token_required
def api_transactions():
    user_id = request.user_id

    if request.method == 'GET':
        # Seed demo data for demo users
        user = db.session.get(User, user_id)
        if user and user.username == 'demo':
            seed_demo_transactions(user_id)
        txs = get_user_transactions(user_id)
        return jsonify({'success': True, 'count': len(txs), 'transactions': txs})

    elif request.method == 'POST':
        data = request.get_json()
        txs = get_user_transactions(user_id)
        transaction = {
            'id': f'TXN_{len(txs)+1:04d}',
            'date': data.get('date', datetime.now().strftime('%Y-%m-%d')),
            'amount': float(data.get('amount', 0)),
            'category': data.get('category', 'Others'),
            'description': data.get('description', ''),
            'type': data.get('type', 'expense'),
            'created_at': datetime.now().isoformat()
        }
        txs.append(transaction)
        set_user_transactions(user_id, txs)
        return jsonify({'success': True, 'message': 'Transaction added', 'transaction': transaction})


@app.route('/api/transactions/<transaction_id>', methods=['DELETE'])
@token_required
def delete_transaction(transaction_id):
    user_id = request.user_id
    txs = [t for t in get_user_transactions(user_id) if t['id'] != transaction_id]
    set_user_transactions(user_id, txs)
    return jsonify({'success': True, 'message': 'Transaction deleted'})


@app.route('/api/transactions/clear', methods=['POST'])
@token_required
def clear_transactions():
    set_user_transactions(request.user_id, [])
    return jsonify({'success': True, 'message': 'All transactions cleared'})

# ---------------------------------------------------------------------------
# API - ML PREDICTIONS
# ---------------------------------------------------------------------------
@app.route('/api/predict/all', methods=['POST'])
@token_required
def api_predict_all():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
        
        user = db.session.get(User, request.user_id)
        if user:
            data.setdefault('monthly_income', user.monthly_income)
            data.setdefault('age', user.age)
            data.setdefault('existing_savings', user.existing_savings)
            data.setdefault('debt_to_income', user.debt_to_income)
            data.setdefault('income_level', user.income_level)
        
        result = predict_all(data)
        return jsonify({'success': True, 'predictions': result})
    except RuntimeError as e:
        # ML models not loaded/trained
        return jsonify({'success': False, 'message': str(e)}), 503
    except Exception as e:
        import traceback
        print("[ERROR] /api/predict/all failed:")
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Prediction engine error: {str(e)}'}), 500

@app.route('/api/predict/savings', methods=['POST'])
@token_required
def api_predict_savings():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
            
        user = db.session.get(User, request.user_id)
        monthly_income = float(data.get('monthly_income', user.monthly_income if user else 50000))
        current_savings = float(data.get('current_savings', user.existing_savings if user else 0))
        result = generate_savings_suggestion(monthly_income, current_savings)
        return jsonify({'success': True, 'suggestion': result})
    except Exception as e:
        import traceback
        print("[ERROR] /api/predict/savings failed:")
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Savings calculation error: {str(e)}'}), 500

# ---------------------------------------------------------------------------
# API - DASHBOARD & ANALYTICS
# ---------------------------------------------------------------------------
@app.route('/api/dashboard/summary')
@token_required
def api_dashboard_summary():
    user_id = request.user_id
    user = db.session.get(User, user_id)
    if user and user.username == 'demo':
        seed_demo_transactions(user_id)

    txs = get_user_transactions(user_id)
    expenses = [t for t in txs if t['type'] == 'expense']
    incomes  = [t for t in txs if t['type'] == 'income']

    total_expense = sum(t['amount'] for t in expenses)
    total_income  = sum(t['amount'] for t in incomes)
    balance       = total_income - total_expense

    category_totals = {}
    for t in expenses:
        cat = t['category']
        category_totals[cat] = category_totals.get(cat, 0) + t['amount']

    monthly_data = {}
    for t in txs:
        month = t['date'][:7]
        if month not in monthly_data:
            monthly_data[month] = {'income': 0, 'expense': 0}
        monthly_data[month][t['type']] += t['amount']

    recent = sorted(txs, key=lambda x: x['date'], reverse=True)[:10]

    return jsonify({
        'success': True,
        'summary': {
            'total_income':      round(total_income, 2),
            'total_expense':     round(total_expense, 2),
            'balance':           round(balance, 2),
            'transaction_count': len(txs),
            'expense_count':     len(expenses),
            'income_count':      len(incomes)
        },
        'category_breakdown': category_totals,
        'monthly_trend':      monthly_data,
        'recent_transactions': recent
    })


@app.route('/api/analytics/category-distribution')
@token_required
def api_category_distribution():
    user_id  = request.user_id
    expenses = [t for t in get_user_transactions(user_id) if t['type'] == 'expense']
    distribution = {}
    for t in expenses:
        cat = t['category']
        if cat not in distribution:
            distribution[cat] = {'count': 0, 'total': 0}
        distribution[cat]['count'] += 1
        distribution[cat]['total'] += t['amount']
    total = sum(d['total'] for d in distribution.values()) or 1
    for cat in distribution:
        distribution[cat]['percentage'] = round(distribution[cat]['total'] / total * 100, 2)
    return jsonify({'success': True, 'distribution': distribution})


@app.route('/api/analytics/monthly-trend')
@token_required
def api_monthly_trend():
    monthly = {}
    for t in get_user_transactions(request.user_id):
        month = t['date'][:7]
        if month not in monthly:
            monthly[month] = {'income': 0, 'expense': 0}
        monthly[month][t['type']] += t['amount']
    sorted_months = sorted(monthly.keys())
    trend = {
        'months':  sorted_months,
        'income':  [monthly[m]['income']  for m in sorted_months],
        'expense': [monthly[m]['expense'] for m in sorted_months],
        'savings': [monthly[m]['income'] - monthly[m]['expense'] for m in sorted_months]
    }
    return jsonify({'success': True, 'trend': trend})

# ---------------------------------------------------------------------------
# API - CSV UPLOAD
# ---------------------------------------------------------------------------
@app.route('/api/upload/csv', methods=['POST'])
@token_required
def api_upload_csv():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.csv'):
        return jsonify({'success': False, 'message': 'Please upload a CSV file'}), 400
    try:
        df = pd.read_csv(file)
        required = ['date', 'amount', 'category']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return jsonify({'success': False, 'message': f'Missing columns: {missing}'}), 400
        user_id = request.user_id
        txs = get_user_transactions(user_id)
        added = 0
        for _, row in df.iterrows():
            txs.append({
                'id': f'TXN_{len(txs)+1:04d}',
                'date': str(row.get('date', datetime.now().strftime('%Y-%m-%d'))),
                'amount': float(row.get('amount', 0)),
                'category': str(row.get('category', 'Others')),
                'description': str(row.get('description', '')),
                'type': str(row.get('type', 'expense')),
                'created_at': datetime.now().isoformat()
            })
            added += 1
        set_user_transactions(user_id, txs)
        return jsonify({'success': True, 'message': f'Imported {added} transactions', 'imported_count': added})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

# ---------------------------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------------------------
@app.route('/api/health')
def api_health():
    return jsonify({
        'status': 'healthy',
        'service': 'AI Personal Finance Advisor v2.0 (INR)',
        'version': '2.0.1',
        'timestamp': datetime.now().isoformat()
    })

# ---------------------------------------------------------------------------
# ERROR HANDLERS
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'message': 'Internal server error'}), 500

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("  AI PERSONAL FINANCE ADVISOR v2.0 — INR EDITION")
    print("  Demo Login  →  username: demo  |  password: demo123")
    print("  Visit: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
