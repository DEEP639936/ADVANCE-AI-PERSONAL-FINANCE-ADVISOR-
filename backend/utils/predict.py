#!/usr/bin/env python3
"""
ML Prediction Engine (FIXED v2.1)
=================================
- Graceful handling when model files are missing
- Auto-trains models if .pkl files not found
- Fixed INR currency symbol in savings suggestion
- Better error messages for debugging
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# MODEL LOADING (lazy singleton pattern)
# ---------------------------------------------------------------------------

_MODELS = {}
_SCALERS = {}
_ENCODERS = {}
_FEATURE_COLS = {}
_MODELS_LOADED = False
_MODEL_ERROR = None

def _get_model_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')

def _list_missing_models():
    """Check which model files are missing."""
    mdir = _get_model_dir()
    required = {
        'expense_classifier.pkl': 'classifier',
        'health_score_model.pkl': 'health',
        'budget_model.pkl': 'budget',
        'scaler_classifier.pkl': 'scaler_classifier',
        'scaler_health.pkl': 'scaler_health',
        'scaler_budget.pkl': 'scaler_budget',
        'label_encoder_category.pkl': 'le_category',
        'label_encoder_income.pkl': 'le_income',
        'label_encoder_age.pkl': 'le_age',
        'feature_cols_classifier.pkl': 'features_classifier',
        'feature_cols_health.pkl': 'features_health',
        'feature_cols_budget.pkl': 'features_budget',
    }
    missing = []
    for fname, key in required.items():
        if not os.path.exists(os.path.join(mdir, fname)):
            missing.append(fname)
    return missing

def _auto_train_models():
    """Automatically run training if models are missing."""
    train_script = os.path.join(_get_model_dir(), 'train_model.py')
    if os.path.exists(train_script):
        print("[INFO] Model files missing. Auto-training models...")
        # Execute training in a subprocess to avoid import conflicts
        import subprocess
        result = subprocess.run(
            [sys.executable, train_script],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(train_script)
        )
        if result.returncode != 0:
            print("[ERROR] Auto-training failed:", result.stderr)
            return False
        print("[INFO] Auto-training completed successfully.")
        return True
    return False

def _load_all():
    """Load all models, scalers, encoders, and feature lists once."""
    global _MODELS, _SCALERS, _ENCODERS, _FEATURE_COLS, _MODELS_LOADED, _MODEL_ERROR
    if _MODELS_LOADED:
        return True

    mdir = _get_model_dir()
    os.makedirs(mdir, exist_ok=True)

    # Check for missing files
    missing = _list_missing_models()
    if missing:
        print(f"[WARN] Missing model files: {missing}")
        # Try auto-training
        if _auto_train_models():
            missing = _list_missing_models()
        if missing:
            _MODEL_ERROR = f"Missing model files: {', '.join(missing)}. Please run: cd backend/model && python train_model.py"
            print(f"[ERROR] {_MODEL_ERROR}")
            return False

    try:
        # Models
        _MODELS['classifier'] = joblib.load(os.path.join(mdir, 'expense_classifier.pkl'))
        _MODELS['health'] = joblib.load(os.path.join(mdir, 'health_score_model.pkl'))
        _MODELS['budget'] = joblib.load(os.path.join(mdir, 'budget_model.pkl'))

        # Scalers
        _SCALERS['classifier'] = joblib.load(os.path.join(mdir, 'scaler_classifier.pkl'))
        _SCALERS['health'] = joblib.load(os.path.join(mdir, 'scaler_health.pkl'))
        _SCALERS['budget'] = joblib.load(os.path.join(mdir, 'scaler_budget.pkl'))

        # Encoders
        _ENCODERS['category'] = joblib.load(os.path.join(mdir, 'label_encoder_category.pkl'))
        _ENCODERS['income'] = joblib.load(os.path.join(mdir, 'label_encoder_income.pkl'))
        _ENCODERS['age'] = joblib.load(os.path.join(mdir, 'label_encoder_age.pkl'))

        # Feature lists
        _FEATURE_COLS['classifier'] = joblib.load(os.path.join(mdir, 'feature_cols_classifier.pkl'))
        _FEATURE_COLS['health'] = joblib.load(os.path.join(mdir, 'feature_cols_health.pkl'))
        _FEATURE_COLS['budget'] = joblib.load(os.path.join(mdir, 'feature_cols_budget.pkl'))

        _MODELS_LOADED = True
        _MODEL_ERROR = None
        print("[INFO] All ML models loaded successfully.")
        return True
    except Exception as e:
        _MODEL_ERROR = f"Failed to load models: {str(e)}"
        print(f"[ERROR] {_MODEL_ERROR}")
        return False

def _ensure_loaded():
    if not _MODELS_LOADED:
        success = _load_all()
        if not success:
            raise RuntimeError(_MODEL_ERROR or "ML models not available. Please train models first.")

# ---------------------------------------------------------------------------
# FEATURE ENGINEERING (must match training exactly)
# ---------------------------------------------------------------------------

def _engineer_single_row(raw_data):
    """
    Convert raw user input into the exact feature vector used during training.
    raw_data: dict with keys like amount, monthly_income, age, etc.
    """
    d = raw_data.copy()

    # Parse date if provided
    if 'date' in d and d['date']:
        try:
            dt = datetime.strptime(str(d['date']), '%Y-%m-%d')
        except ValueError:
            dt = datetime.now()
    else:
        dt = datetime.now()

    day_of_month = d.get('day_of_month', dt.day)
    month = d.get('month', dt.month)
    is_weekend = d.get('is_weekend', 1 if dt.weekday() >= 5 else 0)
    is_month_start = 1 if day_of_month <= 5 else 0
    is_month_end = 1 if day_of_month >= 25 else 0
    is_salary_day = 1 if day_of_month <= 3 else 0
    week_of_month = (day_of_month - 1) // 7 + 1

    # Income level encoding
    income_level = d.get('income_level', 'Medium')
    le_income = _ENCODERS['income']
    # Handle unseen labels gracefully
    if income_level not in le_income.classes_:
        income_level = le_income.classes_[0]
    income_level_encoded = le_income.transform([income_level])[0]

    # Age group encoding
    age = int(d.get('age', 30))
    age_group = pd.cut([age], bins=[0, 25, 35, 50, 100], labels=['Young', 'EarlyCareer', 'MidCareer', 'Senior'])[0]
    age_group_str = str(age_group)
    le_age = _ENCODERS['age']
    # Handle unseen labels gracefully
    if age_group_str not in le_age.classes_:
        age_group_str = le_age.classes_[0]
    age_group_encoded = le_age.transform([age_group_str])[0]

    # Ratios
    monthly_income = float(d.get('monthly_income', 5000))
    amount = float(d.get('amount', 50))
    existing_savings = float(d.get('existing_savings', d.get('current_savings', 5000)))
    debt_to_income = float(d.get('debt_to_income', 0.15))
    prev_month_expenses = float(d.get('prev_month_expenses', monthly_income * 0.7))
    prev_month_savings = monthly_income - prev_month_expenses

    amount_to_income_ratio = amount / monthly_income if monthly_income > 0 else 0
    savings_to_income_ratio = existing_savings / monthly_income if monthly_income > 0 else 0
    expense_stability = prev_month_expenses / (monthly_income * 0.8) if monthly_income > 0 else 0
    savings_rate = prev_month_savings / monthly_income if monthly_income > 0 else 0

    high_expense_flag = 1 if amount > monthly_income * 0.1 else 0
    low_savings_flag = 1 if savings_to_income_ratio < 0.1 else 0

    # Build feature dict
    features = {
        'amount': amount,
        'monthly_income': monthly_income,
        'age': age,
        'existing_savings': existing_savings,
        'debt_to_income': debt_to_income,
        'day_of_month': day_of_month,
        'month': month,
        'is_weekend': is_weekend,
        'is_month_start': is_month_start,
        'is_month_end': is_month_end,
        'is_recurring': int(d.get('is_recurring', 0)),
        'seasonal_multiplier': float(d.get('seasonal_multiplier', 1.0)),
        'prev_month_expenses': prev_month_expenses,
        'prev_month_savings': prev_month_savings,
        'expense_frequency': int(d.get('expense_frequency', 3)),
        'description_keywords': int(d.get('description_keywords', 3)),
        'amount_to_income_ratio': amount_to_income_ratio,
        'savings_to_income_ratio': savings_to_income_ratio,
        'expense_stability': expense_stability,
        'savings_rate': savings_rate,
        'is_salary_day': is_salary_day,
        'week_of_month': week_of_month,
        'high_expense_flag': high_expense_flag,
        'low_savings_flag': low_savings_flag,
        'income_level_encoded': income_level_encoded,
        'age_group_encoded': age_group_encoded
    }

    return features

# ---------------------------------------------------------------------------
# PREDICTION FUNCTIONS
# ---------------------------------------------------------------------------

def predict_category(raw_data):
    """Predict expense category from transaction details."""
    _ensure_loaded()
    features = _engineer_single_row(raw_data)

    # Build DataFrame with exact feature order
    X = pd.DataFrame([{k: features[k] for k in _FEATURE_COLS['classifier']}])
    X_scaled = _SCALERS['classifier'].transform(X)

    pred = _MODELS['classifier'].predict(X_scaled)[0]
    proba = _MODELS['classifier'].predict_proba(X_scaled)[0]

    category = _ENCODERS['category'].inverse_transform([pred])[0]

    # Get top-3 probabilities
    top3_idx = np.argsort(proba)[-3:][::-1]
    top3 = [
        {'category': _ENCODERS['category'].inverse_transform([idx])[0], 'probability': round(float(proba[idx]), 4)}
        for idx in top3_idx
    ]

    return {
        'predicted_category': category,
        'confidence': round(float(proba[pred]), 4),
        'top_3_predictions': top3
    }

def predict_health_score(raw_data):
    """Predict financial health score (0-100)."""
    _ensure_loaded()
    features = _engineer_single_row(raw_data)

    X = pd.DataFrame([{k: features[k] for k in _FEATURE_COLS['health']}])
    X_scaled = _SCALERS['health'].transform(X)

    score = float(_MODELS['health'].predict(X_scaled)[0])
    score = np.clip(score, 0, 100)

    # Determine health status
    if score >= 80:
        status = 'Excellent'
        color = 'success'
    elif score >= 60:
        status = 'Good'
        color = 'info'
    elif score >= 40:
        status = 'Fair'
        color = 'warning'
    else:
        status = 'Poor'
        color = 'danger'

    return {
        'health_score': round(score, 2),
        'status': status,
        'color': color,
        'recommendations': _generate_health_recommendations(score, features)
    }

def predict_budget(raw_data):
    """Predict recommended budget percentage."""
    _ensure_loaded()
    features = _engineer_single_row(raw_data)

    X = pd.DataFrame([{k: features[k] for k in _FEATURE_COLS['budget']}])
    X_scaled = _SCALERS['budget'].transform(X)

    budget_pct = float(_MODELS['budget'].predict(X_scaled)[0])
    budget_pct = np.clip(budget_pct, 50, 85)

    monthly_income = features['monthly_income']
    recommended_budget = round(monthly_income * (budget_pct / 100), 2)
    recommended_savings = round(monthly_income - recommended_budget, 2)

    return {
        'recommended_budget_percentage': round(budget_pct, 2),
        'recommended_monthly_budget': recommended_budget,
        'recommended_monthly_savings': recommended_savings,
        'savings_rate_target': round((100 - budget_pct), 2)
    }

def predict_all(raw_data):
    """Run all predictions in one call."""
    return {
        'category_prediction': predict_category(raw_data),
        'health_score': predict_health_score(raw_data),
        'budget_recommendation': predict_budget(raw_data)
    }

# ---------------------------------------------------------------------------
# ADVICE GENERATION
# ---------------------------------------------------------------------------

def _generate_health_recommendations(score, features):
    """Generate personalized financial advice based on score and features."""
    advice = []

    if score < 40:
        advice.append("Your financial health needs immediate attention. Consider creating an emergency fund.")
        advice.append("Reduce discretionary spending and focus on debt repayment.")
    elif score < 60:
        advice.append("Your finances are stable but have room for improvement. Increase your savings rate.")
        advice.append("Review recurring expenses and identify areas to cut costs.")
    elif score < 80:
        advice.append("Good financial health! Consider diversifying investments.")
        advice.append("Maintain your current savings discipline and plan for long-term goals.")
    else:
        advice.append("Excellent financial health! You're on track for financial independence.")
        advice.append("Consider advanced wealth-building strategies like index fund investing.")

    # Feature-specific advice
    if features['savings_rate'] < 0.1:
        advice.append("Your savings rate is below 10%. Aim for at least 20% of income.")

    if features['debt_to_income'] > 0.36:
        advice.append("Your debt-to-income ratio is high. Prioritize paying down high-interest debt.")

    if features['low_savings_flag']:
        advice.append("You have less than 1 month of expenses saved. Build a 3-6 month emergency fund.")

    if features['high_expense_flag']:
        advice.append("This expense represents >10% of your monthly income. Review if necessary.")

    return advice

def generate_savings_suggestion(monthly_income, current_savings, target_months=6):
    """Generate a savings plan to reach emergency fund target."""
    monthly_expenses = monthly_income * 0.7
    target_amount = monthly_expenses * target_months
    gap = max(0, target_amount - current_savings)

    if gap <= 0:
        return {
            'target_emergency_fund': round(target_amount, 2),
            'current_savings': round(current_savings, 2),
            'gap': 0,
            'monthly_savings_needed': 0,
            'months_to_target': 0,
            'message': "You have reached your emergency fund goal!"
        }

    # Suggest saving 20% of income
    suggested_monthly = monthly_income * 0.20
    months_needed = int(np.ceil(gap / suggested_monthly))

    # FIXED: Use INR symbol ₹ instead of $
    return {
        'target_emergency_fund': round(target_amount, 2),
        'current_savings': round(current_savings, 2),
        'gap': round(gap, 2),
        'monthly_savings_needed': round(suggested_monthly, 2),
        'months_to_target': months_needed,
        'message': f"Save ₹{suggested_monthly:.2f}/month to reach your goal in {months_needed} months."
    }