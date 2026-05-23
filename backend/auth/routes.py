#!/usr/bin/env python3
"""
Authentication Routes
======================
API endpoints for user registration, login, logout.
Includes demo user auto-creation.
"""

from flask import Blueprint, request, jsonify, make_response
from datetime import datetime, timezone
from .models import db, User
from .utils import generate_token, token_required

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

# -------------------------------------------------------
# DEMO USER SEEDING
# -------------------------------------------------------
DEMO_USERNAME = 'demo'
DEMO_PASSWORD = 'demo123'
DEMO_EMAIL    = 'demo@financeai.com'

def ensure_demo_user():
    """Create demo user if it doesn't exist yet."""
    try:
        existing = User.query.filter_by(username=DEMO_USERNAME).first()
        if not existing:
            demo = User(
                username=DEMO_USERNAME,
                email=DEMO_EMAIL,
                full_name='Demo User',
                monthly_income=75000.0,
                age=28,
                existing_savings=150000.0,
                debt_to_income=0.12,
                income_level='Medium'
            )
            demo.set_password(DEMO_PASSWORD)
            db.session.add(demo)
            db.session.commit()
            print("[FinanceAI] Demo user created successfully.")
    except Exception as e:
        print(f"[FinanceAI] Demo user creation error: {e}")


@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    if not data or not data.get('username') or not data.get('email') or not data.get('password'):
        return jsonify({'success': False, 'message': 'Username, email, and password are required'}), 400

    if len(data['password']) < 6:
        return jsonify({'success': False, 'message': 'Password must be at least 6 characters'}), 400

    if data['username'].lower() == DEMO_USERNAME:
        return jsonify({'success': False, 'message': 'Username "demo" is reserved'}), 409

    if User.query.filter_by(username=data['username']).first():
        return jsonify({'success': False, 'message': 'Username already taken'}), 409

    if User.query.filter_by(email=data['email']).first():
        return jsonify({'success': False, 'message': 'Email already registered'}), 409

    user = User(
        username=data['username'],
        email=data['email'],
        full_name=data.get('full_name', ''),
        monthly_income=float(data.get('monthly_income', 50000)),
        age=int(data.get('age', 25)),
        existing_savings=float(data.get('existing_savings', 10000)),
        debt_to_income=float(data.get('debt_to_income', 0.15)),
        income_level=data.get('income_level', 'Medium')
    )
    user.set_password(data['password'])
    db.session.add(user)
    db.session.commit()

    token = generate_token(user.id, user.username)

    response = make_response(jsonify({
        'success': True,
        'message': 'Login successful',
        'token': token,          # ← ADD THIS LINE
        'user': user.to_dict()
    }))
    response.set_cookie('token', token, httponly=True, max_age=7*24*60*60)
    return response


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()

    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'success': False, 'message': 'Username and password are required'}), 400

    user = User.query.filter_by(username=data['username']).first()

    if not user or not user.check_password(data['password']):
        return jsonify({'success': False, 'message': 'Invalid username or password'}), 401

    user.last_login = datetime.now(timezone.utc)
    db.session.commit()

    token = generate_token(user.id, user.username)

    response = make_response(jsonify({
        'success': True,
        'message': 'Login successful',
        'token': token,
        'user': user.to_dict()
    }))
    response.set_cookie('token', token, httponly=True, max_age=7*24*60*60)
    return response


@auth_bp.route('/logout', methods=['POST'])
def logout():
    response = make_response(jsonify({'success': True, 'message': 'Logged out successfully'}))
    response.set_cookie('token', '', expires=0)
    return response


@auth_bp.route('/me', methods=['GET'])
@token_required
def get_current_user():
    user = db.session.get(User, request.user_id)
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    return jsonify({'success': True, 'user': user.to_dict()})


@auth_bp.route('/profile', methods=['PUT'])
@token_required
def update_profile():
    user = db.session.get(User, request.user_id)
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404

    data = request.get_json()
    if 'full_name'        in data: user.full_name        = data['full_name']
    if 'monthly_income'   in data: user.monthly_income   = float(data['monthly_income'])
    if 'age'              in data: user.age               = int(data['age'])
    if 'existing_savings' in data: user.existing_savings = float(data['existing_savings'])
    if 'debt_to_income'   in data: user.debt_to_income   = float(data['debt_to_income'])
    if 'income_level'     in data: user.income_level     = data['income_level']

    db.session.commit()
    return jsonify({'success': True, 'message': 'Profile updated', 'user': user.to_dict()})
