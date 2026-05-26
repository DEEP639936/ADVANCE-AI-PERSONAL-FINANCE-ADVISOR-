#!/usr/bin/env python3
"""
Authentication Utilities
========================
JWT token generation and validation.
"""

import jwt
import datetime
import os
from functools import wraps
from flask import request, jsonify

SECRET_KEY = os.environ.get('SECRET_KEY', 'ai-finance-advisor-jwt-secret-2026')

def generate_token(user_id, username):
    """Generate JWT token for user."""
    payload = {
        'user_id': user_id,
        'username': username,
        'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=7),
        'iat': datetime.datetime.now(datetime.timezone.utc)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def decode_token(token):
    """Decode and validate JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def token_required(f):
    """Decorator to protect routes with JWT."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        # Check Authorization header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({'success': False, 'message': 'Invalid token format'}), 401

        # Check session/cookie
        if not token and 'token' in request.cookies:
            token = request.cookies.get('token')

        if not token:
            return jsonify({'success': False, 'message': 'Authentication required'}), 401

        payload = decode_token(token)
        if not payload:
            return jsonify({'success': False, 'message': 'Invalid or expired token'}), 401

        # Attach user info to request
        request.user_id = payload['user_id']
        request.username = payload['username']

        return f(*args, **kwargs)

    return decorated
