#!/usr/bin/env python3
"""
Authentication Models
=====================
User model with SQLAlchemy for database storage.
"""

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    full_name = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, default=True)

    # Financial profile
    monthly_income = db.Column(db.Float, default=5000.0)
    age = db.Column(db.Integer, default=30)
    existing_savings = db.Column(db.Float, default=10000.0)
    debt_to_income = db.Column(db.Float, default=0.15)
    income_level = db.Column(db.String(20), default='Medium')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'monthly_income': self.monthly_income,
            'age': self.age,
            'existing_savings': self.existing_savings,
            'debt_to_income': self.debt_to_income,
            'income_level': self.income_level
        }
