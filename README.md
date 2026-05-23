# AI Personal Finance Advisor v2.0

> **Premium Fintech Web Application** | Flask + scikit-learn + JWT Auth + Dark/Light Theme

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What's New in v2.0

- **Premium Fintech UI** - Dark navy + cyan theme with glassmorphism
- **Dark/Light Mode** - Toggle with localStorage persistence
- **JWT Authentication** - Login, Register, Logout with secure tokens
- **User Profiles** - Personal + financial data storage
- **Protected Routes** - All features require authentication
- **Toast Notifications** - Modern feedback system
- **Responsive Design** - Mobile, tablet, desktop optimized
- **Animated Charts** - Chart.js with theme-aware colors

---

## Features

| Feature | Description | Status |
|---------|-------------|--------|
| **JWT Auth** | Secure login/register with hashed passwords | ✅ |
| **Dark/Light Theme** | Toggle with smooth transitions | ✅ |
| **Expense Tracking** | CRUD transactions per user | ✅ |
| **Category Prediction** | Random Forest (82.83% accuracy) | ✅ |
| **Health Score** | 0-100 ML-powered assessment | ✅ |
| **Budget AI** | Gradient Boosting recommendations | ✅ |
| **Dashboard** | Interactive charts & analytics | ✅ |
| **CSV Upload** | Bulk transaction import | ✅ |
| **Responsive** | All screen sizes | ✅ |

---

## Tech Stack

**Backend:** Flask 3.0, SQLAlchemy, JWT, bcrypt, scikit-learn 1.5.2
**Frontend:** Bootstrap 5, Chart.js, Vanilla JS, CSS3 Variables
**ML:** Random Forest, Gradient Boosting, 26 engineered features

---

## Quick Start

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Visit: http://127.0.0.1:5000

---

## Project Structure

```
AI-PERSONAL-FINANCE-ADVISOR/
├── backend/
│   ├── app.py              # Main Flask app
│   ├── requirements.txt    # Dependencies
│   ├── auth/
│   │   ├── models.py       # User model (SQLAlchemy)
│   │   ├── utils.py        # JWT helpers
│   │   └── routes.py       # Login/Register API
│   ├── model/
│   │   ├── train_model.py  # ML training
│   │   └── *.pkl           # Trained models
│   └── utils/
│       └── predict.py      # ML inference
├── frontend/
│   ├── templates/
│   │   ├── base.html       # Layout with theme toggle
│   │   ├── index.html      # Landing page
│   │   ├── auth/
│   │   │   ├── login.html  # Premium login
│   │   │   └── register.html
│   │   ├── dashboard.html  # Analytics dashboard
│   │   ├── transactions.html
│   │   ├── predict.html    # AI predictions
│   │   ├── analytics.html
│   │   ├── upload.html
│   │   └── profile.html
│   └── static/
│       ├── css/style.css   # Premium theme CSS
│       └── js/app.js       # Theme + Auth + API
├── dataset/
│   └── sample_transactions.csv
└── README.md
```

---

## ML Metrics (Real)

| Model | Metric | Value |
|-------|--------|-------|
| Expense Classifier | Accuracy | **82.83%** |
| Health Score | R² | **0.52** |
| Budget Recommender | R² | **0.67** |

---

## Interview Points

1. **3-Model Architecture** - Different ML approaches for different tasks
2. **Feature Engineering** - 26 features from raw transaction data
3. **Realistic Metrics** - Honest 82% accuracy (not faked 95%+)
4. **Full-Stack Auth** - JWT tokens, bcrypt hashing, protected routes
5. **Theme System** - CSS variables with localStorage persistence
6. **Production Ready** - Modular code, error handling, responsive UI

---

## License

MIT License
