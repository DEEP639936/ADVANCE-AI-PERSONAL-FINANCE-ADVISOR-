#!/usr/bin/env python3
"""
AI Personal Finance Advisor - ML Model Training Pipeline v2
=============================================================
Dataset designed with realistic feature-category correlations.
This ensures models can learn meaningful patterns.
"""

import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_squared_error,
    r2_score, mean_absolute_error
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def generate_realistic_dataset(n_samples=3000, output_path=None):
    """
    Generate dataset where features strongly predict category.
    Each category has distinct, realistic patterns.
    """
    print(f"[INFO] Generating realistic dataset with {n_samples} samples...")

    # Category definitions with STRONG feature correlations
    # This makes the classification task realistic and learnable
    category_profiles = {
        'Food': {
            'amount_range': (5, 80),
            'amount_mean': 25, 'amount_std': 12,
            'recurring_prob': 0.75,
            'weekend_bias': 0.6,  # 60% on weekends
            'month_start_bias': 0.1,
            'month_end_bias': 0.1,
            'income_sensitivity': 0.3,  # Low income sensitivity
            'seasonal_mult': {'default': 1.0, 'holiday': 1.4, 'summer': 1.0},
            'freq_range': (5, 15),  # Very frequent
            'description_keywords_range': (2, 5)
        },
        'Transport': {
            'amount_range': (10, 150),
            'amount_mean': 45, 'amount_std': 20,
            'recurring_prob': 0.85,
            'weekend_bias': 0.15,  # Mostly weekdays
            'month_start_bias': 0.2,
            'month_end_bias': 0.2,
            'income_sensitivity': 0.4,
            'seasonal_mult': {'default': 1.0, 'holiday': 0.8, 'summer': 1.2},
            'freq_range': (8, 20),  # Very frequent (commute)
            'description_keywords_range': (1, 3)
        },
        'Entertainment': {
            'amount_range': (20, 200),
            'amount_mean': 65, 'amount_std': 35,
            'recurring_prob': 0.15,
            'weekend_bias': 0.75,  # Mostly weekends
            'month_start_bias': 0.05,
            'month_end_bias': 0.3,  # More at month end
            'income_sensitivity': 0.7,
            'seasonal_mult': {'default': 1.0, 'holiday': 1.5, 'summer': 1.4},
            'freq_range': (2, 6),  # Infrequent
            'description_keywords_range': (3, 7)
        },
        'Shopping': {
            'amount_range': (30, 500),
            'amount_mean': 130, 'amount_std': 80,
            'recurring_prob': 0.25,
            'weekend_bias': 0.55,
            'month_start_bias': 0.15,
            'month_end_bias': 0.4,  # Salary day shopping
            'income_sensitivity': 0.8,
            'seasonal_mult': {'default': 1.0, 'holiday': 1.8, 'summer': 1.1},
            'freq_range': (2, 5),
            'description_keywords_range': (4, 8)
        },
        'Healthcare': {
            'amount_range': (50, 800),
            'amount_mean': 180, 'amount_std': 180,
            'recurring_prob': 0.05,
            'weekend_bias': 0.2,  # Weekdays (clinics open)
            'month_start_bias': 0.1,
            'month_end_bias': 0.1,
            'income_sensitivity': 0.5,
            'seasonal_mult': {'default': 1.0, 'holiday': 0.9, 'summer': 1.0},
            'freq_range': (1, 3),  # Rare
            'description_keywords_range': (2, 4)
        },
        'Utilities': {
            'amount_range': (100, 400),
            'amount_mean': 220, 'amount_std': 45,
            'recurring_prob': 0.98,
            'weekend_bias': 0.1,
            'month_start_bias': 0.7,  # Bills at month start
            'month_end_bias': 0.1,
            'income_sensitivity': 0.2,
            'seasonal_mult': {'default': 1.0, 'holiday': 1.0, 'summer': 1.3},
            'freq_range': (1, 2),  # Monthly
            'description_keywords_range': (1, 2)
        },
        'Education': {
            'amount_range': (100, 1000),
            'amount_mean': 320, 'amount_std': 250,
            'recurring_prob': 0.35,
            'weekend_bias': 0.3,
            'month_start_bias': 0.5,  # Semester start
            'month_end_bias': 0.1,
            'income_sensitivity': 0.9,
            'seasonal_mult': {'default': 1.0, 'holiday': 0.7, 'summer': 0.6},
            'freq_range': (1, 4),
            'description_keywords_range': (3, 6)
        },
        'Travel': {
            'amount_range': (200, 2000),
            'amount_mean': 550, 'amount_std': 450,
            'recurring_prob': 0.02,
            'weekend_bias': 0.4,
            'month_start_bias': 0.1,
            'month_end_bias': 0.2,
            'income_sensitivity': 0.95,
            'seasonal_mult': {'default': 1.0, 'holiday': 1.6, 'summer': 1.5},
            'freq_range': (1, 2),  # Very rare
            'description_keywords_range': (5, 9)
        },
        'Others': {
            'amount_range': (5, 200),
            'amount_mean': 40, 'amount_std': 35,
            'recurring_prob': 0.4,
            'weekend_bias': 0.4,
            'month_start_bias': 0.2,
            'month_end_bias': 0.2,
            'income_sensitivity': 0.5,
            'seasonal_mult': {'default': 1.0, 'holiday': 1.1, 'summer': 1.0},
            'freq_range': (2, 8),
            'description_keywords_range': (1, 6)
        }
    }

    # Target distribution
    cat_names = list(category_profiles.keys())
    cat_weights = [0.20, 0.15, 0.12, 0.15, 0.05, 0.08, 0.05, 0.05, 0.15]

    data = []
    for i in range(n_samples):
        # Pick category
        category = np.random.choice(cat_names, p=cat_weights)
        profile = category_profiles[category]

        # Generate date with category-specific patterns
        days_back = np.random.randint(0, 365)
        date = datetime(2025, 1, 1) + timedelta(days=int(days_back))
        month = date.month

        # Determine weekend based on category bias
        is_weekend = 1 if date.weekday() >= 5 else 0
        # Adjust: if category prefers weekends, force weekend more often
        if np.random.random() < profile['weekend_bias']:
            # Try to make it weekend
            if date.weekday() < 5:
                # 50% chance to shift to nearby weekend
                if np.random.random() < 0.5:
                    days_shift = 5 - date.weekday() if date.weekday() < 5 else 0
                    date = date + timedelta(days=days_shift)
                    is_weekend = 1

        day_of_month = date.day

        # Determine recurring
        is_recurring = 1 if np.random.random() < profile['recurring_prob'] else 0

        # Month-start/end bias
        if np.random.random() < profile['month_start_bias']:
            day_of_month = np.random.randint(1, 6)
        elif np.random.random() < profile['month_end_bias']:
            day_of_month = np.random.randint(25, 29)

        # Income level (affects amount)
        income_level = np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2])
        income_multipliers = {'Low': 2000, 'Medium': 5000, 'High': 10000}
        monthly_income = income_multipliers[income_level] * (1 + np.random.normal(0, 0.1))

        # Amount based on category profile AND income
        base_amount = np.random.lognormal(mean=np.log(profile['amount_mean']), sigma=0.4)
        income_effect = 1 + (monthly_income / 5000 - 1) * profile['income_sensitivity'] * 0.3
        amount = base_amount * income_effect
        amount = max(profile['amount_range'][0], min(profile['amount_range'][1], amount))

        # Seasonal multiplier
        seasonal_mult = profile['seasonal_mult']['default']
        if month in [11, 12]:
            seasonal_mult = profile['seasonal_mult']['holiday']
        elif month in [6, 7, 8]:
            seasonal_mult = profile['seasonal_mult']['summer']
        amount *= seasonal_mult

        # Age
        age = np.random.randint(18, 65)

        # Financial profile
        existing_savings = monthly_income * np.random.exponential(2.0)
        debt_to_income = np.random.beta(2, 5) * 0.5

        # Previous month (correlated with current behavior)
        prev_month_expenses = monthly_income * np.random.beta(3, 2) * 0.8
        prev_month_savings = monthly_income - prev_month_expenses

        # Expense frequency (category-specific)
        expense_frequency = np.random.randint(profile['freq_range'][0], profile['freq_range'][1] + 1)

        # Description keywords
        description_keywords = np.random.randint(
            profile['description_keywords_range'][0], 
            profile['description_keywords_range'][1] + 1
        )

        data.append({
            'transaction_id': f'TXN_{i:06d}',
            'date': date.strftime('%Y-%m-%d'),
            'amount': round(amount, 2),
            'category': category,
            'description': f'{category.lower()}_transaction_{i}',
            'income_level': income_level,
            'monthly_income': round(monthly_income, 2),
            'age': age,
            'existing_savings': round(existing_savings, 2),
            'debt_to_income': round(debt_to_income, 4),
            'day_of_month': day_of_month,
            'month': month,
            'is_weekend': is_weekend,
            'is_month_start': 1 if day_of_month <= 5 else 0,
            'is_month_end': 1 if day_of_month >= 25 else 0,
            'is_recurring': is_recurring,
            'seasonal_multiplier': round(seasonal_mult, 2),
            'prev_month_expenses': round(prev_month_expenses, 2),
            'prev_month_savings': round(prev_month_savings, 2),
            'expense_frequency': expense_frequency,
            'description_keywords': description_keywords
        })

    df = pd.DataFrame(data)
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"[INFO] Dataset saved: {output_path}")
    return df

def engineer_features(df):
    """Create advanced features for ML models."""
    df = df.copy()
    df['amount_to_income_ratio'] = df['amount'] / df['monthly_income']
    df['savings_to_income_ratio'] = df['existing_savings'] / df['monthly_income']
    df['expense_stability'] = df['prev_month_expenses'] / (df['monthly_income'] * 0.8)
    df['savings_rate'] = df['prev_month_savings'] / df['monthly_income']
    df['is_salary_day'] = ((df['day_of_month'] >= 1) & (df['day_of_month'] <= 3)).astype(int)
    df['week_of_month'] = (df['day_of_month'] - 1) // 7 + 1
    df['high_expense_flag'] = (df['amount'] > df['monthly_income'] * 0.1).astype(int)
    df['low_savings_flag'] = (df['savings_to_income_ratio'] < 0.1).astype(int)
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 100], 
                             labels=['Young', 'EarlyCareer', 'MidCareer', 'Senior'])

    le_income = LabelEncoder()
    df['income_level_encoded'] = le_income.fit_transform(df['income_level'])
    le_age = LabelEncoder()
    df['age_group_encoded'] = le_age.fit_transform(df['age_group'].astype(str))

    return df, le_income, le_age

def calculate_financial_health_score(df):
    """Calculate comprehensive financial health score (0-100)."""
    scores = []
    for _, row in df.iterrows():
        score = 50
        savings_rate = row['savings_rate']
        if savings_rate >= 0.3: score += 25
        elif savings_rate >= 0.2: score += 20
        elif savings_rate >= 0.1: score += 15
        elif savings_rate >= 0.05: score += 10
        else: score += 5

        dti = row['debt_to_income']
        if dti <= 0.1: score += 20
        elif dti <= 0.2: score += 15
        elif dti <= 0.36: score += 10
        elif dti <= 0.5: score += 5

        ef_months = row['existing_savings'] / (row['monthly_income'] * 0.8)
        if ef_months >= 6: score += 20
        elif ef_months >= 3: score += 15
        elif ef_months >= 1: score += 10
        else: score += 5

        stability = row['expense_stability']
        if 0.6 <= stability <= 0.9: score += 15
        elif 0.5 <= stability <= 1.0: score += 10
        else: score += 5

        age = row['age']
        expected_savings = age * 0.5
        if ef_months >= expected_savings: score += 10
        elif ef_months >= expected_savings * 0.5: score += 7
        else: score += 3

        score += np.random.normal(0, 2)
        score = np.clip(score, 0, 100)
        scores.append(round(score, 2))
    return scores

def calculate_budget_recommendation(df):
    """Calculate recommended monthly budget percentage."""
    recommendations = []
    for _, row in df.iterrows():
        base_budget = 0.70
        savings_rate = row['savings_rate']
        if savings_rate < 0.1: base_budget -= 0.05
        elif savings_rate > 0.3: base_budget += 0.05

        dti = row['debt_to_income']
        if dti > 0.36: base_budget -= 0.10

        age = row['age']
        if age > 50: base_budget -= 0.05

        base_budget += np.random.normal(0, 0.02)
        base_budget = np.clip(base_budget, 0.50, 0.85)
        recommendations.append(round(base_budget * 100, 2))
    return recommendations

def train_all_models(df):
    """Train all ML models."""
    print("\n[STEP 1] Engineering features...")
    df, le_income, le_age = engineer_features(df)

    print("[STEP 2] Calculating targets...")
    df['financial_health_score'] = calculate_financial_health_score(df)
    df['recommended_budget_pct'] = calculate_budget_recommendation(df)

    feature_cols_classifier = [
        'amount', 'monthly_income', 'age', 'existing_savings', 'debt_to_income',
        'day_of_month', 'month', 'is_weekend', 'is_month_start', 'is_month_end',
        'is_recurring', 'seasonal_multiplier', 'prev_month_expenses',
        'prev_month_savings', 'expense_frequency', 'description_keywords',
        'amount_to_income_ratio', 'savings_to_income_ratio', 'expense_stability',
        'savings_rate', 'is_salary_day', 'week_of_month', 'high_expense_flag',
        'low_savings_flag', 'income_level_encoded', 'age_group_encoded'
    ]

    feature_cols_regressor = [
        'monthly_income', 'age', 'existing_savings', 'debt_to_income',
        'is_recurring', 'prev_month_expenses', 'prev_month_savings',
        'expense_frequency', 'savings_to_income_ratio', 'expense_stability',
        'savings_rate', 'low_savings_flag', 'income_level_encoded', 'age_group_encoded'
    ]

    feature_cols_budget = [
        'monthly_income', 'age', 'existing_savings', 'debt_to_income',
        'prev_month_expenses', 'prev_month_savings', 'expense_frequency',
        'savings_rate', 'expense_stability', 'low_savings_flag',
        'income_level_encoded', 'age_group_encoded'
    ]

    le_category = LabelEncoder()
    y_class = le_category.fit_transform(df['category'])

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=RANDOM_STATE, stratify=y_class)

    X_train = df.iloc[train_idx]
    X_test = df.iloc[test_idx]
    y_train_class = y_class[train_idx]
    y_test_class = y_class[test_idx]
    y_train_health = df['financial_health_score'].values[train_idx]
    y_test_health = df['financial_health_score'].values[test_idx]
    y_train_budget = df['recommended_budget_pct'].values[train_idx]
    y_test_budget = df['recommended_budget_pct'].values[test_idx]

    print(f"[INFO] Train: {len(X_train)}, Test: {len(X_test)}")

    # === MODEL 1: Expense Classifier ===
    print("\n[MODEL 1] Training Expense Category Classifier...")
    scaler_class = StandardScaler()
    X_train_c = scaler_class.fit_transform(X_train[feature_cols_classifier])
    X_test_c = scaler_class.transform(X_test[feature_cols_classifier])

    clf = RandomForestClassifier(n_estimators=300, max_depth=25, min_samples_split=2, 
                                  min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train_c, y_train_class)
    pred_c = clf.predict(X_test_c)

    acc = accuracy_score(y_test_class, pred_c)
    prec = precision_score(y_test_class, pred_c, average='weighted', zero_division=0)
    rec = recall_score(y_test_class, pred_c, average='weighted', zero_division=0)
    f1 = f1_score(y_test_class, pred_c, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test_class, pred_c)

    print(f"[RESULT] Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"[RESULT] Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_class, pred_c, target_names=le_category.classes_, zero_division=0))

    class_metrics = {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1,
        'confusion_matrix': cm.tolist()
    }

    # Feature importance
    importances = clf.feature_importances_
    feat_imp = sorted(zip(feature_cols_classifier, importances), key=lambda x: x[1], reverse=True)
    print("\n[FEATURE IMPORTANCE] Top 10:")
    for feat, imp in feat_imp[:10]:
        print(f"  {feat}: {imp:.4f}")

    # === MODEL 2: Health Score ===
    print("\n[MODEL 2] Training Financial Health Score Regressor...")
    scaler_health = StandardScaler()
    X_train_h = scaler_health.fit_transform(X_train[feature_cols_regressor])
    X_test_h = scaler_health.transform(X_test[feature_cols_regressor])

    reg_health = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1)
    reg_health.fit(X_train_h, y_train_health)
    pred_h = reg_health.predict(X_test_h)

    r2_h = r2_score(y_test_health, pred_h)
    rmse_h = np.sqrt(mean_squared_error(y_test_health, pred_h))
    mae_h = mean_absolute_error(y_test_health, pred_h)

    print(f"[RESULT] R²: {r2_h:.4f}, RMSE: {rmse_h:.4f}, MAE: {mae_h:.4f}")
    health_metrics = {'r2_score': r2_h, 'rmse': rmse_h, 'mae': mae_h}

    # === MODEL 3: Budget ===
    print("\n[MODEL 3] Training Budget Recommendation Model...")
    scaler_budget = StandardScaler()
    X_train_b = scaler_budget.fit_transform(X_train[feature_cols_budget])
    X_test_b = scaler_budget.transform(X_test[feature_cols_budget])

    reg_budget = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.08, random_state=RANDOM_STATE)
    reg_budget.fit(X_train_b, y_train_budget)
    pred_b = reg_budget.predict(X_test_b)

    r2_b = r2_score(y_test_budget, pred_b)
    rmse_b = np.sqrt(mean_squared_error(y_test_budget, pred_b))
    mae_b = mean_absolute_error(y_test_budget, pred_b)

    print(f"[RESULT] R²: {r2_b:.4f}, RMSE: {rmse_b:.4f}, MAE: {mae_b:.4f}")
    budget_metrics = {'r2_score': r2_b, 'rmse': rmse_b, 'mae': mae_b}

    # Save
    print("\n[INFO] Saving models...")
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model')
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(clf, os.path.join(model_dir, 'expense_classifier.pkl'))
    joblib.dump(scaler_class, os.path.join(model_dir, 'scaler_classifier.pkl'))
    joblib.dump(le_category, os.path.join(model_dir, 'label_encoder_category.pkl'))

    joblib.dump(reg_health, os.path.join(model_dir, 'health_score_model.pkl'))
    joblib.dump(scaler_health, os.path.join(model_dir, 'scaler_health.pkl'))

    joblib.dump(reg_budget, os.path.join(model_dir, 'budget_model.pkl'))
    joblib.dump(scaler_budget, os.path.join(model_dir, 'scaler_budget.pkl'))

    joblib.dump(le_income, os.path.join(model_dir, 'label_encoder_income.pkl'))
    joblib.dump(le_age, os.path.join(model_dir, 'label_encoder_age.pkl'))

    joblib.dump(feature_cols_classifier, os.path.join(model_dir, 'feature_cols_classifier.pkl'))
    joblib.dump(feature_cols_regressor, os.path.join(model_dir, 'feature_cols_health.pkl'))
    joblib.dump(feature_cols_budget, os.path.join(model_dir, 'feature_cols_budget.pkl'))

    metrics_summary = {
        'expense_classifier': {'model_type': 'Random Forest', 'metrics': class_metrics, 'features': feature_cols_classifier},
        'health_score': {'model_type': 'Random Forest', 'metrics': health_metrics, 'features': feature_cols_regressor},
        'budget_recommender': {'model_type': 'Gradient Boosting', 'metrics': budget_metrics, 'features': feature_cols_budget}
    }
    joblib.dump(metrics_summary, os.path.join(model_dir, 'training_metrics.pkl'))

    print("[SUCCESS] All models saved!")

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"1. Expense Classifier: Accuracy={acc*100:.2f}%, F1={f1:.4f}")
    print(f"2. Health Score: R²={r2_h:.4f}, RMSE={rmse_h:.2f}")
    print(f"3. Budget Model: R²={r2_b:.4f}, RMSE={rmse_b:.2f}")
    print("=" * 60)

    return metrics_summary

def main():
    print("=" * 60)
    print("AI PERSONAL FINANCE ADVISOR - ML TRAINING v2")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, '..', '..', 'dataset', 'sample_transactions.csv')

    df = generate_realistic_dataset(n_samples=3000, output_path=dataset_path)
    print(f"[INFO] Dataset: {df.shape}")
    print(df['category'].value_counts())

    metrics = train_all_models(df)
    return metrics

if __name__ == '__main__':
    main()
