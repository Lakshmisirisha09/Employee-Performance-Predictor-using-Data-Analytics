import os

import numpy as np
import pandas as pd


def generate_synthetic_employee_data(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic employee dataset for performance prediction."""
    np.random.seed(random_state)

    departments = ['Sales', 'HR', 'Finance', 'IT', 'Operations', 'Marketing']
    education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
    salary_levels = ['Low', 'Medium', 'High']
    genders = ['Male', 'Female', 'Other']
    work_life_balance = [1, 2, 3, 4]

    data = {
        'employee_id': np.arange(1001, 1001 + n_samples),
        'age': np.random.randint(22, 60, size=n_samples),
        'gender': np.random.choice(genders, size=n_samples, p=[0.48, 0.48, 0.04]),
        'education_level': np.random.choice(education_levels, size=n_samples, p=[0.2, 0.5, 0.25, 0.05]),
        'department': np.random.choice(departments, size=n_samples),
        'years_experience': np.random.randint(0, 25, size=n_samples),
        'training_hours': np.random.randint(10, 120, size=n_samples),
        'last_performance_rating': np.random.randint(1, 6, size=n_samples),
        'avg_monthly_hours': np.random.randint(140, 270, size=n_samples),
        'salary_level': np.random.choice(salary_levels, size=n_samples, p=[0.3, 0.5, 0.2]),
        'promotion_last_5years': np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]),
        'work_life_balance': np.random.choice(work_life_balance, size=n_samples, p=[0.15, 0.25, 0.35, 0.25]),
        'overtime': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.42, 0.58]),
    }

    df = pd.DataFrame(data)
    df['performance_score'] = _calculate_performance_score(df)
    df['performance_category'] = df['performance_score'].apply(_assign_performance_category)

    return df


def _calculate_performance_score(df: pd.DataFrame) -> pd.Series:
    """Create a performance score from synthetic employee features."""
    salary_map = {'Low': 0, 'Medium': 5, 'High': 10}
    overtime_map = {'Yes': -4, 'No': 0}

    score = (
        df['years_experience'] * 1.6
        + df['training_hours'] * 0.35
        + df['last_performance_rating'] * 7.5
        + df['avg_monthly_hours'] * 0.05
        + df['work_life_balance'] * 3
        + df['promotion_last_5years'] * 5
        + df['salary_level'].map(salary_map)
        + df['overtime'].map(overtime_map)
    )

    noise = np.random.normal(loc=0.0, scale=5.5, size=len(df))
    raw_score = score + noise
    performance_score = np.clip(raw_score, 20, 100).round(1)
    return performance_score


def _assign_performance_category(score: float) -> str:
    """Assign a categorical label based on the performance score."""
    if score >= 70:
        return 'High Performer'
    if score >= 55:
        return 'Solid Performer'
    return 'Needs Improvement'


def save_dataset(df: pd.DataFrame, path: str) -> str:
    """Save the synthetic dataset to CSV and fall back if the target is locked."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback_path = _latest_path(path)
        df.to_csv(fallback_path, index=False)
        return fallback_path


def _latest_path(path: str) -> str:
    root, ext = os.path.splitext(path)
    return f'{root}_latest{ext}'
