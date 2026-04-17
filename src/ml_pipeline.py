import os
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_employee_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Prepare features and target for model training."""
    df_clean = df.copy()

    df_clean.drop(columns=['employee_id'], inplace=True, errors='ignore')

    target = df_clean['performance_category']
    X = df_clean.drop(columns=['performance_category', 'performance_score'], errors='ignore')

    categorical_features = ['gender', 'education_level', 'department', 'salary_level', 'overtime']
    numeric_features = ['age', 'years_experience', 'training_hours', 'last_performance_rating', 'avg_monthly_hours', 'promotion_last_5years', 'work_life_balance']

    transformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ],
        remainder='drop',
    )

    return X, target, transformer


def build_model_pipeline(transformer: ColumnTransformer) -> Pipeline:
    """Build a scikit-learn pipeline with preprocessing and classifier."""
    rf = RandomForestClassifier(n_estimators=120, random_state=42, max_depth=10)
    pipeline = Pipeline([
        ('preprocessing', transformer),
        ('classifier', rf),
    ])
    return pipeline


def train_employee_performance_model(df: pd.DataFrame) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train the model and return the pipeline with splits."""
    X, y, transformer = preprocess_employee_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = build_model_pipeline(transformer)
    pipeline.fit(X_train, y_train)
    return pipeline, X_train, X_test, y_train, y_test


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, output_dir: str) -> dict:
    """Evaluate model performance and generate summary metrics and plots."""
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, output_dict=True)

    cm = confusion_matrix(y_test, y_pred, labels=['High Performer', 'Solid Performer', 'Needs Improvement'])
    plot_confusion_matrix(cm, ['High Performer', 'Solid Performer', 'Needs Improvement'], output_dir)

    metrics = {
        'accuracy': accuracy,
        'weighted_f1': f1,
        'report': report,
    }
    return metrics


def plot_confusion_matrix(cm, labels, output_dir: str) -> None:
    """Plot and save the confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()


def save_model(pipeline: Pipeline, path: str) -> None:
    """Save the trained model pipeline to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path: str) -> Pipeline:
    """Load a saved model pipeline."""
    return joblib.load(path)


def predict_new_employee(pipeline: Pipeline, sample: dict) -> str:
    """Predict performance category for a new employee sample."""
    df_sample = pd.DataFrame([sample])
    prediction = pipeline.predict(df_sample)[0]
    return prediction
