# Employee Performance Predictor using Data Analytics

A machine learning project that predicts employee performance categories based on various employee attributes and work metrics using a Random Forest classifier.

## Overview

This project leverages data analytics and machine learning to predict whether an employee will have **High**, **Medium**, or **Low** performance based on factors such as:
- Employee demographics (age, gender, education level)
- Work experience and training
- Department and salary level
- Work-life balance and overtime
- Historical performance ratings

## Features

✨ **Key Features:**
- **Synthetic Data Generation**: Creates realistic employee datasets for training
- **Data Preprocessing**: Handles categorical and numerical features with appropriate scaling and encoding
- **Model Training**: Trains a Random Forest classifier with optimized hyperparameters
- **Model Evaluation**: Comprehensive metrics including accuracy, F1-score, confusion matrix, and classification reports
- **Performance Visualization**: Generates visual reports of model performance
- **Model Persistence**: Saves trained model for future predictions
- **Sample Predictions**: Demonstrates predictions on new employee data

## Project Structure

```
Employee Performance Predictor using Data Analytics/
├── main.py                              # Main entry point for the project
├── src/
│   ├── data_generator.py               # Synthetic data generation utilities
│   └── ml_pipeline.py                  # ML model training and evaluation
├── data/
│   └── synthetic_employee_data.csv     # Generated training dataset
├── models/
│   └── employee_performance_model.joblib  # Trained ML model
├── outputs/
│   ├── metrics_report.json             # Model evaluation metrics
│   └── sample_prediction.csv           # Sample prediction results
├── images/                              # Visualization outputs
└── README.md                           # This file
```

## Installation

### Requirements
- Python 3.8+
- Libraries: pandas, numpy, scikit-learn, joblib, matplotlib, seaborn

### Setup

1. **Clone or download the project**
   ```bash
   cd "Employee Performance Predictor using Data Analytics"
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn joblib matplotlib seaborn
   ```

## Usage

### Running the Complete Pipeline

Execute the main script to generate data, train the model, and evaluate performance:

```bash
python main.py
```

This will:
1. Generate 1,200 synthetic employee records
2. Split data into training (80%) and testing (20%) sets
3. Train a Random Forest classifier
4. Evaluate model performance
5. Generate a sample prediction
6. Save all outputs (model, metrics, predictions)

### Project Output

The script generates the following outputs:

- **Model**: `models/employee_performance_model.joblib`
- **Metrics**: `outputs/metrics_report.json`
- **Sample Prediction**: `outputs/sample_prediction.csv`
- **Visualizations**: Various charts in `images/` directory

## Model Details

### Algorithm
- **Classifier**: Random Forest
- **Number of Estimators**: 120 trees
- **Max Depth**: 10 levels
- **Random State**: 42 (for reproducibility)

### Features Used

**Numerical Features** (Scaled with StandardScaler):
- Age
- Years of Experience
- Training Hours
- Last Performance Rating
- Average Monthly Hours
- Promotion in Last 5 Years
- Work-Life Balance Score

**Categorical Features** (One-Hot Encoded):
- Gender (Male, Female, Other)
- Education Level (High School, Bachelor, Master, PhD)
- Department (Sales, HR, Finance, IT, Operations, Marketing)
- Salary Level (Low, Medium, High)
- Overtime (Yes, No)

### Target Variable
- **Performance Category**: High, Medium, Low (predicted from performance score)

## Sample Results

The model achieves strong performance on the test set. Check `outputs/metrics_report.json` for detailed metrics including:
- Overall Accuracy
- Weighted F1-Score
- Per-class Precision, Recall, and F1-Score
- Confusion Matrix

## File Descriptions

| File | Description |
|------|-------------|
| `main.py` | Orchestrates the entire pipeline workflow |
| `data_generator.py` | Generates synthetic employee data with realistic distributions |
| `ml_pipeline.py` | Contains functions for training, evaluating, and using the model |
| `synthetic_employee_data.csv` | Training dataset (generated) |
| `employee_performance_model.joblib` | Serialized trained model |
| `metrics_report.json` | Detailed performance metrics |
| `sample_prediction.csv` | Example predictions on new employee data |

## How It Works

### 1. Data Generation
The `data_generator.py` module creates synthetic employee records with realistic distributions for demographic, professional, and performance-related features.

### 2. Data Preprocessing
The `ml_pipeline.py` module:
- Removes non-predictive features (employee_id)
- Scales numerical features using StandardScaler
- Encodes categorical features using OneHotEncoder
- Separates features from target variable

### 3. Model Training
- Splits data: 80% training, 20% testing
- Trains Random Forest classifier
- Saves the complete pipeline (preprocessing + classifier)

### 4. Evaluation & Visualization
- Calculates accuracy and weighted F1-score
- Generates confusion matrix
- Creates classification report with per-class metrics
- Produces visualizations for performance analysis

### 5. Making Predictions
- Loads the trained model
- Accepts new employee data
- Generates performance category predictions

## Example Prediction

```python
sample_employee = {
    'age': 29,
    'gender': 'Male',
    'education_level': 'Master',
    'department': 'IT',
    'years_experience': 5,
    'training_hours': 75,
    'last_performance_rating': 4,
    'avg_monthly_hours': 200,
    'salary_level': 'Medium',
    'promotion_last_5years': 1,
    'work_life_balance': 3,
    'overtime': 'Yes'
}
# Prediction: High Performance
```

## Future Improvements

- Hyperparameter tuning using Grid Search or Random Search
- Cross-validation for robust performance estimation
- Feature importance analysis
- Real employee data integration
- Web API for easy access to predictions
- Dashboard for visualization and monitoring



---

**Note**: This project uses synthetic data for demonstration purposes. For production use, integrate with real employee data while ensuring compliance with data privacy regulations.
