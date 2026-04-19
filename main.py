import os

from src.data_generator import generate_synthetic_employee_data, save_dataset
from src.ml_pipeline import (
    evaluate_model,
    predict_new_employee,
    save_metrics_report,
    save_model,
    save_sample_prediction,
    train_employee_performance_model,
)


def run_project() -> None:
    project_root = os.path.dirname(__file__)
    data_dir = os.path.join(project_root, 'data')
    outputs_dir = os.path.join(project_root, 'outputs')
    model_dir = os.path.join(project_root, 'models')
    images_dir = os.path.join(project_root, 'images')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    dataset_path = os.path.join(data_dir, 'synthetic_employee_data.csv')
    model_path = os.path.join(model_dir, 'employee_performance_model.joblib')
    metrics_path = os.path.join(outputs_dir, 'metrics_report.json')
    prediction_path = os.path.join(outputs_dir, 'sample_prediction.csv')

    print('Generating synthetic employee dataset...')
    df = generate_synthetic_employee_data(n_samples=1200)
    saved_dataset_path = save_dataset(df, dataset_path)
    print(f'Synthetic dataset saved to: {saved_dataset_path}')
    print(df.head(5).to_string(index=False))

    print('\nTraining model...')
    pipeline, X_train, X_test, y_train, y_test = train_employee_performance_model(df)
    saved_model_path = save_model(pipeline, model_path)
    print(f'Model saved to: {saved_model_path}')

    print('\nEvaluating model...')
    metrics = evaluate_model(pipeline, X_test, y_test, images_dir)
    saved_metrics_path = save_metrics_report(metrics, metrics_path)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Weighted F1-score: {metrics['weighted_f1']:.3f}")
    print(f'Metrics report saved to: {saved_metrics_path}')

    sample_employee = {
        'age': 29,
        'gender': 'Female',
        'education_level': 'Master',
        'department': 'IT',
        'years_experience': 6,
        'training_hours': 48,
        'last_performance_rating': 4,
        'avg_monthly_hours': 210,
        'salary_level': 'Medium',
        'promotion_last_5years': 0,
        'work_life_balance': 3,
        'overtime': 'Yes',
    }

    print('\nMaking sample prediction for a new employee:')
    print(sample_employee)
    predicted_label = predict_new_employee(pipeline, sample_employee)
    saved_prediction_path = save_sample_prediction(sample_employee, predicted_label, prediction_path)
    print(f'Predicted performance category: {predicted_label}')
    print(f'Sample prediction saved to: {saved_prediction_path}')

    print('\nProject completed. Check the outputs and images folders for results.')


if __name__ == '__main__':
    run_project()
