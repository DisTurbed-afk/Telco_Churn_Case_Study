# churn_prediction_dag.py
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'churn_prediction_dag',
    default_args=default_args,
    description='Airflow DAG to automate churn prediction model training and evaluation',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    catchup=False,
)


# Define task functions

def load_data():
    file_path = '/path/to/Telco_customer_churn.xlsx'
    df = pd.read_excel(file_path)
    df.to_pickle('/path/to/data.pkl')


def preprocess_data():
    df = pd.read_pickle('/path/to/data.pkl')
    df = df.drop(
        columns=['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long', 'Latitude', 'Longitude'])
    df = df.replace(' ', None)
    df.to_pickle('/path/to/preprocessed_data.pkl')


def feature_engineering():
    df = pd.read_pickle('/path/to/preprocessed_data.pkl')

    # User Descriptive Features
    df['is_male'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
    df['is_senior_citizen'] = df['Senior Citizen'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['has_partner'] = df['Partner'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['has_dependants'] = df['Dependents'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['tenure_months'] = df['Tenure Months']

    # Telco Features
    df['is_phone_service'] = df['Phone Service'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['has_multiple_lines'] = df['Multiple Lines'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['has_online_security'] = df['Online Security'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['has_online_backup'] = df['Online Backup'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['has_device_protection'] = df['Device Protection'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['has_tech_support'] = df['Tech Support'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['has_streaming_tv'] = df['Streaming TV'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['has_streaming_movies'] = df['Streaming Movies'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['has_dsl'] = df['Internet Service'].apply(lambda x: 1 if x == 'DSL' else 0)
    df['has_fiber_optic'] = df['Internet Service'].apply(lambda x: 1 if x == 'Fiber optic' else 0)

    # Payment Features
    df['monthly_charges'] = df['Monthly Charges']
    df['total_charges'] = df['Total Charges']
    df['is_paperless_billing'] = df['Paperless Billing'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['uses_bank_withdrawal'] = df['Payment Method'].apply(lambda x: 1 if x == 'Bank transfer (automatic)' else 0)
    df['uses_credit_card'] = df['Payment Method'].apply(lambda x: 1 if x == 'Credit card (automatic)' else 0)
    df['has_mailed_check'] = df['Payment Method'].apply(lambda x: 1 if x == 'Mailed check' else 0)
    df['uses_electronic_check'] = df['Payment Method'].apply(lambda x: 1 if x == 'Electronic check' else 0)
    df['is_month_to_month'] = df['Contract'].apply(lambda x: 1 if x == 'Month-to-month' else 0)
    df['is_one_year'] = df['Contract'].apply(lambda x: 1 if x == 'One year' else 0)
    df['is_two_year'] = df['Contract'].apply(lambda x: 1 if x == 'Two year' else 0)

    # Select features
    user_descriptive_features = ['is_male', 'is_senior_citizen', 'has_partner', 'has_dependants', 'tenure_months']
    telco_features = ['is_phone_service', 'has_multiple_lines', 'has_dsl', 'has_fiber_optic', 'has_online_security',
                      'has_online_backup', 'has_device_protection', 'has_tech_support', 'has_streaming_tv',
                      'has_streaming_movies']
    payment_features = ['monthly_charges', 'total_charges', 'is_paperless_billing', 'uses_bank_withdrawal',
                        'uses_credit_card', 'has_mailed_check', 'uses_electronic_check', 'is_month_to_month',
                        'is_one_year', 'is_two_year']
    selected_features = user_descriptive_features + telco_features + payment_features

    df = df[selected_features + ['Churn Value']]
    df = df.dropna(subset=selected_features)
    df = df.astype('float')
    df.to_pickle('/path/to/engineered_data.pkl')


def split_data():
    df = pd.read_pickle('/path/to/engineered_data.pkl')
    selected_features = df.columns[:-1]

    X = df[selected_features]
    y = df['Churn Value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Save split data
    X_train.to_pickle('/path/to/X_train.pkl')
    X_test.to_pickle('/path/to/X_test.pkl')
    y_train.to_pickle('/path/to/y_train.pkl')
    y_test.to_pickle('/path/to/y_test.pkl')


def remove_outliers():
    X_train = pd.read_pickle('/path/to/X_train.pkl')
    y_train = pd.read_pickle('/path/to/y_train.pkl')

    p999 = X_train.quantile(0.999)
    p001 = X_train.quantile(0.001)
    X_train = X_train[(X_train <= p999).all(axis=1)]
    X_train = X_train[(X_train >= p001).all(axis=1)]
    y_train = y_train.loc[X_train.index]

    # Save data without outliers
    X_train.to_pickle('/path/to/X_train_no_outliers.pkl')
    y_train.to_pickle('/path/to/y_train_no_outliers.pkl')


def normalize_data():
    X_train = pd.read_pickle('/path/to/X_train_no_outliers.pkl')
    X_test = pd.read_pickle('/path/to/X_test.pkl')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, '/path/to/scaler.pkl')
    joblib.dump(X_train, '/path/to/X_train_scaled.pkl')
    joblib.dump(X_test, '/path/to/X_test_scaled.pkl')


def train_models():
    X_train = joblib.load('/path/to/X_train_scaled.pkl')
    y_train = pd.read_pickle('/path/to/y_train_no_outliers.pkl')
    X_test = joblib.load('/path/to/X_test_scaled.pkl')
    y_test = pd.read_pickle('/path/to/y_test.pkl')

    # Logistic Regression
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, '/path/to/lr_model.pkl')

    # Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, '/path/to/rf_model.pkl')

    # XGBoost
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, '/path/to/xgb_model.pkl')


def evaluate_models():
    X_test = joblib.load('/path/to/X_test_scaled.pkl')
    y_test = pd.read_pickle('/path/to/y_test.pkl')

    def evaluate_model(y_true, y_pred, y_prob):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_prob)

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")
        print(f"ROC-AUC: {roc_auc:.2f}")

    # Logistic Regression
    lr_model = joblib.load('/path/to/lr_model.pkl')
    y_pred_lr = lr_model.predict(X_test)
    y_prob_lr = lr_model.predict_proba(X_test)[:, 1]
    print("Logistic Regression:")
    evaluate_model(y_test, y_pred_lr, y_prob_lr)

    # Random Forest
    rf_model = joblib.load('/path/to/rf_model.pkl')
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    print("\nRandom Forest:")
    evaluate_model(y_test, y_pred_rf, y_prob_rf)

    # XGBoost
    xgb_model = joblib.load('/path/to/xgb_model.pkl')
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    print("\nXGBoost:")
    evaluate_model(y_test, y_pred_xgb, y_prob_xgb)


# Define the tasks
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

feature_engineering_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_engineering,
    dag=dag,
)

split_data_task = PythonOperator(
    task_id='split_data',
    python_callable=split_data,
    dag=dag,
)

remove_outliers_task = PythonOperator(
    task_id='remove_outliers',
    python_callable=remove_outliers,
    dag=dag,
)

normalize_data_task = PythonOperator(
    task_id='normalize_data',
    python_callable=normalize_data,
    dag=dag,
)

train_models_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    dag=dag,
)

evaluate_models_task = PythonOperator(
    task_id='evaluate_models',
    python_callable=evaluate_models,
    dag=dag,
)

# Define the task dependencies
load_data_task >> preprocess_data_task >> feature_engineering_task >> split_data_task >> remove_outliers_task >> normalize_data_task >> train_models_task >> evaluate_models_task
