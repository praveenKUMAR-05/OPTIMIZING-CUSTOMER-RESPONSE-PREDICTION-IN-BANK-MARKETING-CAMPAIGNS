# bank_marketing_model.py (Final Updated Version)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier, plot_importance
import joblib
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from prometheus_client import start_http_server, Counter
import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---------------------------
# 1. DATA PREPARATION
# ---------------------------
def load_and_preprocess(filepath=r"C:\Users\hp-pc\Downloads\bank.csv"):
    """Load and preprocess raw data"""
    df = pd.read_csv(filepath)
    
    # Check for target column
    target_col = None
    possible_targets = ['response', 'subscribed', 'y', 'deposit', 'target']
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        raise ValueError("Could not find target column in dataset. Expected one of: " + str(possible_targets))
    
    # Clean data
    df = df[df['age'] <= 100]  # Remove outliers
    df.fillna({'balance': df['balance'].median()}, inplace=True)
    
    # Feature engineering
    df['age_balance_interaction'] = df['age'] * df['balance']
    df['has_loan'] = (df['housing'] == 'yes') | (df['loan'] == 'yes')
    
    # Handle month if exists
    if 'month' in df.columns:
        df['contact_month_sin'] = np.sin(2*np.pi*pd.to_datetime(df['month'], format='%b').dt.month/12)
    
    # Target encoding
    df[target_col] = df[target_col].map({'yes': 1, 'no': 0, '1': 1, '0': 0}).astype(int)
    df.rename(columns={target_col: 'response'}, inplace=True)
    
    return df

# ---------------------------
# 2. MODEL TRAINING
# ---------------------------
def train_model(df):
    """Train and optimize XGBoost model"""
    X = df.drop('response', axis=1)
    y = df['response']
    
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Create full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(random_state=42, enable_categorical=True))
    ])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Hyperparameter tuning
    params = {
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__n_estimators': [100, 200]
    }
    
    grid = GridSearchCV(pipeline, params, cv=3, scoring='roc_auc', error_score='raise')
    grid.fit(X_train, y_train)
    
    # Evaluate
    best_model = grid.best_estimator_
    y_pred = best_model.predict_proba(X_test)[:, 1]
    print(f"Best AUC: {roc_auc_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, best_model.predict(X_test)))
    
    # Feature importance
    plt.figure(figsize=(10, 6))
    plot_importance(best_model.named_steps['classifier'])
    plt.savefig('feature_importance.png')
    
    return best_model

# ---------------------------
# 3. DEPLOYMENT API
# ---------------------------
app = Flask(__name__)
model = None
API_REQUESTS = Counter('api_requests_total', 'Total API calls')
PREDICTION_SCORE = Counter('prediction_score_sum', 'Sum of prediction scores')

def get_expected_columns():
    """Return all columns expected by the model"""
    return [
        'age', 'job', 'marital', 'education', 'default', 
        'balance', 'housing', 'loan', 'contact', 'day', 
        'month', 'duration', 'campaign', 'pdays', 
        'previous', 'poutcome', 'age_balance_interaction',
        'has_loan', 'contact_month_sin'
    ]

def validate_input(data):
    """Validate API input"""
    required = ['age', 'balance', 'job', 'campaign', 'duration']
    if not all(k in data for k in required):
        raise ValueError(f"Missing required fields: {required}")
    if not 18 <= data['age'] <= 100:
        raise ValueError("Age must be between 18-100")

def prepare_input_data(data):
    """Prepare complete input dataframe with default values"""
    # Default values for optional columns
    defaults = {
        'marital': 'married',
        'education': 'secondary',
        'default': 'no',
        'housing': 'no',
        'loan': 'no',
        'contact': 'cellular',
        'day': 15,
        'month': 'may',
        'pdays': -1,
        'previous': 0,
        'poutcome': 'unknown'
    }
    
    # Create dataframe with all expected columns
    input_df = pd.DataFrame(columns=get_expected_columns())
    
    # Fill in provided values
    for col in data:
        if col in input_df.columns:
            input_df[col] = [data[col]]
    
    # Fill missing columns with defaults
    for col, default_val in defaults.items():
        if col not in data and col in input_df.columns:
            input_df[col] = [default_val]
    
    # Add engineered features
    input_df['age_balance_interaction'] = input_df['age'] * input_df['balance']
    input_df['has_loan'] = (input_df['housing'] == 'yes') | (input_df['loan'] == 'yes')
    
    if 'month' in input_df.columns:
        input_df['contact_month_sin'] = np.sin(2*np.pi*pd.to_datetime(input_df['month'], format='%b').dt.month/12)
    
    return input_df

@app.route('/predict', methods=['POST'])
def predict():
    API_REQUESTS.inc()
    try:
        data = request.get_json()
        validate_input(data)
        
        # Prepare complete input data
        input_df = prepare_input_data(data)
        
        # Predict
        proba = model.predict_proba(input_df)[0, 1]
        PREDICTION_SCORE.inc(proba)
        
        return jsonify({
            'probability': float(proba),
            'decision': 'target' if proba > 0.5 else 'do not target',
            'message': 'Success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Please check your input data',
            'required_fields': ['age', 'balance', 'job', 'campaign', 'duration'],
            'optional_fields': {
                'marital': ['married', 'single', 'divorced'],
                'education': ['primary', 'secondary', 'tertiary'],
                'default': ['yes', 'no'],
                'housing': ['yes', 'no'],
                'loan': ['yes', 'no'],
                'contact': ['cellular', 'telephone'],
                'month': ['jan', 'feb', 'mar', ..., 'dec'],
                'poutcome': ['success', 'failure', 'unknown']
            }
        }), 400

# ---------------------------
# 4. MONITORING
# ---------------------------
def setup_monitoring(port=9090):
    """Start Prometheus metrics server"""
    start_http_server(port)
    logging.info(f"Metrics server running on port {port}")

# ---------------------------
# MAIN EXECUTION
# ---------------------------
if __name__ == '__main__':
    # 1. Train and save model
    print("Training model...")
    df = load_and_preprocess()
    model = train_model(df)
    joblib.dump(model, 'model.pkl')
    
    # 2. Load for API
    model = joblib.load('model.pkl')
    
    # 3. Start monitoring
    setup_monitoring()
    
    # 4. Run API
    print("Starting API...")
    app.run(host='0.0.0.0', port=5000)