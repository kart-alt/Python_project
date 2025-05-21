import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from google.cloud import storage
import streamlit as st
import os
import joblib
from datetime import datetime, timedelta

# Set Seaborn style for nicer plots
sns.set_style('whitegrid')

# Function to load and preprocess the data
def load_and_preprocess_data(file_path):
    """
    Load the telecom customer data and perform initial preprocessing
    """
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Print basic info about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"Missing values:\n{missing_values[missing_values > 0]}")
    
    # Handle missing values (simple imputation for demonstration)
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Handling {missing_values.sum()} missing values")
        
    # For TotalCharges, use median of non-missing values
    if 'TotalCharges' in df.columns and df['TotalCharges'].isnull().sum() > 0:
        median_value = df['TotalCharges'].dropna().median()
        df['TotalCharges'] = df['TotalCharges'].fillna(median_value)
        print(f"Filled missing values in 'TotalCharges' with median: {median_value}")
    
    # Fill any remaining missing values
    df = df.fillna({
        # Add more columns as needed
    })
    
    # Convert 'TotalCharges' to numeric if it's not already
    if df['TotalCharges'].dtype == 'object':
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # Fill any NaN values that resulted from conversion
        df['TotalCharges'].fillna(0, inplace=True)
        print("Converted 'TotalCharges' from object to numeric type")
    
    # Map 'Churn' to binary values if it's not already
    if 'Churn' in df.columns:
        if df['Churn'].dtype == 'object':
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
            print("Converted 'Churn' to binary values")
    
    # Generate a 'CustomerSince' date for time-based analysis (if not present)
    if 'CustomerSince' not in df.columns and 'tenure' in df.columns:
        # Create a date column based on tenure (months)
        today = datetime.now()
        df['CustomerSince'] = df['tenure'].apply(lambda x: today - timedelta(days=int(x*30)))
        df['CustomerSince'] = pd.to_datetime(df['CustomerSince'])
        df['YearMonth'] = df['CustomerSince'].dt.strftime('%Y-%m')
    
    return df

# Function to create visualizations
def create_visualizations(df):
    """
    Create the required visualizations:
    1. Bar plot (churn rates by customer type)
    2. Line plot (churn trend over time)
    3. Heatmap (churn vs. services used)
    """
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    
    # 1. Bar plot: Churn rates by customer type (using Contract type)
    if 'Contract' in df.columns:
        contract_churn = df.groupby('Contract')['Churn'].mean().sort_values(ascending=False)
        contract_churn.plot(kind='bar', ax=axs[0], color='skyblue')
        axs[0].set_title('Churn Rate by Contract Type', fontsize=14)
        axs[0].set_ylabel('Churn Rate')
        axs[0].set_xlabel('Contract Type')
        axs[0].tick_params(axis='x', rotation=0)
        
        # Add percentage labels on top of bars
        for i, v in enumerate(contract_churn):
            axs[0].text(i, v + 0.01, f'{v:.1%}', ha='center')
    
    # 2. Line plot: Churn trend over time
    if 'YearMonth' in df.columns:
        time_churn = df.groupby('YearMonth')['Churn'].mean()
        time_churn.plot(ax=axs[1], marker='o', linestyle='-', color='green')
        axs[1].set_title('Churn Rate Trend Over Time', fontsize=14)
        axs[1].set_ylabel('Churn Rate')
        axs[1].set_xlabel('Month')
        axs[1].tick_params(axis='x', rotation=45)
    
    # 3. Heatmap: Churn vs services used
    # Select service columns (typical for telecom data)
    service_columns = [col for col in df.columns if col in [
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies'
    ]]
    
    if service_columns:
        # Calculate churn rate for each service
        service_churn = {}
        for service in service_columns:
            service_churn[service] = df.groupby(service)['Churn'].mean()
        
        # Convert to DataFrame for heatmap
        service_churn_df = pd.DataFrame(service_churn)
        
        # Create heatmap
        sns.heatmap(service_churn_df, annot=True, cmap='YlGnBu', fmt='.2f', ax=axs[2])
        axs[2].set_title('Churn Rate by Service Usage', fontsize=14)
    
    plt.tight_layout()
    
    # Save visualizations
    plt.savefig('telecom_churn_visuals.png', dpi=300, bbox_inches='tight')
    
    return fig

# Function to build and train the ML model
def build_and_train_model(df):
    """
    Build and train a Random Forest model for churn prediction
    """
    # Define features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove date columns from features
    date_columns = X.select_dtypes(include=['datetime64']).columns.tolist()
    for col in date_columns:
        if col in numerical_features:
            numerical_features.remove(col)
    
    # Also remove CustomerID or similar identifier columns
    exclude_cols = ['customerID', 'CustomerID', 'YearMonth', 'CustomerSince']
    categorical_features = [col for col in categorical_features if col not in exclude_cols]
    numerical_features = [col for col in numerical_features if col not in exclude_cols]
    
    print(f"Categorical features: {categorical_features}")
    print(f"Numerical features: {numerical_features}")
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Create pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train the model
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Save the model
    joblib.dump(pipeline, 'churn_prediction_model.pkl')
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate feature importance
    feature_names = (
        numerical_features +
        list(pipeline.named_steps['preprocessor']
             .transformers_[1][1]  # Get the OneHotEncoder
             .get_feature_names_out(categorical_features))
    )
    
    # Get feature importances from the random forest model
    importances = pipeline.named_steps['classifier'].feature_importances_
    
    # Create a DataFrame for feature importances
    if len(importances) == len(feature_names):
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
    else:
        print("Warning: Feature names length doesn't match importances length")
        feature_importance_df = None
        
    return pipeline, accuracy, cm, feature_importance_df

# Function to upload files to GCP bucket
def upload_to_gcp_bucket(bucket_name, source_file_path, destination_blob_name):
    """
    Upload a file to a GCP storage bucket
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_path)

    print(f"File {source_file_path} uploaded to {destination_blob_name}")

def main():
    # Load and preprocess data
    # For this example, we assume you have a local file or you can modify to download from Kaggle
    df = load_and_preprocess_data('telecom_customer_churn.csv')
    
    # Create visualizations
    fig = create_visualizations(df)
    
    # Build and train model
    model, accuracy, confusion_matrix, feature_importance = build_and_train_model(df)
    
    # Upload to GCP (uncomment and modify as needed)
    # upload_to_gcp_bucket('your-bucket-name', 'churn_prediction_model.pkl', 'models/churn_prediction_model.pkl')
    # upload_to_gcp_bucket('your-bucket-name', 'telecom_churn_visuals.png', 'visualizations/telecom_churn_visuals.png')

if __name__ == "__main__":
    main()