import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from google.cloud import storage
import os
import io
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define functions
@st.cache_data
def load_data(file_path):
    """Load and cache the dataset"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_model(model_path):
    """Load and cache the trained model"""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def load_from_gcp(bucket_name, blob_name):
    """Download a file from GCP bucket"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        content = blob.download_as_bytes()
        return content
    except Exception as e:
        st.error(f"Error downloading from GCP: {e}")
        return None

def make_prediction(model, input_data):
    """Make predictions using the trained model"""
    try:
        # Convert input data to DataFrame with one row
        input_df = pd.DataFrame([input_data])
        # Make prediction
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[:, 1][0]
        return prediction[0], probability
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    return plt

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # For pipelines, get the classifier step's feature_importances_
        importances = model.named_steps['classifier'].feature_importances_

    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 8))
    plt.title('Feature Importances')
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    return plt

# Main app
def main():
    st.title("Telecom Customer Churn Prediction Dashboard")

    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Prediction", "Data Analysis", "Model Details"])

    # GCP bucket and blob names
    bucket_name = st.sidebar.text_input("GCP Bucket Name", "telecom-churn-bucket")
    model_blob_name = "models/churn_prediction_model.pkl"
    data_blob_name = "data/telecom_customer_churn.csv"

    # Local paths fallback
    local_model_path = "churn_prediction_model.pkl"
    local_data_path = "telecom_customer_churn.csv"

    # Data source selection
    data_source = st.sidebar.radio("Data Source", ["Local File", "GCP Bucket"])

    # Load data and model
    if data_source == "Local File":
        data = load_data(local_data_path)
        model = load_model(local_model_path)
    else:
        data = None
        model = None
        if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
            data_content = load_from_gcp(bucket_name, data_blob_name)
            if data_content:
                data = pd.read_csv(io.BytesIO(data_content))
            model_content = load_from_gcp(bucket_name, model_blob_name)
            if model_content:
                model = joblib.load(io.BytesIO(model_content))
        else:
            st.sidebar.warning("GCP credentials not found. Using local files.")
            data = load_data(local_data_path)
            model = load_model(local_model_path)

    # Render selected page
    if page == "Dashboard":
        display_dashboard(data, model)
    elif page == "Prediction":
        display_prediction_page(data, model)
    elif page == "Data Analysis":
        display_data_analysis(data)
    elif page == "Model Details":
        display_model_details(data, model)

def display_dashboard(data, model):
    st.header("Customer Churn Dashboard")
    
    if data is None or data.empty:
        st.warning("No data available. Please check your data source.")
        return
    
    if 'Churn' not in data.columns:
        st.error("'Churn' column not found in data.")
        return
    
    # Convert 'Churn' to numeric if needed
    if not pd.api.types.is_numeric_dtype(data['Churn']):
        churn_map = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}
        data['Churn'] = data['Churn'].map(churn_map)
        data['Churn'] = pd.to_numeric(data['Churn'], errors='coerce').fillna(0).astype(float)
    
    churn_rate = data['Churn'].mean() * 100
    total_customers = len(data)
    churned_customers = data['Churn'].sum()
    retention_rate = 100 - churn_rate
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{int(churned_customers):,}")
    col3.metric("Churn Rate", f"{churn_rate:.2f}%")
    col4.metric("Retention Rate", f"{retention_rate:.2f}%")
    
    st.subheader("Churn Analysis Visualizations")

    try:
        image = Image.open('telecom_churn_visuals.png')
        st.image(image, caption="Churn Analysis Visualizations", use_column_width=True)
    except FileNotFoundError:
        col1, col2 = st.columns(2)
        with col1:
            if 'Contract' in data.columns:
                st.subheader("Churn by Contract Type")
                contract_churn = data.groupby('Contract')['Churn'].mean().sort_values(ascending=False)
                st.bar_chart(contract_churn)
            else:
                st.info("Column 'Contract' not found in data.")
        with col2:
            if 'MonthlyCharges' in data.columns and 'TotalCharges' in data.columns:
                st.subheader("Charges vs Churn")
                fig, ax = plt.subplots()
                sns.scatterplot(x='MonthlyCharges', y='TotalCharges', hue='Churn', data=data, ax=ax)
                st.pyplot(fig)
            else:
                st.info("Columns 'MonthlyCharges' and/or 'TotalCharges' not found in data.")
    
    if 'YearMonth' in data.columns:
        st.subheader("Churn Trend Over Time")
        time_churn = data.groupby('YearMonth')['Churn'].mean()
        st.line_chart(time_churn)
    else:
        st.info("Column 'YearMonth' not found in data for trend analysis.")



def display_prediction_page(data, model):
    st.header("Customer Churn Prediction")

    if data is None or model is None:
        st.warning("Data or model not available. Please check your data source.")
        return

    st.write("Enter customer information to predict churn probability")

    col1, col2 = st.columns(2)
    input_data = {}

    with col1:
        if 'gender' in data.columns:
            input_data['gender'] = st.selectbox("Gender", ['Male', 'Female'])
        if 'SeniorCitizen' in data.columns:
            input_data['SeniorCitizen'] = st.selectbox("Senior Citizen", [0, 1])
        if 'Partner' in data.columns:
            input_data['Partner'] = st.selectbox("Partner", ['Yes', 'No'])
        if 'Dependents' in data.columns:
            input_data['Dependents'] = st.selectbox("Dependents", ['Yes', 'No'])
        if 'tenure' in data.columns:
            input_data['tenure'] = st.slider("Tenure (months)", 0, 72, 12)
        if 'PhoneService' in data.columns:
            input_data['PhoneService'] = st.selectbox("Phone Service", ['Yes', 'No'])
        if 'MultipleLines' in data.columns:
            input_data['MultipleLines'] = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
        if 'InternetService' in data.columns:
            input_data['InternetService'] = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])

    with col2:
        if 'OnlineSecurity' in data.columns:
            input_data['OnlineSecurity'] = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
        if 'OnlineBackup' in data.columns:
            input_data['OnlineBackup'] = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
        if 'DeviceProtection' in data.columns:
            input_data['DeviceProtection'] = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
        if 'TechSupport' in data.columns:
            input_data['TechSupport'] = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
        if 'StreamingTV' in data.columns:
            input_data['StreamingTV'] = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
        if 'StreamingMovies' in data.columns:
            input_data['StreamingMovies'] = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
        if 'Contract' in data.columns:
            input_data['Contract'] = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
        if 'PaperlessBilling' in data.columns:
            input_data['PaperlessBilling'] = st.selectbox("Paperless Billing", ['Yes', 'No'])
        if 'PaymentMethod' in data.columns:
            input_data['PaymentMethod'] = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        if 'MonthlyCharges' in data.columns:
            input_data['MonthlyCharges'] = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
        if 'TotalCharges' in data.columns:
            input_data['TotalCharges'] = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)

    # Map categorical input to model expected format (example)
    def preprocess_input(input_dict):
        mapping_yes_no = {'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0}
        mapping_gender = {'Male': 1, 'Female': 0}
        mapping_multiple_lines = {'Yes': 1, 'No': 0, 'No phone service': 0}
        mapping_internet_service = {'DSL': 1, 'Fiber optic': 2, 'No': 0}
        mapping_contract = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        mapping_payment_method = {
            'Electronic check': 0,
            'Mailed check': 1,
            'Bank transfer (automatic)': 2,
            'Credit card (automatic)': 3
        }

        processed = input_dict.copy()
        # Map fields
        if 'gender' in processed:
            processed['gender'] = mapping_gender.get(processed['gender'], 0)
        for key in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies']:
            if key in processed:
                processed[key] = mapping_yes_no.get(processed[key], 0)
        if 'MultipleLines' in processed:
            processed['MultipleLines'] = mapping_multiple_lines.get(processed['MultipleLines'], 0)
        if 'InternetService' in processed:
            processed['InternetService'] = mapping_internet_service.get(processed['InternetService'], 0)
        if 'Contract' in processed:
            processed['Contract'] = mapping_contract.get(processed['Contract'], 0)
        if 'PaymentMethod' in processed:
            processed['PaymentMethod'] = mapping_payment_method.get(processed['PaymentMethod'], 0)

        return processed

    if st.button("Predict Churn"):
        processed_input = preprocess_input(input_data)
        prediction, probability = make_prediction(model, processed_input)
        if prediction is not None:
            result_text = "Customer will churn" if prediction == 1 else "Customer will NOT churn"
            st.success(f"Prediction: {result_text}")
            st.info(f"Churn Probability: {probability:.2%}")

def display_data_analysis(data):
    st.header("Data Analysis")

    if data is None:
        st.warning("No data available. Please check your data source.")
        return

    st.subheader("Raw Data")
    st.dataframe(data.head(50))

    st.subheader("Summary Statistics")
    st.write(data.describe())

    st.subheader("Data Info")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Plotting
    st.subheader("Churn Count")
    fig, ax = plt.subplots()
    sns.countplot(x='Churn', data=data, ax=ax)
    st.pyplot(fig)

    st.subheader("Monthly Charges Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data['MonthlyCharges'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Churn by Contract Type")
    if 'Contract' in data.columns:
        contract_churn = data.groupby('Contract')['Churn'].mean().sort_values(ascending=False)
        st.bar_chart(contract_churn)

def display_model_details(data, model):
    st.header("Model Performance and Details")

    if data is None or model is None:
        st.warning("Data or model not available. Please check your data source.")
        return

    # Prepare data for evaluation
    target_col = 'Churn'
    drop_cols = ['customerID', 'CustomerID']
    X = data.drop(columns=drop_cols + [target_col], errors='ignore')
    y = data[target_col]

    # Predict on full data
    y_pred = model.predict(X)

    # Accuracy
    acc = accuracy_score(y, y_pred)
    st.subheader("Accuracy")
    st.write(f"The model accuracy on the available dataset is **{acc:.2%}**.")

    # Classification report
    st.subheader("Classification Report")
    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # Confusion matrix
    st.subheader("Confusion Matrix")
    fig_cm = plot_confusion_matrix(y, y_pred)
    st.pyplot(fig_cm)

    # Feature importance
    st.subheader("Feature Importance")
    feature_names = X.columns.tolist()
    fig_fi = plot_feature_importance(model, feature_names)
    st.pyplot(fig_fi)

if __name__ == "__main__":
    main()
