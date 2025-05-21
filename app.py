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
        # Convert input data to DataFrame
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
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = model.named_steps['classifier'].feature_importances_
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Create a bar chart
    plt.figure(figsize=(10, 8))
    plt.title('Feature Importances')
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    return plt

# Main app
def main():
    st.title("Telecom Customer Churn Prediction Dashboard")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Prediction", "Data Analysis", "Model Details"])
    
    # Example GCP bucket details
    bucket_name = st.sidebar.text_input("GCP Bucket Name", "telecom-churn-bucket")
    model_blob_name = "models/churn_prediction_model.pkl"
    data_blob_name = "data/telecom_customer_churn.csv"
    
    # Local paths (use these if not using GCP)
    local_model_path = "churn_prediction_model.pkl"
    local_data_path = "telecom_customer_churn.csv"
    
    # Choose data source
    data_source = st.sidebar.radio("Data Source", ["Local File", "GCP Bucket"])
    
    # Load data and model based on source
    if data_source == "Local File":
        data = load_data(local_data_path)
        model = load_model(local_model_path)
    else:
        # Initialize data and model as None
        data = None
        model = None
        
        # Check if GCP credentials are available
        if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
            # Download data from GCP
            data_content = load_from_gcp(bucket_name, data_blob_name)
            if data_content:
                data = pd.read_csv(io.BytesIO(data_content))
            
            # Download model from GCP
            model_content = load_from_gcp(bucket_name, model_blob_name)
            if model_content:
                model = joblib.load(io.BytesIO(model_content))
        else:
            st.sidebar.warning("GCP credentials not found. Using local files.")
            data = load_data(local_data_path)
            model = load_model(local_model_path)
    
    # Display appropriate page based on selection
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
    
    if data is None:
        st.warning("No data available. Please check your data source.")
        return
        
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics
    churn_rate = data['Churn'].mean() * 100
    total_customers = len(data)
    churned_customers = data['Churn'].sum()
    retention_rate = 100 - churn_rate
    
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{churned_customers:,}")
    col3.metric("Churn Rate", f"{churn_rate:.2f}%")
    col4.metric("Retention Rate", f"{retention_rate:.2f}%")
    
    # Visualizations
    st.subheader("Churn Analysis")
    
    # Display visualizations from the main code
    try:
        image = Image.open('telecom_churn_visuals.png')
        st.image(image, caption="Churn Analysis Visualizations", use_column_width=True)
    except:
        # If image not found, create simplified visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Contract' in data.columns:
                st.subheader("Churn by Contract Type")
                contract_churn = data.groupby('Contract')['Churn'].mean().sort_values(ascending=False)
                st.bar_chart(contract_churn)
        
        with col2:
            if 'MonthlyCharges' in data.columns and 'TotalCharges' in data.columns:
                st.subheader("Charges vs Churn")
                fig, ax = plt.subplots()
                sns.scatterplot(x='MonthlyCharges', y='TotalCharges', hue='Churn', data=data, ax=ax)
                st.pyplot(fig)
    
    # Recent churn trends (if date data available)
    if 'YearMonth' in data.columns:
        st.subheader("Churn Trend Over Time")
        time_churn = data.groupby('YearMonth')['Churn'].mean()
        st.line_chart(time_churn)

def display_prediction_page(data, model):
    st.header("Customer Churn Prediction")
    
    if data is None or model is None:
        st.warning("Data or model not available. Please check your data source.")
        return
    
    st.write("Enter customer information to predict churn probability")
    
    # Create columns for input fields
    col1, col2 = st.columns(2)
    
    # Determine input fields based on available columns in the data
    input_data = {}
    
    # Common telecom data features
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
            input_data['PaymentMethod'] = st.selectbox("Payment Method", [
                'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
            ])
    
    # Add numerical inputs
    col1, col2 = st.columns(2)
    
    with col1:
        if 'MonthlyCharges' in data.columns:
            input_data['MonthlyCharges'] = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=5.0)
    
    with col2:
        if 'TotalCharges' in data.columns:
            # Default value based on tenure × MonthlyCharges
            default_total = input_data.get('tenure', 12) * input_data.get('MonthlyCharges', 70.0)
            input_data['TotalCharges'] = st.number_input("Total Charges ($)", 0.0, 10000.0, default_total, step=100.0)
    
    # Predict button
    if st.button("Predict Churn"):
        # Make prediction
        prediction, probability = make_prediction(model, input_data)
        
        # Display results
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Display prediction
            if prediction == 1:
                st.error("⚠️ Customer is likely to churn")
            else:
                st.success("✅ Customer is likely to stay")
        
        with col2:
            # Display probability
            st.metric("Churn Probability", f"{probability*100:.2f}%")
        
        # Display customer risk level
        risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low"
        risk_color = "red" if risk_level == "High" else "orange" if risk_level == "Medium" else "green"
        
        st.markdown(f"<h3 style='color:{risk_color}'>Risk Level: {risk_level}</h3>", unsafe_allow_html=True)
        
        # Recommendations based on prediction
        st.subheader("Recommendations")
        
        if prediction == 1:
            st.write("Consider the following actions to retain this customer:")
            
            recommendations = []
            
            # Provide targeted recommendations based on input features
            if input_data.get('Contract') == 'Month-to-month':
                recommendations.append("Offer a discount on a long-term contract")
            
            if input_data.get('InternetService') == 'Fiber optic' and (
                input_data.get('OnlineSecurity') == 'No' or 
                input_data.get('OnlineBackup') == 'No' or
                input_data.get('TechSupport') == 'No'
            ):
                recommendations.append("Offer a security and tech support bundle at a discount")
            
            if float(input_data.get('MonthlyCharges', 0)) > 80:
                recommendations.append("Provide a personalized discount or loyalty reward")
            
            if input_data.get('tenure', 0) < 12:
                recommendations.append("Reach out for a satisfaction survey and address concerns")
            
            if not recommendations:
                recommendations = [
                    "Offer a customized retention package",
                    "Provide additional value-added services",
                    "Schedule a follow-up call with a customer service representative",
                    "Consider offering a loyalty discount"
                ]
            
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.write("This customer appears stable, but consider these engagement strategies:")
            st.write("1. Offer upgrades or new services that complement current usage")
            st.write("2. Enroll in a loyalty rewards program")
            st.write("3. Request referrals with incentives")

def display_data_analysis(data):
    st.header("Data Analysis")
    
    if data is None:
        st.warning("No data available. Please check your data source.")
        return
    
    # Show data overview
    with st.expander("Data Overview"):
        st.write(f"Dataset contains {data.shape[0]} customers and {data.shape[1]} features")
        st.dataframe(data.head())
        
        # Display data types
        st.subheader("Data Types")
        st.write(data.dtypes)
    
    # Data distribution
    st.subheader("Data Distribution")
    
    # Select numeric columns for histograms
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Filter out the target variable and any ID columns
    exclude_cols = ['Churn', 'customerID', 'CustomerID']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if numeric_cols:
        selected_col = st.selectbox("Select Feature for Distribution Analysis", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig, ax = plt.subplots()
            sns.histplot(data=data, x=selected_col, hue='Churn', multiple='stack', ax=ax)
            ax.set_title(f"Distribution of {selected_col}")
            st.pyplot(fig)
        
        with col2:
            # Box plot
            fig, ax = plt.subplots()
            sns.boxplot(data=data, x='Churn', y=selected_col, ax=ax)
            ax.set_title(f"{selected_col} by Churn Status")
            st.pyplot(fig)
    
    # Categorical analysis
    st.subheader("Categorical Features Analysis")
    
    # Select categorical columns
    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = [col for col in cat_cols if col not in ['customerID', 'CustomerID']]
    
    if cat_cols:
        selected_cat = st.selectbox("Select Categorical Feature", cat_cols)
        
        # Calculate churn rate by category
        cat_churn = data.groupby(selected_cat)['Churn'].mean().sort_values(ascending=False).reset_index()
        cat_churn['Churn'] = cat_churn['Churn'] * 100
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = sns.barplot(data=cat_churn, x=selected_cat, y='Churn', ax=ax)
        ax.set_title(f"Churn Rate by {selected_cat}")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on top of bars
        for bar in bars.patches:
            bars.annotate(f"{bar.get_height():.1f}%",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha='center', va='bottom',
                        xytext=(0, 5),
                        textcoords='offset points')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    
    # Select numerical columns for correlation
    corr_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if len(corr_cols) > 1:  # Need at least 2 numeric columns for correlation
        corr_matrix = data[corr_cols].corr()
        
        # Plot correlation matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)

def display_model_details(data, model):
    st.header("Model Performance and Details")
    
    if data is None or model is None:
        st.warning("Data or model not available. Please check your data source.")
        return
    
    # Split data into features and target
    X = data.drop(columns=['Churn', 'customerID', 'CustomerID'], errors='ignore')
    y = data['Churn']
    
    # Predict on the dataset to show performance
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
