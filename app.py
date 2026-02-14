"""
Streamlit App for Multiple ML Classification Models
This application hosts multiple machine learning models for classification tasks.
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Biswajit's ML Classification Models Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay

# --- Preprocessing to match training pipeline ---
FREQUENCY_ORDER = ['never', 'less1', '1~3', '4~8', 'gt8']
AGE_ORDER = ['below21', '21', '26', '31', '36', '41', '46', '50plus']
INCOME_ORDER = [
    'Less than $12500', '$12500 - $24999', '$25000 - $37499',
    '$37500 - $49999', '$50000 - $62499', '$62500 - $74999',
    '$75000 - $87499', '$87500 - $99999', '$100000 or More'
]
EXPIRATION_ORDER = ['2h', '1d']
TIME_MAP = {'7AM': 7, '10AM': 10, '2PM': 14, '6PM': 18, '10PM': 22}
FREQUENCY_COLS = ['Bar', 'CoffeeHouse', 'CarryAway', 'RestaurantLessThan20', 'Restaurant20To50']

def preprocess_input(df):
    """Apply the same ordinal encoding and feature engineering used during training."""
    df = df.copy()
    # Drop 'car' column if present and mostly null
    if 'car' in df.columns and df['car'].isnull().mean() > 0.9:
        df = df.drop(columns=['car'])
    # Ordinal encode frequency columns
    freq_map = {v: i for i, v in enumerate(FREQUENCY_ORDER)}
    for col in FREQUENCY_COLS:
        if col in df.columns:
            df[col] = df[col].map(freq_map)
    # Ordinal encode age
    age_map = {v: i for i, v in enumerate(AGE_ORDER)}
    if 'age' in df.columns:
        df['age'] = df['age'].map(age_map)
    # Ordinal encode income
    income_map = {v: i for i, v in enumerate(INCOME_ORDER)}
    if 'income' in df.columns:
        df['income'] = df['income'].map(income_map)
    # Ordinal encode expiration
    exp_map = {v: i for i, v in enumerate(EXPIRATION_ORDER)}
    if 'expiration' in df.columns:
        df['expiration'] = df['expiration'].map(exp_map)
    # Map time to numeric hour
    if 'time' in df.columns:
        df['time'] = df['time'].map(TIME_MAP)
    # Feature engineering
    freq_available = [c for c in FREQUENCY_COLS if c in df.columns]
    if freq_available:
        df['total_visits'] = df[freq_available].sum(axis=1)
    if 'passanger' in df.columns:
        df['is_alone'] = (df['passanger'] == 'Alone').astype(int)
    return df

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .model-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("Biswajit's ML Classification Models Hub")
st.markdown("### Deploy and test multiple classification models in one place")

# Sidebar for model selection
st.sidebar.header("Model Selection")
st.sidebar.markdown("---")

# Define available models
MODELS = {
    "Logistic Regression": {
        "description": "Baseline logistic regression model with standard scaling and one-hot encoding.",
        "path": "model/logistic_regression_pipeline.pkl",
        "metrics_path": "model/logistic_regression_pipeline_metrics.json"
    },
    "K-Nearest Neighbors": {
        "description": "KNN classifier (K=20, distance-weighted, Manhattan metric).",
        "path": "model/knn_pipeline.pkl",
        "metrics_path": "model/knn_pipeline_metrics.json"
    },
    "Decision Tree": {
        "description": "Decision Tree classifier with tuned depth and split parameters.",
        "path": "model/decision_tree_pipeline.pkl",
        "metrics_path": "model/decision_tree_pipeline_metrics.json"
    },
    "Naive Bayes": {
        "description": "Gaussian Naive Bayes classifier with standard preprocessing.",
        "path": "model/naive_bayes_pipeline.pkl",
        "metrics_path": "model/naive_bayes_pipeline_metrics.json"
    },
    "Random Forest": {
        "description": "Random Forest ensemble (200 trees, max_depth=15, sqrt features).",
        "path": "model/random_forest_pipeline.pkl",
        "metrics_path": "model/random_forest_pipeline_metrics.json"
    },
    "XGBoost": {
        "description": "XGBoost gradient boosting classifier (200 rounds, lr=0.1).",
        "path": "model/xgboost_pipeline.pkl",
        "metrics_path": "model/xgboost_pipeline_metrics.json"
    },
}

selected_model = st.sidebar.selectbox(
    "Choose a classification model:",
    list(MODELS.keys())
)

st.sidebar.markdown("---")
st.sidebar.info(f"**Selected Model:** {selected_model}")
st.sidebar.markdown(f"_{MODELS[selected_model]['description']}_")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"## {selected_model}")
    st.markdown("---")
    
    # File upload section
    st.subheader("📁 Upload Data")
    
    # Check if test dataset exists and offer download
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_dataset.csv')
    if os.path.exists(test_data_path):
        with open(test_data_path, 'rb') as f:
            st.download_button(
                label="📥 Download Sample Test Data (csv)",
                data=f,
                file_name="test_dataset.csv",
                mime="text/csv",
                help="Download the test dataset to try out the models"
            )

    uploaded_file = st.file_uploader(
        "Upload your CSV file for prediction",
        type=['csv'],
        help="Upload a CSV file containing the features for classification"
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
        
        with st.expander("👀 Preview Data"):
            st.dataframe(df.head(10), width='content')
        
        # Prediction button
        if st.button("🚀 Run Prediction"):
            with st.spinner("Making predictions..."):
                try:
                    model_path = os.path.join(os.path.dirname(__file__), MODELS[selected_model]['path'])
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Ensure input data matches model requirements
                    # Apply the same preprocessing used during training
                    df_processed = preprocess_input(df)
                    # Drop target column if present (model was trained on features only)
                    if 'Y' in df_processed.columns:
                        df_processed = df_processed.drop(columns=['Y'])
                    
                    predictions = model.predict(df_processed)
                    prediction_proba = model.predict_proba(df_processed)
                    
                    st.success("Predictions completed!")
                    
                    # Add predictions to dataframe
                    results_df = df.copy()
                    results_df['Predicted_Y'] = predictions
                    results_df['Probability_0'] = prediction_proba[:, 0]
                    results_df['Probability_1'] = prediction_proba[:, 1]
                    
                    # Calculate Testing Metrics if ground truth 'Y' is available
                    if 'Y' in df.columns:
                        st.markdown("---")
                        st.subheader("🧪 New Data Evaluation")
                        
                        y_true = df['Y']
                        y_pred = predictions
                        # Use probability of class 1 for AUC
                        y_proba = prediction_proba[:, 1]
                        
                        try:
                            test_accuracy = accuracy_score(y_true, y_pred)
                            test_precision = precision_score(y_true, y_pred, average='weighted')
                            test_recall = recall_score(y_true, y_pred, average='weighted')
                            test_f1 = f1_score(y_true, y_pred, average='weighted')
                            try:
                                test_auc = roc_auc_score(y_true, y_proba)
                            except ValueError:
                                test_auc = 0.0
                            test_mcc = matthews_corrcoef(y_true, y_pred)
                            
                            m_col1, m_col2 = st.columns(2)
                            with m_col1:
                                st.metric("Test Accuracy", f"{test_accuracy:.2%}")
                                st.metric("Test Recall", f"{test_recall:.2%}")
                                st.metric("Test AUC Score", f"{test_auc:.2f}")
                            with m_col2:
                                st.metric("Test Precision", f"{test_precision:.2%}")
                                st.metric("Test F1 Score", f"{test_f1:.2%}")
                                st.metric("Test MCC Score", f"{test_mcc:.2f}")
                            
                            # Generate confusion matrix for the right column
                            cm = confusion_matrix(y_true, y_pred)
                            fig, ax = plt.subplots(figsize=(5, 4))
                            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Accepted', 'Accepted'])
                            disp.plot(ax=ax, cmap='Blues', values_format='d')
                            ax.set_title(f'{selected_model} - Confusion Matrix', fontsize=12, fontweight='bold')
                            fig.tight_layout()
                            st.session_state['confusion_matrix_fig'] = fig
                                
                        except Exception as e:
                            st.warning(f"Could not calculate some metrics: {e}")

                    st.markdown("### 📋 Prediction Results")
                    # Show first few original columns + predictions
                    display_cols = list(df.columns[:8]) + ['Predicted_Y', 'Probability_1']
                    st.dataframe(results_df[display_cols].head(10))
                    
                    # Download results
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"predictions_{selected_model.replace(' ', '_')}.csv",
                        mime="text/csv",
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")
                    st.error("Please ensure your uploaded file has the correct columns.")
    
    else:
        st.info("👆 Please upload a CSV file to get started")

with col2:
    # Confusion matrix will be shown here after prediction if Y column exists
    if 'confusion_matrix_fig' in st.session_state:
        st.subheader("📊 Confusion Matrix")
        st.pyplot(st.session_state['confusion_matrix_fig'])
    
    
    
    st.subheader("📊 Baseline Training Performance")
    # Load metrics from JSON file
    metrics_path = os.path.join(os.path.dirname(__file__), MODELS[selected_model]['metrics_path'])
    m = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            m = json.load(f)
        
        # Train metrics
        train_acc = f"{m.get('train_accuracy', 0):.2%}"
        train_prec = f"{m.get('train_precision', 0):.2%}"
        train_rec = f"{m.get('train_recall', 0):.2%}"
        train_f1 = f"{m.get('train_f1_score', 0):.2%}"
        train_auc = f"{m.get('train_auc_score', 0):.2f}"
        train_mcc = f"{m.get('train_mcc_score', 0):.2f}"
    else:
        train_acc = train_prec = train_rec = train_f1 = train_auc = train_mcc = "N/A"

    # Display metrics
    def metric_row(label, train_val):
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(f"**{label}**")
        with c2:
            st.markdown(f"`{train_val}`")
        st.markdown("---")

    metric_row("Accuracy", train_acc)
    metric_row("Precision (Weighted)", train_prec)
    metric_row("Recall (Weighted)", train_rec)
    metric_row("F1 Score (Weighted)", train_f1)
    metric_row("AUC Score", train_auc)
    metric_row("MCC Score", train_mcc)




# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "ML Classification Hub"
    "</div>",
    unsafe_allow_html=True
)
