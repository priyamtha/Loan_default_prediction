import streamlit as st
import pandas as pd
import os
import joblib
from src.data_processor import load_data, clean_data, get_basic_metrics, plot_distribution, plot_correlation
from src.train_model import train_models

st.set_page_config(page_title="Loan Default Prediction", layout="wide")

st.title("Loan Default Prediction System")

# Sidebar
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Data Overview", "EDA & Statistics", "Model Training & Evaluation", "Prediction"])

# Load Data
DATA_PATH = 'data/Loan_data.csv'
if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found at {DATA_PATH}. Please run download_data.py.")
    st.stop()

@st.cache_data
def get_data():
    df = load_data(DATA_PATH)
    if df is not None:
        df = clean_data(df)
    return df

df = get_data()

if options == "Data Overview":
    st.header("Dataset Overview")
    if df is not None:
        st.write("### Raw Data (First 100 rows)")
        st.dataframe(df.head(100))
        
        metrics = get_basic_metrics(df)
        st.write("### Dataset Metrics")
        for k, v in metrics.items():
            st.write(f"- **{k}**: {v}")
    else:
        st.error("Could not load data.")

elif options == "EDA & Statistics":
    st.header("Exploratory Data Analysis")
    if df is not None:
        # Distribution Plot
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        selected_col = st.selectbox("Select Column for Distribution", numeric_cols)
        if selected_col:
            fig = plot_distribution(df, selected_col)
            st.plotly_chart(fig)
            
        # Correlation Matrix
        st.write("### Correlation Matrix")
        fig_corr = plot_correlation(df)
        st.plotly_chart(fig_corr)
    else:
        st.error("Could not load data.")

elif options == "Model Training & Evaluation":
    st.header("Model Training & Performance")
    if st.button("Train Models"):
        with st.spinner("Training models..."):
            results = train_models(df)
            if results:
                st.success("Training Complete!")
                res_df = pd.DataFrame(results).T
                st.write("### Model Performance Metrics")
                st.dataframe(res_df.style.highlight_max(axis=0))
                
                # Check if we met the criteria
                st.write("### Success Criteria Check (> 87%)")
                best_model_name = res_df['F1 Score'].idxmax()
                st.write(f"Best Model: **{best_model_name}**")
            else:
                st.error("Training failed.")
    
    if os.path.exists('models/best_model.pkl'):
        st.info("A trained model is available.")

elif options == "Prediction":
    st.header("Loan Default Prediction")
    if os.path.exists('models/best_model.pkl'):
        model = joblib.load('models/best_model.pkl')
        
        # Create input fields for features
        # We need the feature names from the model or the dataframe
        # Ideally, we should save feature names during training
        # For now, we will use columns from df (excluding TARGET)
        
        if df is not None:
            input_data = {}
            feature_cols = df.drop(columns=['TARGET'], errors='ignore').columns
            
            with st.form("prediction_form"):
                st.write("Enter Customer Details:")
                cols = st.columns(3)
                for i, col in enumerate(feature_cols):
                    # Check type
                    if pd.api.types.is_numeric_dtype(df[col]):
                        input_data[col] = cols[i % 3].number_input(col, value=float(df[col].mean()))
                    else:
                        unique_vals = df[col].unique()
                        input_data[col] = cols[i % 3].selectbox(col, unique_vals)
                
                submitted = st.form_submit_button("Predict")
                
                if submitted:
                    input_df = pd.DataFrame([input_data])
                    prediction = model.predict(input_df)[0]
                    proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else 0
                    
                    st.write("---")
                    if prediction == 1:
                        st.error(f"Prediction: **Default** (Probability: {proba:.2f})")
                    else:
                        st.success(f"Prediction: **No Default** (Probability: {proba:.2f})")
    else:
        st.warning("Please train the model first in the 'Model Training' section.")
