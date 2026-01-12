import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.preprocessing import StandardScaler

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    return model, scaler, features

@st.cache_data
def load_data():
    df = pd.read_csv("loan_data.csv", nrows=10000)  
    return df


def preprocess_input(user_input, scaler, features):
    input_df = pd.DataFrame([user_input])
    input_encoded = pd.get_dummies(input_df)

    for col in features:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[features]

    scaled_input = scaler.transform(input_encoded)
    return scaled_input


def make_prediction(model, input_array):
    prediction = model.predict(input_array)[0]
    proba = model.predict_proba(input_array)[0][1]
    return prediction, proba


st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title("🏦 Loan Default Prediction App")

model, scaler, features = load_artifacts()
df = load_data()

st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Data", "EDA - Visual", "Prediction"])


if page == "Data":
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(50))

    st.subheader("📊 Model Performance Metrics")
    try:
        metrics_df = pd.read_csv("metrics.csv")
        st.dataframe(metrics_df)
    except FileNotFoundError:
        st.warning("⚠️ metrics.csv not found. Train the model first.")


elif page == "EDA - Visual":
    st.subheader("📊 Exploratory Data Analysis")

    st.markdown("### 🧮 Target Class Distribution")
    target_counts = df["TARGET"].value_counts().reset_index()
    target_counts.columns = ["Class", "Count"]
    fig_pie = px.pie(target_counts, values='Count', names='Class',
                     title='Loan Default Class Distribution',
                     color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("### 📌 Select a column to visualize")
    col = st.selectbox("Choose a column", df.columns)

    if df[col].dtype == 'object' or df[col].nunique() < 20:
        fig_bar = px.histogram(df, x=col, color="TARGET", barmode="group",
                               title=f"Distribution of {col} by Target",
                               color_discrete_sequence=["green", "red"])
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        fig_box = px.box(df, x="TARGET", y=col, color="TARGET",
                         points="all", title=f"{col} vs TARGET",
                         color_discrete_sequence=["green", "red"])
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("### 📈 Show Summary Statistics")
    if st.checkbox("Show describe() output"):
        st.dataframe(df.describe().T)


elif page == "Prediction":
    st.subheader("🧠 Make a Prediction")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)

    income_range = st.selectbox("Annual Income Range", [
        "Below 1 Lakh",
        "1 Lakh - 5 Lakhs",
        "5 Lakhs - 10 Lakhs",
        "10 Lakhs - 25 Lakhs",
        "25 Lakhs - 50 Lakhs",
        "50 Lakhs - 1 Crore",
        "Above 1 Crore"
    ])
    income_mapping = {
        "Below 1 Lakh": 50_000,
        "1 Lakh - 5 Lakhs": 300_000,
        "5 Lakhs - 10 Lakhs": 750_000,
        "10 Lakhs - 25 Lakhs": 1_750_000,
        "25 Lakhs - 50 Lakhs": 3_750_000,
        "50 Lakhs - 1 Crore": 7_500_000,
        "Above 1 Crore": 12_000_000,
    }
    income = income_mapping[income_range]

    employment_status = st.selectbox("Employment Status", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

    input_dict = {
        "AGE": age,
        "ANNUAL_INCOME": income,
        "CREDIT_SCORE": credit_score,
    }

    emp_options = ["Full-time", "Part-time", "Self-employed", "Unemployed"]
    for emp in emp_options:
        col_name = f"EMPLOYMENT_STATUS_{emp}"
        input_dict[col_name] = 1 if employment_status == emp else 0

    if st.button("Predict"):
        try:
            input_array = preprocess_input(input_dict, scaler, features)
            prediction, risk_score = make_prediction(model, input_array)

            result = "🟢 Not Defaulted" if prediction == 0 else "🔴 Defaulted"
            st.success(f"**Prediction:** {result}")
            st.info(f"**Derived Risk Score:** {risk_score:.5f}")

            st.markdown("### 📉 Risk Probability")
            st.progress(risk_score)

            if risk_score < 0.3:
                st.success("🟢 Low Risk")
            elif 0.3 <= risk_score < 0.7:
                st.warning("🟠 Medium Risk")
            else:
                st.error("🔴 High Risk")

            st.write("🧪 Debug Info")
            st.json({
                "Prediction": int(prediction),
                "Risk Score (Raw)": float(risk_score),
                "User Input": input_dict
            })

        except Exception as e:
            st.error(f"Prediction error: {e}")