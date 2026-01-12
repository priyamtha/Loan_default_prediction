# LOAN_DEFAULT_PREDICTION

1) Overview of the Application:
The Loan Default Prediction App is a Streamlit-based web application that predicts the likelihood of a loan default using a trained machine learning model and user input data.

2) Menu Navigation:
The app sidebar allows seamless navigation across three main sections: Data, EDA - Visual, and Prediction, giving users control over exploration and prediction workflows.

3) Model Integration:
The app loads a pre-trained machine learning model (model.pkl), a scaler (scaler.pkl), feature list (features.pkl), and performance metrics (metrics.csv) for real-time inference.

4) Data Exploration Tab:
In the Data tab, a sample of the dataset is displayed, alongside key evaluation metrics such as Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

5) EDA - Visualizations:
The EDA - Visual tab provides interactive plots for both categorical and numerical features, allowing visual exploration of distributions and feature relationships.

6) Correlation Heatmap:
A correlation heatmap built using Plotly shows the relationship between numeric features, aiding feature selection and understanding model inputs.

7) Dynamic Input Interface:
The Prediction tab allows users to input borrower details like Age, Annual Income Range, Employment Status, and Credit Score using intuitive widgets (st.number_input, st.selectbox).

8) Live Prediction and Scoring:
Based on user inputs, the model predicts whether a borrower is likely to default or not, and calculates a Derived Risk Score between 0 and 1.

9) Risk Score Interpretation:
The derived risk score is classified into categories:

🟢 Low Risk: score < 0.3

🟠 Medium Risk: score between 0.3 and 0.7

🔴 High Risk: score > 0.7

10) Preprocessing Pipeline:
The app applies the same preprocessing steps as used in model training—scaling numeric features and one-hot encoding categorical features—to ensure consistent prediction results.

11) Debug Info Display:
A debug section shows the raw prediction, probability, and input values in JSON format, helping with validation and transparency during testing.

12) Efficient Data Loading:
To avoid memory issues with large datasets, the app loads a sample (e.g., 10,000 rows) from the full dataset using pd.read_csv(nrows=10000) with Streamlit caching.

13) Robust Error Handling:
Try-except blocks are used throughout to handle exceptions (e.g., file not found, invalid input), and errors are clearly shown using st.error().

14) Caching for Performance:
Streamlit’s @st.cache_data and @st.cache_resource decorators are used to cache expensive operations like model and data loading, improving performance.

15) Trained ML Models:
The application supports any trained model compatible with scikit-learn (e.g., Random Forest, XGBoost), with a target of evaluation metrics ≥ 0.87.

16) Feature Importance Support:
Feature importance data generated during training can be visualized to help users understand the most influential features affecting prediction.

17) Security and Privacy:
No data is stored or persisted—user inputs are used only during the session for prediction, ensuring privacy and security of sensitive information.

18) Modular and Maintainable Design:
The code is organized into logical components for loading models, preprocessing data, predicting outcomes, and displaying visualizations—ensuring scalability and maintainability.

19) Model Evaluation Metrics Display:
The app reads a metrics.csv file and displays all key evaluation metrics, providing transparency into the performance of the underlying model.

20) Instructions to Run the Project:

Install dependencies: pip install -r requirements.txt

Ensure model files (model.pkl, scaler.pkl, features.pkl, metrics.csv) are present

Run the app: streamlit run app.py

Use the sidebar to explore data, visualize insights, and predict loan defaults
