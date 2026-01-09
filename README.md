# Loan Default Prediction System

This project predicts whether a loan applicant is likely to default using machine learning.

## Prerequisites

- **Python 3.10, 3.11, or 3.12 (64-bit)** is required.
    - *Note: Python 3.13 (32-bit) is currently not supported by many data science libraries.*

## Setup Instructions

1.  **Clone/Open the project** in your terminal.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download Data**:
    ```bash
    python download_data.py
    ```
    *(If the script fails, download `Loan_data.csv` manually and place it in the `data/` folder)*

## How to Run

Run the Streamlit application:
```bash
streamlit run app.py
```

## Application Flow

1.  **Data Overview**: Load the raw dataset and view basic statistics.
2.  **EDA**: visualize distributions and correlations to understand the data.
3.  **Model Training**: Click "Train Models" to train Logistic Regression, Random Forest, and Gradient Boosting models. The best model is saved automatically.
4.  **Prediction**: Enter customer details in the form to get a real-time prediction (Default vs. Non-Default).
