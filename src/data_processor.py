import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(filepath, nrows=None):
    """
    Load data from CSV file.
    
    Args:
        filepath (str): Path to the CSV file.
        nrows (int, optional): Number of rows to read. Useful for large datasets.
    """
    try:
        if nrows:
            df = pd.read_csv(filepath, nrows=nrows)
        else:
            df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """
    Perform basic data cleaning:
    - Handle missing values (imputation)
    - Remove duplicates
    """
    if df is None:
        return None
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Simple imputation for numerical columns (median) and categorical (mode) - simplified
    # This will be refined once we see the actual data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    return df

def get_basic_metrics(df):
    """
    Return basic metrics about the dataset.
    """
    if df is None:
        return {}
    
    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": df.isnull().sum().sum(),
        "Duplicate Rows": df.duplicated().sum() # Should be 0 after cleaning
    }

def plot_distribution(df, column):
    """
    Plot distribution of a numerical column using Plotly.
    """
    fig = px.histogram(df, x=column, title=f"Distribution of {column}")
    return fig

def plot_correlation(df):
    """
    Plot correlation matrix using Plotly.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
    return fig
