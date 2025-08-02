"""
Data Preprocessing Module

Handles data cleaning, feature engineering, and preparation for ML models.

Functions:
    - load_data: Load historical sales and inventory data
    - clean_data: Handle missing values and outliers
    - create_features: Generate time-based features and indicators
    - normalize_data: Scale features for ML models
"""

import pandas as pd
import numpy as np
from typing import Dict, List

def load_data(filepath: str) -> pd.DataFrame:
    """Load raw inventory data from source."""
    try:
        data = pd.read_csv(filepath)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        return pd.DataFrame()
    pass

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data by handling missing values and outliers."""
    # Fill missing values with forward fill method

    df.fillna(method='ffill', inplace=True)
    # Remove outliers based on z-score
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        df = df[z_scores < 3]  # Keep rows with z-score < 3
    # Reset index after cleaning
    df.reset_index(drop=True, inplace=True)
    return df
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the DataFrame."""
    # Fill missing values with forward fill method
    df.fillna(method='ffill', inplace=True)
    # Optionally, drop rows with missing values
    df.dropna(inplace=True)
    return df
def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Handle outliers in the DataFrame."""
    # Remove outliers based on z-score
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        df = df[z_scores < 3]
    # Reset index after cleaning
    df.reset_index(drop=True, inplace=True)
    return df
def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Handle duplicate rows in the DataFrame."""
    """Remove duplicate rows from the DataFrame."""
    df.drop_duplicates(inplace=True)
    return df
def handle_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct data types for each column."""
    """Convert columns to appropriate data types."""
    df['date'] = pd.to_datetime(df['date'])
    df['product_id'] = df['product_id'].astype(str)
    df['sales_quantity'] = pd.to_numeric(df['sales_quantity'], errors='coerce')

    # Convert boolean columns to categorical
    boolean_cols = df.select_dtypes(include=['bool']).columns.tolist()
    for col in boolean_cols:
        df[col] = df[col].astype('category')
    # Convert categorical columns to category type
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    return df
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the DataFrame by cleaning and transforming data."""
    """Preprocess the DataFrame by cleaning and transforming data."""
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = handle_duplicates(df)
    df = handle_data_types(df)
    return df
def preprocess_inventory_data(filepath: str) -> pd.DataFrame:
    """Load and preprocess inventory data from a CSV file."""
    df = load_data(filepath)
    if df.empty:
        return df
    df = preprocess_data(df)
    df = create_features(df)
    df = normalize_data(df)
    return df
# Removed duplicate load_inventory_data function as load_data already exists.
def preprocess_inventory_data(filepath: str) -> pd.DataFrame:
    """Preprocess inventory data by cleaning and transforming."""
    df = load_inventory_data(filepath)
    if df.empty:
        return df
    df = clean_data(df)
    df = create_features(df)
    df = normalize_data(df)
    return df

def load_inventory_data(filepath: str) -> pd.DataFrame:
    """Load inventory data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        return pd.DataFrame()
def preprocess_inventory_data(filepath: str) -> pd.DataFrame:
    """Preprocess inventory data by cleaning and transforming."""
    df = load_inventory_data(filepath)
    if df.empty:
        return df
    df = clean_data(df)
    df = create_features(df)
    df = normalize_data(df)
    return df
def load_inventory_data(filepath: str) -> pd.DataFrame:
    """Load inventory data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        return pd.DataFrame()


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features and indicators."""
    """Generate time-based features and indicators."""
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'] >= 5  # Saturday and Sunday
    # Create lag features for sales quantity
    df['sales_lag_1'] = df.groupby('product_id')['sales_quantity'].shift(1)
    # Fill NaN values created by lagging
    df.fillna(0, inplace=True)
    return df

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize features for ML processing."""
    """Scale numerical features to a range of 0 to 1."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df