import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def process_transaction(transaction_ref):
    """
    Parses transaction reference strings to extract payment type and merchant category.
    """
    parts = str(transaction_ref).split('/')
    payment_type = parts[0].lower() if len(parts) > 0 else 'unknown'
    merchant = parts[3].lower().strip() if len(parts) > 3 else ''
    
    category = "others"
    if merchant:
        if any(kw in merchant for kw in ['swiggy', 'zomato', 'restaurant', 'pizza', 'kfc', 'mcdonalds']): category = "food"
        elif any(kw in merchant for kw in ['netflix', 'spotify', 'prime', 'hotstar', 'multiplex', 'cinema']): category = "entertainment"
        elif any(kw in merchant for kw in ['udemy', 'coursera', 'byjus', 'edx', 'school', 'college']): category = "education"
        elif any(kw in merchant for kw in ['uber', 'ola', 'rapido', 'irctc', 'indigo', 'airindia', 'railway']): category = "travel"
        elif any(kw in merchant for kw in ['amazon', 'flipkart', 'myntra', 'dmart', 'bigbazaar', 'ajio', 'reliance']): category = "shopping"
        elif any(kw in merchant for kw in ['electricity', 'water', 'gas', 'bill', 'bescom', 'recharge']): category = "utilities"
        elif any(kw in merchant for kw in ['salary', 'payroll', 'stipend']): category = "salary"
        elif any(kw in merchant for kw in ['apollo', 'pharmeasy', 'hospital', 'medical', 'pharmacy', 'health']): category = "medical"
        elif any(kw in merchant for kw in ['jio', 'vi', 'airtel', 'bsnl', 'telecom']): category = "recharge"
    return payment_type, category

def preprocess_data(df_input):
    """
    Core preprocessing logic applied to a DataFrame.
    """
    df = df_input.copy()

    # Feature Extraction (Conditional)
    source_col = 'Description' if 'Description' in df.columns else 'Transaction Reference'
    if source_col in df.columns:
        df[['payment_type', 'category']] = df.apply(lambda x: process_transaction(x[source_col]), axis=1, result_type='expand')

    # Parse dates and extract time features if Txn Date is present
    if 'Txn Date' in df.columns:
        df['Txn Date'] = pd.to_datetime(df['Txn Date'], dayfirst=True)
        if 'day' not in df.columns: df['day'] = df['Txn Date'].dt.day
        if 'month' not in df.columns: df['month'] = df['Txn Date'].dt.month

    # Binarize Nominal Features if they were extracted
    nominal_cols = [c for c in ['payment_type', 'category'] if c in df.columns]
    if nominal_cols:
        df = pd.get_dummies(df, columns=nominal_cols, dtype=int)

    # Automatically identify numeric columns for scaling
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Scale ALL numeric columns to [0, 1]
    for col in numeric_cols:
        scaler = MinMaxScaler()
        df[[col]] = scaler.fit_transform(df[[col]])

    # Drop strictly non-numeric or non-date columns remaining (except Txn Date if present)
    cols_to_keep = numeric_cols + (['Txn Date'] if 'Txn Date' in df.columns else [])
    df = df[cols_to_keep]
    
    df.fillna(0, inplace=True)

    return df

def run_preprocessing():
    # Load dataset
    data_path = 'data/RawDataset.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    df_raw = pd.read_csv(data_path)
    df_processed = preprocess_data(df_raw)

    # Save processed data
    output_path = 'data/ProcessedDataset.csv'
    df_save = df_processed.drop(columns=['Txn Date'], errors='ignore')
    df_save.to_csv(output_path, index=False)
    print(f"\nPreprocessing complete. Saved to {output_path}")

if __name__ == "__main__":
    run_preprocessing()
