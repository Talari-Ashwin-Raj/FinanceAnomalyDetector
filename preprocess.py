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

    # Initial Feature Extraction
    source_col = 'Description' if 'Description' in df.columns else 'Transaction Reference'
    df[['payment_type', 'category']] = df.apply(lambda x: process_transaction(x[source_col]), axis=1, result_type='expand')

    # Drop non-feature columns
    df.drop(columns=['Txn Date', 'Value Date', 'Description', 'Transaction Reference', 'Ref No./Cheque No.', 'Balance', 'day', 'month', 'year'], errors='ignore', inplace=True)
    df.fillna(0, inplace=True)
    
    # Binarize Nominal Features
    df = pd.get_dummies(df, columns=['payment_type', 'category'], dtype=int)

    # Scale numeric columns
    for col in ['Debit', 'Credit']:
        if col in df.columns:
            scaler = MinMaxScaler()
            df[[col]] = scaler.fit_transform(df[[col]])

    # Validation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in list(numeric_cols):
        c_min, c_max = df[col].min(), df[col].max()
        if not (0 <= c_min <= 1 and 0 <= c_max <= 1):
            df.drop(columns=[col], inplace=True)

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
    df_processed.to_csv(output_path, index=False)
    print(f"\nPreprocessing complete. Saved to {output_path}")

if __name__ == "__main__":
    run_preprocessing()
