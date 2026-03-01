import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import os

def detect_anomalies(df_processed, eps=0.5, min_samples=5):
    """
    Fits DBSCAN on processed data and returns labels.
    """
    features = df_processed.select_dtypes(include=[np.number])
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features)
    return labels

def clean_and_format_records(df_raw, anomaly_indices):
    """
    Extracts, cleans, and formats anomalous records from the raw dataset.
    """
    if len(anomaly_indices) == 0:
        return []

    from preprocess import process_transaction
    
    df_anomalies = df_raw.iloc[anomaly_indices].copy()
    
    # Format date
    df_anomalies['Txn Date'] = pd.to_datetime(df_anomalies['Txn Date'], dayfirst=True).dt.strftime('%Y-%m-%d')
    
    # Add payment_type and category
    source_col = 'Description' if 'Description' in df_anomalies.columns else 'Transaction Reference'
    df_anomalies[['payment_type', 'category']] = df_anomalies.apply(
        lambda x: process_transaction(x[source_col]), axis=1, result_type='expand'
    )
    
    return df_anomalies.to_dict(orient='records')

def run_dbscan_detection(df_processed, df_raw=None, eps=0.5, min_samples=5):
    """
    Fits DBSCAN and returns cleaned records.
    """
    labels = detect_anomalies(df_processed, eps=eps, min_samples=min_samples)
    
    df_result = df_processed.copy()
    df_result['cluster_label'] = labels
    df_result['is_anomaly'] = (df_result['cluster_label'] == -1).astype(int)
    
    anomaly_indices = df_result[df_result['is_anomaly'] == 1].index
    cleaned_records = []
    if df_raw is not None:
        cleaned_records = clean_and_format_records(df_raw, anomaly_indices)
        
    return df_result, cleaned_records

if __name__ == "__main__":
    data_path = 'data/ProcessedDataset.csv'
    raw_path = 'data/RawDataset.csv'
    if os.path.exists(data_path) and os.path.exists(raw_path):
        df_p = pd.read_csv(data_path)
        df_r = pd.read_csv(raw_path)
        res, records = run_dbscan_detection(df_p, df_r, eps=0.3, min_samples=10)
        print(f"Detected {len(records)} anomalies.")
