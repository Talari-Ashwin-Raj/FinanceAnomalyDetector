import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import os

def detect_anomalies(df_processed, eps=0.5, min_samples=5):
    """
    Fits DBSCAN on processed data and returns labels and anomaly indicators.
    Assumes features are already preprocessed and scaled to [0, 1].
    """
    # Select numeric columns for clustering
    features = df_processed.select_dtypes(include=[np.number])
    
    # Initialize and fit DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = dbscan.fit_predict(features)
    
    return labels

def run_dbscan_detection(eps=0.5, min_samples=5):
    """
    CLI wrapper for anomaly detection.
    """
    data_path = 'data/ProcessedDataset.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run preprocessing.py first.")
        return None

    df = pd.read_csv(data_path)
    labels = detect_anomalies(df, eps=eps, min_samples=min_samples)
    
    df['cluster_label'] = labels
    df['is_anomaly'] = (df['cluster_label'] == -1).astype(int)
    
    # Summary Statistics
    total_samples = len(df)
    n_anomalies = df['is_anomaly'].sum()
    unique_clusters = set(labels) - {-1}
    n_clusters = len(unique_clusters)
    anomaly_percentage = (n_anomalies / total_samples) * 100
    
    print("--- DBSCAN Anomaly Detection Summary ---")
    print(f"Total Samples:        {total_samples}")
    print(f"Clusters Formed:      {n_clusters}")
    print(f"Anomalies Detected:   {n_anomalies}")
    print(f"Anomaly Percentage:   {anomaly_percentage:.2f}%")
    
    return df

if __name__ == "__main__":
    processed_df = run_dbscan_detection(eps=0.3, min_samples=10)
    if processed_df is not None:
        output_path = 'data/AnomaliesDetected.csv'
        processed_df.to_csv(output_path, index=False)
        print(f"\nAnomaly detection complete. Saved to {output_path}")
