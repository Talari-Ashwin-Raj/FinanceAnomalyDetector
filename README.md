# Finance Anomaly Detector
https://financeanomalydetector-7rmh8mnmb6p3d29sdvcech.streamlit.app/

demolink:https://drive.google.com/file/d/1l1D9aP6pl2j5Umuwj0_VSCHX6siSg3XH/view?usp=sharing
ppt:https://drive.google.com/file/d/1Y-CqbOrrdNz1l0K6eNFZd3RERigZP-oR/view?usp=sharing


This repository contains tools for detecting anomalies in financial transaction data using machine learning.

## Components

- `preprocess.py`: Script for normalizing and preparing transaction data for analysis.
- `model_dbscan.py`: Implementation of the DBSCAN clustering algorithm for anomaly detection.
- `anomaly_detection_pipeline.ipynb`: A comprehensive Jupyter notebook demonstrating the end-to-end pipeline.

## Usage

1. **Preprocessing**: Use `preprocess.py` to clean and scale your raw transaction data.
2. **Modeling**: Run `model_dbscan.py` to identify outliers and suspicious activity.
3. **Exploration**: Open `anomaly_detection_pipeline.ipynb` for a detailed walkthrough of the process.

## Data

Processed datasets and detected anomalies are stored in the `data/` directory.f digital banking, users generate extensive transaction histories across bank accounts, UPI platforms, and credit cards. While banks provide basic transaction summaries, they rarely offer intelligent insights into unusual financial behavior.

Unexpected charges, spending spikes, new merchants, and behavioral shifts often go unnoticed until they cause significant financial impact. Users lack a lightweight, proactive system that highlights irregularities in their personal spending patterns in an explainable and user-friendly manner.

## Target Users  
- Individual banking customers  
- Freelancers and salaried professionals  
- Students managing monthly budgets  
- Fintech enthusiasts  
- Personal finance learners  

## Existing Gaps  
- No unified CSV/PDF ingestion tool  
- Lack of personalized anomaly detection  
- No explainable financial insights  
- No lightweight standalone solution  

---

# 2. Problem Understanding & Approach

## Root Cause Analysis  
- Raw transaction data lacks structure  
- Spending behavior differs per user  
- Static rule systems fail personalization  
- No behavioral baseline modeling  

## Solution Strategy  
- Parse and normalize transaction data  
- Automatically categorize transactions  
- Build personalized spending baseline  
- Apply statistical anomaly detection  
- Generate explainable risk scores  
- Visualize financial patterns  

---

# 3. Proposed Solution

## Solution Overview  
A fully interactive **Streamlit-based application** that allows users to upload transaction files, automatically analyze them, and visualize anomalies in an intuitive dashboard.

## Core Idea  
Convert passive transaction history into proactive financial awareness using statistical modeling and behavioral analytics.

## Key Features  
- CSV & PDF statement upload  
- Automatic categorization  
- Amount anomaly detection (Z-score/MAD)  
- Frequency anomaly detection  
- New merchant detection  
- Behavioral spending shift analysis  
- Composite anomaly risk score  
- Interactive dashboard visualization  
- Explainable anomaly reasoning  

---

# 4. System Architecture

## High-Level Flow  

User → Streamlit UI → Data Processing → ML Model → Database → Streamlit Dashboard

---

## Architecture Description  

User Upload (CSV / PDF)
        ↓
Streamlit App UI
        ↓
Parsing & Cleaning Module
        ↓
Auto-Categorization Engine
        ↓
Feature Engineering Layer
        ↓
Clustering based models
        ↓
Risk Scoring + Explanation
        ↓
Interactive Visual Dashboard



# 5. Dataset Selected

## Dataset Name  
Custom User Transaction Dataset

## Source  
- User-uploaded CSV files  
- Synthetic data for model validation  

## Data Type  
Structured transactional financial data  

## Selection Reason  
Represents real-world personal finance behavior and supports anomaly modeling.

## Preprocessing Steps  
- Date normalization  
- Merchant name cleaning  
- Amount standardization  
- Missing value handling  
- Category encoding  

---

# 6. Model Selected

## Model Name  
Hybrid Statistical + clustering based models (example - DBSCAN)

- Z-Score / MAD for amount anomalies  
- Rolling average for frequency anomalies  

## Selection Reasoning  
- Works well without large labeled datasets  
- Handles outliers effectively  
- Suitable for per-user personalization  

## Alternatives Considered  
- One-Class SVM  
- Local Outlier Factor  
- LSTM Time-Series Models  
- K - means
- Knn  

## Evaluation Metrics  
- Precision  
- Recall  
- F1 Score  
- False Positive Rate   

---

# 7. Technology Stack

## Frontend + Backend  
- Streamlit  

## Data Processing  
- Pandas  
- NumPy  

## ML/AI  
- Scikit-learn  
- DBSCAN 
- Statistical models  

## Deployment  
- Streamlit Cloud / AWS / Render   

---

# 8. Application Modules

- File Upload Module  
- Data Preprocessing Module  
- Categorization Engine  
- Anomaly Detection Engine  
- Risk Scoring System  
- Visualization Dashboard  

---

# 9. Module-wise Development & Deliverables

## Checkpoint 1: Research & Planning  
- Architecture design  
- Feature planning   

## Checkpoint 2: Data Processing  
- CSV/PDF parsing  
- Cleaning & normalization
- dataset generation

## Checkpoint 3: Model Development  
- Categorization logic  
- Anomaly detection  

## Checkpoint 4: Dashboard Integration  
- Interactive charts  
- Anomaly highlights  

## Checkpoint 5: Streamlit App Setup  
- File upload interface  
- Layout & UI structure

## Checkpoint 6: Deployment  
- Cloud deployment  
- Testing & optimization 

---

# 11. End-to-End Workflow

1. Upload transaction file  
2. Parse & clean data  
3. Categorize transactions  
4. Establish user baseline  
5. Detect anomalies  
6. Compute risk score  
7. Visualize anomalies  
8. Provide explainable insights  

---

# 12. Demo & Video

Live Demo Link: ()  
Demo Video Link: ()  
GitHub Repository: ()  

---

# 13. Hackathon Deliverables Summary

- Streamlit application  
- Transaction parser  
- Categorization engine  
- Anomaly detection system  
- Interactive dashboard  

---

# 14. Team Roles & Responsibilities

| Member Name | Role | Responsibilities |
|-------------|------|-----------------|
| Koyna Arya | ML Engineer | Model & anomaly detection |
| Sai Teja | Developer | Streamlit UI & integration | Deployment
| Talari Ashwin Raj | Data Engineer | Parsing & preprocessing |

---

# 15. Future Scope & Scalability

## Short-Term  
- contexual based anomaly detection   
- Feedback-based model learning  

## Long-Term  
- Real-time monitoring  
- Personal info based detection
- Mobile app version  
- Financial health score  

---

# 16. Known Limitations

- Contexual based anomaly detection 
- PDF parsing variability  
- Early-stage false positives
- bank statement columns will be different for different banks

---

# 17. Impact

- Increased financial awareness  
- Early anomaly detection  
- Reduced financial stress  
- Better budgeting decisions  
